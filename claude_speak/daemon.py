#!/usr/bin/env python3
"""
claude-speak daemon — persistent TTS service for Claude Code.

Loads Kokoro once, watches the queue directory, normalizes text,
chunks long messages, and plays audio sequentially.
"""

import asyncio
import fcntl
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import Config, CONFIG_PATH, PID_FILE, TOGGLE_FILE, MUTE_FILE, PLAYING_FILE, LOG_FILE, QUEUE_DIR, load_config
from .normalizer import normalize, chunk_text
from .tts import TTSEngine
from .chimes import play_ready_chime, play_error_chime, play_stop_chime
from . import queue as Q

logger = logging.getLogger(__name__)

POLL_INTERVAL = 0.1  # fallback; SIGUSR1 from hook triggers instant processing
LOCK_FILE = Path("/tmp/claude-speak-daemon.lock")
START_TS_FILE = Path("/tmp/claude-speak-daemon.start_ts")
PROCESS_PATTERN = "claude-speak"
CONFIG_RELOAD_INTERVAL = 30  # seconds between mtime checks


def _is_stop_command(text: str, stop_phrases: list[str]) -> bool:
    """Return True if text consists solely of a stop phrase."""
    stripped = text.strip().lower()
    return stripped in [p.lower() for p in stop_phrases]


def _try_reload_config(config: Config, engine: TTSEngine, last_mtime: float) -> tuple[Config, float]:
    """Check config file mtime and hot-reload if changed. Returns (config, mtime)."""
    try:
        current_mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        return config, last_mtime

    if current_mtime == last_mtime:
        return config, last_mtime

    logger.info("Config file changed, reloading...")
    new_config = load_config()

    # Update engine settings
    engine.config = new_config
    engine._resolve_device()
    logger.info(
        "Reloaded config — voice=%s, speed=%s, device=%s",
        new_config.tts.voice, new_config.tts.speed, new_config.tts.device,
    )
    return new_config, current_mtime


async def _wait_for_unmute(poll: float = 0.2) -> None:
    """Block until the mute file is removed."""
    while MUTE_FILE.exists():
        await asyncio.sleep(poll)


async def run_loop(config: Config, engine: TTSEngine, voice_controller=None):
    """Main loop: watch queue, normalize, chunk, speak."""
    Q.ensure_queue_dir()
    logger.info("Watching queue: %s", QUEUE_DIR)

    # SIGUSR1 from hook signals "new item in queue" — skip poll delay
    queue_ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGUSR1, queue_ready.set)

    # Track config file mtime for hot-reload
    try:
        last_config_mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        last_config_mtime = 0.0
    last_config_check = time.time()

    loop_count = 0
    while True:
        loop_count += 1

        # Hot-reload config every CONFIG_RELOAD_INTERVAL seconds
        now = time.time()
        if now - last_config_check >= CONFIG_RELOAD_INTERVAL:
            config, last_config_mtime = _try_reload_config(config, engine, last_config_mtime)
            last_config_check = now

        # Pause if toggle file removed (but don't exit — user might re-enable)
        if not TOGGLE_FILE.exists():
            await asyncio.sleep(1)
            continue

        item = Q.dequeue()
        if item is None:
            # Wait for SIGUSR1 signal OR fallback poll timeout
            queue_ready.clear()
            try:
                await asyncio.wait_for(queue_ready.wait(), timeout=POLL_INTERVAL)
            except asyncio.TimeoutError:
                pass
            continue

        filepath, text = item
        if not text:
            continue

        # Skip excessively large queue items (e.g. full transcript dumps from
        # stale position tracker). 10K chars ~= 2500 words ~= 15 min of speech.
        MAX_QUEUE_CHARS = 10_000
        if len(text) > MAX_QUEUE_CHARS:
            logger.warning("Skipping oversized queue item (%d chars > %d max)", len(text), MAX_QUEUE_CHARS)
            continue

        # Measure time from queue file creation to pickup
        try:
            queue_ts = float(filepath.stem)
            pickup_delay = time.time() - queue_ts
            logger.debug("queue->pickup: %.3fs", pickup_delay)
        except (ValueError, AttributeError):
            pass

        t0 = time.monotonic()

        # Stop word handling: if text is only a stop phrase, abort playback
        if _is_stop_command(text, config.wakeword.stop_phrases):
            logger.info("Stop command received, aborting playback.")
            engine.stop()
            Q.clear()
            if config.audio.chimes:
                play_stop_chime(device=engine._output_device, volume=config.audio.volume)
            continue

        try:
            logger.info("Processing (%d chars)...", len(text))

            # Normalize for speech
            normalized = normalize(text)
            t_norm = time.monotonic()
            logger.debug("normalize: %.0fms", (t_norm - t0) * 1000)

            if not normalized:
                logger.info("Empty after normalization, skipping.")
                continue

            # Chunk for reliable TTS (Kokoro struggles with very long text)
            chunks = chunk_text(normalized, max_chars=config.tts.max_chunk_chars)
            t_chunk = time.monotonic()
            logger.debug("chunk: %.0fms (%d chunks)", (t_chunk - t_norm) * 1000, len(chunks))

            # Mark as playing so VoiceController knows TTS is active
            PLAYING_FILE.touch()
            # Swap to built-in mic during TTS to avoid BT profile switching on AirPods
            if voice_controller and voice_controller._wakeword_listener:
                voice_controller._wakeword_listener.use_builtin_mic()

            try:
                if len(chunks) == 1:
                    # Single chunk — stream directly (check mute before starting)
                    if TOGGLE_FILE.exists() and not MUTE_FILE.exists():
                        await engine.speak(chunks[0])
                        logger.debug("speak: %.0fms", (time.monotonic() - t_chunk) * 1000)
                    elif MUTE_FILE.exists():
                        # Muted — wait for unmute, then play
                        logger.info("Muted, waiting to resume...")
                        await _wait_for_unmute()
                        logger.info("Resumed.")
                        await engine.speak(chunks[0])
                else:
                    # Multiple chunks — generate next while current plays
                    next_audio = await engine.generate_audio(chunks[0])
                    t_gen = time.monotonic()
                    logger.debug("generate chunk[0]: %.0fms", (t_gen - t_chunk) * 1000)

                    for i in range(len(chunks)):
                        if not TOGGLE_FILE.exists():
                            break

                        # Check mute before each chunk
                        if MUTE_FILE.exists():
                            engine.stop()
                            logger.info("Muted at chunk %d/%d, waiting...", i, len(chunks))
                            await _wait_for_unmute()
                            logger.info("Resumed.")

                        current_audio = next_audio

                        if i + 1 < len(chunks):
                            gen_task = asyncio.create_task(engine.generate_audio(chunks[i + 1]))
                            t_play = time.monotonic()
                            await asyncio.to_thread(engine.play_audio, current_audio)
                            logger.debug("play chunk[%d]: %.0fms", i, (time.monotonic() - t_play) * 1000)
                            next_audio = await gen_task
                            logger.debug("generate chunk[%d]: ready", i + 1)
                        else:
                            await asyncio.to_thread(engine.play_audio, current_audio)
            finally:
                PLAYING_FILE.unlink(missing_ok=True)
                MUTE_FILE.unlink(missing_ok=True)  # prevent mute deadlock if TTS finishes while muted
                # Swap back to default mic after TTS finishes
                if voice_controller and voice_controller._wakeword_listener:
                    voice_controller._wakeword_listener.use_default_mic()

            t_done = time.monotonic()
            logger.debug("TOTAL dequeue->done: %.0fms", (t_done - t0) * 1000)
            logger.info("Done speaking.")

        except Exception as e:
            logger.error("Error processing queue item: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def write_pid():
    PID_FILE.write_text(str(os.getpid()))


def read_pid() -> int | None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None


def acquire_lock() -> bool:
    """Try to acquire an exclusive lock. Returns True if acquired."""
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(LOCK_FILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Keep fd open — lock held for process lifetime
        # Store on module so it doesn't get garbage collected
        acquire_lock._fd = lock_fd
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        return True
    except (OSError, IOError):
        return False


def kill_all():
    """Kill ALL claude-speak daemon processes (nuclear option)."""
    my_pid = os.getpid()
    killed = []

    # Method 1: PID file
    pid = read_pid()
    if pid and pid != my_pid:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)

    # Method 2: pkill by pattern (catches orphans)
    try:
        result = subprocess.run(
            ["pgrep", "-f", PROCESS_PATTERN],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            pid = int(line.strip())
            if pid == my_pid:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except ProcessLookupError:
                pass
    except Exception:
        pass

    # Wait briefly, then force-kill survivors
    if killed:
        time.sleep(0.5)
        for pid in killed:
            try:
                os.kill(pid, 0)  # check if still alive
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # Clean up lock, PID, and start timestamp
    LOCK_FILE.unlink(missing_ok=True)
    PID_FILE.unlink(missing_ok=True)
    START_TS_FILE.unlink(missing_ok=True)
    Q.clear()

    if killed:
        print(f"[daemon] Killed {len(killed)} process(es): {killed}")
    else:
        print("[daemon] No processes found to kill.")


def stop_daemon():
    pid = read_pid()
    if pid:
        print(f"[daemon] Stopping (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait up to 3 seconds for graceful shutdown
            for _ in range(30):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
            else:
                # Still alive — force kill
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        PID_FILE.unlink(missing_ok=True)
        LOCK_FILE.unlink(missing_ok=True)
        START_TS_FILE.unlink(missing_ok=True)
        print("[daemon] Stopped.")
    else:
        print("[daemon] Not running.")


def handle_shutdown(signum, frame):
    logger.info("Shutting down...")
    PID_FILE.unlink(missing_ok=True)
    LOCK_FILE.unlink(missing_ok=True)
    START_TS_FILE.unlink(missing_ok=True)
    MUTE_FILE.unlink(missing_ok=True)
    PLAYING_FILE.unlink(missing_ok=True)
    sys.exit(0)


def status():
    pid = read_pid()
    enabled = TOGGLE_FILE.exists()
    depth = Q.depth()
    print(f"Daemon:  {'running (PID ' + str(pid) + ')' if pid else 'stopped'}")
    print(f"Enabled: {'yes' if enabled else 'no'}")
    print(f"Queue:   {depth} items")


def _configure_logging(config: Config, foreground: bool = False):
    """Set up logging with RotatingFileHandler and optional StreamHandler.

    Args:
        config: Config object (used to read log_level).
        foreground: If True, also log to stderr (for non-daemonized mode).
    """
    # Determine log level from env var (highest priority) or config
    level_name = os.environ.get("CLAUDE_SPEAK_LOG_LEVEL", "") or getattr(config, "log_level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any existing handlers (e.g. from basicConfig or prior calls)
    root.handlers.clear()

    # RotatingFileHandler: 5 MB max, 3 backups
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        str(LOG_FILE),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # StreamHandler for foreground (non-daemonized) mode
    if foreground:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(fmt)
        root.addHandler(stream_handler)


def start(daemonize: bool = False):
    # Check PID file first (fast check)
    existing = read_pid()
    if existing:
        print(f"[daemon] Already running (PID {existing}).")
        sys.exit(0)  # Exit cleanly, not an error

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    if daemonize:
        # Launch daemon as a detached subprocess to avoid fork-related crashes
        # with C libraries (PortAudio, ONNX Runtime, CoreML).
        import subprocess as _sp
        project_root = str(Path(__file__).resolve().parent.parent)
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", "")
        if project_root not in env["PYTHONPATH"]:
            env["PYTHONPATH"] = project_root + ((":" + env["PYTHONPATH"]) if env["PYTHONPATH"] else "")
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        _sp.Popen(
            [sys.executable, "-m", "claude_speak.daemon"],
            stdin=devnull_fd,
            stdout=devnull_fd,
            stderr=devnull_fd,
            start_new_session=True,
            env=env,
        )
        os.close(devnull_fd)
        time.sleep(1.5)
        daemon_pid = read_pid() or "?"
        print(f"[daemon] Started (PID {daemon_pid})", flush=True)
        os._exit(0)

    # Configure logging (must happen after fork so the child owns the handlers)
    config_for_log = load_config()
    _configure_logging(config_for_log, foreground=not daemonize)

    # Acquire exclusive lock AFTER fork (child process holds the lock)
    if not acquire_lock():
        logger.info("Another instance holds the lock. Exiting.")
        sys.exit(0)

    write_pid()
    START_TS_FILE.write_text(str(time.time()))
    Q.ensure_queue_dir()

    config = load_config()
    engine = TTSEngine(config)
    logger.info("Loading TTS engine (models will be auto-downloaded if missing)...")
    engine.load()

    # Session greeting: chime + spoken confirmation
    try:
        if config.audio.chimes:
            play_ready_chime(device=engine._output_device, volume=config.audio.volume)
        if config.audio.greeting:
            import asyncio as _aio
            _aio.run(engine.speak(config.audio.greeting))
        logger.info("Greeting complete.")
    except Exception as e:
        logger.warning("Greeting failed: %s", e)

    # Start voice controller (wake word + voice input) if enabled
    voice_controller = None
    if config.wakeword.enabled:
        try:
            from .voice_controller import VoiceController
            voice_controller = VoiceController(config, tts_stop_callback=engine.stop)
            voice_controller.start()
            logger.info("Voice controller started.")
        except Exception as e:
            logger.error("Voice controller failed to start: %s", e)

    try:
        asyncio.run(run_loop(config, engine, voice_controller))
    except KeyboardInterrupt:
        if voice_controller:
            voice_controller.stop()
        handle_shutdown(None, None)


def main():
    args = sys.argv[1:]

    if "--stop" in args:
        stop_daemon()
    elif "--kill-all" in args:
        kill_all()
    elif "--status" in args:
        status()
    elif "--daemon" in args:
        start(daemonize=True)
    elif "--list-voices" in args:
        config = load_config()
        engine = TTSEngine(config)
        for v in engine.list_voices():
            print(v)
    else:
        start(daemonize=False)


if __name__ == "__main__":
    main()
