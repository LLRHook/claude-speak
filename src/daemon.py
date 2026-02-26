#!/usr/bin/env python3
"""
claude-speak daemon — persistent TTS service for Claude Code.

Loads Kokoro once, watches the queue directory, normalizes text,
chunks long messages, and plays audio sequentially.
"""

import asyncio
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import Config, CONFIG_PATH, PID_FILE, TOGGLE_FILE, LOG_FILE, QUEUE_DIR, load_config
from .normalizer import normalize, chunk_text
from .tts import TTSEngine
from .chimes import play_ready_chime, play_error_chime, play_stop_chime
from . import queue as Q

PERF_LOG = os.environ.get("CLAUDE_SPEAK_PERF", "").lower() in ("1", "true", "yes")
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

    print("[daemon] Config file changed, reloading...", flush=True)
    new_config = load_config()

    # Update engine settings
    engine.config = new_config
    engine._resolve_device()
    print(
        f"[daemon] Reloaded config — voice={new_config.tts.voice}, "
        f"speed={new_config.tts.speed}, device={new_config.tts.device}",
        flush=True,
    )
    return new_config, current_mtime


async def run_loop(config: Config, engine: TTSEngine):
    """Main loop: watch queue, normalize, chunk, speak."""
    Q.ensure_queue_dir()
    print(f"[daemon] Watching queue: {QUEUE_DIR}", flush=True)

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

        # Measure time from queue file creation to pickup
        if PERF_LOG:
            try:
                queue_ts = float(filepath.stem)
                pickup_delay = time.time() - queue_ts
                print(f"[perf] queue->pickup: {pickup_delay:.3f}s", flush=True)
            except (ValueError, AttributeError):
                pass

        t0 = time.monotonic()

        # Stop word handling: if text is only a stop phrase, abort playback
        if _is_stop_command(text, config.wakeword.stop_phrases):
            print("[daemon] Stop command received, aborting playback.", flush=True)
            engine.stop()
            Q.clear()
            if config.audio.chimes:
                play_stop_chime(device=engine._output_device, volume=config.audio.volume)
            continue

        try:
            print(f"[daemon] Processing ({len(text)} chars)...", flush=True)

            # Normalize for speech
            normalized = normalize(text)
            t_norm = time.monotonic()
            if PERF_LOG:
                print(f"[perf] normalize: {(t_norm - t0)*1000:.0f}ms", flush=True)

            if not normalized:
                print("[daemon] Empty after normalization, skipping.", flush=True)
                continue

            # Chunk for reliable TTS (Kokoro struggles with very long text)
            chunks = chunk_text(normalized, max_chars=config.tts.max_chunk_chars)
            t_chunk = time.monotonic()
            if PERF_LOG:
                print(f"[perf] chunk: {(t_chunk - t_norm)*1000:.0f}ms ({len(chunks)} chunks)", flush=True)

            if len(chunks) == 1:
                # Single chunk — just stream directly
                if TOGGLE_FILE.exists():
                    await engine.speak(chunks[0])
                    if PERF_LOG:
                        print(f"[perf] speak: {(time.monotonic() - t_chunk)*1000:.0f}ms", flush=True)
            else:
                # Multiple chunks — generate next while current plays
                # Generate first chunk
                next_audio = await engine.generate_audio(chunks[0])
                t_gen = time.monotonic()
                if PERF_LOG:
                    print(f"[perf] generate chunk[0]: {(t_gen - t_chunk)*1000:.0f}ms", flush=True)

                for i in range(len(chunks)):
                    if not TOGGLE_FILE.exists():
                        break

                    current_audio = next_audio

                    if i + 1 < len(chunks):
                        # TRUE parallel: play in thread (frees event loop)
                        # so generate_audio can actually run concurrently
                        gen_task = asyncio.create_task(engine.generate_audio(chunks[i + 1]))
                        t_play = time.monotonic()
                        await asyncio.to_thread(engine.play_audio, current_audio)
                        if PERF_LOG:
                            print(f"[perf] play chunk[{i}]: {(time.monotonic() - t_play)*1000:.0f}ms", flush=True)
                        next_audio = await gen_task
                        if PERF_LOG:
                            print(f"[perf] generate chunk[{i+1}]: ready", flush=True)
                    else:
                        await asyncio.to_thread(engine.play_audio, current_audio)

            t_done = time.monotonic()
            if PERF_LOG:
                print(f"[perf] TOTAL dequeue->done: {(t_done - t0)*1000:.0f}ms", flush=True)
            print(f"[daemon] Done speaking.", flush=True)

        except Exception as e:
            import traceback
            print(f"[daemon] Error: {e}", flush=True)
            traceback.print_exc()


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
    print("\n[daemon] Shutting down...", flush=True)
    PID_FILE.unlink(missing_ok=True)
    LOCK_FILE.unlink(missing_ok=True)
    START_TS_FILE.unlink(missing_ok=True)
    sys.exit(0)


def status():
    pid = read_pid()
    enabled = TOGGLE_FILE.exists()
    depth = Q.depth()
    print(f"Daemon:  {'running (PID ' + str(pid) + ')' if pid else 'stopped'}")
    print(f"Enabled: {'yes' if enabled else 'no'}")
    print(f"Queue:   {depth} items")


def start(daemonize: bool = False):
    # Check PID file first (fast check)
    existing = read_pid()
    if existing:
        print(f"[daemon] Already running (PID {existing}).")
        sys.exit(0)  # Exit cleanly, not an error

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    if daemonize:
        pid = os.fork()
        if pid > 0:
            print(f"[daemon] Started (PID {pid})")
            sys.exit(0)
        os.setsid()
        log_fd = open(LOG_FILE, "a")
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())

    # Acquire exclusive lock AFTER fork (child process holds the lock)
    if not acquire_lock():
        print("[daemon] Another instance holds the lock. Exiting.", flush=True)
        sys.exit(0)

    write_pid()
    START_TS_FILE.write_text(str(time.time()))
    Q.ensure_queue_dir()

    config = load_config()
    engine = TTSEngine(config)
    engine.load()

    # Session greeting: chime + spoken confirmation
    try:
        if config.audio.chimes:
            play_ready_chime(device=engine._output_device, volume=config.audio.volume)
        if config.audio.greeting:
            import asyncio as _aio
            _aio.run(engine.speak(config.audio.greeting))
        print("[daemon] Greeting complete.", flush=True)
    except Exception as e:
        print(f"[daemon] Greeting failed: {e}", flush=True)

    # Start voice controller (wake word + voice input) if enabled
    voice_controller = None
    if config.wakeword.enabled:
        try:
            from .voice_controller import VoiceController
            voice_controller = VoiceController(config, tts_stop_callback=engine.stop)
            voice_controller.start()
            print("[daemon] Voice controller started.", flush=True)
        except Exception as e:
            print(f"[daemon] Voice controller failed to start: {e}", flush=True)

    try:
        asyncio.run(run_loop(config, engine))
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
