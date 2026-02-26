#!/usr/bin/env python3
"""
claude-speak CLI — start, stop, status, test, and configure the daemon.

Usage:
    claude-speak start          # Start daemon in background
    claude-speak stop           # Stop daemon
    claude-speak status         # Show daemon status
    claude-speak restart        # Restart daemon
    claude-speak test "text"    # Speak text immediately (no daemon needed)
    claude-speak say "text"     # Alias for test
    claude-speak speak "text"   # Send text to daemon via socket
    claude-speak voices         # List available TTS voices
    claude-speak stt-models     # List available Whisper STT models
    claude-speak enable         # Enable voice output
    claude-speak disable        # Disable voice output (daemon stays loaded)
    claude-speak pause          # Pause TTS (mute)
    claude-speak resume         # Resume TTS (unmute)
    claude-speak volume <0.1-1.0>  # Set TTS volume
    claude-speak speed <value>  # Set TTS speed
    claude-speak clear          # Clear the TTS queue
    claude-speak log            # Show recent daemon log
    claude-speak config         # Print current config values
    claude-speak setup          # First-time setup (models, hooks, config)
    claude-speak uninstall      # Remove all runtime artifacts and hooks
    claude-speak uninstall --all  # Also remove ~/.claude-speak/ (models + config)
    claude-speak listen         # Start voice controller (wake word + auto-submit)
    claude-speak voice-input    # Trigger a single voice input cycle
    claude-speak preview <voice>          # Speak a sample in the given voice
    claude-speak preview "v1:60+v2:40"   # Speak a sample in a blended voice
    claude-speak preview --all            # Speak a sample in every available voice
    claude-speak train-wakeword "hey claude"     # Train a custom wake word model
    claude-speak train-wakeword "hey claude" --samples 15
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from collections.abc import Callable

from . import queue as Q
from .config import LOG_FILE, TOGGLE_FILE, load_config
from .daemon import START_TS_FILE, kill_all, start, status, stop_daemon
from .ipc import send_message
from .normalizer import chunk_text, normalize
from .tts import TTSEngine


# ---------------------------------------------------------------------------
# Socket helper
# ---------------------------------------------------------------------------

def _send_ipc(msg: dict) -> dict | None:
    """Send a message to the daemon via the IPC socket.

    Returns the response dict, or None if the socket is unavailable.
    """
    return send_message(msg)


def cmd_start() -> None:
    start(daemonize=True)


def cmd_stop() -> None:
    """Stop the daemon. Try socket first, fall back to PID-based stop."""
    resp = _send_ipc({"type": "stop"})
    if resp and resp.get("ok"):
        print("[daemon] Stop command sent via socket.")
        return
    # Fall back to PID-based approach
    stop_daemon()


def cmd_kill_all() -> None:
    kill_all()


def cmd_restart() -> None:
    kill_all()
    time.sleep(1)
    start(daemonize=True)


def cmd_status() -> None:
    """Show daemon status. Try socket first, fall back to PID-based status."""
    resp = _send_ipc({"type": "status"})
    if resp and resp.get("ok"):
        # Socket-based status (richer info from running daemon)
        enabled = resp.get("enabled", False)
        queue_depth = resp.get("queue_depth", 0)
        uptime = resp.get("uptime", 0.0)

        print(f"Daemon:  running (via socket)")
        print(f"Enabled: {'yes' if enabled else 'no'}")
        print(f"Queue:   {queue_depth} items")

        # Format uptime
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            uptime_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            uptime_str = f"{minutes}m {seconds}s"
        else:
            uptime_str = f"{seconds}s"
        print(f"Uptime:  {uptime_str}")
    else:
        # Fall back to PID-based status
        status()

    config = load_config()
    print(f"Voice:   {config.tts.voice}")
    print(f"Speed:   {config.tts.speed}")
    print(f"Device:  {config.tts.device}")

    # Log file
    print(f"Log:     {LOG_FILE}")
    if LOG_FILE.exists():
        try:
            lines = LOG_FILE.read_text().strip().splitlines()
            tail = lines[-5:] if len(lines) >= 5 else lines
            if tail:
                print("  Last log lines:")
                for line in tail:
                    print(f"    {line}")
        except OSError:
            pass

    # Toggle file
    print(f"Toggle:  {TOGGLE_FILE}")


def cmd_speak(text: str) -> None:
    """Send text to the daemon for speaking via socket."""
    resp = _send_ipc({"type": "speak", "text": text})
    if resp and resp.get("ok"):
        print("Text sent to daemon.")
    elif resp:
        print(f"Error: {resp.get('error', 'unknown error')}")
    else:
        # Fall back to file-based queue
        print("Socket unavailable, falling back to file queue.")
        normalized = normalize(text)
        if not normalized:
            print("Nothing to speak after normalization.")
            return
        config = load_config()
        chunks = chunk_text(normalized, max_chars=config.tts.max_chunk_chars)
        paths = Q.enqueue_chunks(chunks)
        print(f"Queued {len(paths)} chunk(s).")


def cmd_pause() -> None:
    """Pause TTS (mute). Try socket, fall back to creating MUTE_FILE."""
    from .config import MUTE_FILE

    resp = _send_ipc({"type": "pause"})
    if resp and resp.get("ok"):
        print("TTS paused.")
    else:
        # Fall back to direct file manipulation
        MUTE_FILE.touch()
        print("TTS paused (via file fallback).")


def cmd_resume() -> None:
    """Resume TTS (unmute). Try socket, fall back to removing MUTE_FILE."""
    from .config import MUTE_FILE

    resp = _send_ipc({"type": "resume"})
    if resp and resp.get("ok"):
        print("TTS resumed.")
    else:
        # Fall back to direct file manipulation
        MUTE_FILE.unlink(missing_ok=True)
        print("TTS resumed (via file fallback).")


def cmd_volume(level: str) -> None:
    """Set TTS volume (0.1-1.0)."""
    try:
        volume = float(level)
    except ValueError:
        print(f"Error: invalid volume value: {level!r}")
        sys.exit(1)

    resp = _send_ipc({"type": "set_volume", "volume": volume})
    if resp and resp.get("ok"):
        print(f"Volume set to {resp['volume']}.")
    elif resp:
        print(f"Error: {resp.get('error', 'unknown error')}")
    else:
        print("Socket unavailable. Daemon is not running.")
        sys.exit(1)


def cmd_speed(level: str) -> None:
    """Set TTS speed."""
    try:
        speed = float(level)
    except ValueError:
        print(f"Error: invalid speed value: {level!r}")
        sys.exit(1)

    resp = _send_ipc({"type": "set_speed", "speed": speed})
    if resp and resp.get("ok"):
        print(f"Speed set to {resp['speed']}.")
    elif resp:
        print(f"Error: {resp.get('error', 'unknown error')}")
    else:
        print("Socket unavailable. Daemon is not running.")
        sys.exit(1)


def cmd_test(text: str) -> None:
    """Speak text directly, bypassing the daemon."""
    import asyncio

    config = load_config()
    engine = TTSEngine(config)
    normalized = normalize(text)
    chunks = chunk_text(normalized, max_chars=config.tts.max_chunk_chars)
    for chunk in chunks:
        asyncio.run(engine.speak(chunk))


def cmd_voices() -> None:
    config = load_config()
    engine = TTSEngine(config)
    voices = engine.list_voices()
    print(f"Available voices ({len(voices)}):")
    for v in voices:
        prefix = "-> " if v == config.tts.voice else "  "
        print(f"  {prefix}{v}")


def cmd_preview(voice: str) -> None:
    """Speak a short sample in the given voice, or all voices if voice is '--all'."""
    import asyncio
    import time as _time

    config = load_config()
    engine = TTSEngine(config)

    if voice == "--all":
        voices = engine.list_voices()
        print(f"Previewing all {len(voices)} voices...")
        for v in voices:
            print(f"  {v}")
            engine.config.tts.voice = v
            engine._voice_style = engine._resolve_voice()
            asyncio.run(engine.speak(f"This is {v}."))
            _time.sleep(1)
    else:
        engine.config.tts.voice = voice
        # load() will be called lazily on first speak(); we need the model loaded
        # before _resolve_voice() can build a blend array, so ensure it now.
        engine.load()
        engine._voice_style = engine._resolve_voice()
        # Build a friendly display name (replace underscores, strip prefix)
        display = voice.replace("_", " ")
        sample = f"Hello, this is {display}."
        asyncio.run(engine.speak(sample))


def cmd_enable() -> None:
    TOGGLE_FILE.touch()
    print("Voice output enabled.")


def cmd_disable() -> None:
    TOGGLE_FILE.unlink(missing_ok=True)
    print("Voice output disabled (daemon stays loaded).")


def cmd_queue(text: str) -> None:
    """Normalize, chunk, and queue text for the daemon to speak."""
    normalized = normalize(text)
    if not normalized:
        print("Nothing to speak after normalization.")
        return
    config = load_config()
    chunks = chunk_text(normalized, max_chars=config.tts.max_chunk_chars)
    paths = Q.enqueue_chunks(chunks)
    print(f"Queued {len(paths)} chunk(s).")


def cmd_clear() -> None:
    """Clear the TTS queue."""
    Q.clear()
    print("Queue cleared.")


def cmd_log() -> None:
    """Show the last 20 lines of the daemon log."""
    if not LOG_FILE.exists():
        print(f"No log file found at {LOG_FILE}")
        return
    try:
        result = subprocess.run(
            ["tail", "-20", str(LOG_FILE)],
            capture_output=True,
            text=True,
            check=False,
        )
        print(result.stdout, end="")
    except Exception as e:
        print(f"Error reading log: {e}")


def cmd_listen() -> None:
    """Start voice controller with wake word detection."""
    from .voice_controller import VoiceController

    config = load_config()
    controller = VoiceController(config)
    controller.start()
    print("Voice controller running. Press Ctrl+C to stop.")
    try:
        import signal

        signal.pause()
    except KeyboardInterrupt:
        controller.stop()
        print("\nVoice controller stopped.")


def cmd_voice_input() -> None:
    """Trigger a single voice input cycle (Superwhisper -> auto-submit)."""
    from .voice_controller import VoiceController

    config = load_config()
    controller = VoiceController(config)
    success = controller.trigger_voice_input()
    if success:
        print("Voice input submitted.")
    else:
        print("Voice input failed or timed out.")


def cmd_stt_models() -> None:
    """List available Whisper STT models and their sizes."""
    from .models import list_stt_models

    registry = list_stt_models()
    print(f"Available STT models ({len(registry)}):")
    for size, info in registry.items():
        print(f"  {size:8s}  {info['hf_repo']}  ({info['size_hint']})")


def cmd_setup() -> None:
    """Run first-time setup."""
    from .setup import run_setup

    run_setup()


def cmd_uninstall() -> None:
    """Remove all claude-speak artifacts. Pass --all to also delete models/config."""
    from .setup import uninstall

    uninstall(remove_models="--all" in sys.argv)


def cmd_train_wakeword() -> None:
    """Train a custom wake word model from recorded examples."""
    from pathlib import Path

    from .train_wakeword import train_wakeword

    # Parse wake phrase (required positional after "train-wakeword")
    if len(sys.argv) < 3:
        print("Usage: claude-speak train-wakeword <wake-phrase> [--samples N] [--output PATH]")
        sys.exit(1)

    wake_phrase = sys.argv[2]
    num_samples = 10
    output_dir = None

    # Parse optional flags
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--samples" and i + 1 < len(sys.argv):
            try:
                num_samples = int(sys.argv[i + 1])
            except ValueError:
                print(f"Error: --samples requires an integer, got '{sys.argv[i + 1]}'")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_dir = Path(sys.argv[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            sys.exit(1)

    train_wakeword(wake_phrase, output_dir=output_dir, num_samples=num_samples)


def cmd_config() -> None:
    """Print current config values from TOML."""
    config = load_config()
    sections = {
        "tts": config.tts,
        "wakeword": config.wakeword,
        "input": config.input,
        "normalization": config.normalization,
        "audio": config.audio,
    }
    for section_name, section in sections.items():
        print(f"[{section_name}]")
        for key, value in vars(section).items():
            print(f"  {key} = {value!r}")
        print()


def main() -> None:
    if platform.system() != "Darwin":
        print(
            "claude-speak currently requires macOS for audio output "
            "(PortAudio/sounddevice) and Superwhisper integration.\n"
            "Linux support is planned for a future release.\n"
            "Track progress: https://github.com/vnicivanov/claude-speak/issues"
        )
        sys.exit(1)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    handlers: dict[str, Callable[[], None]] = {
        "start": cmd_start,
        "stop": cmd_stop,
        "kill-all": cmd_kill_all,
        "restart": cmd_restart,
        "status": cmd_status,
        "voices": cmd_voices,
        "enable": cmd_enable,
        "disable": cmd_disable,
        "pause": cmd_pause,
        "resume": cmd_resume,
        "clear": cmd_clear,
        "log": cmd_log,
        "config": cmd_config,
        "stt-models": cmd_stt_models,
        "setup": cmd_setup,
        "uninstall": cmd_uninstall,
        "listen": cmd_listen,
        "voice-input": cmd_voice_input,
    }

    if cmd in handlers:
        handlers[cmd]()
    elif cmd in ("test", "say"):
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello, this is a test of claude speak."
        cmd_test(text)
    elif cmd == "speak":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        if not text:
            print("Usage: claude-speak speak <text>")
            sys.exit(1)
        cmd_speak(text)
    elif cmd == "queue":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        cmd_queue(text)
    elif cmd == "volume":
        if len(sys.argv) < 3:
            print("Usage: claude-speak volume <0.1-1.0>")
            sys.exit(1)
        cmd_volume(sys.argv[2])
    elif cmd == "speed":
        if len(sys.argv) < 3:
            print("Usage: claude-speak speed <value>")
            sys.exit(1)
        cmd_speed(sys.argv[2])
    elif cmd == "preview":
        voice_arg = sys.argv[2] if len(sys.argv) > 2 else "--all"
        cmd_preview(voice_arg)
    elif cmd == "train-wakeword":
        cmd_train_wakeword()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
