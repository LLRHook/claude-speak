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
    claude-speak voices         # List available voices
    claude-speak enable         # Enable voice output
    claude-speak disable        # Disable voice output (daemon stays loaded)
    claude-speak clear          # Clear the TTS queue
    claude-speak log            # Show recent daemon log
    claude-speak config         # Print current config values
    claude-speak setup          # First-time setup (models, hooks, config)
    claude-speak uninstall      # Remove all runtime artifacts and hooks
    claude-speak uninstall --all  # Also remove ~/.claude-speak/ (models + config)
    claude-speak listen         # Start voice controller (wake word + auto-submit)
    claude-speak voice-input    # Trigger a single voice input cycle
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
from .normalizer import chunk_text, normalize
from .tts import TTSEngine


def cmd_start() -> None:
    start(daemonize=True)


def cmd_stop() -> None:
    stop_daemon()


def cmd_kill_all() -> None:
    kill_all()


def cmd_restart() -> None:
    kill_all()
    time.sleep(1)
    start(daemonize=True)


def cmd_status() -> None:
    status()
    config = load_config()
    print(f"Voice:   {config.tts.voice}")
    print(f"Speed:   {config.tts.speed}")
    print(f"Device:  {config.tts.device}")

    # Uptime
    if START_TS_FILE.exists():
        try:
            start_ts = float(START_TS_FILE.read_text().strip())
            elapsed = time.time() - start_ts
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                uptime_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                uptime_str = f"{minutes}m {seconds}s"
            else:
                uptime_str = f"{seconds}s"
            print(f"Uptime:  {uptime_str}")
        except (ValueError, OSError):
            pass

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


def cmd_setup() -> None:
    """Run first-time setup."""
    from .setup import run_setup

    run_setup()


def cmd_uninstall() -> None:
    """Remove all claude-speak artifacts. Pass --all to also delete models/config."""
    from .setup import uninstall

    uninstall(remove_models="--all" in sys.argv)


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
        "clear": cmd_clear,
        "log": cmd_log,
        "config": cmd_config,
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
    elif cmd == "queue":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        cmd_queue(text)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
