#!/usr/bin/env python3
"""
claude-speak CLI — start, stop, status, test, and configure the daemon.

Usage:
    python3 cli.py start          # Start daemon in background
    python3 cli.py stop           # Stop daemon
    python3 cli.py status         # Show daemon status
    python3 cli.py restart        # Restart daemon
    python3 cli.py test "text"    # Speak text immediately (no daemon needed)
    python3 cli.py say "text"     # Alias for test
    python3 cli.py voices         # List available voices
    python3 cli.py enable         # Enable voice output
    python3 cli.py disable        # Disable voice output (daemon stays loaded)
    python3 cli.py clear          # Clear the TTS queue
    python3 cli.py log            # Show recent daemon log
    python3 cli.py config         # Print current config values
    python3 cli.py listen         # Start voice controller (wake word + auto-submit)
    python3 cli.py voice-input    # Trigger a single voice input cycle
"""

import subprocess
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TOGGLE_FILE, LOG_FILE, CONFIG_PATH, load_config
from src.daemon import start, stop_daemon, kill_all, status, read_pid, START_TS_FILE
from src.tts import TTSEngine
from src.normalizer import normalize, chunk_text
from src import queue as Q


def cmd_start():
    start(daemonize=True)


def cmd_stop():
    stop_daemon()


def cmd_kill_all():
    kill_all()


def cmd_restart():
    kill_all()
    import time
    time.sleep(1)
    start(daemonize=True)


def cmd_status():
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


def cmd_test(text: str):
    """Speak text directly, bypassing the daemon."""
    import asyncio
    config = load_config()
    engine = TTSEngine(config)
    normalized = normalize(text)
    chunks = chunk_text(normalized, max_chars=config.tts.max_chunk_chars)
    for chunk in chunks:
        asyncio.run(engine.speak(chunk))


def cmd_voices():
    config = load_config()
    engine = TTSEngine(config)
    voices = engine.list_voices()
    print(f"Available voices ({len(voices)}):")
    for v in voices:
        prefix = "→ " if v == config.tts.voice else "  "
        print(f"  {prefix}{v}")


def cmd_enable():
    TOGGLE_FILE.touch()
    print("Voice output enabled.")


def cmd_disable():
    TOGGLE_FILE.unlink(missing_ok=True)
    print("Voice output disabled (daemon stays loaded).")


def cmd_queue(text: str):
    """Normalize, chunk, and queue text for the daemon to speak."""
    normalized = normalize(text)
    if not normalized:
        print("Nothing to speak after normalization.")
        return
    config = load_config()
    chunks = chunk_text(normalized, max_chars=config.tts.max_chunk_chars)
    paths = Q.enqueue_chunks(chunks)
    print(f"Queued {len(paths)} chunk(s).")


def cmd_clear():
    """Clear the TTS queue."""
    Q.clear()
    print("Queue cleared.")


def cmd_log():
    """Show the last 20 lines of the daemon log."""
    if not LOG_FILE.exists():
        print(f"No log file found at {LOG_FILE}")
        return
    try:
        result = subprocess.run(
            ["tail", "-20", str(LOG_FILE)],
            capture_output=True, text=True,
        )
        print(result.stdout, end="")
    except Exception as e:
        print(f"Error reading log: {e}")


def cmd_listen():
    """Start voice controller with wake word detection."""
    from src.voice_controller import VoiceController
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


def cmd_voice_input():
    """Trigger a single voice input cycle (Superwhisper -> auto-submit)."""
    from src.voice_controller import VoiceController
    config = load_config()
    controller = VoiceController(config)
    success = controller.trigger_voice_input()
    if success:
        print("Voice input submitted.")
    else:
        print("Voice input failed or timed out.")


def cmd_config():
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


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    handlers = {
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
