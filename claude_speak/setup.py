"""First-time setup for claude-speak."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from .config import (
    LOG_FILE,
    MUTE_FILE,
    PID_FILE,
    PLAYING_FILE,
    PROJECT_DIR,
    QUEUE_DIR,
    TOGGLE_FILE,
    load_config,
)
from .models import ensure_models, ensure_stt_model, list_stt_models

CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
USER_CONFIG_DIR = Path.home() / ".claude-speak"
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.toml"
EXAMPLE_CONFIG_PATH = PROJECT_DIR / "claude-speak.toml.example"


def _print_step(label: str, status: str = "ok") -> None:
    """Print a setup step result."""
    icons = {"ok": "[OK]", "skip": "[--]", "fail": "[!!]", "info": "[->]"}
    icon = icons.get(status, "    ")
    print(f"  {icon} {label}")


def _step_download_models() -> bool:
    """Download TTS models if not already present."""
    print("\n1. Downloading TTS models...")
    try:
        paths = ensure_models()
        for name, path in paths.items():
            _print_step(f"{name} -> {path}")
        return True
    except Exception as e:
        _print_step(f"Failed to download models: {e}", "fail")
        return False


def _step_download_stt_model() -> bool:
    """Optionally pre-download a Whisper STT model (skipped by default)."""
    print("\n2. STT model (optional)...")
    registry = list_stt_models()
    print("  Available Whisper models:")
    for size, info in registry.items():
        print(f"    {size:8s}  {info['size_hint']}")
    print()
    try:
        answer = input(
            "  Pre-download an STT model? "
            "[tiny/base/small/medium/skip] (default: skip): "
        ).strip().lower()
    except EOFError:
        answer = ""

    if answer not in registry:
        _print_step("Skipping STT model pre-download", "skip")
        return True

    try:
        repo = ensure_stt_model(answer)
        _print_step(f"STT model '{answer}' ready ({repo})")
        return True
    except Exception as e:
        _print_step(f"Failed to pre-download STT model: {e}", "fail")
        return False


def _step_create_config() -> bool:
    """Copy example config to user config directory if it doesn't exist."""
    print("\n3. Creating default config...")
    if USER_CONFIG_PATH.exists():
        _print_step(f"Config already exists at {USER_CONFIG_PATH}", "skip")
        return True

    if not EXAMPLE_CONFIG_PATH.exists():
        _print_step(
            f"Example config not found at {EXAMPLE_CONFIG_PATH}, skipping",
            "skip",
        )
        return True

    try:
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EXAMPLE_CONFIG_PATH, USER_CONFIG_PATH)
        _print_step(f"Copied example config to {USER_CONFIG_PATH}")
        return True
    except Exception as e:
        _print_step(f"Failed to create config: {e}", "fail")
        return False


def _step_install_hooks() -> bool:
    """Merge claude-speak hook entries into ~/.claude/settings.json."""
    print("\n4. Installing Claude Code hooks...")

    # Use "python" on Windows (python3 is a Microsoft Store stub), "python3" elsewhere
    import sys as _sys
    _python = "python" if _sys.platform == "win32" else "python3"

    # Define the hooks claude-speak needs
    hook_entries = {
        "SessionStart": {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{_python} -m claude_speak.hooks.daemon_start",
                    "async": True,
                }
            ]
        },
        "SessionEnd": {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{_python} -m claude_speak.hooks.daemon_stop",
                    "async": True,
                }
            ]
        },
        "PostToolUse": {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{_python} -m claude_speak.hooks.speak_response",
                    "timeout": 30,
                    "async": True,
                }
            ]
        },
        "UserPromptSubmit": {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{_python} -m claude_speak.hooks.daemon_restart",
                }
            ]
        },
    }

    try:
        # Read existing settings or start fresh
        settings: dict = {}
        if CLAUDE_SETTINGS_PATH.exists():
            try:
                settings = json.loads(CLAUDE_SETTINGS_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                _print_step(
                    "Could not parse existing settings.json, will merge carefully",
                    "info",
                )
                settings = {}

        if "hooks" not in settings:
            settings["hooks"] = {}

        for event_name, new_entry in hook_entries.items():
            new_command = new_entry["hooks"][0]["command"]

            if event_name in settings["hooks"]:
                # Check if our hook command is already present in any entry
                already_present = False
                for existing_entry in settings["hooks"][event_name]:
                    for hook in existing_entry.get("hooks", []):
                        cmd = hook.get("command", "")
                        # Match if command is a claude-speak hook
                        if "claude_speak.hooks" in cmd or "claude-speak" in cmd:
                            # Update the command to current module invocation
                            hook["command"] = new_command
                            already_present = True
                            break
                    if already_present:
                        break

                if not already_present:
                    settings["hooks"][event_name].append(new_entry)
                    _print_step(f"{event_name}: added hook entry")
                else:
                    _print_step(f"{event_name}: already configured, updated path")
            else:
                settings["hooks"][event_name] = [new_entry]
                _print_step(f"{event_name}: added hook entry")

        # Write settings back
        CLAUDE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CLAUDE_SETTINGS_PATH.write_text(
            json.dumps(settings, indent=2) + "\n"
        )
        _print_step(f"Saved {CLAUDE_SETTINGS_PATH}")
        return True

    except Exception as e:
        _print_step(f"Failed to install hooks: {e}", "fail")
        return False


def _step_create_toggle() -> bool:
    """Create the toggle file to enable voice output."""
    print("\n5. Enabling voice output...")
    try:
        TOGGLE_FILE.touch()
        _print_step(f"Created {TOGGLE_FILE}")
        return True
    except Exception as e:
        _print_step(f"Failed to create toggle file: {e}", "fail")
        return False


def _step_test_audio() -> bool:
    """Try to speak a test phrase. Failures are logged but don't fail setup."""
    print("\n6. Testing audio...")
    try:
        from .tts import TTSEngine

        config = load_config()
        engine = TTSEngine(config)
        asyncio.run(engine.speak("Setup complete."))
        _print_step("Audio test passed")
        return True
    except Exception as e:
        _print_step(f"Audio test failed (non-fatal): {e}", "fail")
        return True  # Don't count audio failure as setup failure


def run_setup() -> None:
    """Run the full first-time setup sequence."""
    print("claude-speak setup")
    print("=" * 40)

    results: dict[str, bool] = {}

    results["models"] = _step_download_models()
    results["stt_model"] = _step_download_stt_model()
    results["config"] = _step_create_config()
    results["hooks"] = _step_install_hooks()
    results["toggle"] = _step_create_toggle()
    results["audio"] = _step_test_audio()

    # Summary
    print("\n" + "=" * 40)
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    if passed == total:
        print(f"Setup complete. All {total} steps succeeded.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Setup finished with issues: {passed}/{total} steps succeeded.")
        print(f"Failed steps: {', '.join(failed)}")
        print("You can run 'claude-speak setup' again to retry.")


def uninstall(remove_models: bool = False) -> None:
    """Remove all claude-speak runtime artifacts and hook entries.

    Args:
        remove_models: If True, also remove ~/.claude-speak/ entirely
                       (models and config).
    """
    print("claude-speak uninstall")
    print("=" * 40)

    # Describe what will be removed so the user can make an informed choice.
    print("\nThe following will be removed:")
    print("  - Runtime state files (PID, mute, playing, position, lock files)")
    print("  - Queue directory")
    print("  - Toggle file (voice enable/disable)")
    print("  - Daemon log file")
    print("  - claude-speak hook entries from ~/.claude/settings.json")
    if remove_models:
        print(f"  - {USER_CONFIG_DIR}/ (models and config)  [--all]")
    else:
        print(
            "  Note: models/config in ~/.claude-speak/ will NOT be removed "
            "(pass --all to include them)"
        )

    print()
    try:
        answer = input("Proceed? [y/N] ").strip().lower()
    except EOFError:
        answer = ""

    if answer != "y":
        print("Aborted.")
        return

    print()

    # 1. Stop daemon if running.
    try:
        from .daemon import stop_daemon

        stop_daemon()
        _print_step("Daemon stopped")
    except Exception as e:
        _print_step(f"Could not stop daemon (may not be running): {e}", "info")

    # 2. Remove individual state files.
    from .platform.paths import hook_lock, lock_file, pos_file
    _POSITION_FILE = pos_file()
    _HOOK_LOCK_FILE = hook_lock()
    _DAEMON_LOCK_FILE = lock_file()

    state_files: list[Path] = [
        TOGGLE_FILE,
        PID_FILE,
        MUTE_FILE,
        PLAYING_FILE,
        _POSITION_FILE,
        _HOOK_LOCK_FILE,
        _DAEMON_LOCK_FILE,
        LOG_FILE,
    ]

    for path in state_files:
        try:
            if path.is_dir():
                shutil.rmtree(path)
                _print_step(f"Removed directory {path}")
            elif path.exists():
                path.unlink()
                _print_step(f"Removed {path}")
            else:
                _print_step(f"Not found, skipping: {path}", "skip")
        except Exception as e:
            _print_step(f"Failed to remove {path}: {e}", "fail")

    # 3. Remove queue directory.
    try:
        if QUEUE_DIR.exists():
            shutil.rmtree(QUEUE_DIR)
            _print_step(f"Removed queue directory {QUEUE_DIR}")
        else:
            _print_step(f"Not found, skipping: {QUEUE_DIR}", "skip")
    except Exception as e:
        _print_step(f"Failed to remove queue directory {QUEUE_DIR}: {e}", "fail")

    # 4. Remove claude-speak hook entries from ~/.claude/settings.json.
    try:
        if CLAUDE_SETTINGS_PATH.exists():
            try:
                settings: dict = json.loads(CLAUDE_SETTINGS_PATH.read_text())
            except (json.JSONDecodeError, OSError) as e:
                _print_step(
                    f"Could not parse {CLAUDE_SETTINGS_PATH}: {e}", "fail"
                )
                settings = {}

            hooks = settings.get("hooks", {})
            changed = False
            for event_name in list(hooks.keys()):
                event_entries = hooks[event_name]
                filtered = []
                for entry in event_entries:
                    inner_hooks = entry.get("hooks", [])
                    kept = [
                        h
                        for h in inner_hooks
                        if "claude-speak" not in h.get("command", "")
                    ]
                    if len(kept) != len(inner_hooks):
                        changed = True
                    if kept:
                        filtered.append({**entry, "hooks": kept})
                    else:
                        changed = True  # whole entry removed

                if filtered:
                    hooks[event_name] = filtered
                else:
                    del hooks[event_name]
                    changed = True

            if changed:
                settings["hooks"] = hooks
                CLAUDE_SETTINGS_PATH.write_text(
                    json.dumps(settings, indent=2) + "\n"
                )
                _print_step(
                    f"Removed claude-speak hooks from {CLAUDE_SETTINGS_PATH}"
                )
            else:
                _print_step(
                    f"No claude-speak hooks found in {CLAUDE_SETTINGS_PATH}",
                    "skip",
                )
        else:
            _print_step(
                f"Not found, skipping: {CLAUDE_SETTINGS_PATH}", "skip"
            )
    except Exception as e:
        _print_step(f"Failed to update {CLAUDE_SETTINGS_PATH}: {e}", "fail")

    # 5. Optionally remove models and config directory.
    if remove_models:
        try:
            if USER_CONFIG_DIR.exists():
                shutil.rmtree(USER_CONFIG_DIR)
                _print_step(f"Removed {USER_CONFIG_DIR}")
            else:
                _print_step(f"Not found, skipping: {USER_CONFIG_DIR}", "skip")
        except Exception as e:
            _print_step(f"Failed to remove {USER_CONFIG_DIR}: {e}", "fail")

    print()
    print("Uninstall complete.")
    if not remove_models:
        print(
            f"Models and config in {USER_CONFIG_DIR}/ were kept. "
            "Run 'claude-speak uninstall --all' to remove them too."
        )
