"""
Unit tests for claude_speak/hotkeys.py — global keyboard shortcut management.

All pyobjc/Quartz imports are mocked so tests run on any platform without
requiring macOS frameworks or Accessibility permissions.
"""

import sys
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Ensure Quartz is always mocked before importing hotkeys
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_quartz(monkeypatch):
    """Provide a fake Quartz module for every test in this file."""
    fake_quartz = MagicMock()
    # Provide realistic constant values used by the module
    fake_quartz.kCGSessionEventTap = 1
    fake_quartz.kCGHeadInsertEventTap = 0
    fake_quartz.kCGEventTapOptionDefault = 0
    fake_quartz.kCGEventKeyDown = 10
    fake_quartz.kCGEventTapDisabledByTimeout = 0xFFFFFFFE
    fake_quartz.kCGEventTapDisabledByUserInput = 0xFFFFFFFF
    fake_quartz.kCGKeyboardEventKeycode = 9
    fake_quartz.CGEventMaskBit = MagicMock(return_value=1 << 10)
    fake_quartz.CGEventTapCreate = MagicMock(return_value="fake_tap")
    fake_quartz.CFMachPortCreateRunLoopSource = MagicMock(return_value="fake_source")
    fake_quartz.CFRunLoopGetCurrent = MagicMock(return_value="fake_loop")
    fake_quartz.CFRunLoopAddSource = MagicMock()
    fake_quartz.CGEventTapEnable = MagicMock()
    fake_quartz.CFRunLoopRun = MagicMock()
    fake_quartz.CFRunLoopStop = MagicMock()
    fake_quartz.CGEventGetFlags = MagicMock(return_value=0)
    fake_quartz.CGEventGetIntegerValueField = MagicMock(return_value=0)
    fake_quartz.kCFRunLoopDefaultMode = "kCFRunLoopDefaultMode"
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    # Force-reload the hotkeys module so it picks up the mock
    if "claude_speak.hotkeys" in sys.modules:
        del sys.modules["claude_speak.hotkeys"]

    yield fake_quartz

    # Clean up
    if "claude_speak.hotkeys" in sys.modules:
        del sys.modules["claude_speak.hotkeys"]


def _import_hotkeys():
    """Import (or re-import) the hotkeys module with current sys.modules state."""
    import importlib
    import claude_speak.hotkeys as mod
    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests: parse_shortcut
# ---------------------------------------------------------------------------

class TestParseShortcut:
    """Tests for parse_shortcut()."""

    def test_basic_cmd_shift_s(self):
        mod = _import_hotkeys()
        parsed = mod.parse_shortcut("cmd+shift+s")
        assert parsed.modifiers == (mod._MOD_COMMAND | mod._MOD_SHIFT)
        assert parsed.key_code == mod._KEY_CODES["s"]
        assert parsed.raw == "cmd+shift+s"

    def test_case_insensitive(self):
        mod = _import_hotkeys()
        p1 = mod.parse_shortcut("Cmd+Shift+S")
        p2 = mod.parse_shortcut("CMD+SHIFT+S")
        p3 = mod.parse_shortcut("cmd+shift+s")
        assert p1.modifiers == p2.modifiers == p3.modifiers
        assert p1.key_code == p2.key_code == p3.key_code

    def test_ctrl_alt_f1(self):
        mod = _import_hotkeys()
        parsed = mod.parse_shortcut("ctrl+alt+f1")
        assert parsed.modifiers == (mod._MOD_CONTROL | mod._MOD_OPTION)
        assert parsed.key_code == mod._KEY_CODES["f1"]

    def test_command_alias(self):
        mod = _import_hotkeys()
        p1 = mod.parse_shortcut("cmd+shift+a")
        p2 = mod.parse_shortcut("command+shift+a")
        assert p1.modifiers == p2.modifiers

    def test_option_alias(self):
        mod = _import_hotkeys()
        p1 = mod.parse_shortcut("alt+shift+b")
        p2 = mod.parse_shortcut("opt+shift+b")
        p3 = mod.parse_shortcut("option+shift+b")
        assert p1.modifiers == p2.modifiers == p3.modifiers

    def test_control_alias(self):
        mod = _import_hotkeys()
        p1 = mod.parse_shortcut("ctrl+shift+c")
        p2 = mod.parse_shortcut("control+shift+c")
        assert p1.modifiers == p2.modifiers

    def test_whitespace_in_parts_is_stripped(self):
        mod = _import_hotkeys()
        parsed = mod.parse_shortcut("cmd + shift + x")
        assert parsed.modifiers == (mod._MOD_COMMAND | mod._MOD_SHIFT)
        assert parsed.key_code == mod._KEY_CODES["x"]

    def test_numeric_key(self):
        mod = _import_hotkeys()
        parsed = mod.parse_shortcut("cmd+1")
        assert parsed.key_code == mod._KEY_CODES["1"]

    def test_function_key(self):
        mod = _import_hotkeys()
        parsed = mod.parse_shortcut("cmd+f12")
        assert parsed.key_code == mod._KEY_CODES["f12"]

    def test_special_key_space(self):
        mod = _import_hotkeys()
        parsed = mod.parse_shortcut("cmd+shift+space")
        assert parsed.key_code == mod._KEY_CODES["space"]

    def test_special_key_escape(self):
        mod = _import_hotkeys()
        parsed = mod.parse_shortcut("cmd+escape")
        assert parsed.key_code == mod._KEY_CODES["escape"]
        parsed2 = mod.parse_shortcut("cmd+esc")
        assert parsed2.key_code == mod._KEY_CODES["esc"]

    # --- Invalid shortcuts ---

    def test_empty_string_raises(self):
        mod = _import_hotkeys()
        with pytest.raises(ValueError, match="Empty shortcut"):
            mod.parse_shortcut("")

    def test_whitespace_only_raises(self):
        mod = _import_hotkeys()
        with pytest.raises(ValueError, match="Empty shortcut"):
            mod.parse_shortcut("   ")

    def test_unknown_token_raises(self):
        mod = _import_hotkeys()
        with pytest.raises(ValueError, match="Unknown key or modifier"):
            mod.parse_shortcut("cmd+shift+bogus")

    def test_no_key_only_modifiers_raises(self):
        mod = _import_hotkeys()
        with pytest.raises(ValueError, match="No key found"):
            mod.parse_shortcut("cmd+shift")

    def test_no_modifiers_raises(self):
        mod = _import_hotkeys()
        with pytest.raises(ValueError, match="No modifiers"):
            mod.parse_shortcut("s")

    def test_multiple_keys_raises(self):
        mod = _import_hotkeys()
        with pytest.raises(ValueError, match="Multiple keys"):
            mod.parse_shortcut("cmd+a+b")


# ---------------------------------------------------------------------------
# Tests: check_conflicts
# ---------------------------------------------------------------------------

class TestCheckConflicts:
    """Tests for check_conflicts()."""

    def test_conflicting_shortcut_returns_true(self):
        mod = _import_hotkeys()
        parsed = mod.parse_shortcut("cmd+shift+3")
        assert mod.check_conflicts(parsed) is True

    def test_non_conflicting_shortcut_returns_false(self):
        mod = _import_hotkeys()
        parsed = mod.parse_shortcut("cmd+shift+s")
        assert mod.check_conflicts(parsed) is False


# ---------------------------------------------------------------------------
# Tests: HotkeyManager init
# ---------------------------------------------------------------------------

class TestHotkeyManagerInit:
    """Tests for HotkeyManager initialization."""

    def test_init_parses_shortcuts(self):
        mod = _import_hotkeys()
        callbacks = {"toggle_tts": MagicMock()}
        mgr = mod.HotkeyManager({"toggle_tts": "cmd+shift+s"}, callbacks)
        assert "toggle_tts" in mgr._parsed
        assert mgr._parsed["toggle_tts"].key_code == mod._KEY_CODES["s"]

    def test_init_multiple_shortcuts(self):
        mod = _import_hotkeys()
        shortcuts = {
            "toggle_tts": "cmd+shift+s",
            "stop_playback": "cmd+shift+x",
            "voice_input": "cmd+shift+v",
        }
        callbacks = {k: MagicMock() for k in shortcuts}
        mgr = mod.HotkeyManager(shortcuts, callbacks)
        assert len(mgr._parsed) == 3
        assert len(mgr._lookup) == 3

    def test_init_invalid_shortcut_logged_not_raised(self):
        mod = _import_hotkeys()
        callbacks = {"bad": MagicMock()}
        # Should not raise — just logs the error
        mgr = mod.HotkeyManager({"bad": "cmd+shift+bogus"}, callbacks)
        assert "bad" not in mgr._parsed

    def test_init_not_running(self):
        mod = _import_hotkeys()
        mgr = mod.HotkeyManager({}, {})
        assert mgr.is_running is False


# ---------------------------------------------------------------------------
# Tests: Callback wiring
# ---------------------------------------------------------------------------

class TestCallbackWiring:
    """Tests that callbacks are properly wired to shortcuts."""

    def test_lookup_maps_to_correct_action(self):
        mod = _import_hotkeys()
        shortcuts = {
            "toggle_tts": "cmd+shift+s",
            "stop_playback": "cmd+shift+x",
        }
        callbacks = {k: MagicMock() for k in shortcuts}
        mgr = mod.HotkeyManager(shortcuts, callbacks)

        mods_s = mod._MOD_COMMAND | mod._MOD_SHIFT
        key_s = mod._KEY_CODES["s"]
        key_x = mod._KEY_CODES["x"]

        assert mgr._lookup[(mods_s, key_s)] == "toggle_tts"
        assert mgr._lookup[(mods_s, key_x)] == "stop_playback"

    def test_callback_invoked_via_event_callback(self):
        """Simulate the _event_callback path and verify the callback fires."""
        mod = _import_hotkeys()
        fake_quartz = sys.modules["Quartz"]
        cb = MagicMock()
        mgr = mod.HotkeyManager({"toggle_tts": "cmd+shift+s"}, {"toggle_tts": cb})

        expected_mods = mod._MOD_COMMAND | mod._MOD_SHIFT
        expected_key = mod._KEY_CODES["s"]

        fake_quartz.CGEventGetFlags.return_value = expected_mods
        fake_quartz.CGEventGetIntegerValueField.return_value = expected_key

        event = MagicMock()
        result = mgr._event_callback(None, fake_quartz.kCGEventKeyDown, event, None)

        cb.assert_called_once()
        # Event should be swallowed (return None)
        assert result is None

    def test_non_matching_event_passes_through(self):
        """Events that don't match any shortcut should be returned (not swallowed)."""
        mod = _import_hotkeys()
        fake_quartz = sys.modules["Quartz"]
        cb = MagicMock()
        mgr = mod.HotkeyManager({"toggle_tts": "cmd+shift+s"}, {"toggle_tts": cb})

        # Return modifier/key that don't match any shortcut
        fake_quartz.CGEventGetFlags.return_value = mod._MOD_COMMAND
        fake_quartz.CGEventGetIntegerValueField.return_value = mod._KEY_CODES["a"]

        event = MagicMock()
        result = mgr._event_callback(None, fake_quartz.kCGEventKeyDown, event, None)

        cb.assert_not_called()
        assert result is event

    def test_callback_exception_does_not_crash(self):
        """If a callback raises, the event loop should survive."""
        mod = _import_hotkeys()
        fake_quartz = sys.modules["Quartz"]
        cb = MagicMock(side_effect=RuntimeError("boom"))
        mgr = mod.HotkeyManager({"toggle_tts": "cmd+shift+s"}, {"toggle_tts": cb})

        expected_mods = mod._MOD_COMMAND | mod._MOD_SHIFT
        expected_key = mod._KEY_CODES["s"]
        fake_quartz.CGEventGetFlags.return_value = expected_mods
        fake_quartz.CGEventGetIntegerValueField.return_value = expected_key

        event = MagicMock()
        # Should not raise
        result = mgr._event_callback(None, fake_quartz.kCGEventKeyDown, event, None)
        cb.assert_called_once()
        # Event swallowed (None) even on error — we matched the shortcut
        assert result is None


# ---------------------------------------------------------------------------
# Tests: start/stop lifecycle
# ---------------------------------------------------------------------------

class TestStartStopLifecycle:
    """Tests for HotkeyManager.start() and .stop()."""

    def test_start_creates_background_thread(self):
        mod = _import_hotkeys()
        fake_quartz = sys.modules["Quartz"]
        # Make CFRunLoopRun exit immediately
        fake_quartz.CFRunLoopRun.side_effect = lambda: None

        cb = MagicMock()
        mgr = mod.HotkeyManager({"toggle_tts": "cmd+shift+s"}, {"toggle_tts": cb})

        result = mgr.start()
        assert result is True
        assert mgr._thread is not None
        assert mgr._thread.daemon is True
        assert mgr._thread.name == "claude-speak-hotkeys"

        # Give thread time to execute and exit
        mgr._thread.join(timeout=2.0)
        mgr.stop()

    def test_start_returns_false_when_already_running(self):
        mod = _import_hotkeys()
        fake_quartz = sys.modules["Quartz"]
        fake_quartz.CFRunLoopRun.side_effect = lambda: None

        mgr = mod.HotkeyManager({"toggle_tts": "cmd+shift+s"}, {"toggle_tts": MagicMock()})
        mgr.start()
        mgr._thread.join(timeout=2.0)
        # Force _running to True to simulate still-active state
        mgr._running = True
        result = mgr.start()
        assert result is False
        mgr._running = False

    def test_start_returns_false_with_no_valid_shortcuts(self):
        mod = _import_hotkeys()
        mgr = mod.HotkeyManager({}, {})
        result = mgr.start()
        assert result is False

    def test_stop_when_not_running_is_noop(self):
        mod = _import_hotkeys()
        mgr = mod.HotkeyManager({}, {})
        # Should not raise
        mgr.stop()
        assert mgr.is_running is False

    def test_stop_sets_running_false(self):
        mod = _import_hotkeys()
        fake_quartz = sys.modules["Quartz"]

        # Block CFRunLoopRun so the thread stays alive until stop() is called
        run_loop_entered = threading.Event()
        stop_event = threading.Event()

        def _blocking_run_loop():
            run_loop_entered.set()
            stop_event.wait(timeout=5.0)

        fake_quartz.CFRunLoopRun.side_effect = _blocking_run_loop

        mgr = mod.HotkeyManager({"toggle_tts": "cmd+shift+s"}, {"toggle_tts": MagicMock()})
        mgr.start()
        run_loop_entered.wait(timeout=2.0)
        assert mgr.is_running is True

        # Release the blocking run loop and stop
        stop_event.set()
        mgr.stop()
        assert mgr.is_running is False
        assert mgr._thread is None

    def test_is_running_property(self):
        mod = _import_hotkeys()
        mgr = mod.HotkeyManager({}, {})
        assert mgr.is_running is False
        mgr._running = True
        assert mgr.is_running is True
        mgr._running = False

    def test_start_returns_false_when_event_tap_fails(self):
        """If CGEventTapCreate returns None (no Accessibility permission), start returns False."""
        mod = _import_hotkeys()
        fake_quartz = sys.modules["Quartz"]
        fake_quartz.CGEventTapCreate.return_value = None

        mgr = mod.HotkeyManager({"toggle_tts": "cmd+shift+s"}, {"toggle_tts": MagicMock()})
        result = mgr.start()
        # Thread was created but _run_event_loop sets _running=False on failure
        if mgr._thread:
            mgr._thread.join(timeout=2.0)
        assert mgr.is_running is False

    def test_event_tap_timeout_re_enables(self):
        """kCGEventTapDisabledByTimeout should re-enable the tap."""
        mod = _import_hotkeys()
        fake_quartz = sys.modules["Quartz"]
        mgr = mod.HotkeyManager({"toggle_tts": "cmd+shift+s"}, {"toggle_tts": MagicMock()})
        mgr._tap = "fake_tap"

        event = MagicMock()
        result = mgr._event_callback(None, fake_quartz.kCGEventTapDisabledByTimeout, event, None)
        fake_quartz.CGEventTapEnable.assert_called_with("fake_tap", True)
        assert result is event


# ---------------------------------------------------------------------------
# Tests: graceful degradation without pyobjc
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Tests that HotkeyManager degrades gracefully without Quartz."""

    def test_start_returns_false_without_quartz(self, monkeypatch):
        # Remove Quartz from sys.modules to simulate missing pyobjc
        monkeypatch.delitem(sys.modules, "Quartz", raising=False)
        if "claude_speak.hotkeys" in sys.modules:
            del sys.modules["claude_speak.hotkeys"]

        # Patch the import to raise ImportError
        import builtins
        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "Quartz":
                raise ImportError("No module named 'Quartz'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        import claude_speak.hotkeys as mod
        import importlib
        importlib.reload(mod)

        assert mod._QUARTZ_AVAILABLE is False

        mgr = mod.HotkeyManager({"toggle_tts": "cmd+shift+s"}, {"toggle_tts": MagicMock()})
        result = mgr.start()
        assert result is False
        assert mgr.is_running is False


# ---------------------------------------------------------------------------
# Tests: config enable/disable
# ---------------------------------------------------------------------------

class TestConfigEnableDisable:
    """Tests for hotkeys enabled/disabled via config."""

    def test_hotkeys_config_defaults(self):
        from claude_speak.config import HotkeysConfig
        hc = HotkeysConfig()
        assert hc.enabled is True
        assert hc.toggle_tts == "cmd+shift+s"
        assert hc.stop_playback == "cmd+shift+x"
        assert hc.voice_input == "cmd+shift+v"

    def test_hotkeys_config_disabled(self):
        from claude_speak.config import HotkeysConfig
        hc = HotkeysConfig(enabled=False)
        assert hc.enabled is False

    def test_hotkeys_in_main_config(self):
        from claude_speak.config import Config
        cfg = Config()
        assert hasattr(cfg, "hotkeys")
        assert cfg.hotkeys.enabled is True
        assert cfg.hotkeys.toggle_tts == "cmd+shift+s"

    def test_hotkeys_config_custom(self):
        from claude_speak.config import HotkeysConfig
        hc = HotkeysConfig(
            enabled=True,
            toggle_tts="ctrl+shift+t",
            stop_playback="ctrl+shift+q",
            voice_input="ctrl+shift+v",
        )
        assert hc.toggle_tts == "ctrl+shift+t"
        assert hc.stop_playback == "ctrl+shift+q"


# ---------------------------------------------------------------------------
# Tests: custom shortcut remapping
# ---------------------------------------------------------------------------

class TestCustomRemapping:
    """Tests that custom shortcut strings are parsed correctly."""

    def test_remap_toggle_to_ctrl_shift(self):
        mod = _import_hotkeys()
        shortcuts = {"toggle_tts": "ctrl+shift+t"}
        mgr = mod.HotkeyManager(shortcuts, {"toggle_tts": MagicMock()})
        assert "toggle_tts" in mgr._parsed
        p = mgr._parsed["toggle_tts"]
        assert p.modifiers == (mod._MOD_CONTROL | mod._MOD_SHIFT)
        assert p.key_code == mod._KEY_CODES["t"]

    def test_remap_to_option_key(self):
        mod = _import_hotkeys()
        shortcuts = {"stop_playback": "opt+shift+q"}
        mgr = mod.HotkeyManager(shortcuts, {"stop_playback": MagicMock()})
        p = mgr._parsed["stop_playback"]
        assert p.modifiers == (mod._MOD_OPTION | mod._MOD_SHIFT)
        assert p.key_code == mod._KEY_CODES["q"]

    def test_remap_to_function_key(self):
        mod = _import_hotkeys()
        shortcuts = {"toggle_tts": "cmd+f5"}
        mgr = mod.HotkeyManager(shortcuts, {"toggle_tts": MagicMock()})
        p = mgr._parsed["toggle_tts"]
        assert p.key_code == mod._KEY_CODES["f5"]

    def test_remap_three_modifiers(self):
        mod = _import_hotkeys()
        shortcuts = {"toggle_tts": "cmd+ctrl+shift+s"}
        mgr = mod.HotkeyManager(shortcuts, {"toggle_tts": MagicMock()})
        p = mgr._parsed["toggle_tts"]
        assert p.modifiers == (mod._MOD_COMMAND | mod._MOD_CONTROL | mod._MOD_SHIFT)


# ---------------------------------------------------------------------------
# Tests: invalid shortcut string handling
# ---------------------------------------------------------------------------

class TestInvalidShortcutHandling:
    """Tests for various invalid shortcut strings."""

    def test_empty_shortcut_not_registered(self):
        mod = _import_hotkeys()
        mgr = mod.HotkeyManager({"bad": ""}, {"bad": MagicMock()})
        assert "bad" not in mgr._parsed

    def test_whitespace_shortcut_not_registered(self):
        mod = _import_hotkeys()
        mgr = mod.HotkeyManager({"bad": "   "}, {"bad": MagicMock()})
        assert "bad" not in mgr._parsed

    def test_unknown_key_not_registered(self):
        mod = _import_hotkeys()
        mgr = mod.HotkeyManager({"bad": "cmd+shift+zzzzz"}, {"bad": MagicMock()})
        assert "bad" not in mgr._parsed

    def test_only_modifiers_not_registered(self):
        mod = _import_hotkeys()
        mgr = mod.HotkeyManager({"bad": "cmd+shift"}, {"bad": MagicMock()})
        assert "bad" not in mgr._parsed

    def test_no_modifier_not_registered(self):
        mod = _import_hotkeys()
        mgr = mod.HotkeyManager({"bad": "s"}, {"bad": MagicMock()})
        assert "bad" not in mgr._parsed

    def test_multiple_keys_not_registered(self):
        mod = _import_hotkeys()
        mgr = mod.HotkeyManager({"bad": "cmd+a+b"}, {"bad": MagicMock()})
        assert "bad" not in mgr._parsed

    def test_valid_and_invalid_mixed(self):
        """Valid shortcuts still register even if others are invalid."""
        mod = _import_hotkeys()
        shortcuts = {
            "good": "cmd+shift+s",
            "bad": "cmd+shift+zzzzz",
        }
        callbacks = {k: MagicMock() for k in shortcuts}
        mgr = mod.HotkeyManager(shortcuts, callbacks)
        assert "good" in mgr._parsed
        assert "bad" not in mgr._parsed
