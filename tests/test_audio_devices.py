"""
Unit tests for claude_speak/audio_devices.py — AudioDeviceManager.

All sounddevice calls are mocked. Tests run without audio hardware.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

import claude_speak.audio_devices as ad_module
from claude_speak.audio_devices import AudioDeviceManager, get_device_manager, _RESOLVE_INTERVAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_sd():
    """Return a MagicMock for sounddevice with common defaults."""
    mock = MagicMock()
    mock.default.device = (0, 1)
    return mock


DEVICE_LIST = [
    {"name": "Built-in Speaker", "max_output_channels": 2, "max_input_channels": 0},
    {"name": "AirPods Pro", "max_output_channels": 2, "max_input_channels": 1},
    {"name": "MacBook Pro Microphone", "max_output_channels": 0, "max_input_channels": 1},
    {"name": "Bose QC45", "max_output_channels": 2, "max_input_channels": 1},
    {"name": "Internal Speakers", "max_output_channels": 2, "max_input_channels": 0},
]


# ---------------------------------------------------------------------------
# Tests: list_output_devices / list_input_devices
# ---------------------------------------------------------------------------

class TestListDevices:
    """Tests for list_output_devices and list_input_devices."""

    def test_list_output_devices(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            outputs = dm.list_output_devices()
            assert len(outputs) == 4
            names = [d["name"] for d in outputs]
            assert "MacBook Pro Microphone" not in names
            assert "Built-in Speaker" in names
            assert "AirPods Pro" in names

    def test_list_input_devices(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            inputs = dm.list_input_devices()
            assert len(inputs) == 3
            names = [d["name"] for d in inputs]
            assert "Built-in Speaker" not in names
            assert "Internal Speakers" not in names
            assert "AirPods Pro" in names
            assert "MacBook Pro Microphone" in names


# ---------------------------------------------------------------------------
# Tests: get_default_output / get_default_input
# ---------------------------------------------------------------------------

class TestGetDefaults:
    """Tests for get_default_output and get_default_input."""

    def test_get_default_output(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (2, 5)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.get_default_output() == 5

    def test_get_default_input(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (3, 7)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.get_default_input() == 3


# ---------------------------------------------------------------------------
# Tests: get_device_by_name
# ---------------------------------------------------------------------------

class TestGetDeviceByName:
    """Tests for get_device_by_name."""

    def test_found_output_device(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = dm.get_device_by_name("airpods", output=True)
            assert result == 1

    def test_found_input_device(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = dm.get_device_by_name("macbook", output=False)
            assert result == 2

    def test_not_found_returns_none(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = dm.get_device_by_name("nonexistent", output=True)
            assert result is None

    def test_case_insensitive(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = dm.get_device_by_name("AIRPODS", output=True)
            assert result == 1

    def test_input_device_not_matched_as_output(self):
        """MacBook Pro Microphone has 0 output channels, should not match output search."""
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = dm.get_device_by_name("macbook", output=True)
            assert result is None


# ---------------------------------------------------------------------------
# Tests: get_device_name
# ---------------------------------------------------------------------------

class TestGetDeviceName:
    """Tests for get_device_name."""

    def test_success(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = {"name": "AirPods Pro"}
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.get_device_name(1) == "AirPods Pro"

    def test_error_fallback(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.side_effect = Exception("no such device")
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.get_device_name(99) == "device 99"

    def test_none_device_id(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = {"name": "Default Device"}
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.get_device_name(None) == "Default Device"


# ---------------------------------------------------------------------------
# Tests: is_bluetooth
# ---------------------------------------------------------------------------

class TestIsBluetooth:
    """Tests for is_bluetooth heuristic."""

    @pytest.mark.parametrize("name,expected", [
        ("AirPods Pro", True),
        ("Bose QC45", True),
        ("Sony WH-1000XM5", True),
        ("Jabra Elite 85t", True),
        ("Beats Fit Pro", True),
        ("Bluetooth Headset", True),
        ("BT Speaker", True),
        ("Built-in Speaker", False),
        ("MacBook Pro Speakers", False),
        ("Internal Speakers", False),
    ])
    def test_various_device_names(self, name, expected):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = {"name": name}
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.is_bluetooth(0) is expected

    def test_error_returns_false(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.side_effect = Exception("no such device")
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.is_bluetooth(99) is False


# ---------------------------------------------------------------------------
# Tests: find_builtin_mic
# ---------------------------------------------------------------------------

class TestFindBuiltinMic:
    """Tests for find_builtin_mic."""

    def test_finds_macbook_mic(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = [
            {"name": "AirPods Pro", "max_input_channels": 1},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        ]
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.find_builtin_mic() == 1

    def test_finds_builtin_mic(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = [
            {"name": "AirPods Pro", "max_input_channels": 1},
            {"name": "Built-in Microphone", "max_input_channels": 1},
        ]
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.find_builtin_mic() == 1

    def test_finds_internal_mic(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = [
            {"name": "External USB Mic", "max_input_channels": 1},
            {"name": "Internal Microphone", "max_input_channels": 2},
        ]
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.find_builtin_mic() == 1

    def test_returns_none_when_no_builtin(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = [
            {"name": "AirPods Pro", "max_input_channels": 1},
            {"name": "External USB Mic", "max_input_channels": 1},
        ]
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.find_builtin_mic() is None

    def test_skips_output_only_devices(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = [
            {"name": "MacBook Pro Speakers", "max_input_channels": 0},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        ]
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.find_builtin_mic() == 1


# ---------------------------------------------------------------------------
# Tests: resolve_output
# ---------------------------------------------------------------------------

class TestResolveOutput:
    """Tests for resolve_output."""

    def test_auto_returns_default(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (0, 5)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.resolve_output("auto") == 5

    def test_numeric_string(self):
        dm = AudioDeviceManager()
        assert dm.resolve_output("3") == 3

    def test_name_substring(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.resolve_output("airpods") == 1

    def test_unknown_name_falls_back_to_default(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        mock_sd.default.device = (0, 0)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.resolve_output("nonexistent") == 0

    def test_empty_string_returns_default(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (0, 2)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.resolve_output("") == 2


# ---------------------------------------------------------------------------
# Tests: resolve_input
# ---------------------------------------------------------------------------

class TestResolveInput:
    """Tests for resolve_input."""

    def test_auto_returns_default(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (3, 0)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.resolve_input("auto") == 3

    def test_numeric_string(self):
        dm = AudioDeviceManager()
        assert dm.resolve_input("2") == 2

    def test_name_substring(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.resolve_input("macbook") == 2

    def test_unknown_name_falls_back_to_default(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = DEVICE_LIST
        mock_sd.default.device = (0, 1)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.resolve_input("nonexistent") == 0


# ---------------------------------------------------------------------------
# Tests: maybe_resolve_output (caching behavior)
# ---------------------------------------------------------------------------

class TestMaybeResolveOutput:
    """Tests for maybe_resolve_output caching."""

    def test_resolves_on_first_call(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (0, 5)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = dm.maybe_resolve_output("auto")
            assert result == 5
            assert dm._output_device == 5
            assert dm._last_resolve_time > 0

    def test_caches_and_skips_if_recently_resolved(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (0, 5)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            dm.maybe_resolve_output("auto")
            assert dm._output_device == 5

            # Change the "default" — should NOT be picked up yet
            mock_sd.default.device = (0, 9)
            result = dm.maybe_resolve_output("auto")
            assert result == 5  # still cached

    def test_re_resolves_after_interval(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (0, 5)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            dm.maybe_resolve_output("auto")
            assert dm._output_device == 5

            # Simulate time passing beyond the resolve interval
            dm._last_resolve_time = time.monotonic() - _RESOLVE_INTERVAL - 1

            mock_sd.default.device = (0, 9)
            result = dm.maybe_resolve_output("auto")
            assert result == 9
            assert dm._output_device == 9


# ---------------------------------------------------------------------------
# Tests: get_device_manager singleton
# ---------------------------------------------------------------------------

class TestGetDeviceManager:
    """Tests for the module-level singleton."""

    def test_returns_same_instance(self):
        # Reset singleton for test isolation
        ad_module._manager = None
        try:
            dm1 = get_device_manager()
            dm2 = get_device_manager()
            assert dm1 is dm2
        finally:
            ad_module._manager = None

    def test_creates_instance_on_first_call(self):
        ad_module._manager = None
        try:
            dm = get_device_manager()
            assert isinstance(dm, AudioDeviceManager)
        finally:
            ad_module._manager = None

    def test_returns_existing_instance(self):
        existing = AudioDeviceManager()
        ad_module._manager = existing
        try:
            dm = get_device_manager()
            assert dm is existing
        finally:
            ad_module._manager = None
