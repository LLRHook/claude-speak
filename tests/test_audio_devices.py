"""
Unit tests for claude_speak/audio_devices.py — AudioDeviceManager and DeviceChangeMonitor.

All sounddevice calls are mocked. Tests run without audio hardware.
"""

import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest

import claude_speak.audio_devices as ad_module
from claude_speak.audio_devices import AudioDeviceManager, DeviceChangeMonitor, get_device_manager, _RESOLVE_INTERVAL


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


# ---------------------------------------------------------------------------
# Tests: is_device_available
# ---------------------------------------------------------------------------

class TestIsDeviceAvailable:
    """Tests for is_device_available."""

    def test_available_device_returns_true(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = {"name": "Built-in Speaker", "max_output_channels": 2}
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.is_device_available(0) is True

    def test_unavailable_device_raises_exception_returns_false(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.side_effect = Exception("Invalid device")
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.is_device_available(99) is False

    def test_query_returns_none_returns_false(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = None
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            assert dm.is_device_available(5) is False

    def test_valid_device_id_passed_through(self):
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = {"name": "AirPods Pro"}
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            dm.is_device_available(3)
            mock_sd.query_devices.assert_called_once_with(3)


# ---------------------------------------------------------------------------
# Tests: invalidate_cache
# ---------------------------------------------------------------------------

class TestInvalidateCache:
    """Tests for invalidate_cache."""

    def test_invalidate_resets_resolve_time(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (0, 5)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            dm.maybe_resolve_output("auto")
            assert dm._last_resolve_time > 0
            dm.invalidate_cache()
            assert dm._last_resolve_time == 0.0

    def test_invalidate_clears_cached_device(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (0, 5)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            dm.maybe_resolve_output("auto")
            assert dm._output_device == 5
            dm.invalidate_cache()
            assert dm._output_device is None

    def test_after_invalidate_next_call_re_resolves(self):
        mock_sd = _mock_sd()
        mock_sd.default.device = (0, 5)
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            dm.maybe_resolve_output("auto")
            assert dm._output_device == 5

            dm.invalidate_cache()

            # Change the default — should now be picked up on next call
            mock_sd.default.device = (0, 9)
            result = dm.maybe_resolve_output("auto")
            assert result == 9
            assert dm._output_device == 9


# ---------------------------------------------------------------------------
# Tests: DeviceChangeMonitor
# ---------------------------------------------------------------------------

def _make_monitor_with_mock_sd(initial_devices, poll_interval=0.05):
    """Create a DeviceChangeMonitor backed by a mock sounddevice.

    Returns (monitor, mock_sd) so tests can mutate mock_sd.query_devices.
    """
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = [{"name": n} for n in initial_devices]
    monitor = DeviceChangeMonitor(poll_interval=poll_interval)
    return monitor, mock_sd


class TestDeviceChangeMonitorLifecycle:
    """Tests for DeviceChangeMonitor start/stop lifecycle."""

    def test_start_spawns_daemon_thread(self):
        monitor, mock_sd = _make_monitor_with_mock_sd(["Speaker"])
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            try:
                assert monitor._thread is not None
                assert monitor._thread.is_alive()
                assert monitor._thread.daemon is True
                assert monitor._thread.name == "device-change-monitor"
            finally:
                monitor.stop()

    def test_stop_joins_thread(self):
        monitor, mock_sd = _make_monitor_with_mock_sd(["Speaker"])
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            thread = monitor._thread
            monitor.stop()
            assert not thread.is_alive()
            assert monitor._thread is None

    def test_start_is_idempotent(self):
        """Calling start() twice should not spawn a second thread."""
        monitor, mock_sd = _make_monitor_with_mock_sd(["Speaker"])
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            first_thread = monitor._thread
            monitor.start()  # second call — should no-op
            try:
                assert monitor._thread is first_thread
            finally:
                monitor.stop()

    def test_stop_without_start_is_safe(self):
        monitor = DeviceChangeMonitor()
        monitor.stop()  # should not raise

    def test_running_event_cleared_after_stop(self):
        monitor, mock_sd = _make_monitor_with_mock_sd(["Speaker"])
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            monitor.stop()
            assert not monitor._running.is_set()


class TestDeviceChangeMonitorCallbacks:
    """Tests for DeviceChangeMonitor callback firing on device changes."""

    def _wait_for_callback(self, callback_mock, timeout=2.0):
        """Poll until the mock is called or timeout expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if callback_mock.called:
                return True
            time.sleep(0.01)
        return False

    def test_callback_fires_when_device_added(self):
        initial = ["Built-in Speaker"]
        monitor, mock_sd = _make_monitor_with_mock_sd(initial, poll_interval=0.05)
        callback = MagicMock()
        monitor.on_change(callback)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            # Give the thread a moment to snapshot the initial device list
            time.sleep(0.1)
            # Simulate AirPods connecting
            mock_sd.query_devices.return_value = [
                {"name": "Built-in Speaker"},
                {"name": "AirPods Pro"},
            ]
            called = self._wait_for_callback(callback)
            monitor.stop()

        assert called, "Callback should have fired when a device was added"
        added, removed = callback.call_args[0]
        assert "AirPods Pro" in added
        assert removed == []

    def test_callback_fires_when_device_removed(self):
        initial = ["Built-in Speaker", "AirPods Pro"]
        monitor, mock_sd = _make_monitor_with_mock_sd(initial, poll_interval=0.05)
        callback = MagicMock()
        monitor.on_change(callback)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            time.sleep(0.1)
            # Simulate AirPods disconnecting
            mock_sd.query_devices.return_value = [{"name": "Built-in Speaker"}]
            called = self._wait_for_callback(callback)
            monitor.stop()

        assert called, "Callback should have fired when a device was removed"
        added, removed = callback.call_args[0]
        assert added == []
        assert "AirPods Pro" in removed

    def test_callback_receives_both_added_and_removed(self):
        """Swapping one device for another yields both added and removed."""
        initial = ["Built-in Speaker", "Old Headphones"]
        monitor, mock_sd = _make_monitor_with_mock_sd(initial, poll_interval=0.05)
        callback = MagicMock()
        monitor.on_change(callback)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            time.sleep(0.1)
            mock_sd.query_devices.return_value = [
                {"name": "Built-in Speaker"},
                {"name": "New Headset"},
            ]
            called = self._wait_for_callback(callback)
            monitor.stop()

        assert called
        added, removed = callback.call_args[0]
        assert "New Headset" in added
        assert "Old Headphones" in removed

    def test_no_callback_when_devices_unchanged(self):
        initial = ["Built-in Speaker"]
        monitor, mock_sd = _make_monitor_with_mock_sd(initial, poll_interval=0.05)
        callback = MagicMock()
        monitor.on_change(callback)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            # Let several poll cycles pass with no change
            time.sleep(0.3)
            monitor.stop()

        callback.assert_not_called()

    def test_multiple_callbacks_all_called(self):
        initial = ["Speaker"]
        monitor, mock_sd = _make_monitor_with_mock_sd(initial, poll_interval=0.05)
        cb1, cb2 = MagicMock(), MagicMock()
        monitor.on_change(cb1)
        monitor.on_change(cb2)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            time.sleep(0.1)
            mock_sd.query_devices.return_value = [
                {"name": "Speaker"},
                {"name": "AirPods Pro"},
            ]
            self._wait_for_callback(cb1)
            monitor.stop()

        assert cb1.called
        assert cb2.called

    def test_callback_exception_does_not_crash_monitor(self):
        """A misbehaving callback should not stop the monitor thread."""
        initial = ["Speaker"]
        monitor, mock_sd = _make_monitor_with_mock_sd(initial, poll_interval=0.05)
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        good_cb = MagicMock()
        monitor.on_change(bad_cb)
        monitor.on_change(good_cb)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            time.sleep(0.1)
            mock_sd.query_devices.return_value = [
                {"name": "Speaker"},
                {"name": "AirPods Pro"},
            ]
            called = self._wait_for_callback(good_cb)
            assert monitor._thread is not None and monitor._thread.is_alive(), (
                "Monitor thread should still be alive despite callback error"
            )
            monitor.stop()

        assert called, "Second callback should still fire after first raises"

    def test_poll_error_does_not_crash_monitor(self):
        """An exception from query_devices() should not kill the poll loop."""
        initial = ["Speaker"]
        monitor, mock_sd = _make_monitor_with_mock_sd(initial, poll_interval=0.05)
        callback = MagicMock()
        monitor.on_change(callback)

        error_then_change = [
            Exception("PortAudio error"),           # first poll: raises
            [{"name": "Speaker"}, {"name": "AirPods Pro"}],  # second poll: change
        ]
        call_count = [0]
        original_return = mock_sd.query_devices.return_value

        def side_effect_factory():
            # Initial snapshot call returns the original list
            # First poll: raises; second poll: new list
            snapshot_done = [False]
            def _side_effect():
                if not snapshot_done[0]:
                    snapshot_done[0] = True
                    return original_return
                idx = call_count[0]
                call_count[0] += 1
                val = error_then_change[min(idx, len(error_then_change) - 1)]
                if isinstance(val, Exception):
                    raise val
                return val
            return _side_effect

        mock_sd.query_devices.side_effect = side_effect_factory()

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            monitor.start()
            called = self._wait_for_callback(callback, timeout=2.0)
            monitor.stop()

        assert called, "Callback should fire after poll recovers from error"


# ---------------------------------------------------------------------------
# Tests: AudioDeviceManager.start_monitoring / stop_monitoring
# ---------------------------------------------------------------------------

class TestAudioDeviceManagerMonitoring:
    """Tests for start_monitoring and stop_monitoring on AudioDeviceManager."""

    def test_start_monitoring_creates_monitor(self):
        dm = AudioDeviceManager()
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = [{"name": "Speaker"}]
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            dm.start_monitoring(poll_interval=0.05)
            try:
                assert dm._monitor is not None
                assert dm._monitor._thread is not None
                assert dm._monitor._thread.is_alive()
            finally:
                dm.stop_monitoring()

    def test_stop_monitoring_removes_monitor(self):
        dm = AudioDeviceManager()
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = [{"name": "Speaker"}]
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            dm.start_monitoring(poll_interval=0.05)
            dm.stop_monitoring()
            assert dm._monitor is None

    def test_stop_monitoring_without_start_is_safe(self):
        dm = AudioDeviceManager()
        dm.stop_monitoring()  # should not raise

    def test_device_change_invalidates_cache(self):
        """When a device change fires, invalidate_cache() is called and the next
        maybe_resolve_output() picks up the newly added device as default."""
        dm = AudioDeviceManager()
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = [{"name": "Speaker"}]
        mock_sd.default.device = (0, 0)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            # Prime the cache
            dm.maybe_resolve_output("auto")
            assert dm._output_device == 0

            dm.start_monitoring(poll_interval=0.05)
            time.sleep(0.1)

            # Simulate device change; also change the reported default
            mock_sd.query_devices.return_value = [
                {"name": "Speaker"},
                {"name": "AirPods Pro"},
            ]
            mock_sd.default.device = (0, 1)

            # Wait for the monitor to detect the change and call invalidate_cache()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and dm._output_device is not None:
                time.sleep(0.01)

            dm.stop_monitoring()

        assert dm._output_device is None, (
            "Cache should be invalidated after device change"
        )
        # Re-resolution should pick up the new default
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = dm.maybe_resolve_output("auto")
        assert result == 1, "Should resolve to the new default after invalidation"

    def test_start_monitoring_is_idempotent(self):
        """Calling start_monitoring() twice should reuse the same monitor."""
        dm = AudioDeviceManager()
        mock_sd = _mock_sd()
        mock_sd.query_devices.return_value = [{"name": "Speaker"}]
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            dm.start_monitoring(poll_interval=0.05)
            first_monitor = dm._monitor
            dm.start_monitoring(poll_interval=0.05)
            try:
                assert dm._monitor is first_monitor
            finally:
                dm.stop_monitoring()
