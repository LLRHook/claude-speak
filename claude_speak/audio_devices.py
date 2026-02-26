"""Centralized audio device management."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

_RESOLVE_INTERVAL = 30  # seconds between device re-checks


class DeviceChangeMonitor:
    """Polls for audio device changes and fires callbacks."""

    def __init__(self, poll_interval: float = 2.0):
        self._poll_interval = poll_interval
        self._callbacks: list[Callable[[list[str], list[str]], None]] = []
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._last_devices: list[str] | None = None

    def on_change(self, callback: Callable[[list[str], list[str]], None]) -> None:
        """Register a callback: fn(added_devices, removed_devices)."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start polling in background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="device-change-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.debug("DeviceChangeMonitor started (poll_interval=%.1fs)", self._poll_interval)

    def stop(self) -> None:
        """Stop polling."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 1.0)
            self._thread = None
        logger.debug("DeviceChangeMonitor stopped.")

    def _poll_loop(self) -> None:
        """Poll sounddevice.query_devices() and detect changes."""
        import sounddevice as sd
        self._last_devices = [d["name"] for d in sd.query_devices()]
        while self._running.is_set():
            self._running.wait(timeout=self._poll_interval)
            if not self._running.is_set():
                break
            try:
                current = [d["name"] for d in sd.query_devices()]
                if current != self._last_devices:
                    added = [d for d in current if d not in self._last_devices]
                    removed = [d for d in self._last_devices if d not in current]
                    self._last_devices = current
                    logger.info("Audio devices changed: +%s -%s", added, removed)
                    for cb in self._callbacks:
                        try:
                            cb(added, removed)
                        except Exception as e:
                            logger.error("Device change callback error: %s", e)
            except Exception as e:
                logger.debug("Device poll error: %s", e)


class AudioDeviceManager:
    """Manages audio input/output device selection and caching."""

    def __init__(self):
        self._output_device: int | None = None
        self._input_device: int | None = None
        self._builtin_mic_id: int | None = None
        self._last_resolve_time: float = 0.0
        self._monitor: DeviceChangeMonitor | None = None

    def list_output_devices(self) -> list[dict]:
        """List all available output devices."""
        import sounddevice as sd
        return [d for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]

    def list_input_devices(self) -> list[dict]:
        """List all available input devices."""
        import sounddevice as sd
        return [d for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]

    def get_default_output(self) -> int:
        """Get the system default output device index."""
        import sounddevice as sd
        return sd.default.device[1]

    def get_default_input(self) -> int:
        """Get the system default input device index."""
        import sounddevice as sd
        return sd.default.device[0]

    def get_device_by_name(self, substring: str, output: bool = True) -> int | None:
        """Find a device by name substring. Returns device index or None."""
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            channels_key = "max_output_channels" if output else "max_input_channels"
            if d[channels_key] > 0 and substring.lower() in d["name"].lower():
                return i
        return None

    def get_device_name(self, device_id: int | None) -> str:
        """Get human-readable name for a device."""
        import sounddevice as sd
        try:
            info = sd.query_devices(device_id)
            return info["name"]
        except Exception:
            return f"device {device_id}"

    def is_bluetooth(self, device_id: int) -> bool:
        """Check if a device is likely Bluetooth (heuristic: name matching)."""
        import sounddevice as sd
        try:
            info = sd.query_devices(device_id)
            name = info["name"].lower()
            return any(kw in name for kw in ("airpods", "bluetooth", "bt ", "bose", "sony wh", "jabra", "beats"))
        except Exception:
            return False

    def find_builtin_mic(self) -> int | None:
        """Find the built-in microphone device index."""
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                name = d["name"].lower()
                if any(kw in name for kw in ("macbook", "built-in", "internal")):
                    return i
        return None

    def resolve_output(self, preference: str = "auto") -> int | None:
        """Resolve output device from preference string.

        Args:
            preference: "auto" for default, device index as string, or name substring
        """
        if preference and preference != "auto":
            try:
                return int(preference)
            except ValueError:
                found = self.get_device_by_name(preference, output=True)
                if found is not None:
                    return found
        return self.get_default_output()

    def resolve_input(self, preference: str = "auto") -> int | None:
        """Resolve input device from preference string."""
        if preference and preference != "auto":
            try:
                return int(preference)
            except ValueError:
                found = self.get_device_by_name(preference, output=False)
                if found is not None:
                    return found
        return self.get_default_input()

    def maybe_resolve_output(self, preference: str = "auto") -> int | None:
        """Resolve output device, but only if enough time has passed since last resolve."""
        now = time.monotonic()
        if now - self._last_resolve_time >= _RESOLVE_INTERVAL:
            self._output_device = self.resolve_output(preference)
            self._last_resolve_time = now
        return self._output_device

    def is_device_available(self, device_id: int) -> bool:
        """Check if a device is still connected/available."""
        import sounddevice as sd
        try:
            info = sd.query_devices(device_id)
            return info is not None
        except Exception:
            return False

    def start_monitoring(self, poll_interval: float = 2.0) -> None:
        """Start background device-change monitoring.

        When a device is added or removed, the cache is invalidated so the
        next call to maybe_resolve_output() picks up the new default device
        immediately instead of waiting up to _RESOLVE_INTERVAL seconds.
        """
        if self._monitor is None:
            self._monitor = DeviceChangeMonitor(poll_interval=poll_interval)
            self._monitor.on_change(lambda added, removed: self.invalidate_cache())
        self._monitor.start()

    def stop_monitoring(self) -> None:
        """Stop background device-change monitoring."""
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None

    def invalidate_cache(self):
        """Force re-resolution of devices on next call."""
        self._last_resolve_time = 0.0
        self._output_device = None


# Module-level singleton
_manager: AudioDeviceManager | None = None


def get_device_manager() -> AudioDeviceManager:
    """Get or create the singleton AudioDeviceManager."""
    global _manager
    if _manager is None:
        _manager = AudioDeviceManager()
    return _manager
