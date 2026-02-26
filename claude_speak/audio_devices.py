"""Centralized audio device management."""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_RESOLVE_INTERVAL = 30  # seconds between device re-checks


class AudioDeviceManager:
    """Manages audio input/output device selection and caching."""

    def __init__(self):
        self._output_device: Optional[int] = None
        self._input_device: Optional[int] = None
        self._builtin_mic_id: Optional[int] = None
        self._last_resolve_time: float = 0.0

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

    def get_device_by_name(self, substring: str, output: bool = True) -> Optional[int]:
        """Find a device by name substring. Returns device index or None."""
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            channels_key = "max_output_channels" if output else "max_input_channels"
            if d[channels_key] > 0 and substring.lower() in d["name"].lower():
                return i
        return None

    def get_device_name(self, device_id: Optional[int]) -> str:
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

    def find_builtin_mic(self) -> Optional[int]:
        """Find the built-in microphone device index."""
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                name = d["name"].lower()
                if any(kw in name for kw in ("macbook", "built-in", "internal")):
                    return i
        return None

    def resolve_output(self, preference: str = "auto") -> Optional[int]:
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

    def resolve_input(self, preference: str = "auto") -> Optional[int]:
        """Resolve input device from preference string."""
        if preference and preference != "auto":
            try:
                return int(preference)
            except ValueError:
                found = self.get_device_by_name(preference, output=False)
                if found is not None:
                    return found
        return self.get_default_input()

    def maybe_resolve_output(self, preference: str = "auto") -> Optional[int]:
        """Resolve output device, but only if enough time has passed since last resolve."""
        now = time.monotonic()
        if now - self._last_resolve_time >= _RESOLVE_INTERVAL:
            self._output_device = self.resolve_output(preference)
            self._last_resolve_time = now
        return self._output_device


# Module-level singleton
_manager: Optional[AudioDeviceManager] = None


def get_device_manager() -> AudioDeviceManager:
    """Get or create the singleton AudioDeviceManager."""
    global _manager
    if _manager is None:
        _manager = AudioDeviceManager()
    return _manager
