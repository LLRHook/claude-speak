"""
Voice input via Superwhisper — trigger dictation and auto-submit to Claude Code.

Superwhisper is a macOS app that records speech, transcribes it locally, and
pastes the result at the cursor position. This module:
  1. Sends a keyboard shortcut (Option+Space) to activate Superwhisper.
  2. Waits for transcription to complete (configurable timeout).
  3. Presses Enter to submit the transcribed text to Claude Code.

All key simulation is done via macOS `osascript` (AppleScript).
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Optional

from .config import InputConfig

logger = logging.getLogger(__name__)


class SuperwhisperError(Exception):
    """Raised when Superwhisper interaction fails."""


def _run_osascript(script: str, timeout: float = 5.0) -> subprocess.CompletedProcess:
    """Execute an AppleScript snippet via osascript.

    Args:
        script: The AppleScript source to execute.
        timeout: Maximum seconds to wait for the command.

    Returns:
        The completed process result.

    Raises:
        SuperwhisperError: If osascript fails or times out.
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise SuperwhisperError(
                f"osascript failed (rc={result.returncode}): {stderr}"
            )
        return result
    except FileNotFoundError:
        raise SuperwhisperError(
            "osascript not found — this module requires macOS"
        )
    except subprocess.TimeoutExpired:
        raise SuperwhisperError(
            f"osascript timed out after {timeout}s"
        )


def _is_superwhisper_running() -> bool:
    """Check whether the Superwhisper app process is currently running."""
    try:
        result = subprocess.run(
            ["pgrep", "-xq", "Superwhisper"],
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_clipboard() -> str:
    """Read the current macOS pasteboard contents."""
    try:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout
    except Exception:
        return ""


def trigger_superwhisper(
    keycode: int = 49,
    modifiers: int = 2048,
) -> None:
    """Send the keyboard shortcut to activate Superwhisper recording.

    Default shortcut: Option + Space (keycode 49, modifier 2048).
    This toggles Superwhisper's recording on/off.

    Args:
        keycode: The macOS virtual keycode to send.
        modifiers: The modifier flags (2048 = Option/Alt).

    Raises:
        SuperwhisperError: If the shortcut cannot be sent.
    """
    # Build modifier list for AppleScript's "key code ... using"
    modifier_names = _modifier_flags_to_names(modifiers)
    using_clause = ""
    if modifier_names:
        modifier_list = ", ".join(modifier_names)
        using_clause = f" using {{{modifier_list}}}"

    script = (
        f'tell application "System Events" to '
        f"key code {keycode}{using_clause}"
    )

    logger.info("Triggering Superwhisper: keycode=%d, modifiers=%d", keycode, modifiers)
    _run_osascript(script)


def _modifier_flags_to_names(flags: int) -> list[str]:
    """Convert macOS modifier flag bitmask to AppleScript modifier names.

    Common flags:
        256   = command
        512   = shift
        2048  = option
        4096  = control
        65536 = function (fn)
    """
    mapping = {
        256: "command down",
        512: "shift down",
        2048: "option down",
        4096: "control down",
    }
    names: list[str] = []
    for flag, name in mapping.items():
        if flags & flag:
            names.append(name)
    return names


def auto_submit() -> None:
    """Press Enter to submit the transcribed text in Claude Code.

    Assumes the cursor is in the Claude Code input area and that
    Superwhisper has already pasted the transcribed text.

    Raises:
        SuperwhisperError: If the keystroke cannot be sent.
    """
    script = 'tell application "System Events" to keystroke return'
    logger.info("Auto-submitting (pressing Enter)")
    _run_osascript(script)


def wait_for_transcription(
    clipboard_before: str,
    timeout: float = 15.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for Superwhisper to finish transcribing by monitoring the clipboard.

    Superwhisper pastes transcribed text at the cursor, which also updates
    the pasteboard. We detect completion by polling until the clipboard
    content changes from the pre-trigger snapshot.

    Args:
        clipboard_before: Clipboard contents captured before triggering Superwhisper.
        timeout: Maximum seconds to wait for transcription.
        poll_interval: Seconds between clipboard checks.

    Returns:
        True if new text appeared on the clipboard, False if timed out
        with no change (meaning nothing was transcribed — do NOT submit).
    """
    logger.info("Waiting for clipboard change (timeout=%.1fs)", timeout)
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        current = _get_clipboard()
        if current != clipboard_before and current.strip():
            logger.info(
                "Clipboard changed (%d chars), transcription complete",
                len(current),
            )
            return True

    logger.warning("No clipboard change after %.1fs — nothing transcribed", timeout)
    return False


def _wait_for_speech_then_silence(
    silence_duration: float = 3.0,
    speech_threshold: float = 80.0,
    timeout: float = 300.0,
) -> bool:
    """Monitor the microphone and detect speech followed by silence.

    Opens the mic, waits for audio energy to rise above the speech threshold
    (user started talking), then waits for it to drop below the threshold
    for `silence_duration` seconds (user stopped talking).

    The timeout is a safety net only (5 minutes). In practice, the silence
    detection is the sole trigger for stopping.

    Args:
        silence_duration: Seconds of silence after speech to consider "done".
        speech_threshold: RMS energy threshold to distinguish speech from silence.
        timeout: Safety-net maximum (should never be hit in practice).

    Returns:
        True if speech-then-silence was detected, False on timeout.
    """
    import numpy as np
    import sounddevice as sd

    sample_rate = 16000
    chunk_samples = 1600  # 100ms chunks
    speech_detected = False
    silence_start = None

    logger.info(
        "Listening for speech then silence (threshold=%.0f, silence=%.1fs, timeout=%.0fs)",
        speech_threshold, silence_duration, timeout,
    )
    deadline = time.monotonic() + timeout

    try:
        with sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="int16", blocksize=chunk_samples,
        ) as stream:
            while time.monotonic() < deadline:
                data, _ = stream.read(chunk_samples)
                rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

                if rms >= speech_threshold:
                    if not speech_detected:
                        logger.debug("Speech started (rms=%.0f)", rms)
                    speech_detected = True
                    silence_start = None  # reset silence timer
                elif speech_detected:
                    # Speech was detected before, now it's quiet
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start >= silence_duration:
                        logger.info(
                            "Silence detected after speech (%.1fs quiet)",
                            silence_duration,
                        )
                        return True

    except Exception as e:
        logger.error("Mic error during silence detection: %s", e)

    if speech_detected:
        logger.info("Timeout reached after speech detected")
        return True  # Speech happened, just timed out waiting for silence
    logger.warning("No speech detected within %.0fs", timeout)
    return False


def voice_input_cycle(
    config: Optional[InputConfig] = None,
) -> bool:
    """Execute a full hands-free voice input cycle.

    Flow:
      1. Snapshot the clipboard.
      2. Trigger Superwhisper to start recording (Option+Space).
      3. Monitor the mic for speech → silence (user finished talking).
         No hard cutoff — only 3s of silence after speech stops recording.
      4. Send Option+Space again to stop Superwhisper recording.
      5. Wait for clipboard to change (transcription pasted).
      6. Only press Enter if new text actually appeared.

    Args:
        config: InputConfig with shortcut and auto_submit settings.
            If None, uses defaults.

    Returns:
        True if the cycle completed successfully, False on timeout or error.
    """
    if config is None:
        config = InputConfig()

    if not config.superwhisper:
        logger.warning("Superwhisper is disabled in config")
        return False

    # Preflight: check if Superwhisper is running
    if not _is_superwhisper_running():
        logger.warning(
            "Superwhisper does not appear to be running. "
            "The shortcut will still be sent, but it may not work."
        )

    try:
        # Step 1: Snapshot clipboard before triggering
        clipboard_before = _get_clipboard()

        # Step 2: Trigger Superwhisper to START recording
        trigger_superwhisper(
            keycode=config.superwhisper_shortcut_keycode,
            modifiers=config.superwhisper_shortcut_modifiers,
        )
        logger.info("Superwhisper recording started")

        # Step 3: Listen for speech then silence (user finished talking)
        # No hard cutoff — only the 3s silence detection stops recording
        _wait_for_speech_then_silence()

        # Step 4: Trigger Superwhisper again to STOP recording
        logger.info("Stopping Superwhisper recording")
        trigger_superwhisper(
            keycode=config.superwhisper_shortcut_keycode,
            modifiers=config.superwhisper_shortcut_modifiers,
        )

        # Step 5: Wait for clipboard to change (transcription pasted)
        transcribed = wait_for_transcription(
            clipboard_before=clipboard_before,
            timeout=10.0,  # transcription should be fast after recording stops
        )

        # Step 6: Only submit if text was actually transcribed
        if not transcribed:
            logger.warning("Nothing transcribed — skipping auto-submit")
            return False

        if config.auto_submit:
            auto_submit()
            logger.info("Voice input cycle completed successfully")
        else:
            logger.info("Voice input cycle completed (auto_submit disabled)")

        return True

    except SuperwhisperError as e:
        logger.error("Voice input cycle failed: %s", e)
        return False
