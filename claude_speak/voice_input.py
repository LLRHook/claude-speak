"""
Voice input — trigger dictation and auto-submit to Claude Code.

Two modes are supported:

1. **Built-in** (default): Records audio from the mic using sounddevice,
   detects speech boundaries with Silero VAD, transcribes via an STT backend
   (mlx-whisper), and pastes the result at the cursor position.

2. **Superwhisper** (legacy): Uses the external Superwhisper macOS app which
   records speech, transcribes it locally, and pastes via clipboard.

Cross-platform clipboard and keyboard helpers are available in
``claude_speak.platform.input_helpers`` for use by other modules.
"""

from __future__ import annotations

import logging
import subprocess
import time

from .config import InputConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backward-compatibility wrappers (old private names used by tests).
# Tests patch ``claude_speak.voice_input.subprocess.run`` and
# ``claude_speak.voice_input._run_osascript`` to control these, so the
# implementations must live here and reference *this* module's ``subprocess``.
# ---------------------------------------------------------------------------


def _is_superwhisper_running() -> bool:
    """Check whether Superwhisper is running (legacy wrapper for tests)."""
    try:
        result = subprocess.run(
            ["pgrep", "-xiq", "superwhisper"],
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_clipboard() -> str:
    """Read the macOS pasteboard via pbpaste (legacy wrapper for tests)."""
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


def _set_clipboard(text: str) -> bool:
    """Write text to the macOS pasteboard via pbcopy (legacy wrapper for tests)."""
    try:
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True,
            timeout=2.0,
        )
        return True
    except Exception as e:
        logger.error("pbcopy failed: %s", e)
        return False


def _paste_at_cursor() -> None:
    """Press Cmd+V via osascript (legacy wrapper for tests)."""
    script = (
        'tell application "System Events" to keystroke "v" using {command down}'
    )
    _run_osascript(script)


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
    except FileNotFoundError as exc:
        raise SuperwhisperError(
            "osascript not found — this module requires macOS"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SuperwhisperError(
            f"osascript timed out after {timeout}s"
        ) from exc


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
    """Press Enter to submit text (legacy wrapper for tests).

    Tests patch ``_run_osascript`` to verify this sends ``keystroke return``,
    so this wrapper must go through osascript rather than ``press_enter()``.
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
    try:
        import sounddevice as sd
    except ImportError:
        logger.error(
            "sounddevice is not installed — mic monitoring unavailable. "
            "Install it with: pip install sounddevice"
        )
        return False

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
    config: InputConfig | None = None,
) -> bool:
    """Execute a full hands-free voice input cycle.

    Flow:
      1. Snapshot the clipboard.
      2. Trigger Superwhisper to start recording (Option+Space).
      3. Monitor the mic for speech -> silence (user finished talking).
         No hard cutoff -- only 3s of silence after speech stops recording.
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

    # Preflight: check if Superwhisper is running — abort immediately if not
    if not _is_superwhisper_running():
        logger.error(
            "Superwhisper is not running. Voice input requires Superwhisper to be active. "
            "Start it from /Applications or install from https://superwhisper.com"
        )
        return False

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


# ---------------------------------------------------------------------------
# Built-in voice input (mic → VAD → STT → paste)
# ---------------------------------------------------------------------------


def builtin_voice_input_cycle(config: InputConfig | None = None) -> bool:
    """Built-in voice input: mic -> VAD -> STT -> paste -> submit.

    Records audio from the default input device, uses Silero VAD to detect
    speech boundaries, transcribes with the configured STT backend, and
    pastes the result at the cursor position.

    Args:
        config: InputConfig with STT backend, model, and auto_submit settings.
            If None, uses defaults.

    Returns:
        True if the cycle completed successfully, False on timeout or error.
    """
    import numpy as np

    try:
        import sounddevice as sd
    except ImportError:
        logger.error(
            "sounddevice is not installed — built-in voice input unavailable. "
            "Install it with: pip install sounddevice"
        )
        return False

    if config is None:
        config = InputConfig()

    # --- Initialise STT recognizer ---
    try:
        from .stt import get_recognizer

        recognizer = get_recognizer(
            backend=config.stt_backend,
            model=config.stt_model,
        )
    except Exception as e:
        logger.error("STT backend not available: %s", e)
        return False

    # --- Initialise VAD ---
    try:
        from .vad import SileroVAD

        vad = SileroVAD(threshold=config.vad_threshold)
    except Exception as e:
        logger.error("VAD initialisation failed: %s", e)
        return False

    # --- Recording parameters ---
    sample_rate = 16000
    # Silero VAD expects 512-sample chunks at 16 kHz (32 ms)
    vad_chunk_size = 512
    speech_timeout = 10.0  # max seconds to wait for speech to start
    silence_frame_limit = 30  # consecutive non-speech frames (~1s at 32ms/frame)

    frames: list[np.ndarray] = []
    speech_started = False
    consecutive_silence = 0
    recording_start = time.monotonic()

    logger.info("Recording started — waiting for speech (timeout=%.0fs)", speech_timeout)

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=vad_chunk_size,
        ) as stream:
            while True:
                data, _ = stream.read(vad_chunk_size)
                # Convert int16 -> float32 for VAD
                chunk_f32 = data.flatten().astype(np.float32) / 32768.0
                is_speech = vad.is_speech(chunk_f32)

                if not speech_started:
                    # Waiting for speech to begin
                    if is_speech:
                        speech_started = True
                        consecutive_silence = 0
                        frames.append(data.copy())
                        logger.info("Speech detected")
                    elif time.monotonic() - recording_start > speech_timeout:
                        logger.warning(
                            "No speech detected within %.0fs — aborting",
                            speech_timeout,
                        )
                        return False
                else:
                    # Recording speech — collect all frames
                    frames.append(data.copy())
                    if is_speech:
                        consecutive_silence = 0
                    else:
                        consecutive_silence += 1
                        if consecutive_silence >= silence_frame_limit:
                            duration = (
                                sum(f.shape[0] for f in frames) / sample_rate
                            )
                            logger.info(
                                "Silence detected (%.1fs of audio)", duration
                            )
                            break

    except Exception as e:
        logger.error("Mic recording error: %s", e)
        return False

    if not frames:
        logger.warning("No audio frames captured")
        return False

    # --- Concatenate and convert audio ---
    audio_int16 = np.concatenate([f.flatten() for f in frames])
    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    # --- Transcribe ---
    t0 = time.monotonic()
    try:
        text = recognizer.transcribe(audio_f32, sample_rate=sample_rate)
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return False

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info("Transcribed: '%s' (%.0fms)", text, elapsed_ms)

    if not text or not text.strip():
        logger.warning("Empty transcription — skipping")
        return False

    # --- Paste text at cursor (cross-platform) ---
    from .platform.input_helpers import set_clipboard as _xplat_set_clipboard
    from .platform.input_helpers import paste_at_cursor as _xplat_paste
    from .platform.input_helpers import press_enter as _xplat_enter

    if not _xplat_set_clipboard(text):
        logger.error("Failed to copy text to clipboard")
        return False

    try:
        _xplat_paste()
    except Exception as e:
        logger.error("Paste failed: %s", e)
        return False

    # --- Auto-submit ---
    if config.auto_submit:
        try:
            _xplat_enter()
            logger.info("Submitted")
        except Exception as e:
            logger.error("Auto-submit failed: %s", e)
            return False
    else:
        logger.info("Voice input pasted (auto_submit disabled)")

    return True
