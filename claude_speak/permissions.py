"""
Check macOS permissions required by claude-speak.

This module provides diagnostic checks for:
- Audio output (sounddevice can list output devices)
- Audio input / Microphone (sounddevice can open an input stream)
- Accessibility (osascript can send key events via System Events)

Each check returns a structured result so the CLI can render a summary.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass
class CheckResult:
    """Result of a single permission check."""

    name: str
    passed: bool
    detail: str  # e.g. device name on success, error hint on failure
    hint: str | None = None  # actionable fix instruction


def _is_macos() -> bool:
    return platform.system() == "Darwin"


# ---------------------------------------------------------------------------
# Audio output check
# ---------------------------------------------------------------------------

def check_audio_output() -> CheckResult:
    """Verify that an audio output device is available via sounddevice."""
    try:
        import sounddevice as sd

        sd.query_devices()  # verify device enumeration works
        default_output = sd.query_devices(kind="output")
        name = default_output.get("name", "unknown") if isinstance(default_output, dict) else "unknown"
        return CheckResult(
            name="Audio output",
            passed=True,
            detail=f"default: {name}",
        )
    except ImportError:
        return CheckResult(
            name="Audio output",
            passed=False,
            detail="sounddevice not installed",
            hint="Install it with: pip install sounddevice",
        )
    except Exception as exc:
        return CheckResult(
            name="Audio output",
            passed=False,
            detail=str(exc),
            hint="Check that an audio output device is connected.",
        )


# ---------------------------------------------------------------------------
# Audio input (microphone) check
# ---------------------------------------------------------------------------

def check_audio_input() -> CheckResult:
    """Verify that a microphone input device is available and can be opened."""
    try:
        import sounddevice as sd

        default_input = sd.query_devices(kind="input")
        name = default_input.get("name", "unknown") if isinstance(default_input, dict) else "unknown"

        # Attempt to briefly open an input stream to trigger the mic permission prompt
        # or confirm that permission has already been granted.
        stream = sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=160)
        stream.start()
        stream.stop()
        stream.close()

        return CheckResult(
            name="Audio input",
            passed=True,
            detail=f"default: {name}",
        )
    except ImportError:
        return CheckResult(
            name="Audio input",
            passed=False,
            detail="sounddevice not installed",
            hint="Install it with: pip install sounddevice",
        )
    except Exception as exc:
        msg = str(exc)
        if "permission" in msg.lower() or "not allowed" in msg.lower():
            return CheckResult(
                name="Audio input",
                passed=False,
                detail="Microphone permission denied",
                hint=(
                    "System Settings > Privacy & Security > Microphone\n"
                    "  Add your terminal app (Terminal.app, iTerm2, etc.)"
                ),
            )
        return CheckResult(
            name="Audio input",
            passed=False,
            detail=msg,
            hint="Check that a microphone is connected and permissions are granted.",
        )


# ---------------------------------------------------------------------------
# Accessibility check
# ---------------------------------------------------------------------------

def check_accessibility() -> CheckResult:
    """Verify that Accessibility permission is granted for osascript key events.

    On non-macOS platforms this returns N/A.
    """
    if not _is_macos():
        return CheckResult(
            name="Accessibility",
            passed=True,
            detail="N/A - macOS only",
        )

    try:
        # Attempt a harmless osascript System Events query.
        # This does NOT simulate a key press — it just checks whether the
        # process is allowed to talk to System Events at all.
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get the name of the first process',
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode == 0:
            return CheckResult(
                name="Accessibility",
                passed=True,
                detail="System Events access OK",
            )
        else:
            stderr = result.stderr.strip()
            return CheckResult(
                name="Accessibility",
                passed=False,
                detail=stderr or "osascript returned non-zero",
                hint=(
                    "System Settings > Privacy & Security > Accessibility\n"
                    "  Add your terminal app (Terminal.app, iTerm2, etc.)"
                ),
            )
    except FileNotFoundError:
        return CheckResult(
            name="Accessibility",
            passed=True,
            detail="N/A - macOS only (osascript not found)",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Accessibility",
            passed=False,
            detail="osascript timed out",
            hint=(
                "System Settings > Privacy & Security > Accessibility\n"
                "  Add your terminal app (Terminal.app, iTerm2, etc.)"
            ),
        )
    except Exception as exc:
        return CheckResult(
            name="Accessibility",
            passed=False,
            detail=str(exc),
            hint=(
                "System Settings > Privacy & Security > Accessibility\n"
                "  Add your terminal app (Terminal.app, iTerm2, etc.)"
            ),
        )


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

def run_all_checks() -> list[CheckResult]:
    """Run all permission checks and return the results."""
    return [
        check_audio_output(),
        check_audio_input(),
        check_accessibility(),
    ]


def format_results(results: list[CheckResult]) -> str:
    """Format check results as a human-readable report string."""
    lines: list[str] = []
    lines.append("claude-speak permission check")
    lines.append("\u2500" * 36)

    # Find the longest name for alignment
    max_name_len = max(len(r.name) for r in results) if results else 0

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        # Pad name with dots for visual alignment
        padding = "." * (max_name_len - len(r.name) + 5)
        line = f"{r.name} {padding} {status}"
        if r.passed and r.detail:
            line += f" ({r.detail})"
        lines.append(line)

        if not r.passed:
            if r.detail:
                lines.append(f"  {r.detail}")
            if r.hint:
                for hint_line in r.hint.split("\n"):
                    lines.append(f"  -> {hint_line.strip()}")

    return "\n".join(lines)
