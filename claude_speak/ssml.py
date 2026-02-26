"""
SSML-like markup parser for speech segments.

Supports a small set of custom tags for controlling speech output:
  - <pause 500ms>        Insert a silence of the specified duration.
  - <slow>text</slow>    Reduce speed by 20% for the enclosed text.
  - <fast>text</fast>    Increase speed by 20% for the enclosed text.
  - <spell>ABC</spell>   Spell out each character with pauses between.

Tags are parsed into a flat list of SpeechSegment dataclasses that the
TTS engine can iterate over to produce audio with the right speed/pauses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SpeechSegment:
    """A single segment of speech with optional modifiers.

    Attributes:
        text:           The text to synthesise (empty for pure pauses).
        speed_modifier: Multiplier applied to the base TTS speed (1.0 = unchanged).
        pause_ms:       Silence to insert *before* any text in this segment.
        spell:          If True, the text has already been expanded to individual
                        characters separated by dots and should be spoken slowly.
    """

    text: str
    speed_modifier: float = 1.0
    pause_ms: int = 0
    spell: bool = False


# ---------------------------------------------------------------------------
# Tag patterns
# ---------------------------------------------------------------------------

# Matches all supported tags in the order they appear in the source text.
# Groups:
#   1 – pause duration (e.g. "500")   (only for <pause …>)
#   2 – "slow" | "fast" | "spell"     (opening tag name)
#   3 – inner text of the tag          (content between open/close)
_TAG_RE = re.compile(
    r"<pause\s+(\d+)\s*ms\s*>"          # <pause 500ms>
    r"|"
    r"<(slow|fast|spell)>(.*?)</\2>",    # <slow>…</slow>, <fast>…</fast>, <spell>…</spell>
    re.DOTALL,
)

# Matches *any* of our SSML-like tags (used by strip_ssml).
_STRIP_RE = re.compile(
    r"<pause\s+\d+\s*ms\s*>"
    r"|"
    r"</?(?:slow|fast|spell)>",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ssml(text: str) -> list[SpeechSegment]:
    """Parse *text* containing SSML-like tags into a list of :class:`SpeechSegment`.

    Plain text between tags becomes a segment with default modifiers.
    Tags are never nested; inner tags of the same type are treated as literal text.
    """
    segments: list[SpeechSegment] = []
    last_end = 0

    for m in _TAG_RE.finditer(text):
        # Emit any plain text before this tag.
        if m.start() > last_end:
            plain = text[last_end : m.start()].strip()
            if plain:
                segments.append(SpeechSegment(text=plain))

        pause_dur = m.group(1)
        tag_name = m.group(2)
        inner = m.group(3)

        if pause_dur is not None:
            # <pause 500ms>
            segments.append(SpeechSegment(text="", pause_ms=int(pause_dur)))
        elif tag_name == "slow":
            segments.append(SpeechSegment(text=inner.strip(), speed_modifier=0.8))
        elif tag_name == "fast":
            segments.append(SpeechSegment(text=inner.strip(), speed_modifier=1.2))
        elif tag_name == "spell":
            expanded = _expand_spell(inner.strip())
            segments.append(
                SpeechSegment(text=expanded, spell=True, speed_modifier=0.7)
            )

        last_end = m.end()

    # Trailing plain text after the last tag.
    if last_end < len(text):
        plain = text[last_end:].strip()
        if plain:
            segments.append(SpeechSegment(text=plain))

    # If the input had no tags at all, return a single default segment.
    if not segments:
        stripped = text.strip()
        if stripped:
            segments.append(SpeechSegment(text=stripped))

    return segments


def strip_ssml(text: str) -> str:
    """Remove all SSML-like tags from *text*, returning plain prose."""
    return _STRIP_RE.sub("", text).strip()


def generate_silence(duration_ms: int, sample_rate: int = 24000) -> np.ndarray:
    """Return a 1-D float32 array of zeros representing silence.

    Parameters:
        duration_ms: Duration in milliseconds.
        sample_rate: Audio sample rate (default 24 000 Hz, Kokoro's native rate).
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_spell(word: str) -> str:
    """Expand *word* into individual characters separated by dots.

    Example: ``"API"`` → ``"A. P. I."``
    """
    return ". ".join(word) + "."
