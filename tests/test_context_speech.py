"""
Unit tests for context-aware speech annotation (task 7.2.2).

Tests cover:
  - Code description detection and <slow> wrapping
  - Error message detection and annotation (pause + slow)
  - Heading detection and pause insertion
  - List item detection and inter-item pauses
  - Config disable (context_aware=False skips annotation)
  - Plain text passthrough (no special context)
  - Valid SSML output that parse_ssml can handle
  - Edge cases (empty text, very short text)
"""

from __future__ import annotations

import pytest

from claude_speak.normalizer import annotate_context, normalize
from claude_speak.ssml import parse_ssml


# ---------------------------------------------------------------------------
# Code description detection
# ---------------------------------------------------------------------------

class TestCodeDescriptionDetection:
    """Lines describing code constructs should be wrapped in <slow>."""

    def test_function_description(self):
        text = "The function processes user input and returns a result."
        result = annotate_context(text)
        assert "<slow>" in result
        assert "</slow>" in result
        assert "function processes user input" in result

    def test_class_description(self):
        text = "This class handles database connections."
        result = annotate_context(text)
        assert "<slow>" in result

    def test_method_description(self):
        text = "The method accepts two arguments and returns a boolean."
        result = annotate_context(text)
        assert "<slow>" in result

    def test_variable_description(self):
        text = "The variable stores the current configuration."
        result = annotate_context(text)
        assert "<slow>" in result

    def test_import_description(self):
        text = "This module imports the logging library."
        result = annotate_context(text)
        assert "<slow>" in result

    def test_decorator_description(self):
        text = "The decorator wraps the original function."
        result = annotate_context(text)
        assert "<slow>" in result

    def test_returns_keyword(self):
        text = "It returns None when the input is empty."
        result = annotate_context(text)
        assert "<slow>" in result

    def test_defines_keyword(self):
        text = "This file defines several utility helpers."
        result = annotate_context(text)
        assert "<slow>" in result

    def test_no_double_wrap(self):
        """Already-tagged content should not be double-wrapped."""
        text = "<slow>The function returns a value.</slow>"
        result = annotate_context(text)
        # Should not add another <slow> wrapper
        assert result.count("<slow>") == 1


# ---------------------------------------------------------------------------
# Error message detection
# ---------------------------------------------------------------------------

class TestErrorMessageDetection:
    """Lines containing error keywords get a pause before and <slow> wrapping."""

    def test_error_keyword(self):
        text = "An error occurred while parsing the config."
        result = annotate_context(text)
        assert "<pause 200ms>" in result
        assert "<slow>" in result
        assert "</slow>" in result

    def test_failed_keyword(self):
        text = "The build failed with exit code 1."
        result = annotate_context(text)
        assert "<pause 200ms>" in result
        assert "<slow>" in result

    def test_exception_keyword(self):
        text = "A RuntimeError exception was raised."
        result = annotate_context(text)
        assert "<pause 200ms>" in result
        assert "<slow>" in result

    def test_traceback_keyword(self):
        text = "Traceback most recent call last."
        result = annotate_context(text)
        assert "<pause 200ms>" in result

    def test_fatal_keyword(self):
        text = "Fatal: unable to connect to the database."
        result = annotate_context(text)
        assert "<pause 200ms>" in result

    def test_case_insensitive(self):
        text = "ERROR: file not found."
        result = annotate_context(text)
        assert "<pause 200ms>" in result
        assert "<slow>" in result

    def test_no_double_wrap_error(self):
        """Already-tagged error content should not be double-wrapped."""
        text = "<pause 200ms><slow>An error occurred.</slow>"
        result = annotate_context(text)
        assert result.count("<pause 200ms>") == 1
        assert result.count("<slow>") == 1


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

class TestHeadingDetection:
    """Short title-like lines should get a pause inserted after them."""

    def test_simple_heading(self):
        text = "Installation"
        result = annotate_context(text)
        assert "<pause 300ms>" in result
        assert "Installation" in result

    def test_heading_with_colon(self):
        text = "Configuration:"
        result = annotate_context(text)
        assert "<pause 300ms>" in result

    def test_multi_word_heading(self):
        text = "Getting Started"
        result = annotate_context(text)
        assert "<pause 300ms>" in result

    def test_heading_in_context(self):
        text = "Overview\nThis is the main content."
        result = annotate_context(text)
        assert "Overview<pause 300ms>" in result

    def test_lowercase_not_heading(self):
        """Lowercase lines should not be detected as headings."""
        text = "this is a regular sentence."
        result = annotate_context(text)
        assert "<pause 300ms>" not in result

    def test_long_line_not_heading(self):
        """Lines longer than ~60 chars should not be treated as headings."""
        text = "A" + " Very Long Title That Goes On And On And On And Keeps Going And Going Still"
        result = annotate_context(text)
        assert "<pause 300ms>" not in result

    def test_no_double_pause_heading(self):
        """Already-paused headings should not get another pause."""
        text = "Overview<pause 300ms>"
        result = annotate_context(text)
        assert result.count("<pause 300ms>") == 1


# ---------------------------------------------------------------------------
# List item detection
# ---------------------------------------------------------------------------

class TestListItemDetection:
    """Consecutive ordinal list items should get pauses between them."""

    def test_two_items_get_pause(self):
        text = "First, install the package.\nSecond, run the setup."
        result = annotate_context(text)
        # Second item should get a pause before it
        assert "<pause 150ms>" in result
        # First item should NOT get a pause before it
        lines = result.split("\n")
        assert not lines[0].startswith("<pause 150ms>")

    def test_three_items(self):
        text = "First, step one.\nSecond, step two.\nThird, step three."
        result = annotate_context(text)
        lines = result.split("\n")
        # First item: no pause
        assert not lines[0].startswith("<pause 150ms>")
        # Second and third: pause before each
        assert lines[1].startswith("<pause 150ms>")
        assert lines[2].startswith("<pause 150ms>")

    def test_single_item_no_pause(self):
        """A single list item should not get a pause."""
        text = "First, do this thing."
        result = annotate_context(text)
        assert "<pause 150ms>" not in result

    def test_non_consecutive_items(self):
        """Items separated by non-list text should not get inter-item pauses."""
        text = "First, do this.\nSome explanation.\nSecond, do that."
        result = annotate_context(text)
        lines = result.split("\n")
        # The second item follows a non-list line, so no pause
        assert not lines[2].startswith("<pause 150ms>")


# ---------------------------------------------------------------------------
# Config disable
# ---------------------------------------------------------------------------

class TestConfigDisable:
    """When context_aware=False, annotation should be skipped entirely."""

    def test_normalize_with_context_aware_false(self):
        text = "An error occurred while parsing the config."
        result = normalize(text, context_aware=False)
        # Should NOT contain any SSML tags from context annotation
        assert "<pause 200ms>" not in result
        assert "<slow>" not in result

    def test_normalize_with_context_aware_true(self):
        text = "An error occurred while parsing the config."
        result = normalize(text, context_aware=True)
        # Should contain SSML tags from context annotation
        assert "<slow>" in result

    def test_normalize_default_is_context_aware(self):
        text = "An error occurred while parsing the config."
        result = normalize(text)
        # Default should be context_aware=True
        assert "<slow>" in result


# ---------------------------------------------------------------------------
# Plain text passthrough
# ---------------------------------------------------------------------------

class TestPlainTextPassthrough:
    """Text without any special context markers should pass through unchanged."""

    def test_simple_sentence(self):
        text = "The weather is nice today."
        result = annotate_context(text)
        assert result == text

    def test_multiple_sentences(self):
        text = "I went to the store. It was a good day."
        result = annotate_context(text)
        assert result == text

    def test_numbers_and_punctuation(self):
        text = "There are 42 items in the list, and 7 are new."
        result = annotate_context(text)
        assert result == text


# ---------------------------------------------------------------------------
# Valid SSML output
# ---------------------------------------------------------------------------

class TestValidSSMLOutput:
    """Annotations should produce valid SSML that parse_ssml can handle."""

    def test_error_annotation_parses(self):
        text = "An error occurred during startup."
        annotated = annotate_context(text)
        segments = parse_ssml(annotated)
        assert len(segments) >= 1
        # Should have a pause segment and a slow segment
        has_pause = any(s.pause_ms > 0 for s in segments)
        has_slow = any(s.speed_modifier < 1.0 for s in segments)
        assert has_pause, "Expected a pause segment for error annotation"
        assert has_slow, "Expected a slow segment for error annotation"

    def test_code_description_parses(self):
        text = "The function validates user input."
        annotated = annotate_context(text)
        segments = parse_ssml(annotated)
        assert len(segments) >= 1
        has_slow = any(s.speed_modifier < 1.0 for s in segments)
        assert has_slow, "Expected a slow segment for code description"

    def test_heading_parses(self):
        text = "Overview"
        annotated = annotate_context(text)
        segments = parse_ssml(annotated)
        assert len(segments) >= 1
        # Should contain a pause segment after the heading text
        has_pause = any(s.pause_ms > 0 for s in segments)
        assert has_pause, "Expected a pause segment after heading"

    def test_mixed_content_parses(self):
        """A realistic block with headings, errors, and code descriptions."""
        text = (
            "Setup Instructions\n"
            "The function initializes the database connection.\n"
            "If it failed, check the configuration."
        )
        annotated = annotate_context(text)
        segments = parse_ssml(annotated)
        assert len(segments) >= 1
        # No crash — the SSML is well-formed enough for parse_ssml

    def test_plain_text_still_parses(self):
        text = "Just a regular sentence."
        annotated = annotate_context(text)
        segments = parse_ssml(annotated)
        assert len(segments) == 1
        assert segments[0].text == "Just a regular sentence."


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: empty text, very short text, whitespace-only."""

    def test_empty_string(self):
        result = annotate_context("")
        assert result == ""

    def test_whitespace_only(self):
        result = annotate_context("   ")
        assert result == "   "

    def test_single_word(self):
        result = annotate_context("hello")
        assert result == "hello"

    def test_single_character(self):
        result = annotate_context("x")
        assert result == "x"

    def test_newlines_only(self):
        result = annotate_context("\n\n\n")
        assert result == "\n\n\n"

    def test_very_short_text(self):
        result = annotate_context("Hi.")
        assert result == "Hi."

    def test_none_like_empty(self):
        """Empty string after strip should return as-is."""
        result = annotate_context("")
        assert result == ""
