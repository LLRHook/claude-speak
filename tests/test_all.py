"""
Comprehensive test suite for claude-speak.

Covers:
    - src/normalizer.py  (text normalization pipeline)
    - src/queue.py       (file-based FIFO queue)
    - src/config.py      (TOML config loader)
"""

import os
import time
from pathlib import Path

from claude_speak.normalizer import (
    expand_units,
    expand_abbreviations,
    expand_stop_words,
    describe_code_blocks,
    narrate_tables,
    improve_lists,
    clean_version_strings,
    speak_decimal_numbers,
    clean_file_paths,
    clean_file_extensions,
    clean_technical_punctuation,
    strip_code_blocks,
    final_cleanup,
    chunk_text,
    normalize,
    # NEW transforms
    clean_urls_and_emails,
    expand_currency,
    expand_percentages,
    expand_ordinals,
    strip_number_commas,
    expand_fractions_ratios,
    expand_time_formats,
    expand_dates,
    expand_temperature,
    expand_math_operators,
    expand_slash_pairs,
    # Pronunciation dictionary
    apply_pronunciation_overrides,
)


# ===================================================================
# Normalizer Tests — Existing Functions
# ===================================================================

# --- expand_units ---

def test_expand_units_megabytes():
    assert "27 megabytes" in expand_units("27MB")


def test_expand_units_megabyte_singular():
    result = expand_units("1MB")
    assert "1 megabyte" in result
    # Should NOT be plural
    assert "megabytes" not in result


def test_expand_units_milliseconds():
    assert "100 milliseconds" in expand_units("100ms")


def test_expand_units_seconds():
    assert "30 seconds" in expand_units("30s")


def test_expand_units_second_singular():
    result = expand_units("1s")
    assert "1 second" in result
    assert "seconds" not in result


def test_expand_units_picoseconds():
    assert "50 picoseconds" in expand_units("50ps")


def test_expand_units_microseconds():
    assert "microseconds" in expand_units("1.5\u03bcs")


def test_expand_units_gigabytes_decimal():
    assert "3.5 gigabytes" in expand_units("3.5GB")


# --- expand_abbreviations ---

def test_expand_abbreviations_api():
    assert "A P I" in expand_abbreviations("the API is ready")


def test_expand_abbreviations_cli():
    assert "C L I" in expand_abbreviations("use the CLI")


def test_expand_abbreviations_json_stays():
    # JSON is mapped to "JSON" (stays as-is since it's pronounceable)
    result = expand_abbreviations("JSON format")
    assert "JSON" in result


def test_expand_abbreviations_llm():
    assert "L L M" in expand_abbreviations("LLM")


# --- expand_stop_words ---

def test_expand_stop_words_eg():
    assert "for example" in expand_stop_words("e.g. Python")


def test_expand_stop_words_ie():
    assert "that is" in expand_stop_words("i.e. both")


def test_expand_stop_words_etc():
    assert "etcetera" in expand_stop_words("etc.")


def test_expand_stop_words_vs():
    assert "versus" in expand_stop_words("vs.")


def test_expand_stop_words_with():
    assert "with" in expand_stop_words("w/ support")


def test_expand_stop_words_without():
    assert "without" in expand_stop_words("w/o issues")


# --- describe_code_blocks ---

def test_describe_code_blocks_python():
    text = "```python\ndef hello():\n    print('hi')\n```"
    result = describe_code_blocks(text)
    assert "Python" in result
    assert "code snippet" in result


def test_describe_code_blocks_bash():
    text = "```bash\npip install foo\n```"
    result = describe_code_blocks(text)
    assert "bash" in result
    assert "command" in result


def test_describe_code_blocks_no_lang():
    text = "```\nsome code\n```"
    result = describe_code_blocks(text)
    # Untagged short blocks are called "command"
    assert "command" in result or "code block" in result


# --- narrate_tables ---

def test_narrate_tables_basic():
    table = "| Name | Type |\n|------|------|\n| voice | string |\n| speed | float |"
    result = narrate_tables(table)
    assert "table with columns" in result
    assert "Row 1" in result


def test_narrate_tables_large():
    # Build a table with 6+ data rows
    header = "| Col1 | Col2 |"
    sep = "|------|------|"
    rows = "\n".join(f"| val{i} | data{i} |" for i in range(1, 8))
    table = f"{header}\n{sep}\n{rows}"
    result = narrate_tables(table)
    assert "table with" in result
    assert "rows" in result


# --- improve_lists ---

def test_improve_lists_numbered():
    text = "1. Install\n2. Run"
    result = improve_lists(text)
    assert "First" in result
    assert "Second" in result


def test_improve_lists_bullets():
    text = "- Fast\n- Simple"
    result = improve_lists(text)
    assert "Fast" in result
    assert "Simple" in result
    # Each item should end with a period
    for line in result.strip().split("\n"):
        if line.strip():
            assert line.strip().endswith(".")


# --- clean_version_strings ---

def test_clean_version_strings_v1_0():
    result = clean_version_strings("v1.0")
    assert "version" in result
    assert "1 point 0" in result


def test_clean_version_strings_V2_5_1():
    result = clean_version_strings("V2.5.1")
    assert "version" in result
    assert "2 point 5 point 1" in result


def test_clean_version_strings_bare_version():
    result = clean_version_strings("version 3.0")
    assert "version" in result
    assert "3 point 0" in result


# --- speak_decimal_numbers ---

def test_speak_decimal_numbers_multiplier():
    result = speak_decimal_numbers("1.5x faster")
    assert "point five" in result


def test_speak_decimal_numbers_plain():
    result = speak_decimal_numbers("0.3 seconds")
    assert "point three" in result


# --- clean_file_paths ---

def test_clean_file_paths_long_path():
    result = clean_file_paths("/usr/local/bin/python")
    assert "python" in result
    # The long leading path should be removed
    assert "/usr/local/bin/" not in result


def test_clean_file_paths_src_prefix():
    """src/ should become source/ for speech."""
    result = clean_file_paths("src/script.py")
    assert "source" in result
    assert "src/" not in result


def test_clean_file_paths_src_in_longer_path():
    """src/ in a longer path should still become source/."""
    result = clean_file_paths("src/utils/helper.py")
    # Long path gets shortened, but src should be replaced
    assert "src/" not in result


# --- clean_file_extensions ---

def test_clean_file_extensions_py():
    result = clean_file_extensions("script.py")
    assert "dot py" in result


def test_clean_file_extensions_yaml():
    result = clean_file_extensions("config.yaml")
    assert "dot YAML" in result


# --- clean_technical_punctuation ---

def test_clean_technical_punctuation_backticks():
    result = clean_technical_punctuation("`hello`")
    assert "`" not in result
    assert "hello" in result


def test_clean_technical_punctuation_tilde_number():
    result = clean_technical_punctuation("~300MB")
    assert "about" in result


def test_clean_technical_punctuation_snake_case():
    result = clean_technical_punctuation("my_function_name")
    assert "my function name" in result


def test_clean_technical_punctuation_arrows():
    result = clean_technical_punctuation("input -> output")
    assert "->" not in result
    assert "," in result


def test_clean_technical_punctuation_ampersand():
    result = clean_technical_punctuation("bread & butter")
    assert "&" not in result
    assert "and" in result


def test_clean_technical_punctuation_hashtag():
    result = clean_technical_punctuation("#python is great")
    assert "hashtag python" in result


def test_clean_technical_punctuation_at_word():
    result = clean_technical_punctuation("mention @user in the chat")
    assert "at user" in result


def test_clean_technical_punctuation_em_dash():
    result = clean_technical_punctuation("word — word")
    assert "—" not in result
    assert "," in result


def test_clean_technical_punctuation_ellipsis():
    result = clean_technical_punctuation("wait for it...")
    assert "..." not in result


def test_clean_technical_punctuation_parenthetical_200():
    """Parenthetical content up to 200 chars should be kept."""
    inner = "a" * 150
    result = clean_technical_punctuation(f"text ({inner}) more text")
    assert inner in result
    assert "(" not in result


# --- strip_code_blocks ---

def test_strip_code_blocks_dollar_prompt():
    text = "$ pip install foo\nThis is prose."
    result = strip_code_blocks(text)
    assert "pip install foo" not in result
    assert "This is prose." in result


def test_strip_code_blocks_python_repl():
    text = ">>> print('hello')\nNormal text here."
    result = strip_code_blocks(text)
    assert "print('hello')" not in result
    assert "Normal text here." in result


def test_strip_code_blocks_keeps_prose():
    text = "This line should stay."
    result = strip_code_blocks(text)
    assert "This line should stay." in result


# --- final_cleanup ---

def test_final_cleanup_collapses_commas():
    result = final_cleanup("hello, , world")
    assert ", ," not in result
    assert "hello" in result
    assert "world" in result


def test_final_cleanup_appends_periods():
    result = final_cleanup("Hello world\nAnother line")
    # Both lines should end with periods in the joined output
    assert "Hello world." in result
    assert "Another line." in result


# --- chunk_text ---

def test_chunk_text_short():
    text = "This is a short sentence."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long():
    # Build text well over 400 chars
    sentence = "This is a moderately long sentence that adds some length. "
    text = sentence * 20  # ~1200 chars
    chunks = chunk_text(text, max_chars=400)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 400


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


# --- Full pipeline (normalize) ---

def test_normalize_full_pipeline():
    """A realistic Claude response should be cleaned into speakable text."""
    text = """## Getting Started

Here's how to set up the API:

1. Install the CLI tool
2. Configure your settings

```bash
pip install my-tool
```

The response is ~50ms, i.e. very fast.
Check the file at `/usr/local/etc/config/settings.yaml` for details.

| Option | Default |
|--------|---------|
| voice  | sarah   |
| speed  | 1.0     |

Use v2.5.1 for the latest features.
"""
    result = normalize(text)
    # Should contain spoken forms, not raw markdown
    assert "```" not in result
    assert "##" not in result
    # Abbreviations expanded
    assert "A P I" in result
    assert "C L I" in result
    # Stop words expanded
    assert "that is" in result
    # Version cleaned
    assert "version" in result
    assert "point" in result
    # Table narrated
    assert "table" in result.lower()
    # List improved
    assert "First" in result
    # Tilde expanded
    assert "about" in result
    # File path shortened (long path gone)
    assert "/usr/local/etc/config/" not in result


# ===================================================================
# Normalizer Tests — NEW Functions
# ===================================================================

# --- clean_urls_and_emails ---

def test_urls_github():
    result = clean_urls_and_emails("Check https://github.com/user/repo for details")
    assert "a github link" in result
    assert "https://" not in result


def test_urls_python_docs():
    result = clean_urls_and_emails("See https://docs.python.org/3/library for info")
    assert "a python docs link" in result


def test_urls_localhost():
    result = clean_urls_and_emails("Running at http://localhost:3000")
    assert "localhost 3000" in result


def test_urls_localhost_no_port():
    result = clean_urls_and_emails("Running at http://localhost")
    assert "localhost" in result


def test_urls_generic():
    result = clean_urls_and_emails("Visit https://example.com/some/path")
    assert "a link to example dot com" in result


def test_urls_email():
    result = clean_urls_and_emails("Contact user@example.com for help")
    assert "user at example dot com" in result


def test_urls_bare_domain():
    result = clean_urls_and_emails("Check example.com for more")
    assert "example dot com" in result


def test_urls_empty_string():
    assert clean_urls_and_emails("") == ""


def test_urls_no_urls():
    text = "This has no URLs at all"
    assert clean_urls_and_emails(text) == text


# --- expand_currency ---

def test_currency_dollars():
    assert "100 dollars" in expand_currency("$100")


def test_currency_dollars_singular():
    result = expand_currency("$1")
    assert "1 dollar" in result
    assert "dollars" not in result


def test_currency_cents():
    result = expand_currency("$99.99")
    assert "99 dollars and 99 cents" in result


def test_currency_euros():
    assert "50 euros" in expand_currency("€50")


def test_currency_pounds():
    assert "75 pounds" in expand_currency("£75")


def test_currency_yen():
    assert "1000 yen" in expand_currency("¥1000")


def test_currency_million():
    result = expand_currency("$1.5M")
    assert "1.5 million dollars" in result


def test_currency_billion():
    result = expand_currency("$2B")
    assert "2 billion dollars" in result


def test_currency_empty():
    assert expand_currency("") == ""


def test_currency_no_match():
    text = "No currency here"
    assert expand_currency(text) == text


# --- expand_percentages ---

def test_percent_basic():
    assert "75 percent" in expand_percentages("75%")


def test_percent_decimal():
    assert "99.9 percent" in expand_percentages("99.9%")


def test_percent_100():
    assert "100 percent" in expand_percentages("100%")


def test_percent_empty():
    assert expand_percentages("") == ""


def test_percent_no_match():
    text = "No percent here"
    assert expand_percentages(text) == text


# --- expand_ordinals ---

def test_ordinal_1st():
    assert expand_ordinals("1st") == "first"


def test_ordinal_2nd():
    assert expand_ordinals("2nd") == "second"


def test_ordinal_3rd():
    assert expand_ordinals("3rd") == "third"


def test_ordinal_4th():
    assert expand_ordinals("4th") == "fourth"


def test_ordinal_11th():
    assert expand_ordinals("11th") == "eleventh"


def test_ordinal_12th():
    assert expand_ordinals("12th") == "twelfth"


def test_ordinal_13th():
    assert expand_ordinals("13th") == "thirteenth"


def test_ordinal_20th():
    assert expand_ordinals("20th") == "twentieth"


def test_ordinal_21st():
    assert expand_ordinals("21st") == "twenty first"


def test_ordinal_22nd():
    assert expand_ordinals("22nd") == "twenty second"


def test_ordinal_23rd():
    assert expand_ordinals("23rd") == "twenty third"


def test_ordinal_30th():
    assert expand_ordinals("30th") == "thirtieth"


def test_ordinal_31st():
    assert expand_ordinals("31st") == "thirty first"


def test_ordinal_99th():
    assert expand_ordinals("99th") == "ninety ninth"


def test_ordinal_in_sentence():
    result = expand_ordinals("The 1st item and the 3rd one")
    assert "first" in result
    assert "third" in result


def test_ordinal_empty():
    assert expand_ordinals("") == ""


def test_ordinal_no_match():
    text = "No ordinals here"
    assert expand_ordinals(text) == text


# --- strip_number_commas ---

def test_number_commas_basic():
    assert strip_number_commas("1,234") == "1234"


def test_number_commas_millions():
    assert strip_number_commas("1,234,567") == "1234567"


def test_number_commas_in_sentence():
    result = strip_number_commas("There are 1,000 items")
    assert "1000" in result
    assert "1,000" not in result


def test_number_commas_not_valid_groups():
    """Should not strip commas from non-standard groupings."""
    text = "12,34"
    assert strip_number_commas(text) == text


def test_number_commas_empty():
    assert strip_number_commas("") == ""


# --- expand_fractions_ratios ---

def test_fraction_half():
    assert expand_fractions_ratios("1/2") == "one half"


def test_fraction_third():
    assert expand_fractions_ratios("1/3") == "one third"


def test_fraction_quarter():
    assert expand_fractions_ratios("1/4") == "one quarter"


def test_fraction_three_quarters():
    assert expand_fractions_ratios("3/4") == "three quarters"


def test_fraction_two_thirds():
    assert expand_fractions_ratios("2/3") == "two thirds"


def test_fraction_two_halves():
    """2/2 should be 'two halves'."""
    assert expand_fractions_ratios("2/2") == "two halves"


def test_ratio_basic():
    assert expand_fractions_ratios("16:9") == "16 to 9"


def test_ratio_one_to_two():
    assert expand_fractions_ratios("1:2") == "1 to 2"


def test_fraction_not_path():
    """Should not match file paths like src/foo."""
    text = "src/foo"
    result = expand_fractions_ratios(text)
    assert result == text


def test_fraction_empty():
    assert expand_fractions_ratios("") == ""


def test_fraction_in_sentence():
    result = expand_fractions_ratios("About 1/4 of the time")
    assert "one quarter" in result


# --- expand_time_formats ---

def test_time_24h_afternoon():
    result = expand_time_formats("14:30")
    assert "2:30 PM" in result


def test_time_24h_morning():
    result = expand_time_formats("09:00")
    assert "9 AM" in result


def test_time_24h_late_night():
    result = expand_time_formats("23:59")
    assert "11:59 PM" in result


def test_time_24h_midnight():
    result = expand_time_formats("00:00")
    assert "12 AM" in result


def test_time_24h_noon():
    result = expand_time_formats("12:00")
    assert "12 PM" in result


def test_time_ampm_lowercase():
    result = expand_time_formats("3:15pm")
    assert "3:15 PM" in result


def test_time_ampm_uppercase():
    result = expand_time_formats("3:15 PM")
    assert "3:15 PM" in result


def test_time_am_lowercase():
    result = expand_time_formats("3:15am")
    assert "3:15 AM" in result


def test_time_empty():
    assert expand_time_formats("") == ""


def test_time_no_match():
    text = "No times here"
    assert expand_time_formats(text) == text


# --- expand_dates ---

def test_date_basic():
    assert expand_dates("2024-02-26") == "February 26, 2024"


def test_date_january_first():
    assert expand_dates("2024-01-01") == "January 1, 2024"


def test_date_december():
    assert expand_dates("2023-12-31") == "December 31, 2023"


def test_date_in_sentence():
    result = expand_dates("Released on 2024-02-26 to the public")
    assert "February 26, 2024" in result


def test_date_empty():
    assert expand_dates("") == ""


def test_date_no_match():
    text = "No dates here"
    assert expand_dates(text) == text


def test_date_invalid_month():
    """Month 13 should not match."""
    text = "2024-13-01"
    assert expand_dates(text) == text


# --- expand_temperature ---

def test_temp_fahrenheit():
    assert expand_temperature("72°F") == "72 degrees Fahrenheit"


def test_temp_celsius():
    assert expand_temperature("22°C") == "22 degrees Celsius"


def test_temp_negative():
    result = expand_temperature("-40°F")
    assert "negative 40 degrees Fahrenheit" in result


def test_temp_decimal():
    result = expand_temperature("98.6°F")
    assert "98.6 degrees Fahrenheit" in result


def test_temp_empty():
    assert expand_temperature("") == ""


def test_temp_no_match():
    text = "No temp here"
    assert expand_temperature(text) == text


# --- expand_math_operators ---

def test_math_plus():
    assert expand_math_operators("x + y") == "x plus y"


def test_math_minus():
    assert expand_math_operators("a - b") == "a minus b"


def test_math_equals():
    assert expand_math_operators("a = b") == "a equals b"


def test_math_not_equals():
    assert expand_math_operators("a != b") == "a not equals b"


def test_math_double_equals():
    assert expand_math_operators("a == b") == "a equals b"


def test_math_gte():
    assert expand_math_operators("a >= b") == "a greater than or equal to b"


def test_math_lte():
    assert expand_math_operators("a <= b") == "a less than or equal to b"


def test_math_gt():
    assert expand_math_operators("a > b") == "a greater than b"


def test_math_lt():
    assert expand_math_operators("a < b") == "a less than b"


def test_math_times():
    assert expand_math_operators("a * b") == "a times b"


def test_math_power_2():
    assert expand_math_operators("x^2") == "x squared"


def test_math_power_3():
    assert expand_math_operators("x^3") == "x cubed"


def test_math_power_n():
    assert expand_math_operators("x^n") == "x to the n"


def test_math_empty():
    assert expand_math_operators("") == ""


def test_math_no_match():
    text = "No math here"
    assert expand_math_operators(text) == text


# --- expand_slash_pairs ---

def test_slash_and_or():
    assert expand_slash_pairs("and/or") == "and or"


def test_slash_true_false():
    assert expand_slash_pairs("true/false") == "true or false"


def test_slash_yes_no():
    assert expand_slash_pairs("yes/no") == "yes or no"


def test_slash_input_output():
    assert expand_slash_pairs("input/output") == "input output"


def test_slash_generic():
    result = expand_slash_pairs("before/after")
    assert "before or after" in result


def test_slash_empty():
    assert expand_slash_pairs("") == ""


def test_slash_no_match():
    text = "No slashes here"
    assert expand_slash_pairs(text) == text


# --- Full pipeline with new features ---

def test_normalize_with_currency():
    result = normalize("It costs $99.99 for the license")
    assert "99 dollars and 99 cents" in result


def test_normalize_with_percentage():
    result = normalize("Success rate is 99.9%")
    assert "percent" in result


def test_normalize_with_ordinals():
    result = normalize("This is the 1st release")
    assert "first" in result


def test_normalize_with_date():
    result = normalize("Released on 2024-02-26")
    assert "February" in result


def test_normalize_with_temperature():
    result = normalize("The temperature is 72°F")
    assert "degrees Fahrenheit" in result


def test_normalize_src_to_source():
    """src/ in file paths should become 'source' in speech."""
    result = normalize("Edit src/script.py")
    assert "source" in result
    assert "src/" not in result


def test_normalize_url_handling():
    result = normalize("Check https://github.com/user/repo for details")
    assert "github link" in result
    assert "https://" not in result


def test_normalize_fractions():
    result = normalize("About 3/4 of the work is done")
    assert "three quarters" in result


def test_normalize_math():
    result = normalize("Calculate x + y for the result")
    assert "plus" in result


def test_normalize_slash_pair():
    result = normalize("Use true/false values")
    assert "true or false" in result


# ===================================================================
# Pronunciation Dictionary Tests
# ===================================================================

import claude_speak.normalizer as normalizer_module


def _write_pron_file(path, content: str):
    """Helper: write a TOML pronunciations file to *path*."""
    path.write_bytes(content.encode())


def test_pron_default_dict_loads(monkeypatch):
    """Built-in pronunciations.toml should load and contain known terms."""
    # Force cache miss so we re-read from the built-in file.
    monkeypatch.setattr(normalizer_module, "_pron_cache", None)
    # Make sure the user file is NOT present.
    monkeypatch.setattr(normalizer_module, "_USER_PRON_PATH",
                        normalizer_module._BUILTIN_PRON_PATH.parent / "nonexistent_user.toml")

    result = apply_pronunciation_overrides("run kubectl now")
    assert "kube control" in result
    assert "kubectl" not in result


def test_pron_default_nginx(monkeypatch):
    """nginx should become 'engine X' from the built-in dict."""
    monkeypatch.setattr(normalizer_module, "_pron_cache", None)
    monkeypatch.setattr(normalizer_module, "_USER_PRON_PATH",
                        normalizer_module._BUILTIN_PRON_PATH.parent / "nonexistent_user.toml")

    result = apply_pronunciation_overrides("configure nginx")
    assert "engine X" in result


def test_pron_custom_file_overrides_default(tmp_path, monkeypatch):
    """A custom pronunciations file should replace the built-in dict entirely."""
    custom = tmp_path / "pronunciations.toml"
    _write_pron_file(custom, '[terms]\nwidget = "wid jet"\n')

    monkeypatch.setattr(normalizer_module, "_pron_cache", None)
    monkeypatch.setattr(normalizer_module, "_USER_PRON_PATH", custom)

    result = apply_pronunciation_overrides("use the widget now")
    assert "wid jet" in result
    # The built-in 'kubectl' key should NOT match because we loaded the custom file.
    result2 = apply_pronunciation_overrides("run kubectl here")
    assert "kubectl" in result2  # not overridden by the custom file


def test_pron_whole_word_only(monkeypatch):
    """Replacements must be whole-word — partial matches must not fire."""
    custom_toml = '[terms]\nfoo = "REPLACED"\n'
    import tempfile
    import pathlib
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="wb") as fh:
        fh.write(custom_toml.encode())
        tmp_path = pathlib.Path(fh.name)

    try:
        monkeypatch.setattr(normalizer_module, "_pron_cache", None)
        monkeypatch.setattr(normalizer_module, "_USER_PRON_PATH", tmp_path)

        # "foo" inside "foobar" should NOT be replaced; standalone "foo" should.
        result = apply_pronunciation_overrides("foobar is not foo")
        assert "foobar" in result       # partial match preserved (no replacement inside word)
        assert "REPLACED" in result     # standalone whole-word 'foo' was replaced
        # The word "foobar" must NOT have been altered to "foobREPLACED" or similar.
        assert "fooREPLACED" not in result
    finally:
        tmp_path.unlink(missing_ok=True)


def test_pron_case_insensitive_matching(tmp_path, monkeypatch):
    """Keys should match regardless of case in the input text."""
    custom = tmp_path / "pronunciations.toml"
    _write_pron_file(custom, '[terms]\nkubectl = "kube control"\n')

    monkeypatch.setattr(normalizer_module, "_pron_cache", None)
    monkeypatch.setattr(normalizer_module, "_USER_PRON_PATH", custom)

    assert "kube control" in apply_pronunciation_overrides("Kubectl apply")
    assert "kube control" in apply_pronunciation_overrides("KUBECTL apply")
    assert "kube control" in apply_pronunciation_overrides("kubectl apply")


def test_pron_cache_reloads_on_mtime_change(tmp_path, monkeypatch):
    """Cache should be invalidated and reloaded when the file's mtime changes."""
    custom = tmp_path / "pronunciations.toml"
    _write_pron_file(custom, '[terms]\nfizz = "fizz pop"\n')

    monkeypatch.setattr(normalizer_module, "_pron_cache", None)
    monkeypatch.setattr(normalizer_module, "_USER_PRON_PATH", custom)

    result1 = apply_pronunciation_overrides("drink some fizz")
    assert "fizz pop" in result1

    # Overwrite the file with different content and bump mtime artificially.
    _write_pron_file(custom, '[terms]\nbuzz = "buzz saw"\n')
    # Touch the file to guarantee a new mtime even on fast filesystems.
    import time as _time
    new_mtime = custom.stat().st_mtime + 1
    os.utime(custom, (new_mtime, new_mtime))

    result2 = apply_pronunciation_overrides("drink some fizz now buzz")
    # Old term should no longer be replaced (new file has different terms).
    assert "fizz pop" not in result2
    # New term should be replaced.
    assert "buzz saw" in result2


def test_pron_missing_user_file_uses_default(tmp_path, monkeypatch):
    """When user file is absent, built-in dict is used gracefully."""
    monkeypatch.setattr(normalizer_module, "_pron_cache", None)
    # Point user path to a non-existent file.
    monkeypatch.setattr(normalizer_module, "_USER_PRON_PATH",
                        tmp_path / "does_not_exist.toml")

    result = apply_pronunciation_overrides("start nginx")
    # Built-in nginx pronunciation should still apply.
    assert "engine X" in result


def test_pron_empty_text(monkeypatch):
    """Empty string should pass through unchanged."""
    monkeypatch.setattr(normalizer_module, "_pron_cache", None)
    monkeypatch.setattr(normalizer_module, "_USER_PRON_PATH",
                        normalizer_module._BUILTIN_PRON_PATH.parent / "nonexistent_user.toml")
    assert apply_pronunciation_overrides("") == ""


def test_pron_no_match_passes_through(tmp_path, monkeypatch):
    """Text with no matching terms should be returned unchanged."""
    custom = tmp_path / "pronunciations.toml"
    _write_pron_file(custom, '[terms]\nalpha = "alpha one"\n')

    monkeypatch.setattr(normalizer_module, "_pron_cache", None)
    monkeypatch.setattr(normalizer_module, "_USER_PRON_PATH", custom)

    text = "nothing to replace here"
    assert apply_pronunciation_overrides(text) == text


def test_pron_wired_into_normalize_pipeline(monkeypatch):
    """apply_pronunciation_overrides should be called as the final normalize() step."""
    monkeypatch.setattr(normalizer_module, "_pron_cache", None)
    monkeypatch.setattr(normalizer_module, "_USER_PRON_PATH",
                        normalizer_module._BUILTIN_PRON_PATH.parent / "nonexistent_user.toml")

    result = normalize("use kubectl to deploy")
    assert "kube control" in result
    assert "kubectl" not in result


def test_pron_custom_pron_config_field():
    """NormalizationConfig should have custom_pronunciations field."""
    from claude_speak.config import NormalizationConfig
    cfg = NormalizationConfig()
    assert hasattr(cfg, "custom_pronunciations")
    assert cfg.custom_pronunciations == ""


def test_pron_custom_pron_config_overridable(tmp_path, monkeypatch):
    """custom_pronunciations in config TOML should be readable."""
    import claude_speak.config as config_module
    custom_path = str(tmp_path / "my_prons.toml")
    toml_content = f'[normalization]\ncustom_pronunciations = "{custom_path}"\n'.encode()
    toml_file = tmp_path / "claude-speak.toml"
    toml_file.write_bytes(toml_content)
    monkeypatch.setattr(config_module, "CONFIG_PATH", toml_file)

    cfg = config_module.load_config()
    assert cfg.normalization.custom_pronunciations == custom_path


# ===================================================================
# Queue Tests
# ===================================================================

import claude_speak.queue as queue_module


def test_enqueue_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, "QUEUE_DIR", tmp_path)
    path = queue_module.enqueue("hello world")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "hello world"
    assert path.suffix == ".txt"


def test_dequeue_returns_oldest(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, "QUEUE_DIR", tmp_path)
    queue_module.enqueue("first")
    time.sleep(0.01)  # ensure distinct timestamps
    queue_module.enqueue("second")

    result = queue_module.dequeue()
    assert result is not None
    _, text = result
    assert text == "first"


def test_dequeue_removes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, "QUEUE_DIR", tmp_path)
    path = queue_module.enqueue("remove me")
    queue_module.dequeue()
    assert not path.exists()


def test_dequeue_empty_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, "QUEUE_DIR", tmp_path)
    assert queue_module.dequeue() is None


def test_enqueue_chunks_creates_multiple(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, "QUEUE_DIR", tmp_path)
    chunks = ["chunk one", "chunk two", "chunk three"]
    paths = queue_module.enqueue_chunks(chunks)
    assert len(paths) == 3
    for p in paths:
        assert p.exists()


def test_enqueue_chunks_preserves_order(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, "QUEUE_DIR", tmp_path)
    chunks = ["first", "second", "third"]
    queue_module.enqueue_chunks(chunks)
    # Dequeue in order
    _, text1 = queue_module.dequeue()
    _, text2 = queue_module.dequeue()
    _, text3 = queue_module.dequeue()
    assert text1 == "first"
    assert text2 == "second"
    assert text3 == "third"


def test_peek_returns_files_without_removing(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, "QUEUE_DIR", tmp_path)
    queue_module.enqueue("peeked")
    files = queue_module.peek()
    assert len(files) == 1
    # File should still exist after peek
    assert files[0].exists()


def test_clear_removes_all(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, "QUEUE_DIR", tmp_path)
    queue_module.enqueue("a")
    queue_module.enqueue("b")
    queue_module.enqueue("c")
    queue_module.clear()
    assert queue_module.depth() == 0


def test_depth_returns_count(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, "QUEUE_DIR", tmp_path)
    assert queue_module.depth() == 0
    queue_module.enqueue("x")
    assert queue_module.depth() == 1
    queue_module.enqueue("y")
    assert queue_module.depth() == 2
    queue_module.dequeue()
    assert queue_module.depth() == 1


def test_fifo_ordering(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, "QUEUE_DIR", tmp_path)
    items = ["alpha", "beta", "gamma", "delta"]
    for item in items:
        queue_module.enqueue(item)
        time.sleep(0.01)  # ensure distinct timestamps

    dequeued = []
    while True:
        result = queue_module.dequeue()
        if result is None:
            break
        dequeued.append(result[1])

    assert dequeued == items


# ===================================================================
# Config Tests
# ===================================================================

import claude_speak.config as config_module
from claude_speak.config import load_config, Config, TTSConfig


def test_load_config_defaults_when_no_file(tmp_path, monkeypatch):
    """When no TOML file exists, load_config returns all defaults."""
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "nonexistent.toml")
    config = load_config()
    assert isinstance(config, Config)


def test_default_voice():
    config = Config()
    assert config.tts.voice == "af_sarah"


def test_default_speed():
    config = Config()
    assert config.tts.speed == 1.0


def test_default_device():
    config = Config()
    assert config.tts.device == "auto"


def test_default_max_chunk_chars():
    config = Config()
    assert config.tts.max_chunk_chars == 400


def test_default_wakeword_disabled():
    config = Config()
    assert config.wakeword.enabled is False


def test_default_audio_volume():
    config = Config()
    assert config.audio.volume == 0.3


def test_toml_overrides_defaults(tmp_path, monkeypatch):
    """Values in the TOML file should override defaults."""
    toml_content = b'[tts]\nvoice = "bf_emma"\nspeed = 1.5\n'
    toml_path = tmp_path / "claude-speak.toml"
    toml_path.write_bytes(toml_content)
    monkeypatch.setattr(config_module, "CONFIG_PATH", toml_path)

    config = load_config()
    assert config.tts.voice == "bf_emma"
    assert config.tts.speed == 1.5
    # Non-overridden defaults should remain
    assert config.tts.device == "auto"


def test_env_vars_override_toml(tmp_path, monkeypatch):
    """Environment variables should take highest priority."""
    toml_content = b'[tts]\nvoice = "bf_emma"\nspeed = 1.5\n'
    toml_path = tmp_path / "claude-speak.toml"
    toml_path.write_bytes(toml_content)
    monkeypatch.setattr(config_module, "CONFIG_PATH", toml_path)

    monkeypatch.setenv("CLAUDE_SPEAK_VOICE", "am_adam")
    monkeypatch.setenv("CLAUDE_SPEAK_SPEED", "2.0")
    monkeypatch.setenv("CLAUDE_SPEAK_DEVICE", "cpu")

    config = load_config()
    assert config.tts.voice == "am_adam"
    assert config.tts.speed == 2.0
    assert config.tts.device == "cpu"
