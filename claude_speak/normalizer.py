"""
Text normalization pipeline for natural speech output.
Transforms Claude's markdown/technical text into speech-friendly prose.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Unit expansions
# ---------------------------------------------------------------------------

UNIT_MAP = {
    "B": "bytes", "KB": "kilobytes", "MB": "megabytes", "GB": "gigabytes",
    "TB": "terabytes", "PB": "petabytes",
    "s": "seconds", "ms": "milliseconds", "μs": "microseconds",
    "ns": "nanoseconds", "ps": "picoseconds",
    "Hz": "hertz", "kHz": "kilohertz", "MHz": "megahertz", "GHz": "gigahertz",
    "px": "pixels", "fps": "frames per second",
    "bps": "bits per second", "Kbps": "kilobits per second",
    "Mbps": "megabits per second", "Gbps": "gigabits per second",
}

# ---------------------------------------------------------------------------
# Abbreviation expansions
# ---------------------------------------------------------------------------

ABBREV_MAP = {
    "API": "A P I", "APIs": "A P I's", "CLI": "C L I",
    "TTS": "T T S", "STT": "S T T",
    "URL": "U R L", "URLs": "U R L's",
    "HTML": "H T M L", "CSS": "C S S",
    "JSON": "JSON", "JSONL": "JSON L",
    "YAML": "YAML", "SQL": "S Q L",
    "SSH": "S S H", "HTTP": "H T T P", "HTTPS": "H T T P S",
    "REST": "REST", "SDK": "S D K", "IDE": "I D E",
    "PR": "P R", "PRs": "P R's",
    "MCP": "M C P", "LLM": "L L M", "LLMs": "L L M's",
    "AI": "A I", "GPU": "G P U", "CPU": "C P U", "RAM": "RAM",
    "ONNX": "onyx", "NPM": "N P M", "PyPI": "pie pie",
    "AWS": "A W S", "GCP": "G C P",
    "JWT": "J W T", "OAuth": "O Auth", "UUID": "U U I D",
    "STDIN": "standard in", "STDOUT": "standard out", "STDERR": "standard error",
    "EOF": "end of file", "PID": "process I D",
    "README": "read me", "TODO": "to do",
    "TCP": "T C P", "UDP": "U D P", "DNS": "D N S",
    "IP": "I P", "OS": "O S", "UI": "U I", "UX": "U X",
    "DB": "database", "ORM": "O R M",
    "WAV": "wave", "MP3": "M P 3", "PDF": "P D F",
    "CSV": "C S V", "XML": "X M L", "SVG": "S V G",
    "PNG": "P N G", "JPG": "J P G", "GIF": "gif",
    "ASYNC": "async", "CORS": "cors", "REGEX": "regex",
    "REPL": "repl", "WSL": "W S L",
    "SaaS": "sass", "CI/CD": "C I C D",
}

# ---------------------------------------------------------------------------
# Stop-word / shorthand expansions (applied before general abbreviations)
# ---------------------------------------------------------------------------

STOP_WORD_MAP = {
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "etcetera",
    "vs.": "versus",
    "w/o": "without",
    "w/": "with",
}

# ---------------------------------------------------------------------------
# Language tag → spoken name for code block descriptions
# ---------------------------------------------------------------------------

LANG_NAMES = {
    "python": "Python", "py": "Python",
    "bash": "bash", "sh": "shell", "zsh": "shell", "shell": "shell",
    "javascript": "JavaScript", "js": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript",
    "json": "JSON", "jsonl": "JSON",
    "yaml": "YAML", "yml": "YAML",
    "html": "HTML", "css": "CSS",
    "sql": "SQL",
    "rust": "Rust", "rs": "Rust",
    "go": "Go", "golang": "Go",
    "java": "Java",
    "ruby": "Ruby", "rb": "Ruby",
    "swift": "Swift",
    "c": "C", "cpp": "C++", "c++": "C++",
    "toml": "TOML", "xml": "XML", "csv": "CSV",
    "dockerfile": "Dockerfile", "docker": "Docker",
    "makefile": "Makefile", "make": "Makefile",
    "markdown": "Markdown", "md": "Markdown",
    "plaintext": "text", "text": "text", "txt": "text",
    "diff": "diff",
    "ini": "config", "conf": "config", "cfg": "config",
}

# ---------------------------------------------------------------------------
# File extension pronunciations
# ---------------------------------------------------------------------------

EXT_MAP = {
    ".py": "dot py", ".sh": "dot sh", ".js": "dot js", ".ts": "dot ts",
    ".json": "dot JSON", ".jsonl": "dot JSON L",
    ".yaml": "dot YAML", ".yml": "dot YAML",
    ".md": "dot md", ".txt": "dot text",
    ".wav": "dot wave", ".mp3": "dot M P 3",
    ".css": "dot C S S", ".html": "dot H T M L",
    ".xml": "dot X M L", ".csv": "dot C S V",
    ".env": "dot env", ".toml": "dot toml",
    ".onnx": "dot onyx", ".bin": "dot bin",
    ".rs": "dot rs", ".go": "dot go", ".java": "dot java",
    ".rb": "dot ruby", ".swift": "dot swift",
}

# ---------------------------------------------------------------------------
# Currency symbols
# ---------------------------------------------------------------------------

CURRENCY_MAP = {
    "$": "dollar",
    "€": "euro",
    "£": "pound",
    "¥": "yen",
}

MAGNITUDE_MAP = {
    "K": "thousand",
    "M": "million",
    "B": "billion",
    "T": "trillion",
}

# ---------------------------------------------------------------------------
# Ordinal data for expand_ordinals
# ---------------------------------------------------------------------------

_ONES_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
}

_TEENS_ORDINALS = {
    10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth",
    14: "fourteenth", 15: "fifteenth", 16: "sixteenth", 17: "seventeenth",
    18: "eighteenth", 19: "nineteenth",
}

_TENS_ORDINALS = {
    20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth",
    60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth",
}

_TENS_CARDINAL = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}

# Cardinal words for fractions
_CARDINAL_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

# Ordinal words for fraction denominators
_FRACTION_DENOM = {
    2: "half", 3: "third", 4: "quarter", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
}

# Month names
_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def _number_to_ordinal(n: int) -> str:
    """Convert an integer (1-99) to its ordinal word form."""
    if n <= 0 or n > 99:
        return f"{n}th"
    if n < 10:
        return _ONES_ORDINALS.get(n, f"{n}th")
    if n < 20:
        return _TEENS_ORDINALS.get(n, f"{n}th")
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS_ORDINALS.get(n, f"{n}th")
    tens_word = _TENS_CARDINAL.get(tens * 10, "")
    ones_ordinal = _ONES_ORDINALS.get(ones, f"{ones}th")
    return f"{tens_word} {ones_ordinal}"


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

# describe_code_blocks
_RE_FENCED_CODE = re.compile(r"```(\w*)\s*\n(.*?)```", re.DOTALL)

# improve_lists
_RE_NUMBERED_ITEM = re.compile(r"^(\d+)[.)]\s+(.*)")
_RE_BULLET_ITEM = re.compile(r"^( *)[-*+]\s+(.*)")

# expand_stop_words
_RE_WITHOUT = re.compile(r"\bw/o\b")
_RE_WITH = re.compile(r"\bw/(?=\s)")
_RE_EG = re.compile(r"\be\.g\.\s?")
_RE_IE = re.compile(r"\bi\.e\.\s?")
_RE_ETC = re.compile(r"\betc\.")
_RE_VS = re.compile(r"\bvs\.\s?")

# expand_units (built from UNIT_MAP)
_SORTED_UNITS = sorted(UNIT_MAP.keys(), key=len, reverse=True)
_RE_UNITS = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" + "|".join(re.escape(u) for u in _SORTED_UNITS) + r")\b"
)

# expand_abbreviations (built from ABBREV_MAP)
_RE_ABBREVS = {
    abbr: re.compile(r"\b" + re.escape(abbr) + r"\b")
    for abbr in ABBREV_MAP
}

# clean_file_paths
_RE_LONG_PATH = re.compile(r"/(?:[\w.~-]+/){2,}[\w.~-]+")
_RE_HOME_PATH = re.compile(r"~(?:/[\w.~-]+){2,}")
_RE_SRC_PREFIX = re.compile(r"\bsrc/")

# clean_file_extensions (built from EXT_MAP)
_RE_FILE_EXTS = {
    ext: re.compile(re.escape(ext) + r"\b")
    for ext in EXT_MAP
}

# speak_decimal_numbers
_RE_DECIMAL = re.compile(r"\b(\d+)\.(\d+)(x)?\b", re.IGNORECASE)

# clean_version_strings
_RE_VERSION_V = re.compile(
    r"(?:(version)\s+)?[vV](\d+(?:\.\d+)+)\b", re.IGNORECASE
)
_RE_VERSION_BARE = re.compile(
    r"\b(version)\s+(\d+(?:\.\d+)+)\b", re.IGNORECASE
)

# clean_technical_punctuation
_RE_EM_DASH = re.compile(r"\s*[\u2014\u2013]\s*")  # em dash + en dash
_RE_CURLY = re.compile(r"[{}]")
_RE_SQUARE = re.compile(r"[\[\]]")
_RE_EMPTY_PARENS = re.compile(r"\(\s*\)")
_RE_PARENTHETICAL = re.compile(r"\(([^)]{1,200})\)")
_RE_MULTI_SPACE = re.compile(r"  +")
_RE_TILDE_NUM = re.compile(r"~(\d)")
_RE_TILDE_PATH = re.compile(r"~/")
_RE_HASH_HEADING = re.compile(r"#+ ?")
_RE_SNAKE_CASE = re.compile(r"\b([a-z]+)_([a-z]+(?:_[a-z]+)*)\b")
_RE_TRIPLE_HYPHEN = re.compile(r"\b(\w+)-(\w+)-(\w+)\b")
_RE_DOUBLE_HYPHEN = re.compile(r"\b(\w+)-(\w+)\b")
_RE_AMPERSAND = re.compile(r"&")
_RE_HASHTAG_WORD = re.compile(r"#(\w+)")
_RE_AT_WORD = re.compile(r"@(\w+)")

# final_cleanup
_RE_DOUBLE_COMMA = re.compile(r",\s*,")
_RE_LEADING_COMMA = re.compile(r"^\s*,\s*", re.MULTILINE)
_RE_TRAILING_COMMA = re.compile(r"\s*,\s*$", re.MULTILINE)
_RE_MULTI_COMMA = re.compile(r",(\s*,)+")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_PUNCT_ONLY_LINE = re.compile(r"^[\s,.:;!?\-]+$")

# chunk_text
_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_RE_COMMA_SPLIT = re.compile(r',\s*')

# clean_urls_and_emails
_RE_GITHUB_URL = re.compile(r"https?://github\.com/\S+")
_RE_PYTHON_DOCS_URL = re.compile(r"https?://docs\.python\.org\S*")
_RE_LOCALHOST_URL = re.compile(r"https?://localhost(?::(\d+))?\S*")
_RE_GENERIC_URL = re.compile(r"https?://([a-zA-Z0-9.-]+)(?:/\S*)?")
_RE_EMAIL = re.compile(r"\b([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b")
_RE_BARE_DOMAIN = re.compile(
    r"(?<!\S)([a-zA-Z0-9-]+\.(?:com|org|net|io|dev|co|edu|gov|app|ai|py))\b"
)

# expand_currency
_RE_CURRENCY_MAG = re.compile(
    r"([$€£¥])(\d+(?:\.\d+)?)\s*([KMBTkmbt])\b"
)
_RE_CURRENCY_CENTS = re.compile(
    r"([$€£¥])(\d+)\.(\d{2})\b"
)
_RE_CURRENCY_PLAIN = re.compile(
    r"([$€£¥])(\d+(?:\.\d+)?)\b"
)

# expand_percentages
_RE_PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")

# expand_ordinals
_RE_ORDINAL = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b")

# strip_number_commas
_RE_NUMBER_COMMAS = re.compile(r"\b(\d{1,3})((?:,\d{3})+)\b")

# expand_fractions_ratios
_RE_FRACTION = re.compile(r"(?<![/\w])(\d+)/(\d+)(?![/\w])")
_RE_RATIO = re.compile(r"\b(\d+):(\d+)\b(?!\s*(?:AM|PM|am|pm))")

# expand_time_formats
_RE_TIME_24H = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_RE_TIME_AMPM = re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)\b")

# expand_dates
_RE_ISO_DATE = re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")

# expand_temperature
_RE_TEMP = re.compile(r"(-?\d+(?:\.\d+)?)\s*°\s*([FCfc])\b")

# expand_math_operators
_RE_NOT_EQUALS = re.compile(r"(\S+)\s*!=\s*(\S+)")
_RE_DOUBLE_EQUALS = re.compile(r"(\S+)\s*==\s*(\S+)")
_RE_GTE = re.compile(r"(\S+)\s*>=\s*(\S+)")
_RE_LTE = re.compile(r"(\S+)\s*<=\s*(\S+)")
_RE_GT = re.compile(r"(\S+) > (\S+)")
_RE_LT = re.compile(r"(\S+) < (\S+)")
_RE_PLUS = re.compile(r"(\S+) \+ (\S+)")
_RE_MINUS = re.compile(r"(\S+) - (\S+)")
_RE_TIMES = re.compile(r"(\S+) \* (\S+)")
_RE_EQUALS = re.compile(r"(\S+) = (\S+)")
_RE_POWER_2 = re.compile(r"(\w+)\^2\b")
_RE_POWER_3 = re.compile(r"(\w+)\^3\b")
_RE_POWER_N = re.compile(r"(\w+)\^(\w+)")

# expand_slash_pairs
_SLASH_COMMON = {
    "and/or": "and or",
    "true/false": "true or false",
    "yes/no": "yes or no",
    "input/output": "input output",
}
_RE_SLASH_PAIR = re.compile(r"\b([a-zA-Z]+)/([a-zA-Z]+)\b")


# ---------------------------------------------------------------------------
# Transform functions
# ---------------------------------------------------------------------------

def describe_code_blocks(text: str) -> str:
    """Replace fenced code blocks with brief spoken descriptions.

    Fenced blocks (```lang ... ```) become a short phrase so the listener
    knows code was present without hearing raw syntax.

    Examples:
        '```python\\ndef hello():\\n    print("hi")\\n```'
            → 'Here is a Python code snippet.'
        '```bash\\npip install foo\\n```'
            → 'Here is a bash command.'
        '```\\nsome code\\n```'
            → 'Here is a code block.'
    """
    def _describe(m: re.Match[str]) -> str:
        lang_tag = (m.group(1) or "").strip().lower()
        body = m.group(2).strip()
        lang_name = LANG_NAMES.get(lang_tag)

        # Determine whether to say "command" or "code snippet"
        command_langs = {"bash", "sh", "zsh", "shell"}
        is_command = lang_tag in command_langs

        # For untagged blocks, peek at the body: if it's a single short
        # line that starts with a common command, call it a command.
        if not lang_tag and body:
            lines = [line for line in body.split("\n") if line.strip()]
            if len(lines) <= 2:
                is_command = True

        if lang_name:
            noun = "command" if is_command else "code snippet"
            return f"Here is a {lang_name} {noun}."
        else:
            noun = "command" if is_command else "code block"
            return f"Here is a {noun}."

    return _RE_FENCED_CODE.sub(_describe, text)


def narrate_tables(text: str) -> str:
    """Convert markdown tables into natural spoken prose.

    Examples:
        '| Name | Type |\\n|------|------|\\n| voice | string |'
            → 'A table with columns Name and Type. Row 1: voice, string.'

    Large tables (more than 5 data rows) are summarized instead of
    reading every row.
    """
    def _oxford_join(items: list[str]) -> str:
        """Join items with commas and 'and' before the last item."""
        if len(items) == 0:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + ", and " + items[-1]

    def _parse_row(line: str) -> list[str]:
        """Extract cell values from a markdown table row."""
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        return [c for c in cells if c]

    def _is_separator(line: str) -> bool:
        """Check if a line is a table separator (|---|---|)."""
        stripped = line.strip().strip("|")
        return bool(stripped) and all(
            c in "-: " for c in stripped.replace("|", "")
        )

    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        # Detect start of a table: a line starting with | followed by
        # a separator line.
        if (
            i + 1 < len(lines)
            and lines[i].strip().startswith("|")
            and _is_separator(lines[i + 1])
        ):
            headers = _parse_row(lines[i])
            # Skip header and separator lines
            i += 2

            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = _parse_row(lines[i])
                if not _is_separator(lines[i]):
                    rows.append(row)
                i += 1

            # Build spoken text
            header_text = _oxford_join(headers)

            if len(rows) > 5:
                # Summarize large tables
                narration = (
                    f"A table with {len(rows)} rows showing "
                    f"{header_text}."
                )
            else:
                parts = [f"A table with columns {header_text}."]
                for idx, row in enumerate(rows, 1):
                    row_text = ", ".join(row)
                    parts.append(f"Row {idx}: {row_text}.")
                narration = " ".join(parts)

            result.append(narration)
        else:
            result.append(lines[i])
            i += 1

    return "\n".join(result)


def improve_lists(text: str) -> str:
    """Make bullet and numbered lists sound natural when spoken.

    - Numbered lists get ordinal prefixes: 'First, ... Second, ...'
    - Bullet lists get periods between items for pauses
    - Nested bullets (indented) are flattened with a 'sub-item' prefix

    Examples:
        '1. Install Python\\n2. Run the script'
            → 'First, Install Python.\\nSecond, Run the script.'
        '- Fast\\n- Simple\\n  - Very simple'
            → 'Fast.\\nSimple.\\nSub-item, Very simple.'
    """
    ORDINALS = [
        "First", "Second", "Third", "Fourth", "Fifth",
        "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
    ]

    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- Numbered list detection ---
        num_match = _RE_NUMBERED_ITEM.match(stripped)
        if num_match:
            # Collect the entire numbered list run
            num_items = []
            while i < len(lines):
                s = lines[i].strip()
                m = _RE_NUMBERED_ITEM.match(s)
                if m:
                    num_items.append(m.group(2))
                    i += 1
                else:
                    break
            for idx, item in enumerate(num_items):
                ordinal = ORDINALS[idx] if idx < len(ORDINALS) else f"Number {idx + 1}"
                # Ensure the item ends with a period for pausing
                item_text = item.rstrip().rstrip(".")
                result.append(f"{ordinal}, {item_text}.")
            continue

        # --- Bullet list detection ---
        bullet_match = _RE_BULLET_ITEM.match(line)
        if bullet_match:
            # Collect the entire bullet list run
            bullet_items = []
            while i < len(lines):
                bm = _RE_BULLET_ITEM.match(lines[i])
                if bm:
                    indent = len(bm.group(1))
                    content = bm.group(2)
                    bullet_items.append((indent, content))
                    i += 1
                else:
                    break
            # Determine base indent level
            base_indent = min(ind for ind, _ in bullet_items) if bullet_items else 0
            for indent, content in bullet_items:
                content_text = content.rstrip().rstrip(".")
                if indent > base_indent:
                    result.append(f"Sub-item, {content_text}.")
                else:
                    result.append(f"{content_text}.")
            continue

        result.append(line)
        i += 1

    return "\n".join(result)


def strip_code_blocks(text: str) -> str:
    """Remove standalone code lines, keep prose that mentions commands.

    Only skip lines that are clearly raw commands (start with $ or >>>)
    or lines that are almost entirely non-alphabetic (syntax/symbols).
    Lines like 'Turn off: rm ~/.claude-speak-enabled' are KEPT because
    they're instructions the user should hear.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        # Only skip lines that are clearly raw shell/code commands
        # (starting with $ prompt or Python REPL prompt)
        if stripped.startswith("$ ") or stripped.startswith(">>> "):
            continue
        # Skip lines that are almost entirely non-alphabetic (likely code/syntax)
        alpha = sum(1 for c in stripped if c.isalpha() or c == " ")
        if len(stripped) > 15 and alpha / max(len(stripped), 1) < 0.3:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# NEW transform functions
# ---------------------------------------------------------------------------

def clean_urls_and_emails(text: str) -> str:
    """Replace URLs and emails with spoken descriptions.

    Examples:
        'https://github.com/user/repo' → 'a github link'
        'https://docs.python.org/3/library' → 'a python docs link'
        'http://localhost:3000' → 'localhost 3000'
        'https://example.com/some/path' → 'a link to example dot com'
        'user@example.com' → 'user at example dot com'
    """
    # GitHub URLs
    text = _RE_GITHUB_URL.sub("a github link", text)
    # Python docs URLs
    text = _RE_PYTHON_DOCS_URL.sub("a python docs link", text)
    # Localhost URLs
    def _localhost(m: re.Match[str]) -> str:
        port = m.group(1)
        if port:
            return f"localhost {port}"
        return "localhost"
    text = _RE_LOCALHOST_URL.sub(_localhost, text)
    # Generic URLs
    def _generic_url(m: re.Match[str]) -> str:
        domain = m.group(1)
        spoken_domain = domain.replace(".", " dot ")
        return f"a link to {spoken_domain}"
    text = _RE_GENERIC_URL.sub(_generic_url, text)
    # Emails
    def _email(m: re.Match[str]) -> str:
        user = m.group(1)
        domain = m.group(2)
        spoken_domain = domain.replace(".", " dot ")
        return f"{user} at {spoken_domain}"
    text = _RE_EMAIL.sub(_email, text)
    # Bare domains (only when clearly a domain with known TLD)
    def _bare_domain(m: re.Match[str]) -> str:
        domain = m.group(1)
        return domain.replace(".", " dot ")
    text = _RE_BARE_DOMAIN.sub(_bare_domain, text)
    return text


def expand_currency(text: str) -> str:
    """Expand currency symbols to spoken form.

    Examples:
        '$100' → '100 dollars'
        '$99.99' → '99 dollars and 99 cents'
        '€50' → '50 euros'
        '$1' → '1 dollar'
        '$1.5M' → '1.5 million dollars'
    """
    # Handle magnitude suffixes first: $1.5M, $2B, etc.
    def _currency_mag(m: re.Match[str]) -> str:
        sym = m.group(1)
        amount = m.group(2)
        mag = m.group(3).upper()
        base = CURRENCY_MAP.get(sym, "dollar")
        mag_word = MAGNITUDE_MAP.get(mag, "")
        # Pluralize
        if base == "dollar":
            plural = "dollars"
        elif base == "euro":
            plural = "euros"
        elif base == "pound":
            plural = "pounds"
        else:
            plural = base  # yen is same singular/plural
        return f"{amount} {mag_word} {plural}"
    text = _RE_CURRENCY_MAG.sub(_currency_mag, text)

    # Handle cents: $99.99
    def _currency_cents(m: re.Match[str]) -> str:
        sym = m.group(1)
        whole = m.group(2)
        cents = m.group(3)
        base = CURRENCY_MAP.get(sym, "dollar")
        if base == "dollar":
            dollar_word = "dollar" if whole == "1" else "dollars"
            cent_word = "cent" if cents == "01" else "cents"
            return f"{whole} {dollar_word} and {cents} {cent_word}"
        elif base == "euro":
            euro_word = "euro" if whole == "1" else "euros"
            cent_word = "cent" if cents == "01" else "cents"
            return f"{whole} {euro_word} and {cents} {cent_word}"
        elif base == "pound":
            pound_word = "pound" if whole == "1" else "pounds"
            pence_word = "pence"
            return f"{whole} {pound_word} and {cents} {pence_word}"
        else:
            return f"{whole} {base}"
    text = _RE_CURRENCY_CENTS.sub(_currency_cents, text)

    # Handle plain amounts: $100, €50, etc.
    def _currency_plain(m: re.Match[str]) -> str:
        sym = m.group(1)
        amount = m.group(2)
        base = CURRENCY_MAP.get(sym, "dollar")
        # Determine singular/plural
        is_singular = (amount == "1" or amount == "1.0")
        if base == "dollar":
            word = "dollar" if is_singular else "dollars"
        elif base == "euro":
            word = "euro" if is_singular else "euros"
        elif base == "pound":
            word = "pound" if is_singular else "pounds"
        else:
            word = base  # yen
        return f"{amount} {word}"
    text = _RE_CURRENCY_PLAIN.sub(_currency_plain, text)

    return text


def expand_percentages(text: str) -> str:
    """Expand percentage signs to spoken form.

    Examples:
        '75%' → '75 percent'
        '99.9%' → '99.9 percent'
    """
    return _RE_PERCENT.sub(r"\1 percent", text)


def expand_ordinals(text: str) -> str:
    """Expand ordinal numbers to spoken form.

    Examples:
        '1st' → 'first'
        '2nd' → 'second'
        '3rd' → 'third'
        '21st' → 'twenty first'
    """
    def _replace_ordinal(m: re.Match[str]) -> str:
        num = int(m.group(1))
        if num < 1 or num > 99:
            return m.group(0)
        return _number_to_ordinal(num)
    return _RE_ORDINAL.sub(_replace_ordinal, text)


def strip_number_commas(text: str) -> str:
    """Remove commas from numbers in standard positions.

    Examples:
        '1,234' → '1234'
        '1,234,567' → '1234567'
    """
    def _strip(m: re.Match[str]) -> str:
        return m.group(1) + m.group(2).replace(",", "")
    return _RE_NUMBER_COMMAS.sub(_strip, text)


def expand_fractions_ratios(text: str) -> str:
    """Expand common fractions and ratios to spoken form.

    Examples:
        '1/2' → 'one half'
        '3/4' → 'three quarters'
        '2/3' → 'two thirds'
        '16:9' → '16 to 9'
    """
    def _replace_fraction(m: re.Match[str]) -> str:
        num = int(m.group(1))
        den = int(m.group(2))
        # Only handle small fractions we have words for
        num_word = _CARDINAL_WORDS.get(num)
        den_word = _FRACTION_DENOM.get(den)
        if num_word is None or den_word is None:
            return m.group(0)
        # Special case: half → halves not halfs
        if den == 2:
            if num == 1:
                return f"{num_word} {den_word}"
            else:
                return f"{num_word} halves"
        # Pluralize: one third vs two thirds, one quarter vs three quarters
        if num == 1:
            return f"{num_word} {den_word}"
        else:
            if den_word == "quarter":
                return f"{num_word} quarters"
            else:
                return f"{num_word} {den_word}s"
    text = _RE_FRACTION.sub(_replace_fraction, text)

    # Ratios: 16:9 → 16 to 9
    text = _RE_RATIO.sub(r"\1 to \2", text)

    return text


def expand_time_formats(text: str) -> str:
    """Expand 24-hour time to 12-hour spoken form.

    Examples:
        '14:30' → '2:30 PM'
        '09:00' → '9 AM'
        '23:59' → '11:59 PM'
        '3:15pm' → '3:15 PM'
    """
    # First, normalize existing am/pm labels to uppercase
    def _normalize_ampm(m: re.Match[str]) -> str:
        h = m.group(1)
        mins = m.group(2)
        ampm = m.group(3).upper()
        return f"{h}:{mins} {ampm}"
    text = _RE_TIME_AMPM.sub(_normalize_ampm, text)

    # Convert 24-hour times (only those NOT already followed by AM/PM)
    def _convert_24h(m: re.Match[str]) -> str:
        # Skip if followed by AM/PM (already handled)
        hour = int(m.group(1))
        minute = int(m.group(2))
        # Check what follows the match to avoid double-converting
        end = m.end()
        rest = text[end:end+5].strip().upper() if end < len(text) else ""
        if rest.startswith("AM") or rest.startswith("PM"):
            return m.group(0)
        # Convert
        if hour == 0:
            spoken_hour = 12
            period = "AM"
        elif hour < 12:
            spoken_hour = hour
            period = "AM"
        elif hour == 12:
            spoken_hour = 12
            period = "PM"
        else:
            spoken_hour = hour - 12
            period = "PM"
        if minute == 0:
            return f"{spoken_hour} {period}"
        else:
            return f"{spoken_hour}:{m.group(2)} {period}"

    # We need to be careful not to match ratios or other patterns.
    # Only match if it looks like a time (hours 0-23, minutes 0-59).
    # Use a function to avoid re-matching already processed times.
    result = []
    last_end = 0
    for m in _RE_TIME_24H.finditer(text):
        hour = int(m.group(1))
        minute = int(m.group(2))
        # Only convert if it's a valid 24-hour time
        if hour > 23 or minute > 59:
            continue
        # Skip if already followed by AM/PM
        end = m.end()
        rest = text[end:end+5].strip().upper() if end < len(text) else ""
        if rest.startswith("AM") or rest.startswith("PM"):
            continue
        result.append(text[last_end:m.start()])
        if hour == 0:
            spoken_hour = 12
            period = "AM"
        elif hour < 12:
            spoken_hour = hour
            period = "AM"
        elif hour == 12:
            spoken_hour = 12
            period = "PM"
        else:
            spoken_hour = hour - 12
            period = "PM"
        if minute == 0:
            result.append(f"{spoken_hour} {period}")
        else:
            result.append(f"{spoken_hour}:{m.group(2)} {period}")
        last_end = m.end()
    result.append(text[last_end:])
    text = "".join(result)

    return text


def expand_dates(text: str) -> str:
    """Expand ISO dates to spoken form.

    Examples:
        '2024-02-26' → 'February 26, 2024'
        '2024-01-01' → 'January 1, 2024'
    """
    def _replace_date(m: re.Match[str]) -> str:
        year = m.group(1)
        month = int(m.group(2))
        day = int(m.group(3))
        month_name = _MONTH_NAMES.get(month, m.group(2))
        return f"{month_name} {day}, {year}"
    return _RE_ISO_DATE.sub(_replace_date, text)


def expand_temperature(text: str) -> str:
    """Expand temperature notations to spoken form.

    Examples:
        '72°F' → '72 degrees Fahrenheit'
        '22°C' → '22 degrees Celsius'
        '-40°F' → 'negative 40 degrees Fahrenheit'
    """
    def _replace_temp(m: re.Match[str]) -> str:
        num = m.group(1)
        scale = m.group(2).upper()
        scale_word = "Fahrenheit" if scale == "F" else "Celsius"
        if num.startswith("-"):
            num = "negative " + num[1:]
        return f"{num} degrees {scale_word}"
    return _RE_TEMP.sub(_replace_temp, text)


def expand_math_operators(text: str) -> str:
    """Expand mathematical operators to spoken form.

    Examples:
        'x + y' → 'x plus y'
        'a != b' → 'a not equals b'
        'a >= b' → 'a greater than or equal to b'
        'x^2' → 'x squared'
    """
    # Order matters: multi-char operators first
    text = _RE_NOT_EQUALS.sub(r"\1 not equals \2", text)
    text = _RE_DOUBLE_EQUALS.sub(r"\1 equals \2", text)
    text = _RE_GTE.sub(r"\1 greater than or equal to \2", text)
    text = _RE_LTE.sub(r"\1 less than or equal to \2", text)
    text = _RE_GT.sub(r"\1 greater than \2", text)
    text = _RE_LT.sub(r"\1 less than \2", text)
    text = _RE_PLUS.sub(r"\1 plus \2", text)
    text = _RE_MINUS.sub(r"\1 minus \2", text)
    text = _RE_TIMES.sub(r"\1 times \2", text)
    text = _RE_EQUALS.sub(r"\1 equals \2", text)
    text = _RE_POWER_2.sub(r"\1 squared", text)
    text = _RE_POWER_3.sub(r"\1 cubed", text)
    text = _RE_POWER_N.sub(r"\1 to the \2", text)
    return text


def expand_slash_pairs(text: str) -> str:
    """Expand word/word slash pairs to spoken form.

    Examples:
        'and/or' → 'and or'
        'true/false' → 'true or false'
        'yes/no' → 'yes or no'
        'input/output' → 'input output'
        'before/after' → 'before or after'

    Does NOT match file paths.
    """
    # Handle known pairs first
    for pair, replacement in _SLASH_COMMON.items():
        text = text.replace(pair, replacement)

    # Generic word/word — only if both sides are purely alphabetic
    # and it doesn't look like a file path (no leading / or . nearby)
    def _replace_slash_pair(m: re.Match[str]) -> str:
        full = m.group(0)
        left = m.group(1)
        right = m.group(2)
        # Check position in text — skip if preceded by / or path-like context
        start = m.start()
        if start > 0 and text[start - 1] in "/.\\":
            return full
        # Skip if followed by / (multi-segment path)
        end = m.end()
        if end < len(text) and text[end] in "/.":
            return full
        return f"{left} or {right}"

    text = _RE_SLASH_PAIR.sub(_replace_slash_pair, text)
    return text


# ---------------------------------------------------------------------------
# Existing transform functions (updated)
# ---------------------------------------------------------------------------

def expand_stop_words(text: str) -> str:
    """Expand common shorthand and Latin abbreviations for speech.

    Examples:
        'e.g. Python' → 'for example Python'
        'etc.' → 'etcetera'
        'w/ support' → 'with support'
        'w/o issues' → 'without issues'
    """
    # Handle w/o before w/ to avoid partial matches
    text = _RE_WITHOUT.sub("without", text)
    text = _RE_WITH.sub("with", text)
    # Handle dotted abbreviations
    text = _RE_EG.sub("for example ", text)
    text = _RE_IE.sub("that is ", text)
    text = _RE_ETC.sub("etcetera", text)
    text = _RE_VS.sub("versus ", text)
    return text


def expand_units(text: str) -> str:
    """Expand '27MB' → '27 megabytes'."""
    def _replace(m: re.Match[str]) -> str:
        number, unit_key = m.group(1), m.group(2)
        unit = UNIT_MAP.get(unit_key, unit_key)
        if number == "1" and unit.endswith("s"):
            unit = unit[:-1]
        return f"{number} {unit}"

    return _RE_UNITS.sub(_replace, text)


def expand_abbreviations(text: str) -> str:
    """Expand tech abbreviations to spoken form."""
    for abbr, expansion in ABBREV_MAP.items():
        text = _RE_ABBREVS[abbr].sub(expansion, text)
    return text


def clean_file_paths(text: str) -> str:
    """Shorten long file paths to just the filename, with src→source."""
    # Replace src/ with source/ for speech
    text = _RE_SRC_PREFIX.sub("source ", text)

    def _shorten(m: re.Match[str]) -> str:
        return m.group(0).rstrip("/").rsplit("/", 1)[-1]

    text = _RE_LONG_PATH.sub(_shorten, text)
    text = _RE_HOME_PATH.sub(_shorten, text)
    return text


def clean_file_extensions(text: str) -> str:
    """Make file extensions speakable."""
    for ext, spoken in EXT_MAP.items():
        text = _RE_FILE_EXTS[ext].sub(" " + spoken, text)
    return text


def speak_decimal_numbers(text: str) -> str:
    """Convert decimal numbers so TTS doesn't pause at the period.

    '1.5' → 'one point five', '1.3x' → 'one point three x'
    Only targets numbers that would confuse TTS (decimals mid-sentence).
    """
    digit_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    }

    def _replace_decimal(m: re.Match[str]) -> str:
        whole = m.group(1)
        frac = m.group(2)
        suffix = m.group(3) or ""
        # Speak each fractional digit: 1.35 → "one point three five"
        frac_spoken = " ".join(digit_words.get(d, d) for d in frac)
        # For whole part, just use the number (TTS handles "1" fine, it's the period that's the problem)
        return f"{whole} point {frac_spoken}{' ' + suffix if suffix else ''}"

    # Match decimal numbers optionally followed by x/X (multiplier)
    # But NOT version strings (handled separately) or IPs
    text = _RE_DECIMAL.sub(_replace_decimal, text)
    return text


def clean_version_strings(text: str) -> str:
    """Convert version strings to fully spoken form in one step.

    v1.0 → 'version 1 point 0'
    v1.0.0 → 'version 1 point 0 point 0'
    V2.5 → 'version 2 point 5'
    """
    def _speak_version(m: re.Match[str]) -> str:
        prefix = m.group(1) or ""  # "version " if already present
        parts = m.group(2).split(".")
        spoken = " point ".join(parts)
        if prefix.strip().lower() == "version":
            return f"version {spoken}"
        return f"version {spoken}"

    # Match: optional "version " prefix + v/V + digits.digits[.digits]
    text = _RE_VERSION_V.sub(_speak_version, text)
    # Also match bare "version 1.0" without v prefix
    text = _RE_VERSION_BARE.sub(_speak_version, text)
    return text


def clean_technical_punctuation(text: str) -> str:
    """Remove or replace technical symbols."""
    text = text.replace("→", ", ").replace("←", ", ")
    text = text.replace("=>", ", ").replace("->", ", ")
    text = _RE_EM_DASH.sub(", ", text)
    text = text.replace("...", ". ")
    text = text.replace("`", "")

    # & → and
    text = _RE_AMPERSAND.sub(" and ", text)

    # # before a word → hashtag (but not markdown headings, those are stripped below)
    # We handle this before stripping heading markers so standalone #word still gets caught
    text = _RE_HASHTAG_WORD.sub(r"hashtag \1", text)

    # @ before a word → at
    text = _RE_AT_WORD.sub(r"at \1", text)

    text = _RE_CURLY.sub("", text)
    text = _RE_SQUARE.sub("", text)
    text = _RE_EMPTY_PARENS.sub("", text)
    # Parenthetical asides — keep content, drop parens (limit raised to 200)
    text = _RE_PARENTHETICAL.sub(r", \1,", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    # ~ before a number means "approximately"
    text = _RE_TILDE_NUM.sub(r"about \1", text)
    # ~ in paths (~/...) — just strip it
    text = _RE_TILDE_PATH.sub("/", text)
    # Standalone ~ — strip
    text = text.replace("~", "")
    text = text.replace("*", "")
    # Strip heading markers (### etc.) — do this AFTER hashtag handling
    # Only strip # at start of line or when followed by space (heading style)
    text = re.sub(r"^#+\s?", "", text, flags=re.MULTILINE)
    text = text.replace("|", " ")
    # snake_case → spoken words
    text = _RE_SNAKE_CASE.sub(
        lambda m: m.group(0).replace("_", " "), text
    )
    # Hyphens in package names → spaces
    text = _RE_TRIPLE_HYPHEN.sub(r"\1 \2 \3", text)
    text = _RE_DOUBLE_HYPHEN.sub(r"\1 \2", text)
    return text


def final_cleanup(text: str) -> str:
    """Final pass — remove artifacts, normalize whitespace, add pauses at line breaks."""
    text = _RE_DOUBLE_COMMA.sub(",", text)
    text = _RE_LEADING_COMMA.sub("", text)
    text = _RE_TRAILING_COMMA.sub("", text)
    text = _RE_MULTI_COMMA.sub(",", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    # Remove empty/punctuation-only lines
    lines = [ln for ln in text.split("\n")
             if ln.strip() and not _RE_PUNCT_ONLY_LINE.match(ln.strip())]
    # Add pause at line breaks: if a line doesn't end with sentence punctuation,
    # append a period so Kokoro pauses naturally between lines.
    for i in range(len(lines)):
        stripped = lines[i].rstrip()
        if stripped and stripped[-1] not in ".!?;:":
            lines[i] = stripped + "."
    # Join with spaces -- periods handle the pauses now
    return " ".join(ln.strip() for ln in lines if ln.strip())


# ---------------------------------------------------------------------------
# Pronunciation dictionary — user-editable overrides
# ---------------------------------------------------------------------------

# Built-in default dictionary, shipped with the package.
_BUILTIN_PRON_PATH = Path(__file__).resolve().parent / "data" / "pronunciations.toml"

# User-level override file location.
_USER_PRON_PATH = Path.home() / ".claude-speak" / "pronunciations.toml"

# Cache: (dict_of_terms, source_path, mtime_at_load)
# mtime is None when the built-in is used (it never changes at runtime).
_pron_cache: tuple[dict[str, str], Path, float | None] | None = None


def _load_toml_pronunciations(path: Path) -> dict[str, str]:
    """Load [terms] section from a TOML pronunciations file."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return {}
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return {k: str(v) for k, v in data.get("terms", {}).items()}
    except Exception:
        return {}


def _get_pronunciation_dict() -> dict[str, str]:
    """Return the cached pronunciation dictionary, reloading if the user file changed.

    Resolution order:
      1. ``~/.claude-speak/pronunciations.toml`` if it exists — loaded and
         cached; reloaded automatically when its mtime changes.
      2. The built-in ``claude_speak/data/pronunciations.toml`` (never reloaded
         at runtime since it is part of the installed package).
    """
    global _pron_cache

    if _USER_PRON_PATH.exists():
        try:
            current_mtime = _USER_PRON_PATH.stat().st_mtime
        except OSError:
            current_mtime = None

        if (
            _pron_cache is not None
            and _pron_cache[1] == _USER_PRON_PATH
            and _pron_cache[2] == current_mtime
        ):
            return _pron_cache[0]

        terms = _load_toml_pronunciations(_USER_PRON_PATH)
        _pron_cache = (terms, _USER_PRON_PATH, current_mtime)
        return terms

    # Fall back to built-in (cached once; mtime sentinel = None)
    if _pron_cache is not None and _pron_cache[1] == _BUILTIN_PRON_PATH:
        return _pron_cache[0]

    terms = _load_toml_pronunciations(_BUILTIN_PRON_PATH)
    _pron_cache = (terms, _BUILTIN_PRON_PATH, None)
    return terms


def apply_pronunciation_overrides(text: str) -> str:
    """Replace known technical terms with their preferred spoken forms.

    - Uses ``~/.claude-speak/pronunciations.toml`` when present, otherwise
      falls back to the built-in ``claude_speak/data/pronunciations.toml``.
    - The loaded dictionary is cached and reloaded only when the user file's
      modification time changes.
    - Matches whole words only (``\\b`` boundaries), case-insensitively on
      the *key*, preserving surrounding text.
    - Applied as the final normalization step so earlier transforms (unit
      expansion, abbreviation expansion, etc.) do not interfere.

    Examples::

        >>> apply_pronunciation_overrides("run kubectl apply")
        'run kube control apply'
        >>> apply_pronunciation_overrides("start nginx now")
        'start engine X now'
    """
    terms = _get_pronunciation_dict()
    if not terms:
        return text

    # Sort by length descending so longer keys match before shorter prefixes.
    for key in sorted(terms, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        replacement = terms[key]
        text = pattern.sub(replacement, text)

    return text


# ---------------------------------------------------------------------------
# Context-aware speech annotation
# ---------------------------------------------------------------------------

# Patterns for content type detection
_RE_ERROR_LINE = re.compile(
    r"^(.*?\b(?:error|failed|failure|exception|traceback|fatal|critical|abort(?:ed)?|panic|segfault)\b.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_RE_HEADING_LINE = re.compile(
    r"^([A-Z][A-Za-z0-9 /&,\-]{2,60}):?\s*$",
    re.MULTILINE,
)
_RE_CODE_DESCRIPTION = re.compile(
    r"^(.*?\b(?:function|method|class|variable|module|parameter|argument|returns?|takes?|accepts?|implements?|defines?|calls?|invokes?|initializ(?:es?|ing)|constructor|decorator|import(?:s|ing)?)\b.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_RE_LIST_ITEM_LINE = re.compile(
    r"^((?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Number \d+),\s.+\.)$",
    re.MULTILINE,
)


def annotate_context(text: str) -> str:
    """Insert SSML tags based on detected content type.

    Content types detected:
      - **error_message**: lines containing "error", "failed", "exception", etc.
        get a brief pause before them and are wrapped in ``<slow>`` for emphasis.
      - **code_description**: lines describing code constructs (function, class,
        variable, etc.) are wrapped in ``<slow>`` for slightly slower delivery.
      - **heading**: short title-like lines (capitalized, standalone) get a
        pause inserted after them.
      - **list_item**: ordinal list items (produced by ``improve_lists``) get a
        brief pause inserted between consecutive items.

    The function is designed to be idempotent when no patterns match —
    plain text passes through unchanged.
    """
    if not text or not text.strip():
        return text

    # --- Error messages: pause before + slow ---
    def _annotate_error(m: re.Match[str]) -> str:
        line = m.group(1).strip()
        # Avoid double-wrapping if already tagged
        if "<slow>" in line or "<pause" in line:
            return m.group(0)
        return f"<pause 200ms><slow>{line}</slow>"
    text = _RE_ERROR_LINE.sub(_annotate_error, text)

    # --- Code descriptions: slow ---
    def _annotate_code_desc(m: re.Match[str]) -> str:
        line = m.group(1).strip()
        if "<slow>" in line or "<pause" in line:
            return m.group(0)
        return f"<slow>{line}</slow>"
    text = _RE_CODE_DESCRIPTION.sub(_annotate_code_desc, text)

    # --- Headings: pause after ---
    def _annotate_heading(m: re.Match[str]) -> str:
        line = m.group(1).strip()
        if "<pause" in line:
            return m.group(0)
        return f"{line}<pause 300ms>"
    text = _RE_HEADING_LINE.sub(_annotate_heading, text)

    # --- List items: pause between ---
    # We insert a small pause before each list item (except the first one
    # in a consecutive run, which already follows a heading/paragraph break).
    lines = text.split("\n")
    annotated_lines: list[str] = []
    prev_was_list_item = False
    for line in lines:
        is_list_item = bool(_RE_LIST_ITEM_LINE.match(line.strip()))
        if is_list_item and prev_was_list_item:
            annotated_lines.append(f"<pause 150ms>{line}")
        else:
            annotated_lines.append(line)
        prev_was_list_item = is_list_item
    text = "\n".join(annotated_lines)

    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(text: str, *, context_aware: bool = True) -> str:
    """Full normalization pipeline: raw text → speech-ready text.

    Parameters:
        text: The raw text to normalize.
        context_aware: When True (default), apply context-aware SSML
            annotations after normalization.  Set to False to skip.
    """
    text = describe_code_blocks(text)
    text = narrate_tables(text)
    text = improve_lists(text)
    text = strip_code_blocks(text)
    text = clean_urls_and_emails(text)      # NEW — before paths/extensions
    text = clean_file_paths(text)           # existing (with src→source fix)
    text = expand_currency(text)            # NEW
    text = expand_percentages(text)         # NEW
    text = expand_ordinals(text)            # NEW
    text = strip_number_commas(text)        # NEW
    text = expand_time_formats(text)        # NEW — before fractions (14:30 is a time, not a ratio)
    text = expand_fractions_ratios(text)    # NEW
    text = expand_dates(text)              # NEW
    text = expand_temperature(text)        # NEW
    text = expand_math_operators(text)     # NEW
    text = expand_units(text)              # existing
    text = clean_version_strings(text)     # existing
    text = speak_decimal_numbers(text)     # existing
    text = clean_file_extensions(text)     # existing
    text = expand_stop_words(text)         # existing
    text = expand_abbreviations(text)      # existing
    text = expand_slash_pairs(text)        # NEW
    text = clean_technical_punctuation(text) # existing (with fixes)
    text = final_cleanup(text)             # existing
    text = apply_pronunciation_overrides(text)  # pronunciation dictionary — final pass
    if context_aware:
        text = annotate_context(text)  # context-aware SSML annotation — after all normalization
    return text


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """Split text into speakable chunks at sentence boundaries.

    Kokoro handles ~400 chars well per call. Splitting at sentence
    boundaries keeps speech natural. Each chunk is queued separately
    so the daemon can start speaking before all chunks are ready.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    chunks = []
    # Split into sentences (period, exclamation, question mark followed by space or EOL)
    sentences = _RE_SENTENCE_SPLIT.split(text)

    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If single sentence exceeds max, split on commas or force-split
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Split long sentence on commas
            parts = _RE_COMMA_SPLIT.split(sentence)
            sub_chunk = ""
            for part in parts:
                if len(sub_chunk) + len(part) + 2 > max_chars:
                    if sub_chunk:
                        chunks.append(sub_chunk.strip())
                    sub_chunk = part
                else:
                    sub_chunk = f"{sub_chunk}, {part}" if sub_chunk else part
            if sub_chunk:
                current_chunk = sub_chunk
            continue

        if len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]
