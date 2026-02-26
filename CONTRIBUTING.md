# Contributing to claude-speak

Thank you for your interest in contributing to claude-speak. This guide covers everything you need to get started: development setup, code style, testing, and the pull request process.

---

## Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/vnicivanov/claude-speak.git
   cd claude-speak
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   Python 3.10 or later is required. Python 3.11+ is recommended (the ruff target version is `py311`).

3. **Install in editable mode with dev dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

   This installs the package along with the development tools: `ruff`, `pytest`, `mypy`, and `pre-commit`.

4. **Install pre-commit hooks** (optional but recommended)

   ```bash
   pre-commit install
   ```

5. **Optional extras**

   If you are working on a specific subsystem, install the relevant extras:

   ```bash
   pip install -e ".[piper]"        # Piper TTS backend
   pip install -e ".[elevenlabs]"   # ElevenLabs TTS backend
   pip install -e ".[wakeword]"     # Wake word detection (openwakeword)
   pip install -e ".[stt]"          # Speech-to-text (mlx-whisper)
   pip install -e ".[macos-extras]" # macOS Quartz bindings for hotkeys
   pip install -e ".[train]"        # Wake word training pipeline
   ```

---

## Code Style

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. The full configuration lives in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "RUF"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["E402", "E501", "F401", "I001", "B011", "SIM300"]
```

Key points:

- **Line length**: 120 characters maximum.
- **Enabled rule sets**: pycodestyle errors/warnings (`E`, `W`), pyflakes (`F`), isort (`I`), pyupgrade (`UP`), flake8-bugbear (`B`), flake8-simplify (`SIM`), and Ruff-specific rules (`RUF`).
- **Test files** have relaxed rules (long lines, import order, and some style checks are ignored).

Before submitting a PR, run the linter and fix any issues:

```bash
ruff check .
ruff format --check .
```

To auto-fix what Ruff can handle:

```bash
ruff check --fix .
ruff format .
```

You can also run type checking (not enforced in CI yet, but encouraged):

```bash
mypy claude_speak/
```

---

## Running Tests

The test suite uses [pytest](https://docs.pytest.org/). Run the full suite with:

```bash
python -m pytest tests/ -q
```

### Test Markers

Two custom markers are defined in `pyproject.toml`:

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.slow` | Tests that take a long time (model loading, integration flows) |
| `@pytest.mark.requires_model` | Tests that need a TTS model file to be present on disk |

To skip slow tests during development:

```bash
python -m pytest tests/ -q -m "not slow"
```

To skip tests that require model files:

```bash
python -m pytest tests/ -q -m "not requires_model"
```

To skip both:

```bash
python -m pytest tests/ -q -m "not slow and not requires_model"
```

### Testing Without Audio Hardware

All tests mock `sounddevice` and other audio dependencies. You do not need a microphone, speakers, or any audio hardware to run the test suite. The project is designed so that:

- `sounddevice.OutputStream` is replaced with `unittest.mock.MagicMock` objects in every test that touches playback.
- TTS model loading is bypassed by injecting mock backends or setting internal attributes directly (e.g., `backend._kokoro = mock_kokoro`).
- Wake word detection, voice input, and any macOS-specific subsystems (AppleScript, Superwhisper) are fully mocked.

This means the test suite runs on any machine with Python 3.10+, including CI servers with no audio devices.

---

## Adding a New TTS Backend

claude-speak uses a backend abstraction layer that makes it straightforward to add new TTS engines. Here is the process:

### 1. Implement the `TTSBackend` interface

Create a new file `claude_speak/tts_<name>.py` and subclass `TTSBackend` from `claude_speak.tts_base`:

```python
"""
<Name> TTS backend for claude-speak.
"""
from __future__ import annotations

from typing import AsyncIterator

import numpy as np

from .config import Config
from .tts_base import TTSBackend


class MyBackend(TTSBackend):

    def __init__(self, config: Config):
        self.config = config
        self._loaded = False

    def load(self) -> None:
        # Initialize the engine, download models if needed
        self._loaded = True

    async def generate(
        self, text: str, voice: str, speed: float = 1.0, lang: str = "en-us"
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        # Yield (samples_float32, sample_rate) tuples
        # Each tuple is one segment of audio
        yield (np.zeros(24000, dtype=np.float32), 24000)

    def list_voices(self) -> list[str]:
        return ["default"]

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def name(self) -> str:
        return "mybackend"
```

The five required methods/properties are:

| Method | Purpose |
|--------|---------|
| `load()` | Load model files and initialize the engine |
| `generate(text, voice, speed, lang)` | Async generator yielding `(np.ndarray, int)` tuples (float32 samples and sample rate) |
| `list_voices()` | Return a list of available voice name strings |
| `is_loaded()` | Return `True` if the engine is ready to generate audio |
| `name` (property) | Return a short, human-readable engine identifier |

See `claude_speak/tts_piper.py` and `claude_speak/tts_elevenlabs.py` for complete real-world examples.

### 2. Register the backend in the factory

Open `claude_speak/tts.py` and update two things:

1. Add the engine name to the `AVAILABLE_ENGINES` list:

   ```python
   AVAILABLE_ENGINES = ["kokoro", "piper", "elevenlabs", "mybackend"]
   ```

2. Add an `elif` branch in `create_backend()`:

   ```python
   elif name == "mybackend":
       try:
           from .tts_mybackend import MyBackend
       except ImportError as exc:
           raise ImportError(
               "MyBackend requires the tts_mybackend module. "
               "Install it with: pip install 'claude-speak[mybackend]'"
           ) from exc
       return MyBackend(config)
   ```

### 3. Add optional dependencies (if any)

If your backend has extra pip dependencies, add them as an optional dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
mybackend = [
    "some-tts-library>=1.0",
]
```

### 4. Write tests

Create `tests/test_tts_mybackend.py`. At minimum, test:

- The backend can be instantiated.
- `is_loaded()` returns `False` before `load()` and `True` after.
- `generate()` yields `(np.ndarray, int)` tuples with correct dtypes.
- `list_voices()` returns a list of strings.
- `name` returns the expected string.

Mock any external libraries so tests run without installing the optional dependency.

---

## Pull Request Process

### Branch Naming

Use descriptive branch names with a prefix:

| Prefix | Use case |
|--------|----------|
| `feat/` | New features (`feat/whisper-backend`) |
| `fix/` | Bug fixes (`fix/airpods-disconnect-crash`) |
| `refactor/` | Code restructuring (`refactor/tts-backend-abc`) |
| `docs/` | Documentation only (`docs/contributing-guide`) |
| `test/` | Test additions or fixes (`test/normalizer-edge-cases`) |

### Commit Messages

Write clear, concise commit messages:

- Use the imperative mood: "Add Piper backend" not "Added Piper backend".
- Keep the subject line under 72 characters.
- Include a body for non-trivial changes explaining **why** the change was made.

```
Add Piper TTS backend with auto-download

Implements TTSBackend for piper-tts with model caching in
~/.claude-speak/models/piper/. Downloads voice models from
HuggingFace on first use.
```

### Before Submitting

1. **Lint passes**: `ruff check .` reports no errors.
2. **Formatting passes**: `ruff format --check .` reports no changes needed.
3. **Tests pass**: `python -m pytest tests/ -q` exits with code 0.
4. **New code has tests**: any new feature or bug fix should include corresponding tests.
5. **No unrelated changes**: keep PRs focused on a single concern.

### Review Process

- All PRs require at least one review before merging.
- CI must pass (lint + tests).
- Squash-merge is preferred for feature branches to keep the main branch history clean.

---

## Issue Reporting Guidelines

When opening an issue, include the following information so we can reproduce and diagnose the problem quickly:

### Bug Reports

- **Description**: A clear, concise summary of the bug.
- **Steps to reproduce**: The exact commands or actions that trigger the issue.
- **Expected behavior**: What you expected to happen.
- **Actual behavior**: What actually happened, including any error messages or log output.
- **Environment**:
  - macOS version (`sw_vers`)
  - Python version (`python3 --version`)
  - claude-speak version (`pip show claude-speak` or git commit hash)
  - Audio setup (speakers, AirPods, external DAC, etc.)
- **Logs**: Relevant output from `claude-speak log` or `daemon.log`.

### Feature Requests

- **Problem statement**: What limitation or pain point does this address?
- **Proposed solution**: How you think it should work.
- **Alternatives considered**: Other approaches you thought about and why you prefer your proposal.

### Security Issues

If you discover a security vulnerability, please do **not** open a public issue. Instead, email victor.n.ivanov@gmail.com directly.

---

## License

By contributing to claude-speak, you agree that your contributions will be licensed under the [MIT License](LICENSE).
