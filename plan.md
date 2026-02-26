# claude-speak: Product Shipping Plan

**Last updated:** 2026-02-26
**Author:** Victor Ivanov
**Status:** Draft

---

## Current State Assessment

### What Works Today

claude-speak is a functional personal voice interface for Claude Code, running as a background daemon on macOS. The core pipeline is:

1. **Hook-driven text capture** (`hooks/speak-response.sh`): A Claude Code PostToolUse/Stop hook reads the conversation transcript, extracts new assistant text, strips markdown, and writes it to a file-based queue at `/tmp/claude-speak-queue/`. Sends SIGUSR1 to the daemon for instant pickup.

2. **Text normalization** (`src/normalizer.py`, 1206 lines): A comprehensive 20+ stage pipeline that transforms Claude's markdown/technical output into speech-friendly prose. Handles code blocks (replaced with descriptions), tables (narrated), lists (ordinals), abbreviations (60+ mappings), units, URLs, currency, fractions, dates, temperatures, math operators, file paths, version strings, and more.

3. **TTS engine** (`src/tts.py`): Kokoro ONNX model loaded once on startup. Supports voice blending (e.g., `bm_george:60+bm_fable:40`), streaming audio via sounddevice, persistent output stream for gapless playback, and chunked generation with look-ahead (generates chunk N+1 while playing chunk N).

4. **Daemon lifecycle** (`src/daemon.py`): Full process management with PID file, exclusive file lock (fcntl), signal handling (SIGTERM/SIGINT for shutdown, SIGUSR1 for queue wakeup), daemonization via fork, hot-reload of config on file change, and mute/toggle state via sentinel files.

5. **Wake word detection** (`src/wakeword.py`): openwakeword-based listener supporting dual models (wake word + stop command). Runs on a background thread with 80ms frame processing. Swaps to built-in mic during TTS playback to avoid Bluetooth SCO profile switching on AirPods.

6. **Voice input** (`src/voice_input.py`): Superwhisper integration via AppleScript keyboard shortcuts. Full cycle: trigger recording, detect speech-then-silence via mic RMS monitoring, stop recording, wait for clipboard change (transcription), auto-submit via Enter keystroke.

7. **Voice controller** (`src/voice_controller.py`): Orchestrates wake word and voice input. Context-aware wake word handling: if TTS is playing, toggles mute; if idle, starts voice input. Thread-safe with input lock to prevent double-triggering.

8. **Audio chimes** (`src/chimes.py`): Programmatic sine-wave tones for state feedback (ready, error, stop, acknowledgment). No external audio files required (except optional ack.wav).

9. **File-based queue** (`src/queue.py`): FIFO queue using timestamped text files. Supports enqueue, dequeue, peek, clear, depth, and bulk chunk enqueue.

10. **Configuration** (`src/config.py`): TOML-based config with dataclass defaults, environment variable overrides, and five config sections (tts, wakeword, input, normalization, audio).

11. **CLI** (`cli.py`): 15 commands including start/stop/restart/status, test/say, voices, enable/disable, queue/clear, log, config, listen, voice-input.

12. **Installer** (`install.sh`): Automated setup script that checks prerequisites, creates venv, installs dependencies, downloads Kokoro models, symlinks hooks into ~/.claude/hooks/, merges hook entries into ~/.claude/settings.json, creates default config.

13. **Custom wake word training pipeline** (`train/`): Three-phase pipeline for training custom openwakeword models using synthetic Kokoro TTS data: clip generation (`generate_clips.py`), data augmentation (`augment_clips.py`), and model training with ONNX export (`train_model.py`).

14. **Test suite** (`tests/test_all.py`): 120+ tests covering normalizer (80+ tests for all transform functions), queue (10 tests), and config (7 tests).

### Known Bugs and Workarounds

| Issue | Severity | Workaround |
|-------|----------|------------|
| Bluetooth profile switching causes audio glitch on AirPods when wake word listener and TTS use same device | Medium | `wakeword.py` swaps to built-in mic during TTS via `use_builtin_mic()` |
| Superwhisper dependency is hard requirement for voice input (macOS-only app) | High | No workaround; voice input simply does not work without it |
| Mute file can cause deadlock if TTS finishes while muted | Medium | `daemon.py` line 205 cleans up mute file in finally block |
| Hook script serialization uses mkdir for locking (no flock on macOS) | Low | 10-second stale lock cleanup and 3-second wait with retry |
| `osascript` commands for key simulation could fail if System Events access is not granted | Medium | User must grant Accessibility permissions manually |
| Config hot-reload replaces engine config but does not reload the Kokoro model (voice blend changes require restart) | Low | Use `cli.py restart` after changing voice blend |
| No log rotation — daemon.log grows unbounded | Low | Manual truncation or restart |
| `pgrep` in `kill_all()` may match unrelated processes containing "claude-speak" | Low | PID file check runs first; pgrep is fallback |
| Queue race condition: two dequeue calls could read same file if perfectly simultaneous | Very Low | File-based queue with unlink; FileNotFoundError caught |
| Voice input RMS threshold (80.0) is hardcoded, not calibrated per environment | Medium | Works in quiet environments; noisy environments may false-trigger |
| Hook scripts resolve `$0` to symlink path instead of target, so `$PROJECT` points to `~/.claude/` instead of the real project directory — `cli.py` not found | High | Fixed: hooks now use `BASH_SOURCE[0]` + `readlink` to follow symlinks before deriving `$PROJECT`; also corrected `cli.py` path to `claude_speak/cli.py` |

### Architecture Diagram

```
                           Claude Code
                               |
                               | (hook events: PostToolUse, Stop, SessionStart, SessionEnd)
                               v
                    +---------------------+
                    | speak-response.sh   |  <-- reads transcript, extracts new text
                    | daemon-start.sh     |  <-- starts daemon on session start
                    | daemon-stop.sh      |  <-- stops daemon on session end
                    | daemon-restart.sh   |  <-- UserPromptSubmit hook for "restart daemon"
                    +---------------------+
                               |
                   writes to   |  SIGUSR1
                   /tmp/queue  |  signal
                               v
               +-------------------------------+
               |         daemon.py             |
               |  (asyncio event loop)         |
               |                               |
               |  +---------+   +----------+   |
               |  | queue.py|-->|normalizer|   |
               |  | (FIFO)  |   | .py      |   |
               |  +---------+   +----------+   |
               |                     |          |
               |                     v          |
               |               +---------+     |
               |               | tts.py  |     |
               |               | (Kokoro |     |
               |               |  ONNX)  |     |
               |               +---------+     |
               |                     |          |
               |                     v          |
               |              sounddevice       |
               |             (PortAudio)        |
               |                                |
               |  +----------------------------+|
               |  | voice_controller.py        ||
               |  |                            ||
               |  |  +------------------+      ||
               |  |  | wakeword.py      |      ||
               |  |  | (openwakeword)   |      ||
               |  |  +------------------+      ||
               |  |           |                ||
               |  |           v                ||
               |  |  +------------------+      ||
               |  |  | voice_input.py   |      ||
               |  |  | (Superwhisper    |      ||
               |  |  |  via osascript)  |      ||
               |  |  +------------------+      ||
               |  +----------------------------+|
               +-------------------------------+
                               |
                               v
                          config.py
                     (claude-speak.toml)

  State files:
    /tmp/claude-speak-daemon.pid     Process ID
    /tmp/claude-speak-daemon.lock    Exclusive lock (fcntl)
    /tmp/claude-speak-daemon.start_ts  Uptime tracking
    /tmp/claude-speak-queue/*.txt    Message queue
    /tmp/claude-speak-muted          Mute state flag
    /tmp/claude-speak-playing        TTS active flag
    /tmp/claude-speak-pos            Transcript position tracker
    ~/.claude-speak-enabled          Enable/disable toggle
```

### Dependency List

**Runtime (required):**
| Package | Version | Purpose |
|---------|---------|---------|
| kokoro-onnx | >=0.4 | TTS model inference |
| sounddevice | >=0.4 | Audio I/O (PortAudio binding) |
| numpy | >=1.24 | Audio sample manipulation |
| onnxruntime | >=1.16 | ONNX model runtime |

**Runtime (optional):**
| Package | Version | Purpose |
|---------|---------|---------|
| openwakeword | >=0.6 | Wake word detection |
| soundfile | (any) | WAV file reading for ack chime |

**System dependencies:**
| Tool | Purpose |
|------|---------|
| jq | JSON parsing in hook scripts |
| osascript | AppleScript execution for Superwhisper and key simulation |
| pbpaste | Clipboard reading for transcription detection |
| pgrep/pkill | Process management in daemon lifecycle |
| curl | Model file downloads in installer |

**Training pipeline (dev only):**
| Package | Version | Purpose |
|---------|---------|---------|
| torch | (any) | Model training |
| scipy | (any) | Audio processing |
| soundfile | (any) | WAV I/O |
| tqdm | (any) | Progress bars |

**Model files (not in git, downloaded by installer):**
| File | Size | Source |
|------|------|--------|
| kokoro-v1.0.onnx | ~87MB | github.com/thewh1teagle/kokoro-onnx |
| voices-v1.0.bin | ~52MB | github.com/thewh1teagle/kokoro-onnx |
| stop.onnx | ~200KB | Locally trained via train/ pipeline |

---

## Phase 1: Foundation & Quality (Weeks 1-2)

### 1.1 Comprehensive Test Suite

#### 1.1.1 Unit Tests for TTS Engine

- **Description:** Write unit tests for `src/tts.py` covering model loading, voice resolution (single + blend), device resolution, stream management, volume scaling, stop mechanism, and the generate_audio/play_audio/speak methods. Mock `kokoro_onnx.Kokoro` and `sounddevice` to avoid requiring actual model files or audio hardware in CI.
- **Files affected:** `tests/test_tts.py` (new), `src/tts.py` (minor refactors for testability)
- **Estimated effort:** L
- **Dependencies:** None
- **Acceptance criteria:** 15+ tests passing. Coverage for: voice blend parsing with weights, voice blend with equal shares, single voice passthrough, device resolution by name substring, device resolution by ID, "auto" device default, stream reuse when sample rate unchanged, stream recreation when sample rate changes, stop() sets event and aborts stream, _write_samples respects _stopped event, volume scaling below 1.0, list_voices returns sorted list.

#### 1.1.2 Unit Tests for Daemon Module

- **Description:** Write unit tests for `src/daemon.py` covering `_is_stop_command`, `_try_reload_config`, PID file management (`write_pid`, `read_pid`), and `acquire_lock`. The main `run_loop` is integration-level and tested separately.
- **Files affected:** `tests/test_daemon.py` (new)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** 10+ tests passing. Coverage for: stop command matching (case-insensitive, whitespace-stripped), config reload on mtime change, config reload skipped when mtime unchanged, PID file write/read round-trip, read_pid returns None when file missing, read_pid returns None when process dead, acquire_lock returns True on first call.

#### 1.1.3 Unit Tests for Wake Word Listener

- **Description:** Write unit tests for `src/wakeword.py` covering model loading, audio processing with mock predictions, cooldown logic, callback firing, pause/resume, and mic swap. Mock `openwakeword.model.Model` and `sounddevice`.
- **Files affected:** `tests/test_wakeword.py` (new)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** 12+ tests passing. Coverage for: _process_audio fires wake callback when score exceeds threshold, _process_audio does NOT fire during cooldown, stop model takes priority over wake model, pause prevents processing, resume re-enables processing, use_builtin_mic / use_default_mic toggle the flag, callbacks are fired in order, callback exceptions are caught and logged.

#### 1.1.4 Unit Tests for Voice Input

- **Description:** Write unit tests for `src/voice_input.py` covering `_modifier_flags_to_names`, `_is_superwhisper_running` (mock subprocess), `_get_clipboard`, `trigger_superwhisper`, `auto_submit`, `wait_for_transcription`, and `voice_input_cycle`. All subprocess calls must be mocked.
- **Files affected:** `tests/test_voice_input.py` (new)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** 15+ tests passing. Coverage for: modifier flag parsing (single flag, combined flags, no flags), osascript failure raises SuperwhisperError, osascript timeout raises SuperwhisperError, clipboard change detection, clipboard no-change returns False, full cycle success path, full cycle when Superwhisper not running, full cycle when nothing transcribed.

#### 1.1.5 Unit Tests for Voice Controller

- **Description:** Write unit tests for `src/voice_controller.py` covering start/stop lifecycle, `_on_wake_word` context awareness (playing vs idle), `handle_stop`, and input lock serialization.
- **Files affected:** `tests/test_voice_controller.py` (new)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** 10+ tests passing. Coverage for: start returns True on first call, start returns False when already running, stop clears running flag, _on_wake_word toggles mute when PLAYING_FILE exists, _on_wake_word starts voice input thread when idle, handle_stop clears queue and calls tts_stop_callback, double-trigger prevented by _input_lock.

#### 1.1.6 Unit Tests for Chimes

- **Description:** Write unit tests for `src/chimes.py` covering tone generation (correct length, fade envelope), and chime composition. Mock `sounddevice.play` to verify samples are generated correctly without actually playing audio.
- **Files affected:** `tests/test_chimes.py` (new)
- **Estimated effort:** S
- **Dependencies:** None
- **Acceptance criteria:** 8+ tests passing. Coverage for: _generate_tone returns correct sample count for duration, fade envelope applied (first/last samples near zero), play_ready_chime generates ascending frequencies, play_error_chime generates descending frequencies, play_stop_chime is single tone, play_ack_chime falls back to tone when asset missing.

#### 1.1.7 Integration Test: Full Pipeline

- **Description:** Write an integration test that exercises the full path from text input through normalization, chunking, and TTS generation (but not audio playback). Verifies the entire speak pipeline produces valid audio samples.
- **Files affected:** `tests/test_integration.py` (new)
- **Estimated effort:** L
- **Dependencies:** Requires Kokoro model files to be present (skip in CI with pytest mark)
- **Acceptance criteria:** Test takes a realistic Claude response, normalizes it, chunks it, generates audio via Kokoro (if available), and verifies output is non-empty float32 numpy arrays with correct sample rate. Marked with `@pytest.mark.slow` and `@pytest.mark.requires_model`.

### 1.2 Error Handling & Recovery

#### 1.2.1 PortAudio Error Recovery in TTS

- **Description:** Add retry logic to `_ensure_stream` and `_write_samples` in `src/tts.py` for `sd.PortAudioError`. Currently a PortAudio failure during stream creation crashes the current speech attempt. Add exponential backoff retry (up to 3 attempts) and fall back to default device if the configured device fails.
- **Files affected:** `src/tts.py`
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** PortAudioError during stream creation is caught, retried up to 3 times with 0.1/0.5/1.0s delays, and falls back to default device. Error chime plays on final failure. Daemon stays alive.

#### 1.2.2 Model Not Found Handling

- **Description:** Add clear error messages and auto-download prompts when Kokoro model files are missing. Currently `TTSEngine.load()` will throw a cryptic ONNX error. Check file existence before loading and provide actionable guidance.
- **Files affected:** `src/tts.py`, `src/config.py`
- **Estimated effort:** S
- **Dependencies:** None
- **Acceptance criteria:** When model files are missing, the error message includes the exact path expected, the download URL, and a command to fix it (`claude-speak setup` or `curl -L -o ...`). Daemon exits cleanly with code 1 (not a crash).

#### 1.2.3 Superwhisper Not Running Error

- **Description:** Improve the voice input flow when Superwhisper is not installed or not running. Currently logs a warning and proceeds to send the keyboard shortcut (which does nothing). Add a clear error chime and skip the entire voice input cycle.
- **Files affected:** `src/voice_input.py`, `src/voice_controller.py`
- **Estimated effort:** S
- **Dependencies:** None
- **Acceptance criteria:** When Superwhisper is not running, `voice_input_cycle` returns False immediately with a log message. Error chime plays. No osascript commands are sent. User sees clear guidance in logs about installing/starting Superwhisper.

#### 1.2.4 Graceful Degradation for Missing Optional Dependencies

- **Description:** Make openwakeword, soundfile, and sounddevice failures produce clear messages instead of import errors. Each module should check at import time and degrade gracefully.
- **Files affected:** `src/wakeword.py` (already has checks), `src/chimes.py`, `src/voice_input.py`, `src/tts.py`
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** If `sounddevice` is not installed, daemon exits with message "pip install sounddevice". If `openwakeword` is not installed, wake word is silently disabled with log message. If `soundfile` is not installed, ack chime falls back to tone (already done) with log message.

#### 1.2.5 Hook Script Error Handling

- **Description:** Improve error handling in `hooks/speak-response.sh` for edge cases: transcript file deleted mid-read, jq parse failures, invalid JSON in transcript, and permission errors on queue directory.
- **Files affected:** `hooks/speak-response.sh`
- **Estimated effort:** S
- **Dependencies:** None
- **Acceptance criteria:** Every possible failure point has error handling that exits cleanly (not propagating errors to Claude Code). A debug log mode (CLAUDE_SPEAK_DEBUG=1) prints error details to stderr.

### 1.3 Logging Overhaul

#### 1.3.1 Structured Logging with Python logging Module

- **Description:** Replace all `print()` calls in the daemon, TTS engine, and other modules with proper Python `logging` calls. `src/voice_controller.py` and `src/wakeword.py` already use `logging`; the rest use `print(f"[daemon] ...")`. Standardize on a single logging configuration.
- **Files affected:** `src/daemon.py`, `src/tts.py`, `src/chimes.py`, `src/queue.py`, `src/normalizer.py`
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** All modules use `logging.getLogger(__name__)`. No more bare `print()` for status messages. Performance logging uses `logger.debug` with a `PERF` log level or a dedicated perf logger. Log format includes timestamp, level, module, and message.

#### 1.3.2 Configurable Log Levels

- **Description:** Add a `log_level` config option and `CLAUDE_SPEAK_LOG_LEVEL` env var. Support DEBUG, INFO, WARNING, ERROR levels. Default to INFO for daemon, WARNING for library modules.
- **Files affected:** `src/config.py`, `src/daemon.py`, `claude-speak.toml.example`
- **Estimated effort:** S
- **Dependencies:** 1.3.1
- **Acceptance criteria:** `log_level = "debug"` in config or `CLAUDE_SPEAK_LOG_LEVEL=DEBUG` env var enables verbose logging. Performance logging (previously gated by CLAUDE_SPEAK_PERF) is now gated by DEBUG level.

#### 1.3.3 Log Rotation

- **Description:** Implement log rotation using `logging.handlers.RotatingFileHandler`. Max 5MB per file, keep 3 backups. Prevents unbounded daemon.log growth in long-running sessions.
- **Files affected:** `src/daemon.py`
- **Estimated effort:** S
- **Dependencies:** 1.3.1
- **Acceptance criteria:** daemon.log never exceeds 5MB. Old logs are rotated to daemon.log.1, daemon.log.2, daemon.log.3. Oldest is deleted. `cli.py log` still shows the most recent entries.

### 1.4 Thread Safety Audit

#### 1.4.1 Document All Shared State

- **Description:** Create a thread safety map documenting every piece of mutable state shared between threads, what lock protects it, and which threads access it. The current shared state includes: `TTSEngine._stream` (protected by `_stream_lock`), `TTSEngine._stopped` (threading.Event, thread-safe), `VoiceController._voice_input_active` (boolean, racy but documented), `VoiceController._input_lock` (threading.Lock), `WakeWordListener._paused/_swap_mic/_running` (booleans set from different threads), sentinel files (MUTE_FILE, PLAYING_FILE, TOGGLE_FILE).
- **Files affected:** `docs/THREAD_SAFETY.md` (new, only if requested; otherwise inline comments)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** Every shared mutable variable has a comment documenting: (a) which threads read it, (b) which threads write it, (c) what synchronization mechanism protects it, (d) whether the current protection is sufficient.

#### 1.4.2 Fix Racy Boolean Flags in WakeWordListener

- **Description:** `_paused`, `_swap_mic`, and `_running` in `src/wakeword.py` are plain booleans read and written from different threads without locks. While Python's GIL makes single-attribute reads/writes atomic for CPython, this is an implementation detail. Use `threading.Event` for these flags (like `_stopped` in TTSEngine) for correctness.
- **Files affected:** `src/wakeword.py`
- **Estimated effort:** S
- **Dependencies:** 1.4.1
- **Acceptance criteria:** `_paused`, `_swap_mic`, and `_running` are `threading.Event` objects. The listen loop checks `.is_set()` instead of reading booleans. `pause()`/`resume()` call `.set()`/`.clear()`.

#### 1.4.3 Sentinel File Atomicity

- **Description:** Audit the sentinel file operations (MUTE_FILE, PLAYING_FILE, TOGGLE_FILE). File creation with `.touch()` and deletion with `.unlink(missing_ok=True)` are effectively atomic on POSIX but can race between the daemon's main loop and the voice controller thread. Document that this is safe due to POSIX filesystem semantics and add try/except around all file operations.
- **Files affected:** `src/daemon.py`, `src/voice_controller.py`
- **Estimated effort:** S
- **Dependencies:** 1.4.1
- **Acceptance criteria:** All sentinel file operations are wrapped in try/except. Comments explain why file-based signaling is safe on POSIX. No behavior change required, just hardening.

### 1.5 Type Annotations and Linting

#### 1.5.1 Add Type Annotations to All Modules

- **Description:** Add complete type annotations to every function and method across all source files. Several modules already have partial annotations (voice_controller, wakeword, voice_input). Complete the rest: daemon.py, tts.py, normalizer.py, chimes.py, queue.py, config.py.
- **Files affected:** All `src/*.py` files, `cli.py`
- **Estimated effort:** L
- **Dependencies:** None
- **Acceptance criteria:** `mypy --strict src/` passes with zero errors. All function signatures have parameter and return type annotations. `Optional` used where None is valid.

#### 1.5.2 Configure Ruff Linter

- **Description:** Add ruff configuration to `pyproject.toml` with a reasonable rule set (E, F, W, I, UP, B, SIM, RUF). Fix all existing violations. Add a `ruff check` command to CI.
- **Files affected:** `pyproject.toml`, potentially all `src/*.py` files for auto-fixable violations
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** `ruff check src/ cli.py tests/` passes with zero violations. Configuration is in `pyproject.toml` under `[tool.ruff]`.

#### 1.5.3 Configure Pre-commit Hooks

- **Description:** Add `.pre-commit-config.yaml` with hooks for ruff (lint + format), mypy, and trailing whitespace. Developers can install with `pre-commit install`.
- **Files affected:** `.pre-commit-config.yaml` (new), `pyproject.toml` (add pre-commit to dev deps)
- **Estimated effort:** S
- **Dependencies:** 1.5.1, 1.5.2
- **Acceptance criteria:** `pre-commit run --all-files` passes. README mentions pre-commit setup for contributors.

---

## Phase 2: Installation & Packaging (Weeks 3-4)

### 2.1 pyproject.toml Overhaul

#### 2.1.1 Fix Package Structure for pip Install

- **Description:** The current `pyproject.toml` has `claude-speak = "cli:main"` as the entry point, but `cli.py` is a top-level script that manipulates `sys.path`. Restructure the package so that `cli.py` becomes `claude_speak/cli.py`, `src/` becomes `claude_speak/core/` (or keep as `claude_speak/`), and the entry point works with a standard pip install.
- **Files affected:** Entire project structure. Move `src/` -> `claude_speak/`, `cli.py` -> `claude_speak/cli.py`, update all imports.
- **Estimated effort:** XL
- **Dependencies:** Phase 1 complete (tests passing before refactor)
- **Acceptance criteria:** `pip install -e .` works. `claude-speak start` launches the daemon. `claude-speak test "hello"` speaks. All imports use absolute package paths (`from claude_speak.tts import TTSEngine`). No `sys.path` hacks.

#### 2.1.2 Dependency Groups

- **Description:** Split dependencies into required, optional, and dev groups. `kokoro-onnx`, `sounddevice`, `numpy`, and `onnxruntime` are required. `openwakeword` is optional (for wake word). Training dependencies (torch, scipy, tqdm) are in a `[train]` extra. Test/lint tools are in a `[dev]` extra.
- **Files affected:** `pyproject.toml`
- **Estimated effort:** S
- **Dependencies:** 2.1.1
- **Acceptance criteria:** `pip install claude-speak` installs only required deps. `pip install claude-speak[wakeword]` adds openwakeword. `pip install claude-speak[dev]` adds pytest, mypy, ruff, pre-commit. `pip install claude-speak[train]` adds torch, scipy, soundfile, tqdm.

#### 2.1.3 Metadata and Classifiers

- **Description:** Add full PyPI metadata: long_description from README.md, project URLs (homepage, repository, issues, changelog), Python version classifiers, topic classifiers, and keywords.
- **Files affected:** `pyproject.toml`
- **Estimated effort:** S
- **Dependencies:** 2.1.1
- **Acceptance criteria:** `python -m build` produces a wheel with correct metadata. `pip show claude-speak` displays description, author, license, homepage.

### 2.2 Auto-download Models on First Run

#### 2.2.1 Model Registry and Downloader

- **Description:** Create a model management module that tracks required model files, their download URLs, expected sizes, and SHA256 checksums. On first run (or when `claude-speak setup` is called), download missing models to a platform-appropriate location (`~/.claude-speak/models/` instead of the project-local `models/` directory).
- **Files affected:** `claude_speak/models.py` (new), `claude_speak/config.py` (update default model paths)
- **Estimated effort:** L
- **Dependencies:** 2.1.1
- **Acceptance criteria:** `claude-speak setup` downloads kokoro-v1.0.onnx and voices-v1.0.bin to `~/.claude-speak/models/`. Progress bar shown during download. SHA256 verified after download. Existing files with correct checksum are skipped. Config defaults point to `~/.claude-speak/models/`.

#### 2.2.2 Lazy Model Loading with Download Prompt

- **Description:** When the daemon starts and model files are missing, print a clear message and offer to download them. In non-interactive mode (daemon), auto-download. In interactive mode (CLI test), prompt the user.
- **Files affected:** `claude_speak/tts.py`, `claude_speak/daemon.py`
- **Estimated effort:** M
- **Dependencies:** 2.2.1
- **Acceptance criteria:** First `claude-speak test "hello"` on a fresh install triggers model download with progress bar, then speaks. Daemon startup auto-downloads without prompting (logs the download). Network failure produces clear error with manual download instructions.

### 2.3 Platform Detection

#### 2.3.1 macOS-only Guard with Clear Error

- **Description:** Add a platform check at the top of the CLI entry point and daemon startup. If not macOS, print a clear message explaining that claude-speak currently requires macOS for audio output (sounddevice/PortAudio) and Superwhisper integration, and exit with code 1.
- **Files affected:** `claude_speak/cli.py`, `claude_speak/daemon.py`
- **Estimated effort:** S
- **Dependencies:** None
- **Acceptance criteria:** Running `claude-speak` on Linux prints "claude-speak currently requires macOS. Linux support is planned for a future release." and exits. The error message includes a link to the GitHub issue tracking Linux support.

#### 2.3.2 Architecture Detection for ONNX Runtime

- **Description:** Detect CPU architecture (arm64 vs x86_64) and select the appropriate onnxruntime package. Apple Silicon Macs should use `onnxruntime` with CoreML execution provider for better performance. Intel Macs use the default CPU provider.
- **Files affected:** `claude_speak/tts.py`, `pyproject.toml`
- **Estimated effort:** M
- **Dependencies:** 2.1.1
- **Acceptance criteria:** On Apple Silicon, Kokoro loads with CoreML provider if available, falling back to CPU. Performance improvement is measurable (log first-segment latency). On Intel, CPU provider is used without error.

### 2.4 One-Command Setup

#### 2.4.1 `claude-speak setup` Command

- **Description:** Create a single CLI command that performs all first-time setup: downloads models, creates default config at `~/.claude-speak/config.toml`, installs Claude Code hooks (symlinks + settings.json merge), creates the toggle file, and verifies audio output works with a test tone.
- **Files affected:** `claude_speak/cli.py`, `claude_speak/setup.py` (new)
- **Estimated effort:** XL
- **Dependencies:** 2.2.1, 2.1.1
- **Acceptance criteria:** A user can `pip install claude-speak && claude-speak setup` and have a fully working installation. The command is idempotent (safe to run multiple times). Each step shows status (checkmarks for completed, download progress for models). Final step plays a test tone and says "Setup complete."

#### 2.4.2 `claude-speak uninstall` Command

- **Description:** Create a command that cleanly removes all claude-speak artifacts: hook symlinks, settings.json entries, toggle file, PID file, queue directory, log files, and optionally model files and config.
- **Files affected:** `claude_speak/cli.py`, `claude_speak/setup.py`
- **Estimated effort:** M
- **Dependencies:** 2.4.1
- **Acceptance criteria:** `claude-speak uninstall` removes all hooks and state files. `claude-speak uninstall --all` also removes models and config. The command prompts for confirmation before deleting anything. Daemon is stopped first if running.

### 2.5 Replace install.sh with Python Setup

#### 2.5.1 Deprecate install.sh

- **Description:** Keep `install.sh` for backward compatibility but add a deprecation notice that points to `claude-speak setup`. Eventually remove in a future major version.
- **Files affected:** `install.sh`
- **Estimated effort:** S
- **Dependencies:** 2.4.1
- **Acceptance criteria:** Running `install.sh` prints a deprecation warning and suggests `pip install claude-speak && claude-speak setup` instead, then proceeds with the old installation flow.

---

## Phase 3: Remove Superwhisper Dependency (Weeks 5-6)

### 3.1 Built-in Speech Recognition

#### 3.1.1 Evaluate STT Backends

- **Description:** Research and benchmark three speech-to-text options for local, on-device transcription on macOS: (a) whisper.cpp via python binding, (b) faster-whisper (CTranslate2), (c) MLX Whisper (Apple Silicon native). Evaluate on latency, accuracy, memory usage, and Apple Silicon optimization.
- **Files affected:** `docs/DECISIONS.md` (new, documenting the evaluation)
- **Estimated effort:** L
- **Dependencies:** None
- **Acceptance criteria:** A written comparison with benchmarks on Apple Silicon (M1/M2/M3). Recommendation for default backend. Latency target: <500ms for 5-second utterance on M1.

#### 3.1.2 Abstract STT Interface

- **Description:** Define a `SpeechRecognizer` abstract base class with methods: `transcribe(audio: np.ndarray, sample_rate: int) -> str` and `is_available() -> bool`. Implement the chosen backend(s).
- **Files affected:** `claude_speak/stt.py` (new), `claude_speak/stt_whisper.py` (new), `claude_speak/stt_mlx.py` (new, if MLX chosen)
- **Estimated effort:** L
- **Dependencies:** 3.1.1
- **Acceptance criteria:** At least one backend works end-to-end. Transcription of "hello world" at 16kHz returns correct text. Model auto-downloads on first use (Whisper tiny/base).

#### 3.1.3 Voice Activity Detection with Silero VAD

- **Description:** Replace the simple RMS threshold in `_wait_for_speech_then_silence()` (voice_input.py, line 197) with Silero VAD. Silero is a small ONNX model (~2MB) that accurately detects speech vs. silence, dramatically reducing false triggers in noisy environments.
- **Files affected:** `claude_speak/vad.py` (new), `claude_speak/voice_input.py`
- **Estimated effort:** M
- **Dependencies:** None (can be done in parallel with 3.1.2)
- **Acceptance criteria:** VAD correctly identifies speech start and end in recordings with background noise. False trigger rate reduced by >80% compared to RMS threshold. Silence detection latency <200ms after speech ends.

#### 3.1.4 Direct Mic -> Transcription -> Paste Pipeline

- **Description:** Build a new voice input flow that does not require Superwhisper: (1) wake word triggers recording, (2) Silero VAD detects speech boundaries, (3) audio is transcribed via built-in STT, (4) text is pasted at cursor position via `osascript` (pbcopy + Cmd+V), (5) Enter is pressed to submit.
- **Files affected:** `claude_speak/voice_input.py` (major rewrite), `claude_speak/voice_controller.py`
- **Estimated effort:** XL
- **Dependencies:** 3.1.2, 3.1.3
- **Acceptance criteria:** Full voice input cycle works without Superwhisper installed. Latency from speech end to text appearing at cursor <1.5 seconds (on M1 with Whisper base model). Accuracy comparable to Superwhisper on standard dictation tasks.

#### 3.1.5 Keep Superwhisper as Optional Backend

- **Description:** Refactor voice_input.py so that Superwhisper is one of multiple input backends. Config option `input.backend = "builtin"` (default) or `input.backend = "superwhisper"`. Users who prefer Superwhisper's UI and accuracy can still use it.
- **Files affected:** `claude_speak/voice_input.py`, `claude_speak/config.py`, `claude-speak.toml.example`
- **Estimated effort:** M
- **Dependencies:** 3.1.4
- **Acceptance criteria:** `input.backend = "superwhisper"` uses the old Superwhisper flow. `input.backend = "builtin"` uses the new direct pipeline. Default is "builtin". Config change does not require daemon restart (hot-reloaded).

### 3.2 STT Model Management

#### 3.2.1 Auto-download STT Models

- **Description:** Extend the model registry (from 2.2.1) to include Whisper models. Support tiny (39MB), base (74MB), small (244MB), and medium (769MB). Default to base. Auto-download on first voice input attempt.
- **Files affected:** `claude_speak/models.py`, `claude_speak/config.py`
- **Estimated effort:** M
- **Dependencies:** 2.2.1, 3.1.2
- **Acceptance criteria:** `claude-speak setup` offers to download a Whisper model. Size and accuracy tradeoff explained in interactive prompt. Model stored in `~/.claude-speak/models/`. Config option `input.stt_model = "base"` to select model size.

---

## Phase 4: Cross-Platform Audio (Weeks 7-8)

### 4.1 Audio Device Management

#### 4.1.1 Abstract Audio Device Layer

- **Description:** Create an `AudioDeviceManager` class that wraps sounddevice and provides: list_output_devices(), list_input_devices(), get_default_output(), get_default_input(), get_device_by_name(substring), is_bluetooth(device_id), and register_device_change_callback(). Centralizes all device logic currently scattered across tts.py and wakeword.py.
- **Files affected:** `claude_speak/audio_devices.py` (new), `claude_speak/tts.py`, `claude_speak/wakeword.py`
- **Estimated effort:** L
- **Dependencies:** None
- **Acceptance criteria:** All device queries go through AudioDeviceManager. Device resolution logic removed from tts.py and wakeword.py. Unit tests cover all methods with mocked sounddevice.

#### 4.1.2 Bluetooth Profile Switching Fix

- **Description:** The current workaround (swap to built-in mic during TTS) is fragile. Implement a proper solution: detect Bluetooth audio devices, and when both input and output are on the same BT device (e.g., AirPods), either (a) use the built-in mic for wake word at all times (not just during TTS), or (b) use a separate InputStream that does not trigger SCO mode. Make this configurable.
- **Files affected:** `claude_speak/audio_devices.py`, `claude_speak/wakeword.py`, `claude_speak/config.py`
- **Estimated effort:** L
- **Dependencies:** 4.1.1
- **Acceptance criteria:** AirPods users do not experience audio quality degradation when wake word is active. The solution works for AirPods, AirPods Pro, AirPods Max, and other BT headphones. Config option `audio.bt_mic_workaround = true` to control behavior.

#### 4.1.3 Test Matrix for Audio Setups

- **Description:** Create a manual test protocol document for verifying audio on different hardware configurations. Include test cases for: built-in speakers + built-in mic, wired headphones, USB microphone, AirPods (both gen), AirPods Pro, other BT headphones, external speakers, HDMI audio, and combinations.
- **Files affected:** `docs/AUDIO_TEST_MATRIX.md` (new)
- **Estimated effort:** M
- **Dependencies:** 4.1.1
- **Acceptance criteria:** Document covers 10+ hardware configurations. Each config has: setup steps, expected behavior, known issues, and pass/fail criteria. CI cannot automate this, so it is a manual checklist for release testing.

### 4.2 Device Change Handling

#### 4.2.1 Graceful Device Disconnection

- **Description:** Handle audio device disconnection mid-session (e.g., AirPods run out of battery). Currently, the PortAudio stream will throw an error, which is caught but not recovered from. Implement automatic fallback to default device when the configured device disappears.
- **Files affected:** `claude_speak/tts.py`, `claude_speak/audio_devices.py`
- **Estimated effort:** M
- **Dependencies:** 4.1.1
- **Acceptance criteria:** When AirPods disconnect during TTS playback: (1) current chunk may be lost, (2) next chunk plays on built-in speakers, (3) log message indicates device change, (4) no crash or hang. When AirPods reconnect, output returns to AirPods on next speech.

#### 4.2.2 Device Change Notification

- **Description:** Detect audio device changes using macOS CoreAudio notifications (via pyobjc or ctypes) and proactively re-resolve the output device. Currently, device re-resolution only happens every 30 seconds (via `_DEVICE_RESOLVE_INTERVAL`).
- **Files affected:** `claude_speak/audio_devices.py`
- **Estimated effort:** L
- **Dependencies:** 4.1.1
- **Acceptance criteria:** Device changes are detected within 1 second. When a new device matching the config pattern appears (e.g., AirPods reconnect), output switches to it. When the current device disappears, output switches to default. Log message on every device change.

---

## Phase 5: Voice & TTS Improvements (Weeks 9-10)

### 5.1 Multiple TTS Backends

#### 5.1.1 Abstract TTS Interface

- **Description:** Extract a `TTSBackend` abstract base class from the current Kokoro-specific implementation. Methods: `load()`, `generate(text, voice, speed, lang) -> AsyncIterator[tuple[np.ndarray, int]]`, `list_voices() -> list[str]`, `get_voice_style(name)`, and `is_loaded() -> bool`.
- **Files affected:** `claude_speak/tts_base.py` (new), `claude_speak/tts.py` (refactor to implement interface)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** `TTSEngine` in tts.py implements `TTSBackend`. All Kokoro-specific code is behind the interface. Adding a new backend requires only implementing the interface, not modifying the engine.

#### 5.1.2 Piper TTS Backend

- **Description:** Add Piper as an alternative TTS backend. Piper is a fast, local TTS engine with many voice options including JARVIS-like voices. Implement `PiperBackend(TTSBackend)`.
- **Files affected:** `claude_speak/tts_piper.py` (new), `claude_speak/config.py`
- **Estimated effort:** L
- **Dependencies:** 5.1.1
- **Acceptance criteria:** `tts.engine = "piper"` in config uses Piper instead of Kokoro. Voice selection works with Piper voice names. Speed control works. Audio quality is acceptable for speech output.

#### 5.1.3 ElevenLabs API Backend

- **Description:** Add ElevenLabs as a cloud-based TTS backend for users who want the highest quality voices and are willing to use an API key. Implement `ElevenLabsBackend(TTSBackend)` with streaming support.
- **Files affected:** `claude_speak/tts_elevenlabs.py` (new), `claude_speak/config.py`
- **Estimated effort:** L
- **Dependencies:** 5.1.1
- **Acceptance criteria:** `tts.engine = "elevenlabs"` in config uses ElevenLabs API. API key stored in `~/.claude-speak/config.toml` or env var `ELEVENLABS_API_KEY`. Streaming audio with <500ms time-to-first-audio. Graceful fallback to Kokoro on network failure.

#### 5.1.4 Engine Hot-Swap via Config

- **Description:** Allow changing the TTS engine without restarting the daemon. When `tts.engine` changes in the config file, the daemon's hot-reload mechanism loads the new engine.
- **Files affected:** `claude_speak/daemon.py`, `claude_speak/tts.py`
- **Estimated effort:** M
- **Dependencies:** 5.1.1
- **Acceptance criteria:** Editing `tts.engine = "piper"` in config while daemon is running causes the daemon to load Piper within 30 seconds. Current playback finishes with old engine; next message uses new engine. Log message confirms engine swap.

### 5.2 Voice Quality Improvements

#### 5.2.1 Pronunciation Dictionary

- **Description:** Add a user-editable pronunciation dictionary for technical terms that TTS engines mispronounce. Format: TOML file at `~/.claude-speak/pronunciations.toml` with entries like `kubectl = "kube control"`, `nginx = "engine X"`, `pytest = "pie test"`. Applied as a normalization step before TTS.
- **Files affected:** `claude_speak/normalizer.py`, `claude_speak/config.py`
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** Pronunciation overrides are applied after standard normalization. User can add custom entries without modifying source code. Default dictionary includes 50+ common developer terms. Dictionary is hot-reloaded when the file changes.

#### 5.2.2 Voice Preview Command

- **Description:** Add `claude-speak preview <voice>` that speaks a sample sentence in the specified voice. Add `claude-speak preview --all` that plays a short sample in every available voice with the voice name announced.
- **Files affected:** `claude_speak/cli.py`
- **Estimated effort:** S
- **Dependencies:** None
- **Acceptance criteria:** `claude-speak preview af_sarah` speaks "Hello, this is af_sarah." in that voice. `claude-speak preview --all` cycles through all voices with a 1-second gap. `claude-speak preview "bm_george:60+bm_fable:40"` previews a blended voice.

#### 5.2.3 SSML-like Markup Support

- **Description:** Add support for inline speech markup in normalized text to control emphasis, pauses, and speed changes. Format: `<pause 500ms>`, `<slow>important text</slow>`, `<spell>API</spell>`. Processed at the TTS engine level by splitting text at markup boundaries and adjusting parameters per segment.
- **Files affected:** `claude_speak/tts.py`, `claude_speak/normalizer.py`
- **Estimated effort:** L
- **Dependencies:** 5.1.1
- **Acceptance criteria:** `<pause 500ms>` inserts 500ms of silence between segments. `<slow>` reduces speed by 20% for the enclosed text. `<spell>` spells out each character with pauses. Markup is stripped before sending to TTS engine; silence and speed changes are applied at the audio level.

---

## Phase 6: Wake Word & Voice Control (Weeks 11-12)

### 6.1 Wake Word Improvements

#### 6.1.1 Real-World Training Data Collection

- **Description:** The current stop.onnx model is trained entirely on synthetic Kokoro TTS data. Collect real-world recordings of wake words from diverse speakers, environments, and microphones. Create a data collection script that records 10-second clips and labels them.
- **Files affected:** `train/collect_data.py` (new), `train/README.md` (new)
- **Estimated effort:** L
- **Dependencies:** None
- **Acceptance criteria:** Data collection script records labeled clips at 16kHz. Supports positive and negative examples. Saves in openwakeword-compatible format. Instructions for contributors to submit recordings.

#### 6.1.2 Custom Wake Word Training Pipeline

- **Description:** Create a user-facing tool that trains a personalized wake word model. User says their chosen wake word 10 times, the tool augments the recordings, downloads negative samples from a shared dataset, trains the model, and exports to ONNX.
- **Files affected:** `claude_speak/train_wakeword.py` (new), `claude_speak/cli.py`
- **Estimated effort:** XL
- **Dependencies:** 6.1.1
- **Acceptance criteria:** `claude-speak train-wakeword "hey claude"` guides the user through 10 recordings, trains a model in <2 minutes on M1, and saves it to `~/.claude-speak/models/`. Model achieves >90% recall and <5% false positive rate in testing.

#### 6.1.3 Wake Word Accuracy Benchmarking

- **Description:** Create a benchmark suite that tests wake word models against a set of positive and negative audio clips. Reports recall, precision, F1 score, and false positive rate at various sensitivity thresholds.
- **Files affected:** `train/benchmark.py` (new)
- **Estimated effort:** M
- **Dependencies:** 6.1.1
- **Acceptance criteria:** Benchmark runs against 100+ positive and 100+ negative clips. Outputs a table of metrics at sensitivity 0.3, 0.4, 0.5, 0.6, 0.7. ROC curve saved as an image. Results reproducible across runs.

### 6.2 Extended Voice Commands

#### 6.2.1 Additional Voice Commands

- **Description:** Add recognition for more voice commands beyond "stop": "pause" (mute TTS, resume on next wake word), "repeat" (replay the last spoken message), "louder" / "quieter" (adjust volume by 20%), "faster" / "slower" (adjust speed by 10%).
- **Files affected:** `claude_speak/voice_controller.py`, `claude_speak/daemon.py`, `claude_speak/config.py`
- **Estimated effort:** L
- **Dependencies:** 3.1.2 (built-in STT for transcribing commands)
- **Acceptance criteria:** Each command is recognized after wake word activation. "Repeat" replays the most recent message. Volume and speed changes persist across messages. Changes are logged. All commands configurable in config (can be remapped or disabled).

#### 6.2.2 Media Key Support

- **Description:** Add support for hardware media keys (play/pause, volume up/down) to control TTS playback. On macOS, intercept media key events using `pyobjc` or CGEvent tap. Play/pause toggles mute. Volume keys adjust TTS volume.
- **Files affected:** `claude_speak/media_keys.py` (new), `claude_speak/daemon.py`, `claude_speak/config.py`
- **Estimated effort:** L
- **Dependencies:** None
- **Acceptance criteria:** Play/pause button on keyboard or AirPods toggles TTS mute. Volume buttons adjust TTS volume (not system volume). Works when Claude Code is not the frontmost app. Can be disabled via config. No conflict with media apps (only intercepts when TTS is active).

#### 6.2.3 Configurable Keyboard Shortcuts

- **Description:** Add configurable global keyboard shortcuts for common actions: toggle TTS, stop playback, start voice input. Use pyobjc CGEvent tap for global hotkey registration.
- **Files affected:** `claude_speak/hotkeys.py` (new), `claude_speak/config.py`, `claude_speak/daemon.py`
- **Estimated effort:** L
- **Dependencies:** None
- **Acceptance criteria:** Default shortcuts: Cmd+Shift+S (toggle TTS), Cmd+Shift+X (stop), Cmd+Shift+V (voice input). All configurable in config file. Shortcuts work globally regardless of focused app. Conflicts with existing shortcuts are detected and warned.

---

## Phase 7: Claude Code Integration (Weeks 13-14)

### 7.1 Improved Hook Architecture

#### 7.1.1 Replace Shell Hooks with Python

- **Description:** Rewrite `speak-response.sh` in Python for better error handling, type safety, and testability. The shell script currently uses jq, sed, tail, and perl for text processing. A Python version can import the normalizer directly and handle edge cases better.
- **Files affected:** `claude_speak/hooks/speak_response.py` (new), `hooks/speak-response.sh` (deprecated wrapper)
- **Estimated effort:** L
- **Dependencies:** 2.1.1 (package structure)
- **Acceptance criteria:** Python hook handles the same hook events (PostToolUse, Stop) with identical behavior. Transcript parsing uses json module instead of jq. Text extraction and markdown stripping use normalizer functions. Position tracking is atomic (file locks). Performance is within 10% of shell version. Shell wrapper calls Python script for backward compatibility.

#### 7.1.2 Hook Communication Protocol

- **Description:** Replace the SIGUSR1 signal + file-based queue with a Unix domain socket for communication between hooks and the daemon. This eliminates race conditions, enables acknowledgment of message receipt, and supports bidirectional communication.
- **Files affected:** `claude_speak/ipc.py` (new), `claude_speak/daemon.py`, `claude_speak/hooks/speak_response.py`
- **Estimated effort:** L
- **Dependencies:** 7.1.1
- **Acceptance criteria:** Daemon listens on `/tmp/claude-speak.sock`. Hook sends messages via socket (non-blocking). Daemon acknowledges receipt. File-based queue kept as fallback if socket is unavailable. Latency from hook to daemon speech is unchanged or improved.

### 7.2 Bidirectional Communication

#### 7.2.1 Control API via Unix Socket

- **Description:** Expose a JSON-based control API on the Unix domain socket for programmatic control of the daemon. Commands: speak(text), stop(), pause(), resume(), set_voice(voice), set_speed(speed), set_volume(volume), status(), queue_depth().
- **Files affected:** `claude_speak/ipc.py`, `claude_speak/daemon.py`
- **Estimated effort:** L
- **Dependencies:** 7.1.2
- **Acceptance criteria:** External programs can connect to `/tmp/claude-speak.sock` and send JSON commands. Responses include success/failure status. `cli.py` uses the socket API instead of PID file signals.

#### 7.2.2 Context-Aware Speech

- **Description:** Use different voice parameters for different types of content. Code descriptions get a slightly different tone than explanations. Error messages get a distinct treatment. The hook script or normalizer adds metadata tags that the daemon uses to select parameters.
- **Files affected:** `claude_speak/normalizer.py`, `claude_speak/daemon.py`, `claude_speak/tts.py`
- **Estimated effort:** M
- **Dependencies:** 5.2.3 (SSML-like markup)
- **Acceptance criteria:** Code block descriptions use a slightly lower speed. Error messages use a slightly different voice blend or tone. The changes are subtle and configurable. Users can disable context-aware speech via config.

#### 7.2.3 Interrupt Handling

- **Description:** When the user speaks a new prompt while Claude is still responding (and TTS is playing), immediately stop TTS, clear the queue, and allow the new prompt to be processed. Currently, voice input already stops TTS, but the transition should be seamless.
- **Files affected:** `claude_speak/voice_controller.py`, `claude_speak/daemon.py`
- **Estimated effort:** M
- **Dependencies:** 3.1.4 (built-in voice input)
- **Acceptance criteria:** User says "hey jarvis" during TTS -> TTS stops immediately (<100ms), voice input activates, new prompt is submitted. Previous incomplete response is discarded. No audio artifacts (clicks, pops) during interruption.

---

## Phase 8: Documentation & Community (Weeks 15-16)

### 8.1 User Documentation

#### 8.1.1 README Rewrite

- **Description:** Rewrite the README for an open-source audience. Include: project description, demo GIF/video, quick start (3 steps), feature list, configuration reference, voice list with audio samples (linked), architecture overview, FAQ, and contributing link.
- **Files affected:** `README.md`
- **Estimated effort:** L
- **Dependencies:** Phase 2 complete (installation story)
- **Acceptance criteria:** A developer who has never seen the project can go from zero to working setup in under 5 minutes by following the README. Demo GIF shows the full flow: Claude Code responding -> voice output. All config options documented with descriptions and defaults.

#### 8.1.2 Architecture Documentation

- **Description:** Write detailed architecture documentation covering: system overview, data flow (text -> normalization -> chunking -> TTS -> audio), threading model, IPC mechanisms, state management (sentinel files), and extension points.
- **Files affected:** `docs/ARCHITECTURE.md` (new)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** A new contributor can understand the system architecture by reading this document. Includes updated ASCII diagrams, sequence diagrams for key flows (speak, wake word, voice input), and a module dependency graph.

#### 8.1.3 Configuration Reference

- **Description:** Generate a comprehensive configuration reference documenting every config option, its type, default value, valid range, and example values. Include environment variable overrides.
- **Files affected:** `docs/CONFIGURATION.md` (new)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** Every field in every config section is documented. Examples for common setups (JARVIS voice, fast speed, AirPods device). Troubleshooting section for common config mistakes.

### 8.2 Contributor Documentation

#### 8.2.1 Contributing Guide

- **Description:** Write a contributing guide covering: development setup, code style (ruff config), testing (pytest commands), PR process, commit message format, and issue templates.
- **Files affected:** `CONTRIBUTING.md` (new)
- **Estimated effort:** M
- **Dependencies:** 1.5 (linting setup)
- **Acceptance criteria:** A new contributor can set up the development environment, run tests, and submit a PR by following this guide. Includes instructions for testing without audio hardware (mock mode).

#### 8.2.2 Issue Templates

- **Description:** Create GitHub issue templates for bug reports, feature requests, and audio device issues. Bug reports should include: OS version, Python version, audio device info, config file, and daemon log.
- **Files affected:** `.github/ISSUE_TEMPLATE/bug_report.md` (new), `.github/ISSUE_TEMPLATE/feature_request.md` (new), `.github/ISSUE_TEMPLATE/audio_issue.md` (new)
- **Estimated effort:** S
- **Dependencies:** None
- **Acceptance criteria:** Creating a new issue on GitHub shows template options. Bug report template includes all diagnostic fields. `claude-speak diagnostics` command generates a paste-friendly report with all relevant system info.

### 8.3 CI/CD

#### 8.3.1 GitHub Actions CI

- **Description:** Set up GitHub Actions workflow for: linting (ruff), type checking (mypy), unit tests (pytest), and build verification (python -m build). Tests run on macOS runners (required for sounddevice). Integration tests marked as optional (require model files).
- **Files affected:** `.github/workflows/ci.yml` (new)
- **Estimated effort:** L
- **Dependencies:** 1.5 (linting), 1.1 (tests)
- **Acceptance criteria:** CI runs on every push and PR. Three jobs: lint, typecheck, test. All jobs pass on the main branch. macOS-latest runner. Test job skips tests marked `@pytest.mark.requires_model`. Badge in README shows CI status.

#### 8.3.2 Release Automation

- **Description:** Set up GitHub Actions workflow for automated releases. On git tag (v*), build wheel, publish to PyPI, create GitHub Release with changelog, and upload wheel as release artifact.
- **Files affected:** `.github/workflows/release.yml` (new), `pyproject.toml`
- **Estimated effort:** M
- **Dependencies:** 8.3.1, 2.1 (package structure)
- **Acceptance criteria:** Pushing a `v1.0.0` tag triggers: PyPI publish, GitHub Release with auto-generated changelog, and wheel artifact. PyPI publish uses trusted publisher (OIDC). Version in pyproject.toml matches the git tag.

#### 8.3.3 Changelog Automation

- **Description:** Set up automated changelog generation from commit messages or PR titles. Use conventional commits format or GitHub's auto-generated release notes.
- **Files affected:** `CHANGELOG.md` (new), `.github/workflows/release.yml`
- **Estimated effort:** S
- **Dependencies:** 8.3.2
- **Acceptance criteria:** Each release has a changelog section with: new features, bug fixes, breaking changes, and dependency updates. Changelog is included in the GitHub Release and published to the docs.

### 8.4 Demo Materials

#### 8.4.1 Demo Video

- **Description:** Record a 2-minute demo video showing: (1) installation via pip, (2) first-run setup, (3) Claude Code session with voice output, (4) wake word activation, (5) voice input submitting a question, (6) stop command. Upload to YouTube and embed in README.
- **Files affected:** `README.md` (embed link)
- **Estimated effort:** M
- **Dependencies:** Phase 2, Phase 3
- **Acceptance criteria:** Video is <2 minutes. Shows the full user experience. Audio is clear. Captions included. Linked from README and docs.

---

## Phase 9: Polish & Launch (Weeks 17-18)

### 9.1 Performance

#### 9.1.1 Performance Profiling

- **Description:** Profile the full speech pipeline (hook -> daemon -> normalize -> chunk -> TTS -> audio) using Python's cProfile and custom timing. Identify bottlenecks. Current perf logging is ad-hoc (CLAUDE_SPEAK_PERF env var). Make it systematic.
- **Files affected:** `claude_speak/profiler.py` (new), `claude_speak/daemon.py`
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** `claude-speak benchmark` runs 10 representative texts through the full pipeline and reports P50, P95, P99 latencies for: hook-to-daemon, normalization, chunking, TTS generation, audio playback, and end-to-end. Results saved to a JSON file. Target: <300ms from hook to first audio for short messages.

#### 9.1.2 Memory Leak Detection

- **Description:** Run the daemon for 24 hours with continuous speech output and monitor memory usage. Use `tracemalloc` to detect leaks. The Kokoro ONNX session, sounddevice streams, and numpy arrays are the primary suspects.
- **Files affected:** `claude_speak/daemon.py` (add optional tracemalloc)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** Memory usage after 24 hours is within 20% of usage after 1 hour. No monotonically increasing allocations. `claude-speak status` shows current memory usage. Any leaks found are fixed.

#### 9.1.3 Battery Impact Assessment

- **Description:** Measure battery impact of continuous wake word listening on MacBook. The wake word listener opens a 16kHz microphone input stream and runs inference on every 80ms frame. Measure CPU usage and estimate battery drain.
- **Files affected:** `docs/BATTERY.md` (new, or section in README)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** Document includes: CPU usage with wake word active vs. inactive, estimated battery drain per hour, and recommendations (e.g., disable wake word on battery, reduce frame rate). If battery impact is >5% per hour, implement optimizations (lower frame rate, batch inference).

### 9.2 Security

#### 9.2.1 Security Audit

- **Description:** Audit all code paths for security issues: osascript command injection (user text passed to AppleScript), hook script injection (transcript content passed to shell), file permission on queue directory (world-readable /tmp), and secrets in config files.
- **Files affected:** Multiple files (fixes applied where needed)
- **Estimated effort:** L
- **Dependencies:** None
- **Acceptance criteria:** No command injection possible via transcript text or config values. Queue directory has restricted permissions (700). Config file with API keys (ElevenLabs) has restricted permissions (600). osascript calls use parameterized input (not string interpolation). Hook scripts quote all variables.

#### 9.2.2 Restrict File Permissions

- **Description:** Set restrictive permissions on all runtime files: PID file (600), lock file (600), queue directory (700), log file (600), config file (600). Currently, these files are created with default umask permissions.
- **Files affected:** `claude_speak/daemon.py`, `claude_speak/queue.py`, `claude_speak/config.py`
- **Estimated effort:** S
- **Dependencies:** None
- **Acceptance criteria:** All runtime files created by claude-speak have owner-only permissions. Queue directory is 700. No sensitive data (API keys, conversation content) is logged at INFO level.

### 9.3 Accessibility

#### 9.3.1 Accessibility Permissions Guide

- **Description:** Create a guide for setting up macOS Accessibility permissions required by osascript key simulation. Include screenshots, troubleshooting for common issues (permission denied, System Preferences changes between macOS versions), and a diagnostic command.
- **Files affected:** `docs/ACCESSIBILITY.md` (new), `claude_speak/cli.py` (add `claude-speak check-permissions`)
- **Estimated effort:** M
- **Dependencies:** None
- **Acceptance criteria:** `claude-speak check-permissions` tests Accessibility access and reports pass/fail with instructions. Guide covers macOS 13 (Ventura), 14 (Sonoma), and 15 (Sequoia). Includes screenshots for granting Terminal.app and iTerm2 Accessibility access.

### 9.4 Launch

#### 9.4.1 Beta Testing Program

- **Description:** Recruit 10-20 beta testers from the Claude Code community. Create a feedback form covering: installation experience, audio quality, latency perception, wake word accuracy, and feature requests. Run for 2 weeks before public launch.
- **Files affected:** None (external coordination)
- **Estimated effort:** L
- **Dependencies:** All previous phases
- **Acceptance criteria:** 10+ testers complete the feedback form. Installation success rate >80%. No critical bugs reported. Average satisfaction score >7/10. Top feature requests documented and prioritized.

#### 9.4.2 PyPI Publication

- **Description:** Publish claude-speak 1.0.0 to PyPI. Verify the package installs correctly from PyPI (not just local). Test on a clean machine / clean virtualenv.
- **Files affected:** `pyproject.toml` (version bump), `.github/workflows/release.yml`
- **Estimated effort:** S
- **Dependencies:** 8.3.2
- **Acceptance criteria:** `pip install claude-speak` installs from PyPI. `claude-speak setup` works on a clean machine. Version is 1.0.0. PyPI page has README, classifiers, and links.

#### 9.4.3 Launch Blog Post

- **Description:** Write a blog post / Show HN submission announcing claude-speak. Cover: motivation (why voice output for Claude Code), architecture decisions, demo video, installation instructions, and future roadmap.
- **Files affected:** None (external)
- **Estimated effort:** M
- **Dependencies:** 9.4.2
- **Acceptance criteria:** Post clearly explains the value proposition. Includes demo video or GIF. Installation is 3 commands or fewer. Links to GitHub repo and PyPI. Submitted to Hacker News and shared on Twitter/X.

---

## Appendix

### A. Risk Register

| Risk | Phase | Impact | Probability | Mitigation |
|------|-------|--------|-------------|------------|
| Kokoro ONNX API changes break TTS engine | 1-9 | High | Low | Pin version, abstract interface (5.1.1) |
| openwakeword library abandoned | 6 | Medium | Low | Abstract interface, alternative backends |
| Superwhisper app discontinued | 3 | High | Medium | Phase 3 removes dependency entirely |
| macOS audio API changes break sounddevice | 4 | High | Low | Pin sounddevice version, test on each macOS release |
| Apple Silicon ONNX performance regression | 2 | Medium | Low | Benchmark on each onnxruntime release |
| Claude Code hook format changes | 7 | High | Medium | Monitor Claude Code changelog, abstract hook interface |
| PyPI name "claude-speak" already taken | 2 | Medium | Low | Check availability early, have backup names (claude-voice, claude-tts) |
| Bluetooth audio issues across different headphones | 4 | Medium | High | Extensive hardware testing matrix (4.1.3) |
| False wake word activations in noisy environments | 6 | Medium | High | Real-world training data (6.1.1), adjustable sensitivity |
| Memory leaks in long-running daemon | 9 | Medium | Medium | 24-hour soak test (9.1.2), tracemalloc monitoring |
| osascript Accessibility permission complexity | 3, 9 | Medium | High | Permissions guide (9.3.1), diagnostic command |
| Whisper model size too large for some users | 3 | Low | Medium | Multiple model size options, default to "tiny" |
| Network required for first-run model download | 2 | Low | Low | Offline install option with pre-bundled models |

### B. Decision Log

| Decision | Date | Rationale | Alternatives Considered |
|----------|------|-----------|------------------------|
| Kokoro ONNX for TTS | Pre-project | Fast, local, good quality, small model size, MIT license | Piper (less natural), Coqui (larger models), ElevenLabs (requires API key) |
| File-based queue over IPC | Pre-project | Simple, debuggable (can inspect queue via ls), no additional dependencies | Unix socket (planned for Phase 7), Redis (overkill), shared memory (complex) |
| openwakeword for wake word | Pre-project | Pre-trained models, ONNX inference, customizable, MIT license | Porcupine (proprietary), Snowboy (deprecated), custom RNN (too much work) |
| Superwhisper for voice input | Pre-project | High-quality local transcription, macOS native, no additional model needed | Whisper.cpp (planned for Phase 3), Apple Dictation (no API), Google STT (cloud) |
| SIGUSR1 for queue notification | Pre-project | Zero-latency wake-up of daemon event loop, no polling needed | Filesystem watcher (inotify not on macOS), polling (adds latency), socket (planned for Phase 7) |
| Sentinel files for state | Pre-project | Simple inter-process signaling, inspectable, no IPC needed | Shared memory (complex), socket messages (requires connection), database (overkill) |
| Synthetic data for wake word training | Phase 0 | Kokoro can generate diverse training data with many voices and speeds | Recording real speakers (time-consuming), crowd-sourcing (complex) |
| TOML for config | Pre-project | Human-readable, Python 3.11+ built-in (tomllib), simple structure | YAML (requires pyyaml), JSON (no comments), INI (limited nesting) |

### C. Dependency Matrix

```
Phase 1: Foundation & Quality
  (no external dependencies — can start immediately)

Phase 2: Installation & Packaging
  depends on: Phase 1 (tests passing before refactoring)
  blocks: Phase 3, Phase 7

Phase 3: Remove Superwhisper Dependency
  depends on: Phase 2 (package structure for new modules)
  blocks: Phase 6 (voice commands need STT), Phase 7 (interrupt handling)

Phase 4: Cross-Platform Audio
  depends on: nothing (can run in parallel with Phase 3)
  blocks: nothing directly

Phase 5: Voice & TTS Improvements
  depends on: nothing (can start after Phase 1)
  blocks: Phase 7 (context-aware speech needs SSML support)

Phase 6: Wake Word & Voice Control
  depends on: Phase 3 (voice commands need built-in STT)
  blocks: nothing directly

Phase 7: Claude Code Integration
  depends on: Phase 2 (package structure), Phase 3 (interrupt handling)
  blocks: nothing directly

Phase 8: Documentation & Community
  depends on: Phase 2 (installation story for docs)
  blocks: Phase 9 (launch needs docs)

Phase 9: Polish & Launch
  depends on: All previous phases
  blocks: nothing (this is the end)
```

**Critical path:** Phase 1 -> Phase 2 -> Phase 3 -> Phase 7 -> Phase 8 -> Phase 9

**Parallelizable:** Phase 4 and Phase 5 can run in parallel with Phases 3-6.

### D. Metrics to Track

#### Latency Metrics
| Metric | Measurement Point | Target | Tool |
|--------|-------------------|--------|------|
| Hook-to-daemon latency | Time from hook script queue write to daemon dequeue | <50ms | CLAUDE_SPEAK_PERF timestamps |
| Normalization latency | Time to normalize a 500-char message | <10ms | time.monotonic() in daemon |
| TTS first-segment latency | Time from text input to first audio samples generated | <200ms (Kokoro), <500ms (Whisper STT) | Perf logger in tts.py |
| End-to-end time-to-speech | Time from Claude response to first audio output | <500ms | Hook timestamp -> first audio sample |
| Wake word detection latency | Time from wake word spoken to callback fired | <200ms | openwakeword frame time + model inference |
| Voice input cycle latency | Time from wake word to transcribed text submitted | <3s (excluding speech duration) | Timestamps in voice_controller.py |

#### Accuracy Metrics
| Metric | Target | Tool |
|--------|--------|------|
| Wake word recall (true positive rate) | >95% at sensitivity 0.5 | Benchmark suite (6.1.3) |
| Wake word false positive rate | <2 per hour in quiet office | 1-hour soak test |
| Stop command detection accuracy | >90% recall | Benchmark suite |
| STT word error rate (WER) | <10% on standard dictation | LibriSpeech test set |
| Normalization correctness | >99% of transforms produce valid speech text | Test suite (1.1) |

#### Resource Metrics
| Metric | Target | Tool |
|--------|--------|------|
| Daemon memory usage (idle) | <200MB | `claude-speak status` / ps |
| Daemon memory usage (peak, during TTS) | <400MB | tracemalloc |
| Daemon CPU usage (idle, wake word active) | <5% of one core | Activity Monitor / top |
| Daemon CPU usage (during TTS generation) | <50% of one core | Activity Monitor / top |
| Battery drain (wake word active) | <3% per hour on MacBook Pro | Empirical measurement (9.1.3) |
| Model disk usage (all models) | <300MB | du -sh ~/.claude-speak/models/ |
| Queue processing throughput | >10 messages/second | Benchmark with rapid enqueue |

#### User Experience Metrics (Beta)
| Metric | Target | Tool |
|--------|--------|------|
| Installation success rate | >90% on first attempt | Beta feedback form |
| Time from clone to first speech | <5 minutes | Beta feedback form |
| User satisfaction score | >7/10 | Beta feedback form |
| Most requested missing feature | Documented and prioritized | Beta feedback form |
| Critical bugs found | 0 blockers at launch | GitHub issues |
