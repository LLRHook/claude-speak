# Architecture

Technical architecture documentation for claude-speak, a voice interface for Claude Code
that provides text-to-speech output, wake word detection, and voice input.

---

## 1. System Overview

```
                         TEXT-TO-SPEECH PATH
                         ==================

  Claude Code               Hook                    Daemon
 +-----------+    +---------------------+    +-------------------+
 | Assistant |    | speak_response.py   |    |    daemon.py      |
 | response  |--->| (PostToolUse/Stop)  |--->|                   |
 |           |    |                     |    |  +-------------+  |
 +-----------+    | 1. Read transcript  |    |  | run_loop()  |  |
                  | 2. Extract new text |    |  |   (asyncio) |  |
                  | 3. Strip markdown   |    |  +------+------+  |
                  | 4. Send via IPC     |    |         |         |
                  +-----+---------------+    +---------+---------+
                        |                              |
                   IPC Socket                          |
                 /tmp/claude-speak.sock          +-----v------+
                        |                        | normalize() |
                        |                        +-----+------+
                   +----v----+                         |
                   |  File   |                   +-----v------+
                   |  Queue  |<-- fallback       | chunk_text |
                   | /tmp/   |                   +-----+------+
                   +---------+                         |
                                                 +-----v------+
                                                 | TTSEngine  |
                                                 | .speak()   |
                                                 +-----+------+
                                                       |
                                                 +-----v-------+
                                                 | TTSBackend  |
                                                 | (Kokoro /   |
                                                 |  Piper /    |
                                                 |  ElevenLabs)|
                                                 +-----+-------+
                                                       |
                                                 +-----v-------+
                                                 | sounddevice |
                                                 | OutputStream|
                                                 +-----+-------+
                                                       |
                                                    Speakers
                                                   / AirPods


                        VOICE INPUT PATH
                        ================

  Microphone          Wake Word           Voice Input
 +----------+    +----------------+    +------------------+
 |  Audio   |--->| WakeWordListener|-->| VoiceController  |
 |  Stream  |    | (openwakeword) |    | ._on_wake_word() |
 | (16kHz)  |    +----------------+    +--------+---------+
 +----------+                                   |
                                          +-----v------+
                                          | Record mic |
                                          | (Silero VAD|
                                          |  endpoint) |
                                          +-----+------+
                                                |
                                          +-----v------+
                                          | STT Engine |
                                          | (MLX       |
                                          |  Whisper)  |
                                          +-----+------+
                                                |
                                          +-----v------+
                                          | Paste +    |
                                          | auto-submit|
                                          | to Claude  |
                                          +------------+
```

---

## 2. Component Descriptions

### `daemon.py` -- Daemon Process

The long-running process that loads the TTS engine once and processes speech
requests. Manages the full lifecycle: PID file, exclusive lock, signal handlers,
config hot-reload, and graceful shutdown.

**Key functions:**
- `start(daemonize)` -- Entry point. Loads config, initializes TTSEngine,
  starts IPC server, voice controller, media key handler, and hotkey manager.
- `run_loop(config, engine, voice_controller)` -- Main async loop. Dequeues
  text from the file queue, normalizes, chunks, and calls `engine.speak()`.
- `_try_reload_config()` -- Checks config mtime every 30s, hot-swaps TTS
  backend if the engine field changed.
- `_create_ipc_server()` -- Wires up all IPC message handlers (speak, stop,
  status, pause, resume, set_voice, set_speed, set_volume, queue_depth,
  list_voices).

**Sentinel files:**
- `/tmp/claude-speak-daemon.pid` -- PID of running daemon.
- `/tmp/claude-speak-daemon.lock` -- Exclusive flock to prevent duplicates.
- `/tmp/claude-speak-daemon.start_ts` -- Epoch timestamp for uptime tracking.
- `~/.claude-speak-enabled` -- Toggle file; daemon idles when absent.
- `/tmp/claude-speak-muted` -- Mute sentinel; TTS pauses while present.
- `/tmp/claude-speak-playing` -- Exists while audio is actively playing.

### `tts.py` -- TTS Engine & Kokoro Backend

Two responsibilities: (1) the `KokoroBackend` class implementing Kokoro-ONNX
synthesis, and (2) the `TTSEngine` class handling device routing and playback.

**KokoroBackend:**
- Loads Kokoro ONNX model and voice style files.
- Resolves voice config: single name (e.g., `af_sarah`) or blend syntax
  (`bm_george:60+bm_fable:40`) producing a weighted numpy array.
- `generate()` -- Async generator yielding `(samples, sample_rate)` tuples
  via Kokoro's streaming API.

**TTSEngine:**
- `load()` -- Delegates to backend, resolves audio output device.
- `speak(text)` -- Parses SSML tags via `parse_ssml()`, then iterates
  segments: generates audio through the backend and writes samples to a
  persistent `sounddevice.OutputStream`.
- `generate_audio(text)` / `play_audio(segments)` -- Split generation and
  playback for pipelining (generate chunk N+1 while playing chunk N).
- `stop()` -- Thread-safe immediate stop. Sets `_stopped` event then
  aborts/closes the PortAudio stream under lock.
- `swap_backend(backend)` -- Hot-swap to a different TTS backend at runtime.
- `_ensure_stream(sample_rate)` -- Creates or reuses the output stream with
  retry logic and device fallback.

**Factory:** `create_backend(engine_name, config)` returns a `KokoroBackend`,
`PiperBackend`, or `ElevenLabsBackend`.

### `tts_base.py` -- Backend ABC

```python
class TTSBackend(ABC):
    def load() -> None
    async def generate(text, voice, speed, lang) -> AsyncIterator[tuple[ndarray, int]]
    def list_voices() -> list[str]
    def is_loaded() -> bool
    @property name -> str
```

All backends implement this interface. The engine can swap backends at runtime
without restarting.

### `config.py` -- Configuration

Dataclass-based configuration with TOML loading and env var overrides.

**Config sections (dataclasses):**
- `TTSConfig` -- engine, voice, speed, lang, device, model paths, volume,
  max_chunk_chars, ElevenLabs API key.
- `WakeWordConfig` -- enabled, engine, model, stop_model, sensitivity,
  stop_phrases.
- `InputConfig` -- backend (builtin/superwhisper/auto), auto_submit, VAD
  threshold, STT backend and model.
- `NormalizationConfig` -- skip_code, expand_units, expand_abbreviations,
  shorten_paths, custom_pronunciations path, context_aware.
- `AudioConfig` -- chimes, greeting, volume, bt_mic_workaround,
  media_keys_enabled.
- `HotkeysConfig` -- enabled, toggle_tts, stop_playback, voice_input shortcuts.
- `VoiceCommandsConfig` -- word mappings for pause, resume, repeat, louder,
  quieter, faster, slower, stop.

**Loading priority:** defaults < `claude-speak.toml` < environment variables
(`CLAUDE_SPEAK_VOICE`, `CLAUDE_SPEAK_SPEED`, `CLAUDE_SPEAK_DEVICE`).

### `normalizer.py` -- Text Normalization

Transforms markdown/technical text into speech-friendly prose through a
multi-stage pipeline. See section 7 for the full stage list.

**Public API:**
- `normalize(text, context_aware=True)` -- Full pipeline.
- `chunk_text(text, max_chars=400)` -- Splits at sentence boundaries for
  reliable TTS synthesis.

### `ssml.py` -- SSML-like Markup

Custom tag parser for controlling speech output.

**Supported tags:**
- `<pause 500ms>` -- Insert silence.
- `<slow>text</slow>` -- 0.8x speed modifier.
- `<fast>text</fast>` -- 1.2x speed modifier.
- `<spell>ABC</spell>` -- Expand to `A. P. I.` at 0.7x speed.

**Key types:**
- `SpeechSegment(text, speed_modifier, pause_ms, spell)` -- Dataclass
  consumed by `TTSEngine.speak()`.
- `parse_ssml(text) -> list[SpeechSegment]` -- Parser.
- `generate_silence(duration_ms, sample_rate)` -- Returns zero-filled ndarray.

### `ipc.py` -- IPC Protocol

Unix domain socket IPC using newline-delimited JSON.

**Server (daemon side):**
- `IPCServer` -- Listens on `/tmp/claude-speak.sock` in a background thread.
  Uses `selectors` for non-blocking accept. Dispatches messages to registered
  handlers.
- `register_handler(msg_type, handler)` -- Maps message type strings to
  `(dict) -> dict` callables.

**Client (hook/CLI side):**
- `send_message(msg, timeout=2.0)` -- Connect, send JSON + newline, read
  response, close.
- `is_daemon_running()` -- Probes socket reachability.

### `voice_controller.py` -- Voice Controller

High-level orchestrator tying wake word detection to voice input.

**VoiceController:**
- `start()` / `stop()` -- Manages WakeWordListener lifecycle.
- `_on_wake_word()` -- Context-aware callback:
  - If TTS is playing (PLAYING_FILE exists): interrupt (stop engine, clear
    queue), then start voice input.
  - If TTS is idle: start voice input directly.
- `_handle_wake()` -- Acquires `_input_lock`, pauses wake word listener,
  runs the voice input cycle (builtin or Superwhisper), resumes listener.
- `match_voice_command(text)` / `handle_voice_command(command)` -- Dispatches
  recognized voice commands (pause, resume, repeat, louder, quieter, etc.).
- `handle_stop(phrase)` -- Clears queue and stops TTS playback.

### `wakeword.py` -- Wake Word Detection

**WakeWordListener:**
- Uses openwakeword with ONNX inference to detect wake words (default:
  `hey_jarvis`) and optional stop commands (`stop.onnx`).
- Runs a continuous mic input loop in a dedicated thread.
- Bluetooth workaround: when the output device is Bluetooth, always uses the
  built-in microphone to avoid SCO profile switching (which degrades audio).
- `pause()` / `resume()` -- Releases/reclaims the mic for voice input.
- `use_builtin_mic()` / `use_default_mic()` -- Per-TTS mic swap signals.

### `voice_input.py` -- Voice Input

Two flows:

1. **Built-in** (`builtin_voice_input_cycle`): Record mic via sounddevice,
   detect speech boundaries with Silero VAD, transcribe with MLX Whisper,
   paste via Cmd+V, optionally press Enter to submit.
2. **Superwhisper** (`voice_input_cycle`): Toggle the external Superwhisper
   app via keyboard shortcut, monitor mic for speech-then-silence, toggle
   Superwhisper again to stop, wait for clipboard change, auto-submit.

### `stt.py` -- Speech-to-Text

Abstract `SpeechRecognizer` base class with `MLXWhisperRecognizer`
implementation. Uses `mlx-whisper` for Apple Silicon native transcription
via Metal GPU. Models auto-download from Hugging Face.

### `vad.py` -- Voice Activity Detection

`SileroVAD` wraps the Silero VAD v5 ONNX model. Processes 512-sample chunks
at 16 kHz (32ms). Returns speech probability; maintains hidden state across
calls. Model auto-downloads to `~/.claude-speak/models/`.

### `hooks/speak_response.py` -- Claude Code Hook

Invoked by Claude Code on `PostToolUse` and `Stop` events.

**Flow:**
1. Check toggle file exists (gate).
2. Acquire directory-based lock (serialize concurrent hook invocations).
3. Parse hook JSON from stdin (transcript path, session ID).
4. Read position file to find where we left off in the JSONL transcript.
5. Extract new assistant text blocks.
6. Strip markdown.
7. Send to daemon via IPC socket (fast path) or fall back to file queue +
   SIGUSR1.
8. Update position file.

### `queue.py` -- File-Based Queue

FIFO queue using timestamped `.txt` files in `/tmp/claude-speak-queue/`.

- `enqueue(text)` / `enqueue_chunks(chunks)` -- Write.
- `dequeue()` -- Pop oldest file, return `(path, text)`.
- `peek()` / `depth()` / `clear()` -- Inspection and cleanup.

### `audio_devices.py` -- Audio Device Management

**AudioDeviceManager (singleton):**
- Resolves output/input devices by name substring, index, or `"auto"`.
- Caches resolved device for 30 seconds (`maybe_resolve_output`).
- Detects Bluetooth devices by name heuristic.
- Finds the built-in microphone for BT workaround.

**DeviceChangeMonitor:**
- Background thread polling `sounddevice.query_devices()` every 2 seconds.
- Fires callbacks on device additions/removals, invalidating the cache.

### `cli.py` -- Command-Line Interface

Entry point: `claude-speak <command>`. Dispatches to handler functions.
Communicates with the daemon primarily via IPC socket, falling back to
file-based operations.

---

## 3. Data Flow: Text to Spoken Audio

Detailed walk-through of a single message from Claude Code to audio output:

```
1. Claude Code writes assistant response to JSONL transcript file
         |
2. Claude Code fires PostToolUse/Stop hook
         |
3. hooks/speak-response.sh invokes speak_response.py with hook JSON on stdin
         |
4. speak_response.py:
   a. Checks ~/.claude-speak-enabled (skip if absent)
   b. Acquires directory lock /tmp/claude-speak-hook.lock
   c. Reads /tmp/claude-speak-pos for session position
   d. Opens transcript, skips to position, extracts new assistant text
   e. Strips markdown (bold, italic, links, images, HTML)
   f. Tries IPC socket: send_message({"type":"speak","text":...})
      - On success: daemon receives text directly in _handle_speak()
      - On failure: writes to /tmp/claude-speak-queue/<timestamp>.txt
                    sends SIGUSR1 to daemon PID
   g. Updates position file
         |
5. daemon.py run_loop():
   a. Dequeues text (from IPC handler or file queue)
   b. Checks voice commands and stop phrases
   c. normalize(text):
      - 20+ transform stages (see section 7)
      - Returns speech-ready text with optional SSML tags
   d. chunk_text(normalized, max_chars=400):
      - Splits at sentence boundaries
         |
6. For each chunk:
   a. parse_ssml(chunk) -> list[SpeechSegment]
   b. For each segment:
      - If pause_ms > 0: generate silence, write to stream
      - Compute effective_speed = base_speed * segment.speed_modifier
      - backend.generate(text, voice, speed, lang) -> async iterator
      - For each (samples, sample_rate) yielded:
        * _ensure_stream(sample_rate): create/reuse OutputStream
        * _write_samples(samples): volume scaling, chunked write (4096)
         |
7. Audio flows to speakers via sounddevice/PortAudio
```

For multi-chunk messages, the daemon pipelines generation and playback:
generate chunk N+1 concurrently while playing chunk N.

---

## 4. Backend Abstraction

The `TTSBackend` ABC in `tts_base.py` defines the contract all engines must
implement:

```
TTSBackend (ABC)
    |
    +-- KokoroBackend    (tts.py)        -- Kokoro ONNX, local, streaming
    +-- PiperBackend     (tts_piper.py)  -- Piper TTS, local
    +-- ElevenLabsBackend(tts_elevenlabs.py) -- ElevenLabs cloud API
```

**Hot-swap mechanism:**
1. Config file changes detected by `_try_reload_config()` (mtime check).
2. If `tts.engine` field differs from current, `create_backend()` instantiates
   the new backend.
3. `new_backend.load()` initializes the engine.
4. `engine.swap_backend(new_backend)` stops playback and replaces the backend
   atomically.
5. If the swap fails, the old backend is retained and config is reverted.

The `TTSEngine` never calls backend-specific methods -- all synthesis goes
through `backend.generate()`.

---

## 5. IPC Protocol

**Transport:** Unix domain socket at `/tmp/claude-speak.sock`.

**Wire format:** Newline-delimited JSON. One message per connection
(connect, send, receive, close).

**Message structure:**
```json
// Request (client -> daemon)
{"type": "<message_type>", ...params}

// Response (daemon -> client)
{"ok": true, ...data}
{"ok": false, "error": "description"}
```

**Available message types:**

| Type          | Parameters                  | Response fields          |
|---------------|-----------------------------|--------------------------|
| `speak`       | `text: str`                 | -                        |
| `stop`        | -                           | -                        |
| `status`      | -                           | `queue_depth`, `enabled`, `uptime` |
| `pause`       | -                           | -                        |
| `resume`      | -                           | -                        |
| `set_voice`   | `voice: str`                | `voice`                  |
| `set_speed`   | `speed: float`              | `speed`                  |
| `set_volume`  | `volume: float (0.1-1.0)`   | `volume`                 |
| `queue_depth` | -                           | `depth`                  |
| `list_voices` | -                           | `voices: list[str]`      |

**Handler registration:** In `_create_ipc_server()`, each message type maps
to a closure that captures the engine, config, queue_ready event, and asyncio
loop references.

**Max message size:** 256 KB. **Client timeout:** 2 seconds.

---

## 6. Configuration System

**File:** `claude-speak.toml` in the project root.

**Loading (`load_config()`):**
1. Create `Config()` with dataclass defaults.
2. If `claude-speak.toml` exists, parse with `tomllib` (Python 3.11+) or
   `tomli` fallback.
3. For each TOML section (`[tts]`, `[wakeword]`, etc.), iterate key-value
   pairs and `setattr` onto the matching dataclass if the attribute exists.
4. Override with environment variables (highest priority):
   - `CLAUDE_SPEAK_VOICE` -> `config.tts.voice`
   - `CLAUDE_SPEAK_SPEED` -> `config.tts.speed`
   - `CLAUDE_SPEAK_DEVICE` -> `config.tts.device`
5. Resolve model paths: if user-level `~/.claude-speak/models/` is missing,
   fall back to project-local `models/` directory.

**Hot-reload:**
- `_try_reload_config()` runs every 30 seconds in the main loop.
- Compares `CONFIG_PATH.stat().st_mtime` to cached value.
- On change: loads fresh config, detects engine changes, hot-swaps backend
  if needed, updates engine settings (voice, speed, device).

**Key paths (defined in `config.py`):**
```
PROJECT_DIR     = <repo root>
CONFIG_PATH     = PROJECT_DIR / "claude-speak.toml"
MODELS_DIR      = ~/.claude-speak/models/
LOG_FILE        = PROJECT_DIR / "daemon.log"
QUEUE_DIR       = /tmp/claude-speak-queue/
PID_FILE        = /tmp/claude-speak-daemon.pid
TOGGLE_FILE     = ~/.claude-speak-enabled
MUTE_FILE       = /tmp/claude-speak-muted
PLAYING_FILE    = /tmp/claude-speak-playing
WAKEWORD_DIR    = PROJECT_DIR / wakeword/
```

---

## 7. Audio Pipeline

### 7.1 Normalization Stages

The `normalize()` function in `normalizer.py` applies transforms in this
order:

```
 1. describe_code_blocks   -- Replace fenced code blocks with "Here is <lang> code: <summary>"
 2. narrate_tables         -- Convert markdown tables to spoken prose
 3. improve_lists          -- Convert bullet/numbered lists to "First, ... Second, ..."
 4. strip_code_blocks      -- Remove any remaining fenced blocks
 5. clean_urls_and_emails  -- Replace URLs with domain name, emails with spoken form
 6. clean_file_paths       -- Shorten /long/paths/to/file.py to file.py; src/ -> source/
 7. expand_currency        -- $100 -> "100 dollars"
 8. expand_percentages     -- 50% -> "50 percent"
 9. expand_ordinals        -- 1st -> "first", 2nd -> "second"
10. strip_number_commas    -- 1,000 -> 1000 (TTS handles bare numbers)
11. expand_time_formats    -- 14:30 -> "2:30 PM"
12. expand_fractions_ratios-- 1/2 -> "one half", 16:9 -> "16 to 9"
13. expand_dates           -- 2024-02-26 -> "February 26, 2024"
14. expand_temperature     -- 72F -> "72 degrees Fahrenheit"
15. expand_math_operators  -- x + y -> "x plus y", >= -> "greater than or equal to"
16. expand_units           -- 27MB -> "27 megabytes"
17. clean_version_strings  -- v1.0 -> "version 1 point 0"
18. speak_decimal_numbers  -- 1.5 -> "one point five"
19. clean_file_extensions  -- .py -> "python file", .json -> "JSON file"
20. expand_stop_words      -- e.g. -> "for example", w/o -> "without"
21. expand_abbreviations   -- API -> "A P I", ONNX -> "onyx"
22. expand_slash_pairs     -- true/false -> "true or false"
23. clean_technical_punct  -- Remove backticks, arrows, braces; snake_case -> spaces
24. final_cleanup          -- Normalize whitespace, add periods at line breaks
25. apply_pronunciation_overrides  -- Custom dictionary (pronunciations.toml)
26. annotate_context       -- Insert SSML tags: <slow> for errors, <pause> after headings
```

### 7.2 Pronunciation Overrides

Two-tier dictionary system:
1. **User file:** `~/.claude-speak/pronunciations.toml` (checked first,
   auto-reloaded on mtime change).
2. **Built-in file:** `claude_speak/data/pronunciations.toml` (packaged
   default).

Format:
```toml
[terms]
kubectl = "kube control"
nginx = "engine X"
```

Applied as the final normalization step using whole-word regex replacement.

### 7.3 SSML Parsing

After normalization, `TTSEngine.speak()` passes text through `parse_ssml()`
which extracts `SpeechSegment` objects. Each segment carries:
- `text` to synthesize
- `speed_modifier` (multiplied with base speed)
- `pause_ms` (silence inserted before text)

### 7.4 TTS Generation

The backend's `generate()` async iterator yields `(samples, sample_rate)`
tuples. Kokoro's native rate is 24 kHz. Samples are float32 numpy arrays.

### 7.5 Playback

- A persistent `sounddevice.OutputStream` is reused across segments for
  gapless playback.
- Volume scaling: `samples * config.tts.volume` if volume < 1.0.
- Samples are written in 4096-sample chunks, checking `_stopped` event
  between chunks for responsive interruption.
- Device re-resolution happens every 30 seconds via
  `AudioDeviceManager.maybe_resolve_output()`, with immediate re-resolution
  on device disconnection.

---

## 8. Voice Input Pipeline

### 8.1 Wake Word Detection

```
Mic (16kHz, int16, 80ms chunks)
    |
    v
openwakeword Model.predict(samples)
    |
    +-- wake word score >= sensitivity --> fire wake callbacks
    +-- stop model score >= stop_sensitivity --> fire stop callbacks
```

- Cooldown: 2 seconds between detections to prevent rapid-fire.
- Stop model is checked before wake word (stop takes priority).
- Model state is reset after each detection.

### 8.2 Built-in Voice Input Flow

```
1. Wake word detected
2. VoiceController._on_wake_word()
   - If PLAYING_FILE exists: interrupt TTS (stop engine + clear queue)
3. Spawn voice-input-cycle thread
4. Pause wake word listener (release mic)
5. Initialize Silero VAD + MLX Whisper recognizer
6. Open mic InputStream (16kHz, 512-sample chunks)
7. Wait for speech start (VAD probability > threshold, max 10s)
8. Record speech frames
9. Detect silence (30 consecutive non-speech frames ~= 1 second)
10. Concatenate frames, convert int16 -> float32
11. Transcribe with MLX Whisper
12. Paste at cursor (pbcopy + Cmd+V via osascript)
13. Auto-submit (press Enter via osascript) if enabled
14. Resume wake word listener
```

### 8.3 Interrupt Handling

When the wake word fires during active TTS playback:
1. `_interrupt_callback()` fires: calls `engine.stop()` + `Q.clear()`.
2. `PLAYING_FILE` and `MUTE_FILE` are cleaned up.
3. Voice input begins immediately (no waiting for TTS to finish).

Target latency for interrupt: < 100ms.

---

## 9. Thread Model

```
+--------------------------------------------------+
|  Main Thread (asyncio event loop)                |
|  - daemon.run_loop(): dequeue, normalize,        |
|    chunk, engine.speak()                          |
|  - Signal handlers (SIGUSR1, SIGTERM, SIGINT)    |
|  - Config hot-reload timer                        |
+--------------------------------------------------+

+--------------------------------------------------+
|  IPC Server Thread ("ipc-server")                |
|  - IPCServer._accept_loop()                       |
|  - selectors-based non-blocking accept            |
|  - Handles client connections inline              |
|  - Calls registered handlers (may interact with   |
|    queue and engine)                               |
+--------------------------------------------------+

+--------------------------------------------------+
|  Wake Word Listener Thread ("wakeword-listener") |
|  - WakeWordListener._listen_loop()                |
|  - Continuous mic InputStream                     |
|  - openwakeword inference per 80ms chunk           |
|  - Fires wake/stop callbacks                       |
+--------------------------------------------------+

+--------------------------------------------------+
|  Voice Input Cycle Thread ("voice-input-cycle")   |
|  - Spawned on-demand by _on_wake_word()            |
|  - Acquires _input_lock (non-blocking)             |
|  - Records audio, runs VAD, runs STT               |
|  - Pastes and submits text                          |
+--------------------------------------------------+

+--------------------------------------------------+
|  Device Change Monitor Thread                     |
|  ("device-change-monitor")                         |
|  - Polls sounddevice.query_devices() every 2s     |
|  - Fires callbacks on device add/remove            |
|  - Invalidates AudioDeviceManager cache            |
+--------------------------------------------------+

+--------------------------------------------------+
|  Audio Playback (asyncio.to_thread)               |
|  - engine.play_audio() runs in thread pool         |
|  - Writes samples to OutputStream                  |
|  - _stopped event checked per 4096-sample chunk    |
+--------------------------------------------------+
```

All background threads are daemon threads (process exits when main exits).

**Synchronization primitives:**
- `threading.Lock`: `TTSEngine._stream_lock` (audio stream), `VoiceController._input_lock`.
- `threading.Event`: `TTSEngine._stopped`, `WakeWordListener._running_event`,
  `_paused_event`, `_swap_mic_event`, `DeviceChangeMonitor._running`.
- `asyncio.Event`: `queue_ready` (SIGUSR1 -> main loop wakeup).
- File sentinels: `MUTE_FILE`, `PLAYING_FILE`, `TOGGLE_FILE` (cross-thread
  signaling via POSIX atomic create/unlink).

---

## 10. File/Directory Layout

```
claude-speak/
|-- claude-speak.toml           # Runtime config (user-edited)
|-- claude-speak.toml.example   # Annotated example config
|-- claude-hooks.json           # Claude Code hook definitions
|-- pyproject.toml              # Package metadata, deps, tool config
|-- requirements.txt            # Minimal pip requirements
|-- install.sh                  # First-time setup script
|-- daemon.log                  # Rotating log (5MB x 3 backups)
|
|-- claude_speak/               # Main Python package
|   |-- __init__.py             # Package version
|   |-- __main__.py             # python -m claude_speak entry
|   |-- cli.py                  # CLI dispatcher (claude-speak <cmd>)
|   |-- daemon.py               # Daemon process, main loop, lifecycle
|   |-- config.py               # Config dataclasses, TOML loader, paths
|   |-- tts.py                  # TTSEngine, KokoroBackend, backend factory
|   |-- tts_base.py             # TTSBackend ABC
|   |-- tts_piper.py            # PiperBackend implementation
|   |-- tts_elevenlabs.py       # ElevenLabsBackend implementation
|   |-- normalizer.py           # 25+ text normalization transforms
|   |-- ssml.py                 # SSML tag parser, SpeechSegment
|   |-- ipc.py                  # Unix socket IPC server + client
|   |-- queue.py                # File-based FIFO queue
|   |-- voice_controller.py     # Wake word -> voice input orchestrator
|   |-- wakeword.py             # openwakeword listener
|   |-- voice_input.py          # Built-in and Superwhisper voice input
|   |-- stt.py                  # SpeechRecognizer ABC, MLX Whisper
|   |-- vad.py                  # Silero VAD (ONNX)
|   |-- audio_devices.py        # Device manager, BT detection, monitoring
|   |-- chimes.py               # Audio feedback chimes (ready, ack, error)
|   |-- models.py               # Model registry, download, ensure_models()
|   |-- setup.py                # First-time setup wizard
|   |-- hotkeys.py              # Global keyboard shortcuts
|   |-- media_keys.py           # Hardware media key interception
|   |-- train_wakeword.py       # Custom wake word training
|   |-- data/
|   |   |-- pronunciations.toml # Built-in pronunciation dictionary
|   |-- assets/                 # Audio assets (chime sounds)
|   |-- hooks/
|       |-- __init__.py
|       |-- speak_response.py   # Claude Code hook (PostToolUse/Stop)
|
|-- hooks/                      # Shell hook wrappers
|   |-- speak-response.sh       # Invokes speak_response.py
|   |-- daemon-start.sh         # SessionStart hook
|   |-- daemon-stop.sh          # SessionEnd hook
|
|-- models/                     # Local model files (git-ignored)
|-- wakeword/                   # Custom wake word models
|-- train/                      # Wake word training utilities
|-- tests/                      # pytest test suite (29 test files)
|-- docs/                       # Additional documentation
```

**Runtime artifacts (not in repo):**
```
~/.claude-speak/
|-- models/                     # Downloaded models (Kokoro, Silero VAD)
|   |-- kokoro-v1.0.onnx
|   |-- voices-v1.0.bin
|   |-- silero_vad.onnx
|-- pronunciations.toml         # User pronunciation overrides (optional)

/tmp/
|-- claude-speak-daemon.pid
|-- claude-speak-daemon.lock
|-- claude-speak-daemon.start_ts
|-- claude-speak.sock           # IPC Unix domain socket
|-- claude-speak-queue/         # File-based message queue
|-- claude-speak-pos            # Hook transcript position tracker
|-- claude-speak-muted          # Mute sentinel
|-- claude-speak-playing        # Playback sentinel
|-- claude-speak-hook.lock/     # Hook serialization lock (directory)
|-- claude-speak-perf.log       # Performance log (CLAUDE_SPEAK_PERF=1)

~/.claude-speak-enabled         # Toggle sentinel (TTS on/off)
```
