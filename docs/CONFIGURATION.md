# Configuration Reference

Complete reference for all claude-speak configuration options.

---

## Config File Location

claude-speak reads configuration from a [TOML](https://toml.io/) file. The loader checks the following location:

| Path | Description |
|------|-------------|
| `<project-root>/claude-speak.toml` | Primary config file, alongside your project code |

The project root is the parent directory of the `claude_speak/` Python package.

For ElevenLabs credentials specifically, a user-level config is also supported:

| Path | Description |
|------|-------------|
| `~/.claude-speak/config.toml` | User-level config (currently only `[elevenlabs] api_key`) |

If no config file is found, all settings use their built-in defaults.

### Precedence Order (highest to lowest)

1. **Environment variables** (e.g. `CLAUDE_SPEAK_VOICE`) -- always win
2. **`claude-speak.toml`** values -- override defaults
3. **Built-in defaults** from the dataclass definitions

---

## Sections

- [`[tts]`](#tts) -- Text-to-speech engine, voice, speed, volume
- [`[wakeword]`](#wakeword) -- Wake word detection (e.g. "Hey Jarvis")
- [`[input]`](#input) -- Voice input / dictation settings
- [`[normalization]`](#normalization) -- Text preprocessing for speech
- [`[audio]`](#audio) -- Chimes, greetings, system audio
- [`[hotkeys]`](#hotkeys) -- Global keyboard shortcuts
- [`[voice_commands]`](#voice_commands) -- Spoken command words

---

## `[tts]`

Controls which TTS engine is used, voice selection, speed, language, and audio output.

| Field | Type | Default | Valid Values | Description |
|-------|------|---------|--------------|-------------|
| `engine` | string | `"kokoro"` | `"kokoro"`, `"piper"`, `"elevenlabs"` | TTS backend engine. Kokoro is the built-in local engine. Piper is an alternative local engine. ElevenLabs uses a cloud API. |
| `voice` | string | `"af_sarah"` | See [Voice Names](#voice-names) | Voice name or blend specification. For Kokoro, use voice IDs like `af_sarah`, `bm_george`. For blends, use the format `name1:weight+name2:weight` (weights are percentages that should sum to 100). For ElevenLabs, use a voice ID from your account. For Piper, use model names like `en_US-lessac-medium`. |
| `speed` | float | `1.0` | `0.1` -- `5.0` (practical range) | Speech rate multiplier. `1.0` is normal speed, `1.5` is 50% faster, `0.5` is half speed. |
| `lang` | string | `"en-us"` | `"en-us"`, `"en-gb"`, etc. | Language code passed to the TTS engine. Kokoro uses this for pronunciation rules. Use `en-gb` for British English pronunciation. |
| `device` | string | `"auto"` | `"auto"` or device name substring | Audio output device. `"auto"` selects the system default. To target a specific device, provide a substring of its name (e.g. `"AirPods"`, `"BlackHole"`). The engine performs fuzzy matching. |
| `model_path` | string | `"~/.claude-speak/models/kokoro-v1.0.onnx"` | Absolute or relative file path | Path to the Kokoro ONNX model file. Auto-downloaded on first run if missing. Also checks `<project>/models/` as a fallback. |
| `voices_path` | string | `"~/.claude-speak/models/voices-v1.0.bin"` | Absolute or relative file path | Path to the Kokoro voice styles file. Auto-downloaded on first run if missing. Also checks `<project>/models/` as a fallback. |
| `max_chunk_chars` | int | `400` | `50` -- `2000` | Maximum characters per text chunk sent to the TTS engine. Smaller values produce faster first-audio latency (text is synthesized in parallel chunks). |
| `volume` | float | `1.0` | `0.0` -- `1.0` | TTS speech volume. `1.0` is full volume, `0.0` is silent. This controls the speech audio only (not chimes -- see `[audio].volume`). |
| `elevenlabs_api_key` | string | `""` (empty) | Any valid API key | ElevenLabs API key. Only needed when `engine = "elevenlabs"`. The `ELEVENLABS_API_KEY` environment variable takes priority over this field. |

### Voice Names

**Kokoro voices** follow the pattern `{gender}{region}_{name}`:
- Prefix: `af` = American female, `am` = American male, `bf` = British female, `bm` = British male
- Examples: `af_sarah`, `af_nicole`, `am_adam`, `am_michael`, `bf_emma`, `bm_george`, `bm_fable`

**Kokoro voice blending** lets you mix multiple voices:
- Format: `voice1:weight+voice2:weight` where weights are percentages
- Example: `bm_george:60+bm_fable:40` (60% George, 40% Fable)
- Weights should sum to 100. If omitted, equal weights are applied.

**Piper voices** follow the pattern `{lang}_{REGION}-{speaker}-{quality}`:
- Available: `en_US-lessac-medium`, `en_US-ryan-medium`, `en_GB-alan-medium`
- Custom models: place `.onnx` + `.onnx.json` files in `~/.claude-speak/models/piper/`

**ElevenLabs voices** use voice IDs from your ElevenLabs account. Use the ElevenLabs dashboard or API to find your voice IDs. Cloned voices work by using their voice ID.

---

## `[wakeword]`

Controls hands-free wake word detection. When enabled, saying the wake word (e.g. "Hey Jarvis") triggers voice input mode.

| Field | Type | Default | Valid Values | Description |
|-------|------|---------|--------------|-------------|
| `enabled` | bool | `false` | `true`, `false` | Enable or disable wake word detection. When disabled, voice input is only triggered via hotkeys. |
| `engine` | string | `"openwakeword"` | `"openwakeword"` | Wake word detection engine. Currently only openwakeword is supported. Requires the `openwakeword` Python package. |
| `model` | string | `"hey_jarvis"` | Pre-trained model name or path | Wake word model to use. `"hey_jarvis"` is a built-in openwakeword model. You can also provide a path to a custom `.onnx` model. |
| `stop_model` | string | `""` (empty/disabled) | Path to `.onnx` file | Path to a stop-word ONNX model for instant TTS interruption (~80ms latency). Set to `""` to disable stop detection. Example: `"models/stop.onnx"`. |
| `sensitivity` | float | `0.5` | `0.0` -- `1.0` | Wake word detection threshold. Higher values require a more confident match (fewer false positives, more missed detections). Lower values are more sensitive (more false positives). |
| `stop_sensitivity` | float | `0.5` | `0.0` -- `1.0` | Separate detection threshold for the stop model. Works the same as `sensitivity` but applies only to stop-word detection. |
| `stop_phrases` | array of strings | `["stop", "quiet", "shut up"]` | Any list of strings | Phrases that, when detected by the STT engine during wake word listening, will stop TTS playback. These are matched against transcribed speech, not wake word models. |

---

## `[input]`

Controls voice input (speech-to-text / dictation) behavior.

| Field | Type | Default | Valid Values | Description |
|-------|------|---------|--------------|-------------|
| `backend` | string | `"builtin"` | `"builtin"`, `"superwhisper"`, `"auto"` | Voice input backend. `"builtin"` uses the integrated mic recording + Whisper STT pipeline. `"superwhisper"` uses the external Superwhisper macOS app. `"auto"` tries Superwhisper first, then falls back to builtin. |
| `auto_submit` | bool | `true` | `true`, `false` | Automatically submit (press Enter) after voice input is transcribed and pasted. When `false`, text is pasted but you must press Enter manually. |
| `superwhisper_shortcut_keycode` | int | `49` | macOS virtual key code | The virtual key code used to trigger Superwhisper's recording shortcut. Default `49` is the Space key. Only relevant when `backend` is `"superwhisper"` or `"auto"`. |
| `superwhisper_shortcut_modifiers` | int | `2048` | macOS modifier flags | The modifier key flags for the Superwhisper shortcut. Default `2048` is the Option key. Combined with the keycode to form the full shortcut (default: Option+Space). |
| `vad_threshold` | float | `0.5` | `0.0` -- `1.0` | Silero VAD (Voice Activity Detection) speech probability threshold. Audio frames with a speech probability above this threshold are considered speech. Lower = more sensitive, higher = more strict. Only used by the `"builtin"` backend. |
| `stt_backend` | string | `"auto"` | `"auto"`, `"mlx"`, `"whisper_cpp"` | Speech-to-text engine for the builtin input backend. `"auto"` selects the best available (prefers MLX Whisper on Apple Silicon). `"mlx"` forces MLX Whisper. `"whisper_cpp"` forces whisper.cpp. |
| `stt_model` | string | `"base"` | `"tiny"`, `"base"`, `"small"`, `"medium"` | Whisper model size for speech recognition. Larger models are more accurate but slower and use more memory. `"tiny"` ~39 MB, `"base"` ~74 MB, `"small"` ~244 MB, `"medium"` ~769 MB. |

---

## `[normalization]`

Controls how text is preprocessed before being sent to the TTS engine. The normalizer transforms markdown, code, technical notation, and abbreviations into speech-friendly prose.

| Field | Type | Default | Valid Values | Description |
|-------|------|---------|--------------|-------------|
| `skip_code` | bool | `true` | `true`, `false` | Skip code blocks (fenced with triple backticks) rather than reading them aloud. When `true`, code blocks are replaced with a brief mention like "code block". |
| `expand_units` | bool | `true` | `true`, `false` | Expand technical units into spoken words. For example, `100MB` becomes "100 megabytes", `50ms` becomes "50 milliseconds". |
| `expand_abbreviations` | bool | `true` | `true`, `false` | Expand common technical abbreviations. For example, `API` becomes "A P I", `ONNX` becomes "onyx", `README` becomes "read me". |
| `shorten_paths` | bool | `true` | `true`, `false` | Shorten long file paths to just the filename. For example, `/Users/victor/projects/claude-speak/claude_speak/config.py` becomes just `config.py`. |
| `custom_pronunciations` | string | `""` (empty) | File path to a `.toml` file | Path to a custom pronunciations TOML file. When empty, uses the built-in pronunciation dictionary. The custom file can override or extend the default pronunciations. |
| `context_aware` | bool | `true` | `true`, `false` | Insert SSML-like tags based on detected content type. Enables speed adjustments for different content (e.g., slower for technical terms, pauses around headings). |

---

## `[audio]`

Controls system audio behavior: chimes, greetings, and Bluetooth workarounds.

| Field | Type | Default | Valid Values | Description |
|-------|------|---------|--------------|-------------|
| `chimes` | bool | `true` | `true`, `false` | Play audio chimes on state changes (e.g., daemon start, wake word detected, voice input start/stop). |
| `greeting` | string | `"Ready."` | Any text string | Text spoken aloud when the daemon starts. Set to `""` to disable the startup greeting. |
| `volume` | float | `0.3` | `0.0` -- `1.0` | Volume for chime sounds. `1.0` is full volume. This is separate from TTS speech volume (`[tts].volume`). |
| `bt_mic_workaround` | bool | `true` | `true`, `false` | When the output device is Bluetooth (e.g., AirPods), use the built-in microphone instead of the BT mic for recording. This avoids the macOS issue where BT audio quality degrades to phone-call quality (HFP profile) when the BT mic is active. |
| `media_keys_enabled` | bool | `true` | `true`, `false` | Intercept hardware media keys (play/pause, next, previous) for TTS playback control. When enabled, media keys control claude-speak instead of Music/Spotify while TTS is playing. |

---

## `[hotkeys]`

Global keyboard shortcuts that work system-wide on macOS. Requires Accessibility permissions (System Settings > Privacy & Security > Accessibility).

| Field | Type | Default | Valid Values | Description |
|-------|------|---------|--------------|-------------|
| `enabled` | bool | `true` | `true`, `false` | Enable or disable all global hotkeys. Set to `false` to rely solely on CLI commands and voice control. |
| `toggle_tts` | string | `"cmd+shift+s"` | Shortcut string | Toggle TTS on/off. When toggled off, new text is not spoken. |
| `stop_playback` | string | `"cmd+shift+x"` | Shortcut string | Immediately stop current TTS playback. |
| `voice_input` | string | `"cmd+shift+v"` | Shortcut string | Trigger voice input (start recording for speech-to-text). |

### Shortcut String Format

Shortcuts are written as modifier keys joined with `+`, followed by the main key:

- **Modifiers**: `cmd` / `command`, `shift`, `ctrl` / `control`, `alt` / `opt` / `option`
- **Keys**: `a`-`z`, `0`-`9`, `f1`-`f12`, `space`, `return`, `tab`, `escape`, `delete`, arrow keys (`up`, `down`, `left`, `right`), and punctuation (`-`, `=`, `[`, `]`, etc.)
- At least one modifier is required for global shortcuts.
- Examples: `cmd+shift+s`, `ctrl+alt+f1`, `cmd+option+space`

**Avoid conflicts** with macOS system shortcuts: `cmd+shift+3` (screenshot), `cmd+shift+4` (area screenshot), `cmd+shift+5` (screenshot bar), `cmd+space` (Spotlight), `cmd+tab` (app switcher).

---

## `[voice_commands]`

Configurable words for spoken commands. These are recognized during wake word listening and control TTS behavior. Set any command to `""` (empty string) to disable it.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pause` | string | `"pause"` | Pause TTS playback |
| `resume` | string | `"resume"` | Resume paused TTS playback |
| `repeat` | string | `"repeat"` | Repeat the last spoken text |
| `louder` | string | `"louder"` | Increase TTS volume |
| `quieter` | string | `"quieter"` | Decrease TTS volume |
| `faster` | string | `"faster"` | Increase TTS speed |
| `slower` | string | `"slower"` | Decrease TTS speed |
| `stop` | string | `"stop"` | Stop TTS playback entirely |

---

## Environment Variables

Environment variables override config file values and are the highest-priority configuration source.

### TTS Overrides

| Variable | Overrides | Description |
|----------|-----------|-------------|
| `CLAUDE_SPEAK_VOICE` | `tts.voice` | Override the TTS voice. Useful for quick testing without editing the config file. |
| `CLAUDE_SPEAK_SPEED` | `tts.speed` | Override the TTS speed (parsed as float). |
| `CLAUDE_SPEAK_DEVICE` | `tts.device` | Override the audio output device. |

### API Keys

| Variable | Overrides | Description |
|----------|-----------|-------------|
| `ELEVENLABS_API_KEY` | `tts.elevenlabs_api_key` | ElevenLabs API key. Takes priority over both the config field and the `~/.claude-speak/config.toml` file. |

### Debug and Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_SPEAK_DEBUG` | unset (disabled) | Set to `1` to enable debug logging to stderr. Logs hook invocations, text extraction, queue writes, and daemon signals. |
| `CLAUDE_SPEAK_PERF` | unset (disabled) | Set to `1` to enable performance logging to `/tmp/claude-speak-perf.log`. Records timestamps for each stage of the hook-to-audio pipeline. |

### Usage Example

```bash
# Quick voice test without editing config
CLAUDE_SPEAK_VOICE=am_adam CLAUDE_SPEAK_SPEED=1.3 claude-speak start

# Debug hook issues
CLAUDE_SPEAK_DEBUG=1 claude-speak start

# Use ElevenLabs without putting the key in a file
export ELEVENLABS_API_KEY="sk-your-key-here"
```

---

## Example Configurations

### JARVIS Voice Setup

Male British voice with blended warmth, slightly faster than normal:

```toml
[tts]
voice = "bm_george:60+bm_fable:40"   # 60% deep authority + 40% warm narrative
speed = 1.1
lang = "en-gb"                         # British English pronunciation
max_chunk_chars = 150                  # small chunks for low latency
volume = 0.4

[wakeword]
enabled = true
model = "hey_jarvis"
stop_model = "models/stop.onnx"
sensitivity = 0.5

[audio]
chimes = true
greeting = "Ready."
```

### Minimal Config (Just Change Voice)

The simplest possible config -- only override what you need:

```toml
[tts]
voice = "af_nicole"
```

Everything else uses defaults: Kokoro engine, 1.0 speed, American English, auto device, full volume.

### AirPods Device Targeting

Route TTS audio to AirPods while using the built-in mic for wake word detection:

```toml
[tts]
device = "AirPods"       # fuzzy-matches your AirPods device name
volume = 0.6             # AirPods can be loud at full volume

[audio]
bt_mic_workaround = true  # use built-in mic (avoids BT HFP quality drop)
volume = 0.2              # quieter chimes on AirPods

[wakeword]
enabled = true
sensitivity = 0.6         # slightly higher threshold to reduce false triggers
```

### Voice Cloning with ElevenLabs

Use a cloned voice from your ElevenLabs account:

```toml
[tts]
engine = "elevenlabs"
voice = "your-cloned-voice-id"    # voice ID from ElevenLabs dashboard
speed = 1.0
max_chunk_chars = 300             # ElevenLabs handles longer chunks well
```

Set the API key via environment variable (recommended -- avoids putting secrets in files):

```bash
export ELEVENLABS_API_KEY="sk-your-key-here"
```

Or in the config file (less secure):

```toml
[tts]
engine = "elevenlabs"
voice = "your-cloned-voice-id"
elevenlabs_api_key = "sk-your-key-here"
```

Or in the user-level config at `~/.claude-speak/config.toml`:

```toml
[elevenlabs]
api_key = "sk-your-key-here"
```

API key resolution priority: environment variable > `claude-speak.toml` field > `~/.claude-speak/config.toml`.

### Piper TTS Setup

Use the Piper local TTS engine (lighter weight than Kokoro, runs on any platform):

```toml
[tts]
engine = "piper"
voice = "en_US-lessac-medium"     # auto-downloads on first use
speed = 1.0
```

Available built-in Piper voices:

| Voice Name | Description |
|------------|-------------|
| `en_US-lessac-medium` | US English, Lessac (female, general purpose) |
| `en_US-ryan-medium` | US English, Ryan (male) |
| `en_GB-alan-medium` | British English, Alan (male) |

Custom Piper voices: place your `.onnx` and `.onnx.json` files in `~/.claude-speak/models/piper/` and set `voice` to the filename stem (without `.onnx`).

### Full Featured Config

All sections with non-default values for a comprehensive setup:

```toml
[tts]
engine = "kokoro"
voice = "bm_george:60+bm_fable:40"
speed = 1.1
lang = "en-gb"
device = "auto"
max_chunk_chars = 150
volume = 0.4

[wakeword]
enabled = true
engine = "openwakeword"
model = "hey_jarvis"
stop_model = "models/stop.onnx"
sensitivity = 0.5
stop_sensitivity = 0.5
stop_phrases = ["stop", "quiet", "shut up"]

[input]
backend = "auto"
auto_submit = true
stt_backend = "auto"
stt_model = "base"
vad_threshold = 0.5

[normalization]
skip_code = true
expand_units = true
expand_abbreviations = true
shorten_paths = true
context_aware = true

[audio]
chimes = true
greeting = "Ready."
volume = 0.3
bt_mic_workaround = true
media_keys_enabled = true

[hotkeys]
enabled = true
toggle_tts = "cmd+shift+s"
stop_playback = "cmd+shift+x"
voice_input = "cmd+shift+v"

[voice_commands]
pause = "pause"
resume = "resume"
repeat = "repeat"
louder = "louder"
quieter = "quieter"
faster = "faster"
slower = "slower"
stop = "stop"
```

---

## Troubleshooting

### Common Config Mistakes

**Wrong voice name for the engine**

Each TTS engine has its own voice naming scheme. Using a Kokoro voice name with Piper (or vice versa) will fail:

```toml
# WRONG: "af_sarah" is a Kokoro voice, not a Piper voice
[tts]
engine = "piper"
voice = "af_sarah"

# CORRECT: use a Piper voice name
[tts]
engine = "piper"
voice = "en_US-lessac-medium"
```

**Voice blend syntax errors**

The blend format is `name:weight+name:weight`. Weights are integers (percentages), not decimals:

```toml
# WRONG: weights as decimals
voice = "bm_george:0.6+bm_fable:0.4"

# CORRECT: weights as percentages
voice = "bm_george:60+bm_fable:40"
```

**ElevenLabs key not found**

If you get "ElevenLabs API key not found", check these in order:
1. Is `ELEVENLABS_API_KEY` set in your shell environment?
2. Is `elevenlabs_api_key` set in `[tts]` in `claude-speak.toml`?
3. Is `[elevenlabs] api_key` set in `~/.claude-speak/config.toml`?

```bash
# Verify the env var is set
echo $ELEVENLABS_API_KEY
```

**Volume set but no audible change**

There are two separate volume controls:
- `[tts].volume` -- controls speech volume
- `[audio].volume` -- controls chime volume

Make sure you are adjusting the right one. Both default to different values (`1.0` for speech, `0.3` for chimes).

**Device name not matching**

The `device` field uses substring matching. If your device is not found:
1. Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list all audio devices
2. Use a unique substring from the device name (e.g., `"AirPods"` not `"AirPods Pro"` if the full name includes a model number)
3. Set to `"auto"` to use the system default

**Hotkeys not working**

Global hotkeys require macOS Accessibility permissions:
1. Go to System Settings > Privacy & Security > Accessibility
2. Add your terminal app (Terminal, iTerm2, etc.) or the Python interpreter
3. Restart claude-speak after granting permissions

Also check for conflicts with existing system shortcuts or other apps.

**Wake word not triggering**

- Ensure `[wakeword].enabled = true`
- The `openwakeword` package must be installed: `pip install openwakeword`
- Try lowering `sensitivity` (e.g., `0.3`) for easier triggering
- Check that your microphone is working and has permission in System Settings > Privacy & Security > Microphone

**TOML syntax errors**

If your config file has syntax errors, claude-speak silently falls back to all defaults. Common mistakes:
- Missing quotes around string values: `voice = af_sarah` should be `voice = "af_sarah"`
- Using YAML syntax: `enabled: true` should be `enabled = true`
- Wrong boolean format: `enabled = True` should be `enabled = true` (TOML uses lowercase)

To validate your TOML file:
```bash
python -c "import tomllib; tomllib.load(open('claude-speak.toml', 'rb')); print('Valid TOML')"
```

**Config changes not taking effect**

The config file is read when the daemon starts. After editing `claude-speak.toml`, restart the daemon:

```bash
claude-speak stop && claude-speak start
```

Environment variable overrides are also read at startup. Export them before starting the daemon.
