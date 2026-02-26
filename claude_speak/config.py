"""
Configuration loader for claude-speak.
Reads claude-speak.toml and provides defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "claude-speak.toml"
MODELS_DIR = Path.home() / ".claude-speak" / "models"
WAKEWORD_DIR = PROJECT_DIR / "wakeword"
QUEUE_DIR = Path("/tmp/claude-speak-queue")
PID_FILE = Path("/tmp/claude-speak-daemon.pid")
TOGGLE_FILE = Path.home() / ".claude-speak-enabled"
MUTE_FILE = Path("/tmp/claude-speak-muted")
PLAYING_FILE = Path("/tmp/claude-speak-playing")
LOG_FILE = PROJECT_DIR / "daemon.log"


@dataclass
class TTSConfig:
    engine: str = "kokoro"  # "kokoro", "piper", "elevenlabs"
    voice: str = "af_sarah"  # single voice name or blend like "bm_george:60+bm_fable:40"
    speed: float = 1.0
    lang: str = "en-us"  # language code for Kokoro (en-us, en-gb, etc.)
    device: str = "auto"
    model_path: str = str(MODELS_DIR / "kokoro-v1.0.onnx")
    voices_path: str = str(MODELS_DIR / "voices-v1.0.bin")
    max_chunk_chars: int = 400  # split text into chunks of this size
    volume: float = 1.0  # TTS speech volume (0.0-1.0)
    elevenlabs_api_key: str = ""  # ElevenLabs API key (env var ELEVENLABS_API_KEY takes priority)


@dataclass
class WakeWordConfig:
    enabled: bool = False
    engine: str = "openwakeword"
    model: str = "hey_jarvis"  # pre-trained wake word model
    stop_model: str = ""  # path to stop.onnx for instant stop detection
    sensitivity: float = 0.5
    stop_sensitivity: float = 0.5  # separate threshold for stop model
    stop_phrases: list[str] = field(default_factory=lambda: ["stop", "quiet", "shut up"])


@dataclass
class InputConfig:
    backend: str = "builtin"  # "builtin" (default), "superwhisper", or "auto"
    auto_submit: bool = True
    superwhisper_shortcut_keycode: int = 49  # Space
    superwhisper_shortcut_modifiers: int = 2048  # Option
    vad_threshold: float = 0.5  # Silero VAD speech probability threshold
    stt_backend: str = "auto"  # "auto", "mlx", "whisper_cpp"
    stt_model: str = "base"  # "tiny", "base", "small", "medium"


@dataclass
class NormalizationConfig:
    skip_code: bool = True
    expand_units: bool = True
    expand_abbreviations: bool = True
    shorten_paths: bool = True
    custom_pronunciations: str = ""  # path to custom pronunciations.toml (empty = use default)
    context_aware: bool = True  # insert SSML tags based on detected content type


@dataclass
class AudioConfig:
    chimes: bool = True
    greeting: str = "Ready."
    volume: float = 0.3
    bt_mic_workaround: bool = True  # use built-in mic when output is BT
    media_keys_enabled: bool = True  # intercept hardware media keys for TTS control


@dataclass
class HotkeysConfig:
    """Global keyboard shortcut configuration."""
    enabled: bool = True
    toggle_tts: str = "cmd+shift+s"
    stop_playback: str = "cmd+shift+x"
    voice_input: str = "cmd+shift+v"


@dataclass
class VoiceCommandsConfig:
    """Configurable voice command words.

    Each field maps a command action to the word/phrase used to trigger it.
    Set a field to an empty string to disable that command.
    """
    pause: str = "pause"
    resume: str = "resume"
    repeat: str = "repeat"
    louder: str = "louder"
    quieter: str = "quieter"
    faster: str = "faster"
    slower: str = "slower"
    stop: str = "stop"


@dataclass
class Config:
    tts: TTSConfig = field(default_factory=TTSConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
    input: InputConfig = field(default_factory=InputConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    hotkeys: HotkeysConfig = field(default_factory=HotkeysConfig)
    voice_commands: VoiceCommandsConfig = field(default_factory=VoiceCommandsConfig)


def load_config() -> Config:
    """Load config from TOML file, falling back to defaults."""
    config = Config()

    if not CONFIG_PATH.exists():
        return config

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return config

    try:
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return config

    # Apply TOML values over defaults
    if "tts" in data:
        for k, v in data["tts"].items():
            if hasattr(config.tts, k):
                setattr(config.tts, k, v)
    if "wakeword" in data:
        for k, v in data["wakeword"].items():
            if hasattr(config.wakeword, k):
                setattr(config.wakeword, k, v)
    if "input" in data:
        for k, v in data["input"].items():
            if hasattr(config.input, k):
                setattr(config.input, k, v)
    if "normalization" in data:
        for k, v in data["normalization"].items():
            if hasattr(config.normalization, k):
                setattr(config.normalization, k, v)
    if "audio" in data:
        for k, v in data["audio"].items():
            if hasattr(config.audio, k):
                setattr(config.audio, k, v)
    if "hotkeys" in data:
        for k, v in data["hotkeys"].items():
            if hasattr(config.hotkeys, k):
                setattr(config.hotkeys, k, v)
    if "voice_commands" in data:
        for k, v in data["voice_commands"].items():
            if hasattr(config.voice_commands, k):
                setattr(config.voice_commands, k, v)

    # Override with env vars (highest priority)
    if os.environ.get("CLAUDE_SPEAK_VOICE"):
        config.tts.voice = os.environ["CLAUDE_SPEAK_VOICE"]
    if os.environ.get("CLAUDE_SPEAK_SPEED"):
        config.tts.speed = float(os.environ["CLAUDE_SPEAK_SPEED"])
    if os.environ.get("CLAUDE_SPEAK_DEVICE"):
        config.tts.device = os.environ["CLAUDE_SPEAK_DEVICE"]

    # Resolve model paths: fall back to project-local models/ if user-level missing
    _LOCAL_MODELS = PROJECT_DIR / "models"
    for attr in ("model_path", "voices_path"):
        path = Path(getattr(config.tts, attr))
        if not path.exists() and _LOCAL_MODELS.exists():
            local = _LOCAL_MODELS / path.name
            if local.exists():
                setattr(config.tts, attr, str(local))

    return config
