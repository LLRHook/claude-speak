# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added

- **TTS engine support** with three backends:
  - Kokoro ONNX as the default engine with voice blending (e.g., `bm_george:60+bm_fable:40`),
    chunked generation with look-ahead, and persistent output stream for gapless playback
  - Piper TTS as an alternative local engine
  - ElevenLabs as a cloud-based engine for premium voice quality
- **Voice input** with two modes:
  - Built-in speech-to-text via mlx-whisper with voice activity detection (VAD)
  - Superwhisper integration for external STT with clipboard-based handoff
- **Wake word detection** using openwakeword with support for custom-trained models
  and a training pipeline (`train_wakeword.py`) for recording, augmenting, and
  training personalized wake word models
- **Audio device management** with automatic device selection, Bluetooth audio
  workaround for macOS, and runtime device switching
- **Unix socket IPC** with a JSON-based control API for external tool integration,
  supporting commands for playback control, status queries, and configuration
- **SSML markup support** for fine-grained speech control including pauses, emphasis,
  prosody adjustment, and phoneme-level pronunciation
- **Pronunciation dictionary** with 60+ abbreviation mappings and a comprehensive
  20-stage text normalization pipeline that handles code blocks, tables, lists,
  URLs, currency, fractions, dates, math operators, file paths, and version strings
- **Context-aware speech** that adapts normalization and delivery based on content
  type (code explanation, conversational reply, error message, etc.)
- **Voice commands** for hands-free daemon control (mute, unmute, stop, skip,
  change voice) processed through the voice input pipeline
- **Media key integration** for play/pause and skip control on macOS
- **Global hotkeys** via Quartz event taps for system-wide keyboard shortcuts
- **Python hook system** for Claude Code integration with `PostToolUse` and `Stop`
  hooks that capture assistant output, strip markdown, and queue it for speech
- **Daemon lifecycle management** with PID file locking, signal handling
  (SIGTERM/SIGINT for shutdown, SIGUSR1 for queue wakeup), daemonization via fork,
  hot-reload of configuration on file change, and mute/toggle state via sentinel files
- **CLI interface** (`claude-speak`) with subcommands for starting, stopping,
  restarting the daemon, checking status, and managing configuration
- **Chime system** for audible feedback on state transitions (start, stop, mute,
  unmute, wake word detected)
- **CI/CD pipeline** with GitHub Actions for linting (ruff), type checking (mypy),
  testing (pytest), automated releases to GitHub Releases, and publishing to PyPI
  via trusted publishing
- **Pre-commit hooks** with ruff formatting and linting checks
- **Example configuration** (`claude-speak.toml.example`) documenting all available
  settings with sensible defaults

[unreleased]: https://github.com/vnicivanov/claude-speak/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vnicivanov/claude-speak/releases/tag/v0.1.0
