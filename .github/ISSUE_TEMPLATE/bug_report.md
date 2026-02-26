---
name: Bug Report
about: Report a bug or unexpected behavior in claude-speak
title: "[Bug] "
labels: bug
assignees: ""
---

## Environment

- **OS**: <!-- e.g. macOS 15.2 (Sequoia), macOS 14.5 (Sonoma) -->
- **Python version**: <!-- output of `python --version` -->
- **claude-speak version**: <!-- output of `pip show claude-speak | grep Version` or commit hash -->
- **Installation method**: <!-- pip install, pip install -e ., install.sh, etc. -->

## Audio Devices

<details>
<summary>Output of <code>python -m sounddevice</code></summary>

```
PASTE OUTPUT HERE
```

</details>

- **Output device in use**: <!-- e.g. "MacBook Pro Speakers", "AirPods Pro", "auto" -->
- **Input device in use**: <!-- e.g. "MacBook Pro Microphone", "auto" -->

## Configuration

<details>
<summary>Contents of <code>claude-speak.toml</code> (remove any sensitive values)</summary>

```toml
PASTE CONFIG HERE
```

</details>

## Daemon Log

<details>
<summary>Relevant excerpt from <code>daemon.log</code></summary>

<!-- Tip: run with log_level = "DEBUG" in claude-speak.toml to get more detail -->

```
PASTE LOG EXCERPT HERE
```

</details>

## Description

<!-- A clear and concise description of the bug. -->

## Steps to Reproduce

1. <!-- Step 1 -->
2. <!-- Step 2 -->
3. <!-- Step 3 -->

## Expected Behavior

<!-- What you expected to happen. -->

## Actual Behavior

<!-- What actually happened. Include any error messages or tracebacks. -->

## Additional Context

<!-- Any other relevant details: screenshots, screen recordings, related issues, etc. -->
