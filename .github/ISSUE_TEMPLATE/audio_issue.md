---
name: Audio Issue
about: Report audio-specific problems (no sound, wrong device, crackling, latency)
title: "[Audio] "
labels: bug, audio
assignees: ""
---

## Issue Type

<!-- Check all that apply -->

- [ ] No sound output at all
- [ ] Audio plays on the wrong device
- [ ] Crackling, popping, or distorted audio
- [ ] Audio cuts out or drops intermittently
- [ ] High latency / long delay before audio plays
- [ ] Bluetooth audio quality degrades (SCO/telephony mode)
- [ ] Wake word detection not hearing me
- [ ] Other (describe below)

## Environment

- **OS**: <!-- e.g. macOS 15.2 (Sequoia) -->
- **Python version**: <!-- output of `python --version` -->
- **claude-speak version**: <!-- output of `pip show claude-speak | grep Version` or commit hash -->

## Audio Device List

<details>
<summary>Output of <code>python -m sounddevice</code></summary>

```
PASTE FULL OUTPUT HERE
```

</details>

## Bluetooth Information

<!-- If using Bluetooth audio, fill this section. Otherwise write "N/A". -->

- **Bluetooth device**: <!-- e.g. AirPods Pro 2, Sony WH-1000XM5 -->
- **Connected profile**: <!-- if known: A2DP (stereo) or SCO/HFP (telephony) -->
- **bt_mic_workaround setting**: <!-- true or false from your config -->

<details>
<summary>Bluetooth system info (optional)</summary>

<!-- On macOS: Option-click the Bluetooth icon in the menu bar, or run:
     system_profiler SPBluetoothDataType -->

```
PASTE BLUETOOTH INFO HERE
```

</details>

## Audio Configuration

<details>
<summary>Audio-related sections from <code>claude-speak.toml</code></summary>

<!-- Paste the [tts], [audio], and [wakeword] sections from your config -->

```toml
PASTE CONFIG SECTIONS HERE
```

</details>

## Description

<!-- Describe the audio problem in detail. When does it occur? Is it consistent or intermittent? -->

## Steps to Reproduce

1. <!-- Step 1 -->
2. <!-- Step 2 -->
3. <!-- Step 3 -->

## Expected Audio Behavior

<!-- What should the audio sound like / which device should it play on? -->

## Actual Audio Behavior

<!-- What actually happens? Include descriptions of any artifacts, silence, wrong device, etc. -->

## Daemon Log

<details>
<summary>Relevant excerpt from <code>daemon.log</code></summary>

<!-- Run with log_level = "DEBUG" for the most useful audio diagnostics -->

```
PASTE LOG EXCERPT HERE
```

</details>

## Troubleshooting Already Attempted

<!-- Check any steps you have already tried -->

- [ ] Verified the correct device is set in `claude-speak.toml` (`[tts] device`)
- [ ] Tested with `device = "auto"` (system default)
- [ ] Tested with a different audio output device
- [ ] Checked macOS Sound preferences for the correct output device
- [ ] Restarted the claude-speak daemon
- [ ] Tested audio with another application (e.g. Music, YouTube)
- [ ] Toggled `bt_mic_workaround` setting (for Bluetooth issues)

## Additional Context

<!-- Any other relevant details: screenshots, audio recordings, related issues, etc. -->
