# Accessibility & Permissions Guide

claude-speak uses several macOS system APIs that require explicit user consent.
This document explains **why** each permission is needed, **how** to grant it,
and what to do when things are not working.

## Why permissions are needed

### Accessibility

claude-speak registers **global keyboard shortcuts** (e.g. Cmd+Shift+S to
toggle TTS, Cmd+Shift+V for voice input) using the macOS `CGEventTap` API from
the Quartz framework. It also uses `osascript` to send synthetic key events
(pasting transcribed text via Cmd+V, pressing Enter to auto-submit).

Both of these mechanisms require the calling process to have **Accessibility**
permission. Without it:

- `CGEventTapCreate` returns `nil` and global hotkeys silently fail to register.
- `osascript` commands that send keystrokes to System Events are rejected.

### Microphone

The built-in voice input pipeline records audio from the default input device
using `sounddevice` (PortAudio). macOS requires **Microphone** permission for
any process that opens an audio input stream. Without it, the stream open call
raises a permission error and voice input / wake word detection cannot function.

## Required permissions

| Permission    | Used by                                         | Purpose                                  |
|---------------|------------------------------------------------|------------------------------------------|
| Accessibility | `hotkeys.py`, `media_keys.py`, `voice_input.py` | Global hotkeys (CGEventTap), osascript key simulation |
| Microphone    | `voice_input.py`, `voice_controller.py`, `wakeword.py` | Audio recording for STT and wake word detection |

## How to grant permissions

The steps are the same on macOS 13 (Ventura), macOS 14 (Sonoma), and macOS 15
(Sequoia).

### Accessibility

1. Open **System Settings**.
2. Navigate to **Privacy & Security** (in the sidebar).
3. Click **Accessibility**.
4. Click the **+** button (you may need to unlock with your password or Touch ID).
5. Add your **terminal emulator** — this is the app that _runs_ claude-speak:
   - **Terminal.app** — `/Applications/Utilities/Terminal.app`
   - **iTerm2** — `/Applications/iTerm.app`
   - **WezTerm** — `/Applications/WezTerm.app`
   - **Alacritty** — `/Applications/Alacritty.app`
   - **Kitty** — `/Applications/kitty.app`
   - **VS Code integrated terminal** — add Visual Studio Code itself
   - **Cursor** — add the Cursor app itself
6. Make sure the toggle next to the app is **on**.

### Microphone

1. Open **System Settings**.
2. Navigate to **Privacy & Security** (in the sidebar).
3. Click **Microphone**.
4. If your terminal app is already listed, make sure its toggle is **on**.
5. If it is not listed, the first time claude-speak tries to access the
   microphone macOS will show a permission dialog — click **Allow**.
   - If you dismissed the dialog or chose "Don't Allow", you can add the app
     manually by toggling it off and on, or by using the `tccutil` command
     described below.

## Verifying permissions

Run the built-in diagnostic command:

```
claude-speak check-permissions
```

Example output when everything is configured correctly:

```
claude-speak permission check
────────────────────────────────────
Audio output ..... PASS (default: MacBook Pro Speakers)
Audio input ...... PASS (default: MacBook Pro Microphone)
Accessibility .... PASS (System Events access OK)
```

Example output when Accessibility is missing:

```
claude-speak permission check
────────────────────────────────────
Audio output ..... PASS (default: MacBook Pro Speakers)
Audio input ...... PASS (default: MacBook Pro Microphone)
Accessibility .... FAIL
  osascript returned non-zero
  -> System Settings > Privacy & Security > Accessibility
  -> Add your terminal app (Terminal.app, iTerm2, etc.)
```

## Troubleshooting

### Permission granted but still not working

macOS sometimes caches the permission state. Try:

1. Open **System Settings > Privacy & Security > Accessibility**.
2. **Remove** your terminal app from the list (select it and click the minus button).
3. **Re-add** it using the plus button.
4. **Quit and relaunch** your terminal app.

### Resetting the permissions database

If the UI approach does not help, you can reset the TCC (Transparency, Consent,
and Control) database for specific services:

```bash
# Reset Accessibility permissions for all apps
tccutil reset Accessibility

# Reset Microphone permissions for all apps
tccutil reset Microphone
```

After resetting, re-launch your terminal and run `claude-speak check-permissions`
again. macOS will re-prompt you for consent on next use.

> **Note:** `tccutil reset` affects _all_ apps for that permission category, not
> just your terminal. Other apps will need to be re-authorized as well.

### Global hotkeys not working despite Accessibility being granted

- Make sure another application has not claimed the same shortcut. macOS delivers
  global key events to the _first_ tap that matches.
- Check that `pyobjc-framework-Quartz` is installed: `pip show pyobjc-framework-Quartz`.
- Run `claude-speak status` to confirm the daemon is running and hotkeys are
  enabled in your config.
- Try remapping the shortcuts in `claude-speak.toml` under `[hotkeys]` to avoid
  conflicts.

### Voice input not transcribing

- Confirm the microphone check passes: `claude-speak check-permissions`.
- Check that a default input device is set in **System Settings > Sound > Input**.
- Verify the STT model is downloaded: `claude-speak stt-models`.

## What works without Accessibility

The following features do **not** require Accessibility permission and will
function normally without it:

- TTS output (speaking text aloud)
- IPC control (`claude-speak speak`, `pause`, `resume`, `volume`, etc.)
- Voice commands (audio-only pipeline — wake word + STT, without hotkey trigger)
- All CLI commands (`start`, `stop`, `status`, `config`, `voices`, etc.)
- The daemon itself

Only **global hotkeys** (Cmd+Shift+S/X/V) and **voice-input paste** (simulating
Cmd+V / Enter via osascript) require Accessibility.
