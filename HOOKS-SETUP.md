# Claude-Speak Hooks Setup

Claude-speak integrates with Claude Code through **hooks** -- event-driven shell
scripts that Claude Code invokes at specific points during a session. Four hooks
across four events make up the full integration.

---

## Hook Overview

### 1. `SessionStart` -- daemon-start.sh

| Field   | Value |
|---------|-------|
| Script  | `hooks/daemon-start.sh` |
| Event   | Fires once when a new Claude Code session begins |
| Purpose | Starts the TTS daemon process (`cli.py start`) so it is ready to process speech before any responses arrive |
| Async   | Yes -- runs in the background so it never blocks Claude Code startup |

The script is guarded by `~/.claude-speak-enabled`; if the toggle file does not
exist the hook exits immediately and the daemon is not started. It also uses
`flock` to prevent duplicate spawns when multiple hooks fire close together.

### 2. `PostToolUse` -- speak-response.sh

| Field   | Value |
|---------|-------|
| Script  | `hooks/speak-response.sh` |
| Event   | Fires after **every** tool invocation (Bash, Read, Edit, etc.) |
| Purpose | Reads the session transcript, extracts any new assistant text since the last invocation, strips markdown formatting, and writes the cleaned text to the daemon's queue directory (`/tmp/claude-speak-queue/`) |
| Timeout | 30 seconds |
| Async   | Yes |

This is the primary hook that feeds text to the TTS pipeline. It tracks its
read position in `/tmp/claude-speak-pos` so it never re-speaks old content.
After writing a queue file it sends `SIGUSR1` to the daemon to trigger
immediate processing (no polling delay).

### 3. `Stop` -- speak-response.sh (again)

| Field   | Value |
|---------|-------|
| Script  | `hooks/speak-response.sh` |
| Event   | Fires when Claude Code finishes its response (all tool calls complete) |
| Purpose | Catches any final assistant text that was emitted after the last tool call but before the turn ended. Without this hook the last paragraph of a response could be silently dropped. |
| Timeout | 30 seconds |
| Async   | Yes |

The script detects the `Stop` event via the `hook_event_name` field in the JSON
input and adds a short `sleep 0.3` to let the transcript file flush before
reading.

### 4. `SessionEnd` -- daemon-stop.sh

| Field   | Value |
|---------|-------|
| Script  | `hooks/daemon-stop.sh` |
| Event   | Fires when the Claude Code session is closed |
| Purpose | Gracefully shuts down the TTS daemon (`cli.py stop`) so no orphan processes remain |
| Async   | Yes |

---

## The Template File: `claude-hooks.json`

`claude-hooks.json` contains the exact JSON structure that must appear inside
the `"hooks"` key of `~/.claude/settings.json`. It uses the placeholder
`REPO_DIR` which the installer (`install.sh`) replaces with the actual
absolute path to the claude-speak repository.

---

## Manual Installation

If `install.sh` does not work (or you prefer to set things up by hand), follow
these steps:

### Step 1 -- Enable the toggle

```bash
touch ~/.claude-speak-enabled
```

### Step 2 -- Make hook scripts executable

```bash
chmod +x /path/to/claude-speak/hooks/*.sh
```

### Step 3 -- Edit `~/.claude/settings.json`

Open the file and merge the hook entries below into the existing `"hooks"`
object. If you already have entries for a given event (e.g. you already have a
`Stop` hook), add the new hook object to the existing array -- do **not**
replace the array.

Replace `REPO_DIR` with the absolute path to your claude-speak checkout
(e.g. `/Users/you/projects/claude-speak`).

```jsonc
{
  "hooks": {
    "PostToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "REPO_DIR/hooks/speak-response.sh",
            "timeout": 30,
            "async": true
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "REPO_DIR/hooks/speak-response.sh",
            "timeout": 30,
            "async": true
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "REPO_DIR/hooks/daemon-start.sh",
            "async": true
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "REPO_DIR/hooks/daemon-stop.sh",
            "async": true
          }
        ]
      }
    ]
  }
}
```

### Step 4 -- Verify

Restart Claude Code. You should see the daemon start (check with
`ps aux | grep claude-speak`) and hear responses spoken aloud.

---

## Merging with Existing Hooks

If your `settings.json` already has hooks for the same events, you need to
**append** rather than replace. For example, if your `Stop` event already
contains a notification hook:

```jsonc
"Stop": [
  {
    "hooks": [
      {
        "type": "command",
        "command": "~/.claude/notify.sh 'Task completed'",
        "async": true
      },
      {
        "type": "command",
        "command": "REPO_DIR/hooks/speak-response.sh",
        "timeout": 30,
        "async": true
      }
    ]
  }
]
```

Multiple hooks in the same array entry run for every invocation of that event.

---

## Disabling Specific Hooks

### Disable all of claude-speak (recommended)

Remove the toggle file. Every hook script checks for it first and exits
immediately if it is absent:

```bash
rm ~/.claude-speak-enabled
```

To re-enable:

```bash
touch ~/.claude-speak-enabled
```

### Disable individual hooks

Remove or comment out the specific hook entry from `~/.claude/settings.json`.

| To disable...                 | Remove the entry from... |
|-------------------------------|--------------------------|
| Speech during tool use        | `PostToolUse` array -- remove the `speak-response.sh` entry |
| Speech at end of response     | `Stop` array -- remove the `speak-response.sh` entry |
| Auto-start of daemon          | `SessionStart` array -- remove the `daemon-start.sh` entry |
| Auto-stop of daemon           | `SessionEnd` array -- remove the `daemon-stop.sh` entry |

If you remove the `SessionStart` hook, you will need to start the daemon
manually before speech will work:

```bash
python3 /path/to/claude-speak/cli.py start
```

### Disable speech but keep the daemon running

Remove both `speak-response.sh` entries (from `PostToolUse` and `Stop`) but
leave `SessionStart` and `SessionEnd` in place. The daemon will run but receive
no text.

---

## Troubleshooting

- **No speech output**: Check that `~/.claude-speak-enabled` exists and that the
  daemon is running (`ps aux | grep claude-speak`).
- **Duplicate speech**: The position file `/tmp/claude-speak-pos` may be stale.
  Remove it: `rm /tmp/claude-speak-pos`.
- **Hook errors in Claude Code**: Check `/tmp/claude-speak-perf.log` for timing
  data and errors.
- **Daemon won't start**: Check that `cli.py` is functional by running
  `python3 /path/to/claude-speak/cli.py start` directly.
