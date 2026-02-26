#!/bin/bash
# UserPromptSubmit hook: detect "restart daemon" phrases and restart automatically.
# Blocks the prompt (exit 2) so Claude doesn't waste tokens processing it.
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SOURCE" ]]; do SOURCE="$(readlink "$SOURCE")"; done
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"

# Read the prompt from stdin JSON
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

# Match "restart daemon", "daemon restart", "restart the daemon", etc.
# Case-insensitive. Only triggers on short, standalone requests.
if echo "$PROMPT" | grep -iqE '^\s*(restart\s+(the\s+)?d[ae]+mon|d[ae]+mon\s+restart)\s*[.!?]*\s*$'; then
  # Run restart: stop first, then launch daemon directly (not via cli's daemonize fork)
  PYTHONPATH="$PROJECT" python3 -m claude_speak.cli stop 2>/dev/null || true
  sleep 1
  PYTHONPATH="$PROJECT" python3 -m claude_speak.daemon &>/tmp/claude-speak-stderr.log &
  disown
  sleep 1

  # Block the prompt and provide feedback via stderr
  echo "Daemon restarted." >&2
  exit 2
fi

# Not a daemon restart request — let the prompt through
exit 0
