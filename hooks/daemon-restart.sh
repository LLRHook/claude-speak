#!/bin/bash
# UserPromptSubmit hook: detect "restart daemon" phrases and restart automatically.
# Blocks the prompt (exit 2) so Claude doesn't waste tokens processing it.
set -euo pipefail

# Read the prompt from stdin JSON
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

# Match "restart daemon", "daemon restart", "restart the daemon", etc.
# Case-insensitive. Only triggers on short, standalone requests.
if echo "$PROMPT" | grep -iqE '^\s*(restart\s+(the\s+)?d[ae]+mon|d[ae]+mon\s+restart)\s*[.!?]*\s*$'; then
  python3 -m claude_speak.cli stop 2>/dev/null || true
  sleep 1
  python3 -m claude_speak.daemon &>/tmp/claude-speak-stderr.log &
  disown
  sleep 1

  echo "Daemon restarted." >&2
  exit 2
fi

exit 0
