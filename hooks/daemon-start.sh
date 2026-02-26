#!/bin/bash
# Start the claude-speak daemon when Claude Code launches.
# Uses flock to prevent concurrent spawns from multiple hooks.
set -euo pipefail

TOGGLE_FILE="$HOME/.claude-speak-enabled"
DAEMON_PID="/tmp/claude-speak-daemon.pid"
LOCK="/tmp/claude-speak-daemon.lock"
SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SOURCE" ]]; do SOURCE="$(readlink "$SOURCE")"; done
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"

[[ ! -f "$TOGGLE_FILE" ]] && exit 0

# Only start if not already running
if [[ -f "$DAEMON_PID" ]]; then
  pid=$(cat "$DAEMON_PID" 2>/dev/null || echo "")
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && exit 0
fi

# flock prevents race: if another hook is already starting the daemon, skip
exec 200>"$LOCK"
flock -n 200 || exit 0

PYTHONPATH="$PROJECT" python3 -m claude_speak.daemon &>/tmp/claude-speak-stderr.log &
disown
exit 0
