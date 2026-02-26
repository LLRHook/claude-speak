#!/bin/bash
# Start the claude-speak daemon when Claude Code launches.
# Uses mkdir-based locking (portable — flock is not available on macOS).
set -euo pipefail

TOGGLE_FILE="$HOME/.claude-speak-enabled"
DAEMON_PID="/tmp/claude-speak-daemon.pid"
LOCK_DIR="/tmp/claude-speak-start.lock"

[[ ! -f "$TOGGLE_FILE" ]] && exit 0

# Only start if not already running
if [[ -f "$DAEMON_PID" ]]; then
  pid=$(cat "$DAEMON_PID" 2>/dev/null || echo "")
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && exit 0
fi

# mkdir is atomic on POSIX — prevents concurrent spawns from multiple hooks.
# Remove stale locks older than 30 seconds.
if [[ -d "$LOCK_DIR" ]]; then
  _lock_age=$(( $(date +%s) - $(stat -f%m "$LOCK_DIR" 2>/dev/null || echo 0) ))
  [[ $_lock_age -gt 30 ]] && rmdir "$LOCK_DIR" 2>/dev/null || true
fi
mkdir "$LOCK_DIR" 2>/dev/null || exit 0
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

python3 -m claude_speak.daemon > /tmp/claude-speak-stderr.log 2>&1 &
disown
exit 0
