#!/bin/bash
# speak-response.sh — Claude Code hook (PostToolUse + Stop).
# Thin: parse transcript, strip markdown, write to queue.
# Does NOT manage the daemon — SessionStart hook handles that.

set -euo pipefail

TOGGLE_FILE="$HOME/.claude-speak-enabled"
QUEUE_DIR="/tmp/claude-speak-queue"
POS_FILE="/tmp/claude-speak-pos"
PERF_LOG="/tmp/claude-speak-perf.log"
PERF_ENABLED="${CLAUDE_SPEAK_PERF:-}"
HOOK_LOCK="/tmp/claude-speak-hook.lock"

# --- Gate ---
[[ ! -f "$TOGGLE_FILE" ]] && exit 0

# --- Serialize: prevent parallel hook instances from reading the same position ---
# mkdir is atomic on all POSIX systems — perfect for macOS which lacks flock
# Remove stale locks older than 10 seconds (crashed hook)
if [[ -d "$HOOK_LOCK" ]]; then
  _lock_age=$(( $(date +%s) - $(stat -f%m "$HOOK_LOCK" 2>/dev/null || echo 0) ))
  [[ $_lock_age -gt 10 ]] && rmdir "$HOOK_LOCK" 2>/dev/null
fi
_lock_acquired=0
for _i in $(seq 1 30); do  # wait up to 3 seconds (30 x 0.1s)
  if mkdir "$HOOK_LOCK" 2>/dev/null; then
    _lock_acquired=1
    trap 'rmdir "$HOOK_LOCK" 2>/dev/null' EXIT
    break
  fi
  sleep 0.1
done
[[ $_lock_acquired -eq 0 ]] && exit 0  # give up if still locked

[[ -n "$PERF_ENABLED" ]] && T_START=$(perl -MTime::HiRes=time -e 'printf "%.3f\n", time()')

# --- Read hook input (single jq call instead of three) ---
INPUT=$(cat)
eval "$(echo "$INPUT" | jq -r '
  "TRANSCRIPT_PATH=\(.transcript_path // "")",
  "SESSION_ID=\(.session_id // "")",
  "HOOK_EVENT=\(.hook_event_name // "")"
')"

[[ -z "$TRANSCRIPT_PATH" || ! -f "$TRANSCRIPT_PATH" ]] && exit 0

[[ -n "$PERF_ENABLED" ]] && T_PARSE=$(perl -MTime::HiRes=time -e 'printf "%.3f\n", time()')

# --- Position tracking ---
LAST_POS=0
if [[ -f "$POS_FILE" ]]; then
  STORED_SESSION=$(head -1 "$POS_FILE" 2>/dev/null || echo "")
  STORED_POS=$(tail -1 "$POS_FILE" 2>/dev/null || echo "0")
  [[ "$STORED_SESSION" == "$SESSION_ID" ]] && LAST_POS="$STORED_POS"
fi

# --- For Stop hook, wait for transcript flush ---
[[ "$HOOK_EVENT" == "Stop" ]] && sleep 0.3

# --- Count lines ---
TOTAL_LINES=$(wc -l < "$TRANSCRIPT_PATH" | tr -d ' ')
[[ "$TOTAL_LINES" -le "$LAST_POS" ]] && exit 0

# --- Extract new assistant text ---
NEW_LINES=$((TOTAL_LINES - LAST_POS))
TEXT=$(tail -n "$NEW_LINES" "$TRANSCRIPT_PATH" \
  | jq -r '
    select(.type == "assistant")
    | .message.content[]?
    | select(.type == "text")
    | .text
  ' 2>/dev/null)

[[ -n "$PERF_ENABLED" ]] && T_EXTRACT=$(perl -MTime::HiRes=time -e 'printf "%.3f\n", time()')

# --- Update position ---
printf '%s\n%s\n' "$SESSION_ID" "$TOTAL_LINES" > "$POS_FILE"

[[ -z "$TEXT" ]] && exit 0

# --- Strip markdown (light pass — normalizer handles code blocks, tables, lists) ---
CLEAN=$(echo "$TEXT" | sed -E '
  s/\*{1,3}([^*]*)\*{1,3}/\1/g
  s/_{1,3}([^_]*)_{1,3}/\1/g
  s/\[([^]]*)\]\([^)]*\)/\1/g
  s/!\[([^]]*)\]\([^)]*\)/\1/g
  s/^[-*_]{3,}$//
  s/<[^>]*>//g
')

CLEAN=$(echo "$CLEAN" | sed '/^[[:space:]]*$/d' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
[[ -z "$CLEAN" ]] && exit 0

[[ -n "$PERF_ENABLED" ]] && T_SED=$(perl -MTime::HiRes=time -e 'printf "%.3f\n", time()')

# --- Write to queue ---
mkdir -p "$QUEUE_DIR"
TIMESTAMP=$(perl -MTime::HiRes=time -e 'printf "%.6f\n", time()')
echo "$CLEAN" > "$QUEUE_DIR/${TIMESTAMP}.txt"

# Signal daemon to process immediately (no poll delay)
DAEMON_PID=$(cat /tmp/claude-speak-daemon.pid 2>/dev/null)
[[ -n "$DAEMON_PID" ]] && kill -USR1 "$DAEMON_PID" 2>/dev/null || true

if [[ -n "$PERF_ENABLED" ]]; then
  T_END=$(perl -MTime::HiRes=time -e 'printf "%.3f\n", time()')
  echo "[hook-perf] parse=$(echo "$T_PARSE - $T_START" | bc)s extract=$(echo "$T_EXTRACT - $T_PARSE" | bc)s sed=$(echo "$T_SED - $T_EXTRACT" | bc)s write=$(echo "$T_END - $T_SED" | bc)s TOTAL=$(echo "$T_END - $T_START" | bc)s lines=$NEW_LINES chars=${#CLEAN}" >> "$PERF_LOG"
fi

exit 0
