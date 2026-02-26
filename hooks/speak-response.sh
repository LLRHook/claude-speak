#!/bin/bash
# speak-response.sh — Claude Code hook (PostToolUse + Stop).
#
# DEPRECATED: This shell script is a compatibility wrapper.  The actual logic
# now lives in claude_speak/hooks/speak_response.py which provides better
# error handling, type safety, and testability.
#
# Behaviour:
#   1. Try to run the Python hook.
#   2. If the Python hook is not available or fails, fall back to the original
#      shell logic below.
#
# Error handling: every failure path exits 0 so errors never propagate to
# Claude Code. Set CLAUDE_SPEAK_DEBUG=1 to log errors to stderr.

# Intentionally NOT using set -euo pipefail — every failure is handled
# explicitly and must exit 0.

DEBUG="${CLAUDE_SPEAK_DEBUG:-}"

debug_log() {
  [[ -n "$DEBUG" ]] && echo "[claude-speak-hook] $*" >&2
  return 0
}

# --- Read stdin into a variable (must happen before any subprocess consumes it) ---
INPUT=$(cat)

# --- Try the Python hook first (pip-installed claude-speak package) ---
if command -v python3 &>/dev/null; then
  debug_log "Delegating to Python hook"
  echo "$INPUT" | python3 -m claude_speak.hooks.speak_response 2>/dev/null
  PY_EXIT=$?
  if [[ $PY_EXIT -eq 0 ]]; then
    exit 0
  fi
  debug_log "Python hook exited with $PY_EXIT, falling back to shell"
fi

# =========================================================================
# FALLBACK: Original shell logic (kept for backward compatibility)
# =========================================================================

TOGGLE_FILE="$HOME/.claude-speak-enabled"
QUEUE_DIR="/tmp/claude-speak-queue"
POS_FILE="/tmp/claude-speak-pos"
PERF_LOG="/tmp/claude-speak-perf.log"
PERF_ENABLED="${CLAUDE_SPEAK_PERF:-}"
HOOK_LOCK="/tmp/claude-speak-hook.lock"

# --- Gate ---
if [[ ! -f "$TOGGLE_FILE" ]]; then
  exit 0
fi

# --- Serialize: prevent parallel hook instances from reading the same position ---
# mkdir is atomic on all POSIX systems — perfect for macOS which lacks flock
# Remove stale locks older than 10 seconds (crashed hook)
if [[ -d "$HOOK_LOCK" ]]; then
  _lock_age=$(( $(date +%s) - $(stat -f%m "$HOOK_LOCK" 2>/dev/null || echo 0) ))
  [[ $_lock_age -gt 10 ]] && rmdir "$HOOK_LOCK" 2>/dev/null || true
fi
_lock_acquired=0
for _i in $(seq 1 30); do  # wait up to 3 seconds (30 x 0.1s)
  if mkdir "$HOOK_LOCK" 2>/dev/null; then
    _lock_acquired=1
    trap 'rmdir "$HOOK_LOCK" 2>/dev/null || true' EXIT
    break
  fi
  sleep 0.1
done
if [[ $_lock_acquired -eq 0 ]]; then
  debug_log "Could not acquire lock after 3s, giving up"
  exit 0  # give up if still locked
fi

[[ -n "$PERF_ENABLED" ]] && T_START=$(perl -MTime::HiRes=time -e 'printf "%.3f\n", time()')

# Check that jq is available
if ! command -v jq &>/dev/null; then
  debug_log "jq is not installed — cannot parse hook input"
  exit 0
fi

# Parse JSON input with jq — extract each field separately to avoid eval
# (eval of jq output was a shell injection vector if transcript content
# contained shell metacharacters)
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // ""' 2>/dev/null) || true
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // ""' 2>/dev/null) || true
HOOK_EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // ""' 2>/dev/null) || true

if [[ -z "$TRANSCRIPT_PATH" || ! -f "$TRANSCRIPT_PATH" ]]; then
  debug_log "Transcript path empty or file missing: '$TRANSCRIPT_PATH'"
  exit 0
fi

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

# --- Count lines (handle missing/unreadable transcript) ---
TOTAL_LINES=$(wc -l < "$TRANSCRIPT_PATH" 2>/dev/null | tr -d ' ')
if [[ -z "$TOTAL_LINES" ]]; then
  debug_log "Could not read transcript file: $TRANSCRIPT_PATH"
  exit 0
fi
[[ "$TOTAL_LINES" -le "$LAST_POS" ]] && exit 0

# --- Extract new assistant text ---
NEW_LINES=$((TOTAL_LINES - LAST_POS))
TEXT=$(tail -n "$NEW_LINES" "$TRANSCRIPT_PATH" 2>/dev/null \
  | jq -r '
    select(.type == "assistant")
    | .message.content[]?
    | select(.type == "text")
    | .text
  ' 2>/dev/null) || true

[[ -n "$PERF_ENABLED" ]] && T_EXTRACT=$(perl -MTime::HiRes=time -e 'printf "%.3f\n", time()')

# --- Update position ---
printf '%s\n%s\n' "$SESSION_ID" "$TOTAL_LINES" > "$POS_FILE" 2>/dev/null || {
  debug_log "Failed to write position file: $POS_FILE"
}

if [[ -z "$TEXT" ]]; then
  debug_log "No new assistant text found"
  exit 0
fi

# --- Strip markdown (light pass — normalizer handles code blocks, tables, lists) ---
CLEAN=$(echo "$TEXT" | sed -E '
  s/\*{1,3}([^*]*)\*{1,3}/\1/g
  s/_{1,3}([^_]*)_{1,3}/\1/g
  s/\[([^]]*)\]\([^)]*\)/\1/g
  s/!\[([^]]*)\]\([^)]*\)/\1/g
  s/^[-*_]{3,}$//
  s/<[^>]*>//g
' 2>/dev/null) || true

CLEAN=$(echo "$CLEAN" | sed '/^[[:space:]]*$/d' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//' 2>/dev/null) || true
if [[ -z "$CLEAN" ]]; then
  debug_log "Text was empty after markdown stripping"
  exit 0
fi

[[ -n "$PERF_ENABLED" ]] && T_SED=$(perl -MTime::HiRes=time -e 'printf "%.3f\n", time()')

# --- Write to queue ---
mkdir -p "$QUEUE_DIR" 2>/dev/null || {
  debug_log "Failed to create queue directory: $QUEUE_DIR"
  exit 0
}
chmod 700 "$QUEUE_DIR" 2>/dev/null || true
TIMESTAMP=$(perl -MTime::HiRes=time -e 'printf "%.6f\n", time()')
QUEUE_FILE="$QUEUE_DIR/${TIMESTAMP}.txt"
echo "$CLEAN" > "$QUEUE_FILE" 2>/dev/null || {
  debug_log "Failed to write queue file: $QUEUE_FILE"
  exit 0
}
chmod 600 "$QUEUE_FILE" 2>/dev/null || true

# Signal daemon to process immediately (no poll delay)
DAEMON_PID=$(cat /tmp/claude-speak-daemon.pid 2>/dev/null) || true
if [[ -n "$DAEMON_PID" ]]; then
  kill -USR1 "$DAEMON_PID" 2>/dev/null || true
fi

if [[ -n "$PERF_ENABLED" ]]; then
  T_END=$(perl -MTime::HiRes=time -e 'printf "%.3f\n", time()')
  echo "[hook-perf] parse=$(echo "$T_PARSE - $T_START" | bc)s extract=$(echo "$T_EXTRACT - $T_PARSE" | bc)s sed=$(echo "$T_SED - $T_EXTRACT" | bc)s write=$(echo "$T_END - $T_SED" | bc)s TOTAL=$(echo "$T_END - $T_START" | bc)s lines=$NEW_LINES chars=${#CLEAN}" >> "$PERF_LOG" 2>/dev/null || true
fi

exit 0
