#!/bin/bash
# Stop the claude-speak daemon when Claude Code session ends.
set -euo pipefail
SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SOURCE" ]]; do SOURCE="$(readlink "$SOURCE")"; done
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"
PYTHONPATH="$PROJECT" python3 -m claude_speak.cli stop 2>/dev/null || true
exit 0
