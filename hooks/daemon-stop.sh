#!/bin/bash
# Stop the claude-speak daemon when Claude Code session ends.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"
python3 "$PROJECT/cli.py" stop 2>/dev/null || true
exit 0
