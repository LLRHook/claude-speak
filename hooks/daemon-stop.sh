#!/bin/bash
# Stop the claude-speak daemon when Claude Code session ends.
set -euo pipefail
python3 -m claude_speak.cli stop 2>/dev/null || true
exit 0
