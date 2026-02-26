#!/bin/bash
# install.sh — Automated setup for claude-speak
# Safe to run multiple times (idempotent).

# --- Deprecation Notice ---
echo ""
echo "⚠️  install.sh is deprecated and will be removed in a future version."
echo "   Please use the new setup flow instead:"
echo ""
echo "     pip install claude-speak"
echo "     claude-speak setup"
echo ""
echo "   Continuing with legacy installation in 5 seconds..."
echo "   Press Ctrl+C to abort."
echo ""
sleep 5

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors & helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { printf "${BLUE}[info]${NC}  %s\n" "$*"; }
ok()      { printf "${GREEN}[ok]${NC}    %s\n" "$*"; }
warn()    { printf "${YELLOW}[warn]${NC}  %s\n" "$*"; }
err()     { printf "${RED}[error]${NC} %s\n" "$*"; }
step()    { printf "\n${BOLD}==> %s${NC}\n" "$*"; }

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLAUDE_DIR="$HOME/.claude"
SETTINGS_JSON="$CLAUDE_DIR/settings.json"
HOOKS_DIR="$CLAUDE_DIR/hooks"
MODELS_DIR="$PROJECT_DIR/models"
VENV_DIR="$PROJECT_DIR/.venv"
TOGGLE_FILE="$HOME/.claude-speak-enabled"

MODEL_ONNX_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
MODEL_VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
step "Checking prerequisites"

# macOS only
if [[ "$(uname -s)" != "Darwin" ]]; then
    err "claude-speak requires macOS (detected: $(uname -s)). Exiting."
    exit 1
fi
ok "macOS detected"

# Python 3.10+
if ! command -v python3 &>/dev/null; then
    err "python3 not found. Install Python 3.10+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || { [[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -lt 10 ]]; }; then
    err "Python 3.10+ required (found $PYTHON_VERSION). Exiting."
    exit 1
fi
ok "Python $PYTHON_VERSION"

# pip
if ! python3 -m pip --version &>/dev/null; then
    err "pip not found. Install pip (python3 -m ensurepip) and try again."
    exit 1
fi
ok "pip available"

# jq (needed by hook scripts)
if ! command -v jq &>/dev/null; then
    warn "jq not found — the hook scripts require it."
    info "Install with: brew install jq"
fi

# curl (needed for model downloads)
if ! command -v curl &>/dev/null; then
    err "curl not found. Install curl and try again."
    exit 1
fi
ok "curl available"

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------
step "Setting up Python virtual environment"

if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# Activate the venv for the remainder of this script
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Activated venv ($(python3 --version))"

# ---------------------------------------------------------------------------
# 3. Install Python dependencies
# ---------------------------------------------------------------------------
step "Installing Python dependencies"

if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
    err "requirements.txt not found in $PROJECT_DIR"
    exit 1
fi

pip install --upgrade pip --quiet
pip install -r "$PROJECT_DIR/requirements.txt" --quiet
ok "Dependencies installed from requirements.txt"

# ---------------------------------------------------------------------------
# 4. Download model files
# ---------------------------------------------------------------------------
step "Downloading model files"

mkdir -p "$MODELS_DIR"

download_model() {
    local url="$1"
    local dest="$2"
    local name
    name="$(basename "$dest")"

    if [[ -f "$dest" ]]; then
        local size
        size=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null || echo 0)
        if [[ "$size" -gt 1000 ]]; then
            ok "$name already exists ($(( size / 1024 / 1024 )) MB) — skipping"
            return 0
        else
            warn "$name exists but looks incomplete ($size bytes) — re-downloading"
        fi
    fi

    info "Downloading $name ..."
    if curl -L --progress-bar -o "$dest" "$url"; then
        ok "$name downloaded successfully"
    else
        err "Failed to download $name"
        rm -f "$dest"
        return 1
    fi
}

download_model "$MODEL_ONNX_URL" "$MODELS_DIR/kokoro-v1.0.onnx"
download_model "$MODEL_VOICES_URL" "$MODELS_DIR/voices-v1.0.bin"

# ---------------------------------------------------------------------------
# 5. Configure Claude Code hooks
# ---------------------------------------------------------------------------
step "Configuring Claude Code hooks"

mkdir -p "$HOOKS_DIR"

# Symlink hook scripts into ~/.claude/hooks/
symlink_hook() {
    local src="$PROJECT_DIR/hooks/$1"
    local dest="$HOOKS_DIR/$1"

    if [[ ! -f "$src" ]]; then
        warn "Hook source not found: $src — skipping"
        return 0
    fi

    chmod +x "$src"

    if [[ -L "$dest" ]]; then
        local current_target
        current_target="$(readlink "$dest")"
        if [[ "$current_target" == "$src" ]]; then
            ok "Symlink $1 already correct"
            return 0
        else
            warn "Symlink $1 points to $current_target — updating to $src"
            ln -sf "$src" "$dest"
            ok "Symlink $1 updated"
            return 0
        fi
    elif [[ -f "$dest" ]]; then
        warn "$dest exists as a regular file — backing up to ${dest}.bak"
        mv "$dest" "${dest}.bak"
    fi

    ln -s "$src" "$dest"
    ok "Symlinked $1 -> $src"
}

symlink_hook "speak-response.sh"
symlink_hook "daemon-start.sh"
symlink_hook "daemon-stop.sh"

# --- Merge hooks into ~/.claude/settings.json ---

# The hook entries we want to ensure exist
SPEAK_RESPONSE_CMD="~/.claude/hooks/speak-response.sh"
DAEMON_START_CMD="~/.claude/hooks/daemon-start.sh"
DAEMON_STOP_CMD="~/.claude/hooks/daemon-stop.sh"

if ! command -v jq &>/dev/null; then
    warn "jq not installed — cannot auto-configure settings.json"
    warn "You will need to manually add the hook entries to $SETTINGS_JSON"
    warn "See the project README for the required hook configuration."
else
    # Build settings.json if it doesn't exist
    if [[ ! -f "$SETTINGS_JSON" ]]; then
        echo '{}' > "$SETTINGS_JSON"
        info "Created $SETTINGS_JSON"
    fi

    CURRENT_SETTINGS=$(cat "$SETTINGS_JSON")

    # Check what needs to be added
    needs_post_tool=false
    needs_stop=false
    needs_session_start=false
    needs_session_end=false

    # PostToolUse: speak-response.sh
    if echo "$CURRENT_SETTINGS" | jq -e '.hooks.PostToolUse // [] | map(.hooks[]? | select(.command == "'"$SPEAK_RESPONSE_CMD"'")) | length > 0' &>/dev/null; then
        ok "PostToolUse hook already configured"
    else
        needs_post_tool=true
    fi

    # Stop: speak-response.sh
    if echo "$CURRENT_SETTINGS" | jq -e '.hooks.Stop // [] | map(.hooks[]? | select(.command == "'"$SPEAK_RESPONSE_CMD"'")) | length > 0' &>/dev/null; then
        ok "Stop hook (speak-response) already configured"
    else
        needs_stop=true
    fi

    # SessionStart: daemon-start.sh
    if echo "$CURRENT_SETTINGS" | jq -e '.hooks.SessionStart // [] | map(.hooks[]? | select(.command == "'"$DAEMON_START_CMD"'")) | length > 0' &>/dev/null; then
        ok "SessionStart hook already configured"
    else
        needs_session_start=true
    fi

    # SessionEnd: daemon-stop.sh
    if echo "$CURRENT_SETTINGS" | jq -e '.hooks.SessionEnd // [] | map(.hooks[]? | select(.command == "'"$DAEMON_STOP_CMD"'")) | length > 0' &>/dev/null; then
        ok "SessionEnd hook already configured"
    else
        needs_session_end=true
    fi

    if $needs_post_tool || $needs_stop || $needs_session_start || $needs_session_end; then
        echo ""
        info "The following hook entries will be added to $SETTINGS_JSON:"
        echo ""

        if $needs_post_tool; then
            printf "  ${YELLOW}PostToolUse${NC}: %s (async, timeout 30s)\n" "$SPEAK_RESPONSE_CMD"
        fi
        if $needs_stop; then
            printf "  ${YELLOW}Stop${NC}:        %s (async, timeout 30s)\n" "$SPEAK_RESPONSE_CMD"
        fi
        if $needs_session_start; then
            printf "  ${YELLOW}SessionStart${NC}: %s (async)\n" "$DAEMON_START_CMD"
        fi
        if $needs_session_end; then
            printf "  ${YELLOW}SessionEnd${NC}:  %s (async)\n" "$DAEMON_STOP_CMD"
        fi

        echo ""
        printf "  Existing hooks in settings.json will ${BOLD}not${NC} be modified.\n"
        echo ""
        printf "  Proceed? [Y/n] "
        read -r CONFIRM
        CONFIRM="${CONFIRM:-Y}"

        if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
            UPDATED="$CURRENT_SETTINGS"

            if $needs_post_tool; then
                UPDATED=$(echo "$UPDATED" | jq '
                    .hooks.PostToolUse = (.hooks.PostToolUse // []) + [{
                        "hooks": [{
                            "type": "command",
                            "command": "'"$SPEAK_RESPONSE_CMD"'",
                            "timeout": 30,
                            "async": true
                        }]
                    }]
                ')
            fi

            if $needs_stop; then
                UPDATED=$(echo "$UPDATED" | jq '
                    .hooks.Stop = (.hooks.Stop // []) + [{
                        "hooks": [{
                            "type": "command",
                            "command": "'"$SPEAK_RESPONSE_CMD"'",
                            "timeout": 30,
                            "async": true
                        }]
                    }]
                ')
            fi

            if $needs_session_start; then
                UPDATED=$(echo "$UPDATED" | jq '
                    .hooks.SessionStart = (.hooks.SessionStart // []) + [{
                        "hooks": [{
                            "type": "command",
                            "command": "'"$DAEMON_START_CMD"'",
                            "async": true
                        }]
                    }]
                ')
            fi

            if $needs_session_end; then
                UPDATED=$(echo "$UPDATED" | jq '
                    .hooks.SessionEnd = (.hooks.SessionEnd // []) + [{
                        "hooks": [{
                            "type": "command",
                            "command": "'"$DAEMON_STOP_CMD"'",
                            "async": true
                        }]
                    }]
                ')
            fi

            echo "$UPDATED" | jq '.' > "$SETTINGS_JSON"
            ok "Hooks added to $SETTINGS_JSON"
        else
            warn "Skipped hook configuration. You can re-run this script or add them manually."
        fi
    else
        ok "All hooks already configured in settings.json"
    fi
fi

# ---------------------------------------------------------------------------
# 6. Default config
# ---------------------------------------------------------------------------
step "Setting up default configuration"

CONFIG_FILE="$PROJECT_DIR/claude-speak.toml"
EXAMPLE_CONFIG="$PROJECT_DIR/claude-speak.toml.example"

if [[ -f "$CONFIG_FILE" ]]; then
    ok "claude-speak.toml already exists"
elif [[ -f "$EXAMPLE_CONFIG" ]]; then
    cp "$EXAMPLE_CONFIG" "$CONFIG_FILE"
    ok "Created claude-speak.toml from example"
else
    # Generate a sensible default config
    cat > "$CONFIG_FILE" << 'TOML'
# claude-speak configuration

[tts]
voice = "af_sarah"
speed = 1.3
device = "auto"          # "auto" = system default, or name substring e.g. "AirPods"
max_chunk_chars = 150    # small chunks = fast first audio; parallel gen eliminates gaps

[wakeword]
enabled = true           # wake word detection active
engine = "openwakeword"
model = "hey_jarvis"     # swap to "wakeword/hey-claude.onnx" after training
sensitivity = 0.5
stop_phrases = ["stop", "quiet", "shut up"]

[input]
superwhisper = true
auto_submit = true

[audio]
chimes = true              # play chimes on state changes
greeting = "Ready."        # spoken when daemon starts
volume = 0.3               # chime volume (0.0-1.0)

[normalization]
skip_code = true
expand_units = true
expand_abbreviations = true
shorten_paths = true
TOML
    ok "Created default claude-speak.toml"
fi

# ---------------------------------------------------------------------------
# 7. Toggle file
# ---------------------------------------------------------------------------
step "Creating toggle file"

if [[ -f "$TOGGLE_FILE" ]]; then
    ok "$TOGGLE_FILE already exists"
else
    touch "$TOGGLE_FILE"
    ok "Created $TOGGLE_FILE"
fi

# ---------------------------------------------------------------------------
# Done!
# ---------------------------------------------------------------------------
echo ""
printf "${GREEN}${BOLD}============================================${NC}\n"
printf "${GREEN}${BOLD}  claude-speak installed successfully!${NC}\n"
printf "${GREEN}${BOLD}============================================${NC}\n"
echo ""
printf "  ${BOLD}Next steps:${NC}\n"
echo ""
printf "  1. Test TTS:       ${BLUE}cd %s && source .venv/bin/activate && python3 cli.py test \"Hello world\"${NC}\n" "$PROJECT_DIR"
printf "  2. Start daemon:   ${BLUE}python3 cli.py start${NC}\n"
printf "  3. Check status:   ${BLUE}python3 cli.py status${NC}\n"
printf "  4. Edit config:    ${BLUE}%s/claude-speak.toml${NC}\n" "$PROJECT_DIR"
echo ""
printf "  The daemon will auto-start with Claude Code sessions via hooks.\n"
printf "  Toggle voice on/off:  ${BLUE}python3 cli.py enable${NC} / ${BLUE}python3 cli.py disable${NC}\n"
echo ""
