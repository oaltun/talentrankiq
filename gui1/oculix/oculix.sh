#!/bin/bash
# Run the Oculix (Sikuli) flow against the TalentRankIQ GUI.
# Usage: ./oculix.sh --theme=light|dark [--scenes=scene1,scene2|all] [extra args...]

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GUI_DIR="$REPO_ROOT/talentrankiq_gui"
CLI_DIR="$REPO_ROOT/talentrankiq_cli"

THEME=""
SCENES="all"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --theme=*)
            THEME="${1#*=}"
            shift
            ;;
        --theme)
            THEME="$2"
            shift 2
            ;;
        --scenes=*)
            SCENES="${1#*=}"
            shift
            ;;
        --scenes)
            SCENES="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$THEME" ]]; then
    echo "ERROR: --theme=light|dark is required"
    exit 1
fi

if [[ "$THEME" != "light" && "$THEME" != "dark" ]]; then
    echo "ERROR: --theme must be 'light' or 'dark'"
    exit 1
fi

# --- 1. Build triq CLI (the GUI spawns it as a subprocess) ---
if [ ! -x "$CLI_DIR/target/debug/triq" ] || [ ! -x "$CLI_DIR/target/release/triq" ]; then
    echo "Building triq CLI..."
    ( cd "$CLI_DIR" && cargo build && cargo build --release ) || { echo "ERROR: failed to build triq CLI"; exit 1; }
fi
export TRIQ_BIN="$CLI_DIR/target/debug/triq"
export TRIQ_CLI_DIR="$CLI_DIR"

# --- Sikuli mode: run against tauri dev ---
echo "Starting Oculix flow for theme: $THEME"

is_gui_running() {
    pgrep -f "target/(debug|release)/tauri-native" >/dev/null 2>&1
}

STARTED_GUI=0
if is_gui_running; then
    echo "GUI already running, reusing it"
else
    echo "Starting GUI (npm run tauri dev)..."
    ( cd "$GUI_DIR" && npm run tauri dev > /tmp/oculix-tauri-dev.log 2>&1 ) &
    GUI_PID=$!
    STARTED_GUI=1
    trap 'if [ "$STARTED_GUI" = "1" ]; then echo "Stopping GUI (PID $GUI_PID)..."; kill "$GUI_PID" 2>/dev/null; pkill -P "$GUI_PID" 2>/dev/null; fi' EXIT

    echo "Waiting for GUI window..."
    for i in $(seq 1 90); do
        if is_gui_running; then
            sleep 3
            echo "GUI is up"
            break
        fi
        sleep 2
    done
    if ! is_gui_running; then
        echo "ERROR: GUI failed to start. See /tmp/oculix-tauri-dev.log"
        exit 1
    fi
fi

# --- Run the Oculix flow ---
echo "Running Oculix flow: oculix_final.sikuli"
cd "$SCRIPT_DIR"
java -jar oculixide-4.1.0-rc1-macos.jar -c -r ./flows/oculix_final.sikuli "${EXTRA_ARGS[@]}"

# Force exit after run (in case jar doesn't exit)
pkill -f "oculixide-4.1.0-rc1-macos.jar" 2>/dev/null || true
