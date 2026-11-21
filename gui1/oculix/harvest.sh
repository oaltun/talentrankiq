#!/bin/bash
# Harvest mode: run Playwright against `bun run dev` (not tauri) to capture images.
# Usage: ./harvest.sh --theme=light|dark [--scenes=scene1,scene2|all] [extra args...]

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

# --- Harvest mode: run Playwright against bun run dev (not tauri) ---
echo "Starting harvest mode for theme: $THEME, scenes: $SCENES"

cd "$GUI_DIR"

export OCULIX_THEME="$THEME"
export OCULIX_SCENES="$SCENES"

# Start vite dev server in background if not already running
if ! curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo "Starting Vite dev server (bun run dev)..."
    ( bun run dev > /tmp/oculix-vite-dev.log 2>&1 ) &
    VITE_PID=$!
    trap 'if [ -n "${VITE_PID:-}" ]; then echo "Stopping Vite (PID $VITE_PID)..."; kill "$VITE_PID" 2>/dev/null; pkill -P "$VITE_PID" 2>/dev/null; fi' EXIT

    echo "Waiting for Vite dev server..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:3000 >/dev/null 2>&1; then
            echo "Vite dev server is up"
            break
        fi
        sleep 2
    done
    if ! curl -s http://localhost:3000 >/dev/null 2>&1; then
        echo "ERROR: Vite dev server failed to start. See /tmp/oculix-vite-dev.log"
        exit 1
    fi
else
    echo "Vite dev server already running on port 3000"
fi

# Run Playwright harvest
echo "Running Playwright harvest..."
npx playwright test -c oculix/harvest/playwright.config.ts "${EXTRA_ARGS[@]}"

# Post-process: remove images that are too small or single-color
MIN_SIZE=100
IMG_BASE="$GUI_DIR/ocr/ulix/img"
if ! command -v identify >/dev/null 2>&1; then
    echo "WARNING: 'identify' (ImageMagick) not found, skipping image size/color checks"
else
    if [ -d "$IMG_BASE" ]; then
        LATEST_DIR=$(find "$IMG_BASE" -maxdepth 1 -type d -name "img-*" | sort -V | tail -1)
        if [ -n "$LATEST_DIR" ] && [ -d "$LATEST_DIR/$THEME" ]; then
            THEME_DIR="$LATEST_DIR/$THEME"
            echo "Checking images in $THEME_DIR (min size: ${MIN_SIZE}px, single-color filter)..."
            for img in "$THEME_DIR"/*.png; do
                [ -f "$img" ] || continue
                # Get width, height, unique color count
                read width height colors <<< $(identify -format "%w %h %k" "$img" 2>/dev/null)
                if [ -z "$width" ] || [ -z "$height" ] || [ -z "$colors" ]; then
                    echo "  Skipping $img (identify failed)"
                    continue
                fi
                if [ "$width" -lt "$MIN_SIZE" ] || [ "$height" -lt "$MIN_SIZE" ]; then
                    echo "  Removing $img (too small: ${width}x${height})"
                    rm "$img"
                    continue
                fi
                if [ "$colors" -eq 1 ]; then
                    echo "  Removing $img (single color)"
                    rm "$img"
                    continue
                fi
            done
        fi
    fi
fi

echo "Harvest complete. Images written to oculix/img-N/$THEME/"
exit 0
