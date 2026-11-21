#!/bin/bash
# Test runner for Tauri GUI E2E tests
# This script runs both tauri-pilot and Playwright E2E tests
# Reuses an existing Tauri app if one is already running.

set -euo pipefail

GUI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GUI_DIR/.." && pwd)"
CLI_DIR="$REPO_ROOT/talentrankiq_cli"

# Build triq CLI (debug + release)
echo "Building triq CLI (debug + release)..."
(cd "$CLI_DIR" && cargo build && cargo build --release) || { echo "ERROR: failed to build triq CLI"; exit 1; }
export TRIQ_BIN="$CLI_DIR/target/debug/triq"
export TRIQ_CLI_DIR="$CLI_DIR"

# Build GUI Rust part (debug + release)
echo "Building GUI Rust part (debug + release)..."
(cd "$GUI_DIR/src-tauri" && cargo build && cargo build --release) || { echo "ERROR: failed to build GUI Rust part"; exit 1; }

# Build GUI
cd "$GUI_DIR"
npm install
npm run build

# Create test fixtures for change-paths scenario
mkdir -p /tmp/test_cv_dir_change /tmp/test_cv_dir_button
echo "This is a test job ad for tauri-pilot testing." > /tmp/test_job_ad_change.txt

# Create comparator artifact fixtures for the comparator-add scenario
mkdir -p /tmp/triq-e2e-comparator
echo '{"feature_spec":{"dim":1024}}' > /tmp/triq-e2e-comparator/comparator-harrier-600m-v1.json
echo '{"feature_spec":{"dim":640}}' > /tmp/triq-e2e-comparator/comparator-harrier-270m-v1.json

# Always start a fresh Tauri app for tauri-pilot tests to ensure clean CLI subprocess
STARTED_TAURI=1

# Kill any existing stale instances
pkill -f "tauri.*dev" 2>/dev/null || true
pkill -f "tauri-native" 2>/dev/null || true
pkill -f "talentrankiq" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
pkill -f "triq serve" 2>/dev/null || true
sleep 3

# Clean cache for a fresh tauri-pilot test run
rm -rf "$HOME/Library/Caches/com.razalt.talentrankiq" "$HOME/Library/Application Support/com.razalt.talentrankiq" 2>/dev/null || true
rm -rf "$HOME/Library/WebKit/talentrankiq" "$HOME/Library/WebKit/talentrankiq-gui" "$HOME/Library/WebKit/tauri-native" 2>/dev/null || true

# Start the process_in_waves scenarios from zero progress: the waves pipeline
# keeps its own cv/cv_embedding tables (separate from candidates), and a
# green e2e must see embedded_done move from 0, not inherit a previous run.
TRIQ_DB="$HOME/Library/Application Support/com.razalt.talentrankiq/triq/triqui.db"
if [ -f "$TRIQ_DB" ]; then
  sqlite3 "$TRIQ_DB" "DELETE FROM cv_embedding; DELETE FROM cv_similarity; DELETE FROM file_state; DELETE FROM waves_rank_state; DELETE FROM cv;" 2>/dev/null || true
fi

# Start the Tauri app for tauri-pilot tests
npm run tauri dev > /tmp/tauri-test.log 2>&1 &
TAURI_PID=$!
echo "Started Tauri dev server (PID: $TAURI_PID)"

# Wait for the app to be ready
echo "Waiting for Tauri app to be ready..."
for i in $(seq 1 60); do
  if tauri-pilot ping 2>/dev/null | grep -q "Connected"; then
    echo "Tauri app is ready!"
    break
  fi
  sleep 2
done

# Run tauri-pilot scenarios
echo "Running tauri-pilot E2E tests..."
TAURI_PILOT_RESULT=0

SCENARIOS=(
  app-shell
  models-set-default
  job-details-allows-downloaded-default-model
  cv-watch-directory-autostart
  job-details-ad-card-path-input
  job-details-cv-watch-directory-layout
  process-in-waves-job2000
  models-quick-remove-default
  language-settings
  # Add more scenario names here as they are created
)

for scenario in "${SCENARIOS[@]}"; do
  scenario_file="$GUI_DIR/scenarios/$scenario.toml"
  if [ ! -f "$scenario_file" ]; then
    echo "SKIP: missing scenario $scenario.toml"
    continue
  fi
  echo "Running scenario: $scenario"
  tauri-pilot run --rpc-timeout 120 "$scenario_file" || TAURI_PILOT_RESULT=$?
done

# Clean up Tauri app
kill "$TAURI_PID" 2>/dev/null || true
wait "$TAURI_PID" 2>/dev/null || true
pkill -f "tauri.*dev" 2>/dev/null || true
pkill -f "tauri-native" 2>/dev/null || true
pkill -f "talentrankiq" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 2

# Clean cache for Playwright tests
rm -rf "$HOME/Library/Caches/com.razalt.talentrankiq" "$HOME/Library/Application Support/com.razalt.talentrankiq" 2>/dev/null || true
rm -rf "$HOME/Library/WebKit/talentrankiq" "$HOME/Library/WebKit/talentrankiq-gui" "$HOME/Library/WebKit/tauri-native" 2>/dev/null || true

# Run Playwright tests (webServer in playwright.config.ts starts Vite dev server)
echo "Running Playwright E2E tests..."
PLAYWRIGHT_RESULT=0
npx playwright test "$@" || PLAYWRIGHT_RESULT=$?

# Exit with failure if any test suite failed
if [ "$TAURI_PILOT_RESULT" -ne 0 ] || [ "$PLAYWRIGHT_RESULT" -ne 0 ]; then
  exit 1
fi

exit 0