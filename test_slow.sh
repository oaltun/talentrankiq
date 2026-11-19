#!/usr/bin/env bash
# Slow test suite - runs before final release.
# Uses the nightly toolchain so cargo's test harness emits per-test
# timings (--format json). After all tests finish, prints a listing
# of every successful test sorted by duration (slowest first).

set -uo pipefail

CLI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CLI_DIR"

echo "=== Slow Test Suite ==="
echo "Start: $(date)"

START_TOTAL=$(date +%s.%N)
GLOBAL_DURATIONS="$(mktemp)"
: > "$GLOBAL_DURATIONS"

SUCCESS_GROUPS=()
SUCCESS_GROUP_DURS=()
FAILED_GROUPS=()

run_timed() {
    local name="$1"
    shift
    echo "--- $name ---"
    local start=$(date +%s.%N)
    local logfile
    logfile="$(mktemp)"
    if "$@" -- -Z unstable-options --format json >"$logfile" 2>&1; then
        local end=$(date +%s.%N)
        local dur=$(echo "$end - $start" | bc)
        printf "  ✓ %s: %.3fs\n" "$name" "$dur"
        SUCCESS_GROUPS+=("$name")
        SUCCESS_GROUP_DURS+=("$dur")
        python3 - "$logfile" "$name" "$GLOBAL_DURATIONS" <<'PYINNER'
import json, sys
logfile, group, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
count = 0
with open(logfile) as f:
    for line in f:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "test" and e.get("event") == "ok":
            t = e.get("exec_time", 0.0)
            nm = e.get("name", "?")
            with open(outfile, "a") as out:
                out.write(f"{group}\t{nm}\t{t}\n")
            count += 1
print(f"  ({count} individual tests passed)")
PYINNER
        rm -f "$logfile"
        return 0
    else
        local rc=$?
        local end=$(date +%s.%N)
        local dur=$(echo "$end - $start" | bc)
        printf "  ✗ %s: %.3fs (FAILED, exit %d)\n" "$name" "$dur" "$rc"
        FAILED_GROUPS+=("$name")
        echo "  --- tail of output ---"
        tail -20 "$logfile" | sed 's/^/  /'
        rm -f "$logfile"
        return 1
    fi
}

FAILED=0

# All unit tests
run_timed "unit tests (all)" cargo +nightly test --lib --workspace || FAILED=1

# All integration tests
run_timed "cli::download_test" cargo +nightly test -p cli --test download_test || FAILED=1
run_timed "cli::logging_test" cargo +nightly test -p cli --test logging_test || FAILED=1
run_timed "cli::model_workflow_test" cargo +nightly test -p cli --test model_workflow_test || FAILED=1
run_timed "process-in-waves::smoke" cargo +nightly test -p libprocess-in-waves --test smoke || FAILED=1
run_timed "process-in-waves-db-monitor::smoke" cargo +nightly test -p libprocess-in-waves-db-monitor --test smoke || FAILED=1
run_timed "tokenizer::real_tokenizer" cargo +nightly test -p libtokenizer --test real_tokenizer || FAILED=1
run_timed "mlxembed::parity" cargo +nightly test -p libmlxembed --test parity || FAILED=1

# Doc tests
run_timed "doc tests" cargo +nightly test --doc --workspace || FAILED=1

END_TOTAL=$(date +%s.%N)
TOTAL_DUR=$(echo "$END_TOTAL - $START_TOTAL" | bc)

echo ""
echo "=== Successful test durations (per test, slowest first) ==="
python3 - "$GLOBAL_DURATIONS" <<'PY'
import sys
rows = []
with open(sys.argv[1]) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        group, name, t = parts
        try:
            t = float(t)
        except ValueError:
            continue
        rows.append((t, group, name))
rows.sort(reverse=True)
if not rows:
    print("  (no successful tests)")
else:
    for t, group, name in rows:
        print(f"  {t:9.4f}s  {group:34s}  {name}")
print(f"\n  Total successful tests: {len(rows)}")
PY

echo ""
echo "=== Summary ==="
printf "Total time: %.3fs\n" "$TOTAL_DUR"
echo "Passed groups: ${#SUCCESS_GROUPS[@]}  Failed groups: ${#FAILED_GROUPS[@]}"
if [ ${#FAILED_GROUPS[@]} -gt 0 ]; then
    echo "Failed groups:"
    for n in "${FAILED_GROUPS[@]}"; do
        echo "  ✗ $n"
    done
fi
echo "End: $(date)"

rm -f "$GLOBAL_DURATIONS"
exit $FAILED
