#!/usr/bin/env bash
#
# Run the full daeFinder test suite once and save a timestamped report.
#
# Usage:
#   ./tests/run_tests.sh                 # run everything with the default python3
#   PYTHON=python3.11 ./tests/run_tests.sh   # pick the interpreter
#   ./tests/run_tests.sh -k noise        # forward extra args to pytest (filter, -x, etc.)
#   ./tests/run_tests.sh tests/test_library.py   # run a single file
#
# Reports are written to tests/reports/ (git-ignored): a human-readable .txt log
# and a JUnit .xml. A coverage summary is included automatically if pytest-cov is
# installed.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-python3}"
REPORT_DIR="$SCRIPT_DIR/reports"
mkdir -p "$REPORT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
TXT_REPORT="$REPORT_DIR/test_run_${STAMP}.txt"
XML_REPORT="$REPORT_DIR/test_run_${STAMP}.xml"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: interpreter '$PYTHON' not found. Set PYTHON=<path> and retry." >&2
    exit 2
fi
if ! "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
    echo "ERROR: pytest is not installed for $PYTHON." >&2
    echo "       Install it with: $PYTHON -m pip install -r tests/requirements-test.txt" >&2
    exit 2
fi

# Add a coverage report if pytest-cov is available.
COV_ARGS=()
if "$PYTHON" -c "import pytest_cov" >/dev/null 2>&1; then
    COV_ARGS=(--cov=daeFinder --cov-report=term-missing)
fi

{
    echo "=========================================================="
    echo " daeFinder test run"
    echo " when:    $STAMP"
    echo " python:  $("$PYTHON" --version 2>&1)  ($("$PYTHON" -c 'import sys; print(sys.executable)'))"
    echo " repo:    $REPO_ROOT"
    echo "=========================================================="
} | tee "$TXT_REPORT"

# Note: "${arr[@]+"${arr[@]}"}" expands safely even when the array is empty under
# `set -u` on bash 3.2 (the version shipped with macOS).
"$PYTHON" -m pytest tests/ -v --junitxml="$XML_REPORT" \
    ${COV_ARGS[@]+"${COV_ARGS[@]}"} "$@" 2>&1 | tee -a "$TXT_REPORT"
STATUS=${PIPESTATUS[0]}

{
    echo
    echo "=========================================================="
    if [ "$STATUS" -eq 0 ]; then
        echo " RESULT: PASS"
    else
        echo " RESULT: FAIL (pytest exit code $STATUS)"
    fi
    echo " text report: $TXT_REPORT"
    echo " junit  xml : $XML_REPORT"
    echo "=========================================================="
} | tee -a "$TXT_REPORT"

exit "$STATUS"
