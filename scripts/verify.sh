#!/usr/bin/env bash
# One-command local verification: static checks + full offline test suite +
# synthetic end-to-end example. Requires the dev dependencies:
#   pip install -r requirements.txt -r requirements-dev.txt
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ruff (static checks) =="
python -m ruff check .

echo "== pytest (offline suite; no API key needed) =="
python -m pytest

echo "== offline end-to-end example =="
python examples/offline_demo.py

echo
echo "All verification steps passed."
