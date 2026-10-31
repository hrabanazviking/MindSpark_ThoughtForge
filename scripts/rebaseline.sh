#!/bin/sh
# TF-014 — campaign baseline re-measurement.
# Re-measures the frozen zero line exactly as docs/audit/TF-014_BASELINE_FREEZE.md
# records it. Non-destructive: the only writes are a synthetic store under a
# temporary directory. Run from anywhere; it cds to the repo root itself.
set -u
cd "$(dirname "$0")/.."

echo "== tree =="
git rev-parse HEAD
git branch --show-current
python3 --version
python3 -m pytest --version | head -1

echo "== suite (PYTHONPATH=src python3 -m pytest tests/ -q) =="
PYTHONPATH=src python3 -m pytest tests/ -q 2>&1 | tail -3

echo "== forge_doctor.py --audit --json =="
python3 forge_doctor.py --audit --json

echo "== entry points =="
PYTHONPATH=src python3 run_thoughtforge.py --help >/dev/null 2>&1
echo "run_thoughtforge.py --help exit: $?"
PYTHONPATH=src python3 run_thoughtforge.py "What is ThoughtForge?" 2>&1 | grep -E "no knowledge|Confidence|Enforcement" || true
TMPD=$(mktemp -d)
PYTHONPATH=src python3 forge_memory.py --db "$TMPD/synthetic.db" init >/dev/null 2>&1
echo "forge_memory.py init (synthetic) exit: $?"
PYTHONPATH=src python3 forge_memory.py --db "$TMPD/synthetic.db" status 2>&1 | grep -E "Size|entities" || true
rm -rf "$TMPD"
