#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
.venv/bin/python -m unittest discover -s tests -p test_adaptation_core.py -v
.venv/bin/python -u -m dtra.adapt "$@"
