#!/usr/bin/env bash
set -euo pipefail

# Reproduce the figure for task #3: 1d-multi-method-data.csv
# Assumes dependencies are installed (e.g., via `uv sync` or equivalent).

python workflow/format_1d_multi_method.py
python workflow/viz_1d_multi_method.py
