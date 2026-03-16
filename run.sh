#!/usr/bin/env bash
set -euo pipefail

mkdir -p figs workflow/processed

run_py () {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  elif command -v python3 >/dev/null 2>&1; then
    python3 "$@"
  else
    python "$@"
  fi
}

# Dataset 1: 1d-data
run_py workflow/format_1d_data.py
run_py workflow/viz_1d_data.py

# Dataset 2: 2d-data
run_py workflow/format_2d_data.py
run_py workflow/viz_2d_data.py

# Dataset 3: 1d-multi-method
run_py workflow/format_1d_multi_method.py
run_py workflow/viz_1d_multi_method.py

# Dataset 4: digits
run_py workflow/format_digits.py
run_py workflow/viz_digits.py