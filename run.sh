#!/usr/bin/env bash
set -e

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

# Dataset 3: 1d-multi-method
run_py workflow/format_1d_multi_method.py
run_py workflow/viz_1d_multi_method.py