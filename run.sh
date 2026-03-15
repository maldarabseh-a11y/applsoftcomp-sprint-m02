#!/usr/bin/env bash
set -e

mkdir -p figs workflow/processed

# Dataset 1: 1d-data
uv run python workflow/format_1d_data.py
uv run python workflow/viz_1d_data.py
