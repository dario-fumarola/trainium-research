#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
root=$PWD
check() (
  cd "$root/$1"
  uv run --no-project --python 3.11 --with torch==2.9.1 --with transformers==5.15.1 --with pytest python -m pytest -q "$2"
)
check studies/03-autonomous-compute tests/test_interleaved_sequential_reference.py
check studies/02-pause-pilot tests/test_pause_pretraining.py
check studies/01-useful-thoughts tests/test_scaffold_learning.py
