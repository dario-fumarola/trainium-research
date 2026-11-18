#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../replay"
python3 uniform-lm-baseline-20261022/replay_uniform_lm.py uniform-lm-baseline-20261022/evidence
python3 scaffold-learning-20261022/replay.py scaffold-learning-20261022/evidence
python3 uniform-mixture-20261022/replay.py uniform-mixture-20261022/evidence
