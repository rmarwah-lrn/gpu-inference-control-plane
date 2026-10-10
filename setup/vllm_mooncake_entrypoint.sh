#!/usr/bin/env sh
set -eu

# vllm-openai does not ship the Mooncake wheel. Install the CUDA-12 wheel
# selected for the v0.11 image before starting the real connector workers.
python3 -m pip install --quiet --no-cache-dir "${MOONCAKE_PACKAGE:-mooncake-transfer-engine-cuda13}"
exec vllm serve "$@"
