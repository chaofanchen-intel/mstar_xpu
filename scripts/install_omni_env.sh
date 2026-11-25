#!/usr/bin/env bash
# Build a vLLM-Omni XPU venv for a given vLLM release (same recipe as the v0.32.0 env).
# Usage: scripts/install_omni_env.sh <vllm-version e.g. 0.34.0> <venv path>
set -euo pipefail
VLLM_VER="${1:?}"; VENV="${2:?}"; WS="$(cd "$(dirname "$0")/.." && pwd)"
uv venv --python 3.12 --seed "$VENV"
uv pip install --python "$VENV/bin/python" \
  --index-url "https://wheels.vllm.ai/$VLLM_VER/xpu" \
  --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple \
  --extra-index-url https://vllm-project.github.io/vllm-xpu-kernels/whl --extra-index-url https://wheels.vllm.ai/xpu/ \
  --index-strategy unsafe-best-match "vllm==${VLLM_VER}+xpu"
uv pip install --python "$VENV/bin/python" -r "$WS/external/vllm-omni/requirements/xpu.txt"
VLLM_OMNI_TARGET_DEVICE=xpu uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps \
  --editable "$WS/external/vllm-omni"
uv pip check --python "$VENV/bin/python" || true
# Optional two-stage connectors compared in docs/report (SharedMemory needs nothing):
#   Mooncake TransferEngine (TCP protocol works without RDMA) and NIXL (UCX backend; the "nixl" wheel pulls nixl-cu12).
uv pip install --python "$VENV/bin/python" mooncake-transfer-engine-non-cuda nixl
