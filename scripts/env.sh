#!/usr/bin/env bash
# Common environment for the M* vs vLLM-Omni BAGEL benchmarks on Intel Arc Pro B70.
# Usage: source scripts/env.sh
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WS
export MSTAR_VENV="$WS/.venv"
export OMNI_VENV="${OMNI_VENV:-$HOME/workspaces/.venvs/vllm-omni-main}"   # vLLM 0.34.0+xpu + vllm-omni main; old env: vllm-omni-v0.32.0
export FFMPEG_HOME="${FFMPEG_HOME:-$HOME/workspaces/.native/ffmpeg}"
export MODEL_ID="ByteDance-Seed/BAGEL-7B-MoT"

export HF_HUB_OFFLINE=1
export HF_HUB_CACHE="$WS/models/cache"

# --- OpenCL / oneDNN fix --------------------------------------------------------------
# $FFMPEG_HOME/lib (a conda env) ships its own libOpenCL.so.1 (conda `ocl-icd`). With that
# directory on LD_LIBRARY_PATH it shadows the system ICD loader and only reads
# $FFMPEG_HOME/etc/OpenCL/vendors (empty), so clinfo sees 0 platforms and oneDNN fails with
#   "Intel(R) Arc(TM) Pro B70 Graphics OpenCL device not found ... could not create a primitive"
# on every torch SDPA call. Point the loader at the system vendor directory instead.
# OCL_FIX=0 reproduces the broken state for diagnostics only.
if [[ "${OCL_FIX:-1}" == 1 ]]; then export OCL_ICD_VENDORS=/etc/OpenCL/vendors; else unset OCL_ICD_VENDORS; fi

export PATH="$FFMPEG_HOME/bin:$PATH"

# Fail early if this shell cannot open the render nodes (new SSH/VS Code sessions have the group).
if ! id -nG | tr ' ' '\n' | grep -qx render; then
  echo "env.sh: current shell is not in the 'render' group; re-login or wrap commands in 'sg render -c ...'" >&2
fi
