#!/usr/bin/env bash
# Diagnostic queue: separate the effects of (OpenCL fix / SDPA backend / card pair).
cd "$(dirname "$0")/.."
R=results/20261122_rerun
C="t2i,i2t_1tok"; A="--repeats 3"
OCL_FIX=0 scripts/run_one.sh $R diag-omni-single-no-ocl-fix omni configs/vllm_omni/bagel_single_stage_tp2.yaml 2 $C $A > $R/diag-omni-single-no-ocl-fix.driver.log 2>&1
MATH_SDPA=1 scripts/run_one.sh $R diag-omni-single-mathsdpa omni configs/vllm_omni/bagel_single_stage_tp2.yaml 2 $C $A > $R/diag-omni-single-mathsdpa.driver.log 2>&1
MATH_SDPA=1 scripts/run_one.sh $R diag-mstar-mathsdpa mstar configs/mstar/bagel_tp2_2gpu.yaml 2 $C $A > $R/diag-mstar-mathsdpa.driver.log 2>&1
echo done > $R/queue2.done
