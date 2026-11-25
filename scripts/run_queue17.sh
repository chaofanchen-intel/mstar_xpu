#!/usr/bin/env bash
# M* per-phase timing on the diffusion cases (T2I, I2I), eager, 3-card CFG parallel.
cd "$(dirname "$0")/.."
R=results/20261123_graph; P="Replace the bagel with a croissant."
MSTAR_PHASE_TIMING=25 scripts/run_one.sh $R mstar-phase-diff mstar configs/mstar/bagel_cfgpar_3gpu.yaml 3 t2i,i2i --mstar-i2i-preprocess default --i2i-prompt "$P" > $R/mstar-phase-diff.driver.log 2>&1
echo done > $R/queue17.done
