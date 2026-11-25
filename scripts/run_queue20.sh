#!/usr/bin/env bash
# M* TP2 (3 cards) graph vs eager with the XCCL capture workarounds (all_gather out of compile, capture sizes <= 4 MiB messages).
cd "$(dirname "$0")/.."
R=results/20261124_tp2graph
scripts/run_one.sh $R mstar-tp2-graph-fixed mstar configs/mstar/bagel_tp2_3gpu_graph.yaml 3 i2t_1tok,i2t_64tok,t2i > $R/mstar-tp2-graph-fixed.driver.log 2>&1
scripts/run_one.sh $R mstar-tp2-eager-rerun mstar configs/mstar/bagel_tp2_3gpu.yaml 3 i2t_1tok,i2t_64tok,t2i > $R/mstar-tp2-eager-rerun.driver.log 2>&1
echo done > $R/queue20.done
