# M* vs vLLM-Omni — BAGEL-7B-MoT on Intel Arc Pro B70

Host: 3 × Arc Pro B70 (32 GiB), shared with other users. Both frameworks run natively (no Docker).

```
mstar_xpu/
├── README.md                    this file
├── report/                      汇报文档（Beamer 幻灯片，main.pdf）：支持范围、接入工作量、BAGEL B70 实测；bash report/build.sh
├── docs/
│   ├── BAGEL_B70_BEST_CONFIG_COMPARISON.md best-config comparison (current headline, 2026-11-22)
│   ├── VLLM_OMNI_BAGEL_B70_VALIDATION.md   same-default-config comparison + audit of the initial run + OpenCL fix
│   └── BAGEL_XPU_HOST_VALIDATION.md        M* host setup record (2026-11-18)
├── configs/
│   ├── mstar/bagel_{tp2,cfgpar}_2gpu{,_graph}.yaml   M*: TP2 or CFG-parallel (TP1 per card), eager or XPU graph
│   └── vllm_omni/bagel_{single_stage_tp2,single_stage_cfgpar2,two_stage_2gpu,two_stage_2gpu_graph}.yaml   (@GPU…@ placeholders)
├── scripts/
│   ├── env.sh                   common env, includes the OpenCL fix (OCL_ICD_VENDORS)
│   ├── gpu_guard.sh             prints idle XPU ids; refuses (exit 1) if fewer than N are idle
│   ├── serve_mstar.sh / serve_omni.sh / wait_ready.sh / stop_server.sh
│   ├── run_one.sh               guard → serve → wait → benchmark → stop (always), GPU timeline sampler
│   ├── bench_bagel.py           ONE client for both frameworks (same workload definition)
│   ├── make_report.py           timings.csv → Markdown tables
│   ├── make_tex_tables.py       timings.csv → report/generated/*.tex（报告里的所有数字）
│   ├── install_omni_env.sh      builds a vLLM-Omni XPU venv for a given vLLM release
│   ├── run_queue1..8.sh         the exact experiment queues (1,2: results/20261122_rerun; 3-6: results/20261122_best_cfg; 7: 20261122_factors connector/cache; 8: 20261122_i2t_parity)
│   └── diagnostics/             ocl_sdpa_smoke.py, allreduce_probe.py, gpu_clock_probe.py, hooks/ (A/B only)
├── assets/                      benchmark input image (512×512) and the original
├── results/
│   ├── 20261122_factors/          two-stage connectors (SharedMemory/Mooncake/NIXL), prefix cache
│   ├── 20261122_i2t_parity/       M* I2T with default preprocessing (1369 patches = Omni Thinker's 1386-token prompt)
│   ├── 20261122_best_cfg/{<candidate>,final/<label>}/   best-config search + final matrix (vLLM-Omni main, vLLM 0.34)
│   ├── 20261122_rerun/<label>/    same-default-config rerun (vLLM-Omni 0.32.0)  timings.csv, manifest.json, sample outputs, server.log, gpu_timeline.csv, effective_config.yaml
│   └── archive/20261118_initial/   first-pass data, configs and report (superseded, kept for reference)
├── external/{mstar,vllm-omni}   upstream checkouts (unmodified)
├── models -> ../models          compatibility symlink; canonical cache: ~/workspaces/models/cache
└── .venv                        M* environment  (vLLM-Omni envs: ~/workspaces/.venvs/vllm-omni-main = vLLM 0.34 [default], vllm-omni-v0.32.0 = old)
```

## Rules for running on this host
1. Never launch on a card that is not idle. `scripts/run_one.sh` calls `scripts/gpu_guard.sh` and exits if too few cards are idle.
2. Pass physical XPU ids. M* gets them through `ZE_AFFINITY_MASK`; vLLM-Omni overwrites `ZE_AFFINITY_MASK` from each stage's `devices`, so `serve_omni.sh` substitutes them into the deploy config.
3. Always `source scripts/env.sh` first — it exports `OCL_ICD_VENDORS=/etc/OpenCL/vendors` (see the report: without it oneDNN SDPA fails whenever `~/workspaces/.native/ffmpeg/lib` is on `LD_LIBRARY_PATH`).

## Notes
- `external/vllm-omni` is now detached at upstream `main` (needs vLLM 0.34). To go back to the 0.32 results: `git -C external/vllm-omni checkout v0.32.0` and `OMNI_VENV=~/workspaces/.venvs/vllm-omni-v0.32.0`.
- I2T workload parity: Omni Thinker sees 1386 prompt tokens (1369 ViT patches); M* must use default preprocessing for I2T (`--mstar-i2t-preprocess default`, the bench default), `vllm` (4900 patches) only for I2I. M* KV prefix cache is on by default, Omni's off: compare cache-off vs cache-off and on vs on.
- NIXL connector does not work on XPU (UCX cannot register XPU VRAM); Mooncake-TCP and SharedMemory do.
- Omni two-stage I2T must use `--omni-prompt-style chat` on vLLM-Omni main; two-stage I2I currently returns a blank image on main (regression), so it is excluded.

## Reproduce
```bash
cd ~/workspaces/mstar_xpu
scripts/run_one.sh results/<run> mstar-tp2 mstar configs/mstar/bagel_tp2_2gpu.yaml 2 t2i,i2t_1tok,i2t_64tok,i2i
scripts/run_one.sh results/<run> omni-single-tp2 omni configs/vllm_omni/bagel_single_stage_tp2.yaml 2 t2i,i2t_1tok,i2t_64tok,i2i
scripts/run_one.sh results/<run> omni-two-stage-2gpu omni configs/vllm_omni/bagel_two_stage_2gpu.yaml 2 t2i,i2t_1tok,i2t_64tok --omni-ignore-eos --omni-prompt-style chat
.venv/bin/python -I scripts/make_report.py results/<run>
```
A different card pair is picked automatically (first idle ones). Re-run when all three cards are idle to also measure the 3-card layouts (M* `external/mstar/configs/bagel_xpu_tp2.yaml`, Omni Thinker-TP2 + DiT).
