bagel_tp2_2gpu.yaml  — M* BAGEL, LLM TP2 over 2 ranks; ViT/VAE co-located with rank 0 (eager, xpu_paged attention).
                       Same layout as the upstream external/mstar/configs/bagel_xpu_tp2.yaml except the
                       ViT/VAE group shares rank 0 instead of using a third GPU (so both frameworks use 2 cards).
