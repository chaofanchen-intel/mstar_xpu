import torch, torch.nn.functional as F, sys
dev = torch.device("xpu", 0)
# ViT-like shape (head_dim 72, 16 heads) with a block-diagonal bool mask, and a VAE-like (1 head, dim 512)
for name, (h, s, d, masked) in {"vit_h72_mask": (16, 1024, 72, True), "vae_h1_d512": (1, 1024, 512, False)}.items():
    q = torch.randn(1, h, s, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    mask = torch.ones(s, s, dtype=torch.bool, device=dev).tril() if masked else None
    try:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask); torch.xpu.synchronize()
        print(name, "OK", tuple(o.shape))
    except Exception as e:
        print(name, "FAIL", str(e)[:120])
