"""2-rank all-reduce latency/bandwidth between two XPUs (the TP2 communication path).
Usage: ZE_AFFINITY_MASK=a,b torchrun --nproc_per_node 2 allreduce_probe.py"""
import os, time
import torch, torch.distributed as dist

rank = int(os.environ["RANK"]); torch.xpu.set_device(rank)
dist.init_process_group("xccl")
dev = torch.device("xpu", rank)
for numel, label in [(4096 * 1, "8 KB (decode-like)"), (1024 * 3584, "7 MB (512x512 image-token activations)")]:
    x = torch.ones(numel, dtype=torch.bfloat16, device=dev)
    for _ in range(20): dist.all_reduce(x)
    torch.xpu.synchronize(); n = 300; t = time.perf_counter()
    for _ in range(n): dist.all_reduce(x)
    torch.xpu.synchronize(); dt = (time.perf_counter() - t) / n
    if rank == 0:
        print(f"allreduce {label:42s}: {dt * 1e3:7.3f} ms/op  ({numel * 2 / dt / 1e9:6.2f} GB/s algbw)", flush=True)
dist.destroy_process_group()
