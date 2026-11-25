"""all_gather capture on XPU by message size / API.  ROWS=<n> torchrun --nproc_per_node 2 xccl_gather_repro.py <dist|func>"""
import os, sys, torch, torch.distributed as dist
variant = sys.argv[1]; rows = int(os.environ.get("ROWS", "1")); cols = int(os.environ.get("COLS", "76032"))
rank = int(os.environ["RANK"]); torch.xpu.set_device(rank)
dist.init_process_group("xccl", device_id=torch.device("xpu", rank)); dev = torch.device("xpu", rank)
x = torch.ones(rows, cols, device=dev, dtype=torch.bfloat16)
def step(t):
    if variant == "func":
        y = torch.ops._c10d_functional.all_gather_into_tensor.default(t, 2, dist.group.WORLD.group_name)
        return torch.ops._c10d_functional.wait_tensor.default(y)
    out = torch.empty(2 * rows, cols, device=dev, dtype=t.dtype)
    dist.all_gather_into_tensor(out, t); return out
for _ in range(3): step(x)
torch.xpu.synchronize()
g = torch.xpu.XPUGraph()
try:
    with torch.xpu.graph(g): o = step(x)
    g.replay(); torch.xpu.synchronize()
    print(f"rank{rank} {variant} rows={rows} cols={cols} ({rows*cols*2//1024} KiB/rank): CAPTURE OK", flush=True)
except Exception as e:
    print(f"rank{rank} {variant} rows={rows} cols={cols} ({rows*cols*2//1024} KiB/rank): CAPTURE FAILED: {str(e)[:90]}", flush=True)
