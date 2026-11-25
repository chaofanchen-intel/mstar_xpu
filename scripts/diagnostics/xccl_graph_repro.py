"""Minimal repro: can an XCCL all_reduce be captured into an XPU graph?  torchrun --nproc_per_node 2 xccl_graph_repro.py <variant>
variants: dist (dist.all_reduce) | func (c10d_functional.all_reduce_) | func_compiled (torch.compile'd func) | dist_sidestream"""
import os, sys, torch, torch.distributed as dist
variant = sys.argv[1] if len(sys.argv) > 1 else "dist"
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.xpu.set_device(rank)
dist.init_process_group("xccl", device_id=torch.device("xpu", rank))
dev = torch.device("xpu", rank)
rows = int(os.environ.get("ROWS", "1"))
x = torch.ones(rows, 3584, device=dev, dtype=torch.bfloat16)

def step(t):
    y = t * 2
    if variant == "compiled":
        dist.all_reduce(y)
    elif variant.startswith("func"):
        y = torch.ops._c10d_functional.all_reduce_.default(y, "sum", dist.group.WORLD.group_name)
        y = torch.ops._c10d_functional.wait_tensor.default(y)
    else:
        dist.all_reduce(y)
    return y + 1

fn = torch.compile(step) if variant in ("compiled", "compiled_nowarm") else step
for _ in range(0 if variant == "compiled_nowarm" else 3): fn(x)
torch.xpu.synchronize()
g = torch.xpu.XPUGraph()
inp = x.clone()
try:
    if variant == "dist_sidestream":
        s = torch.xpu.Stream()
        with torch.xpu.stream(s), torch.xpu.graph(g, stream=s):
            out = fn(inp)
    else:
        with torch.xpu.graph(g):
            out = fn(inp)
    inp.fill_(3); g.replay(); torch.xpu.synchronize()
    exp = (3 * 2 * world) + 1
    print(f"rank{rank} {variant}: CAPTURE OK, replay={out[0,0].item()} expected={exp}", flush=True)
except Exception as e:
    print(f"rank{rank} {variant}: CAPTURE FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
if variant == "dist" and os.environ.get("POISON"):
    big = torch.ones(2048, 3584, device=dev, dtype=torch.bfloat16); small = torch.ones(1, 3584, device=dev, dtype=torch.bfloat16)
    for name, t in (("big", big), ("small", small)):
        for _ in range(2): step(t)
        torch.xpu.synchronize(); g2 = torch.xpu.XPUGraph()
        try:
            with torch.xpu.graph(g2): o = step(t)
            g2.replay(); torch.xpu.synchronize(); print(f"rank{rank} poison-test {name}: OK", flush=True)
        except Exception as e:
            print(f"rank{rank} poison-test {name}: FAILED {str(e)[:60]}", flush=True)
dist.destroy_process_group()
