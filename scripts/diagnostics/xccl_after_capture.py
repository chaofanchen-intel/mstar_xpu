"""After a successful small-message all_reduce capture: do eager collectives / synchronize still work?
torchrun --nproc_per_node 2 xccl_after_capture.py"""
import os, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); torch.xpu.set_device(rank)
dist.init_process_group("xccl", device_id=torch.device("xpu", rank)); dev = torch.device("xpu", rank)
small = torch.ones(1, 3584, device=dev, dtype=torch.bfloat16)
def step(t):
    y = t * 2; dist.all_reduce(y); return y + 1
def log(m): print(f"rank{rank} {m}", flush=True)
for _ in range(2): step(small)
torch.xpu.synchronize(); log("warm ok")
g = torch.xpu.XPUGraph(); inp = small.clone()
with torch.xpu.graph(g): out = step(inp)
log("capture ok")
try: torch.xpu.synchronize(); log("sync after capture ok")
except Exception as e: log(f"sync after capture FAILED: {e}")
try: g.replay(); torch.xpu.synchronize(); log(f"replay ok {out[0,0].item()}")
except Exception as e: log(f"replay FAILED: {e}")
try: step(small); torch.xpu.synchronize(); log("eager small ok")
except Exception as e: log(f"eager small FAILED: {e}")
try: step(torch.ones(2048, 3584, device=dev, dtype=torch.bfloat16)); torch.xpu.synchronize(); log("eager big ok")
except Exception as e: log(f"eager big FAILED: {e}")
dist.destroy_process_group()
