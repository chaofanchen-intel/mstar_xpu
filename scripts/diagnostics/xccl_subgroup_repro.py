"""Is the failure due to a lazily-initialised sub-communicator?  torchrun --nproc_per_node 3 xccl_subgroup_repro.py <warm|nowarm>
World of 3, TP group (1,2) like M* bagel_tp2_3gpu; ranks 1,2 capture an all_reduce on the sub-group."""
import os, sys, torch, torch.distributed as dist
mode = sys.argv[1]
rank = int(os.environ["RANK"])
torch.xpu.set_device(rank)
import threading
composite = "composite" in mode
dist.init_process_group("cpu:gloo,xpu:xccl" if composite else "xccl", device_id=torch.device("xpu", rank))
dev = torch.device("xpu", rank)
tp = dist.new_group([1, 2])
if rank in (1, 2):
    x = torch.ones(1, 3584, device=dev, dtype=torch.bfloat16)
    if mode == "warm":
        dist.all_reduce(x.clone(), group=tp); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph(); inp = x.clone()
    def cap():
        try:
            kw = {}
            if "pool" in mode:
                kw["pool"] = torch.xpu.graph_pool_handle()
            if "stream" in mode:
                kw["stream"] = torch.xpu.Stream()
            with torch.xpu.graph(g, **kw):
                y = inp * 2; dist.all_reduce(y, group=tp); res["out"] = y + 1
            inp.fill_(3); g.replay(); torch.xpu.synchronize()
            print(f"rank{rank} {mode}: CAPTURE OK replay={res['out'][0,0].item()} expected=13", flush=True)
        except Exception as e:
            print(f"rank{rank} {mode}: CAPTURE FAILED: {str(e)[:160]}", flush=True)
    res = {}
    if "thread" in mode:
        t = threading.Thread(target=cap); t.start(); t.join()
    else:
        cap()
dist.barrier()
dist.destroy_process_group()
