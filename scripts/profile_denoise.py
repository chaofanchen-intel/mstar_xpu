"""Record a kernel-level torch.profiler window inside the denoise loop of one T2I (or I2I) request.
Usage: profile_denoise.py <mstar|omni> <url> <profile_dir> [--case t2i|i2i] [--at 2.0] [--window 1.5]
The server must run with BENCH_TORCH_PROFILE_DIR=<profile_dir> and scripts/diagnostics/hooks on PYTHONPATH."""
import argparse, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import bench_bagel as B
import requests

p = argparse.ArgumentParser()
p.add_argument("framework"); p.add_argument("url"); p.add_argument("pdir", type=Path)
p.add_argument("--case", default="t2i"); p.add_argument("--at", type=float, default=2.0)
p.add_argument("--window", type=float, default=1.5); p.add_argument("--warmups", type=int, default=2)
p.add_argument("--i2i-prompt", default="Replace the bagel with a croissant.")
a = p.parse_args()
B.CASES["i2i"] = dict(B.CASES["i2i"], prompt=a.i2i_prompt)
ns = argparse.Namespace(framework=a.framework, url=a.url, size=512, steps=50, omni_prompt_style="bagel_raw",
                        omni_ignore_eos=False, omni_i2i_size=0, mstar_i2t_preprocess="default",
                        mstar_i2i_preprocess="default")
case = B.CASES[a.case]
image = B.DEFAULT_IMAGE.read_bytes() if a.case == "i2i" else None
build = B.mstar_request if a.framework == "mstar" else B.omni_request
url, kw, _ = build(ns, case, a.case, image)
for _ in range(a.warmups):
    t = time.time(); requests.post(url, timeout=900, **kw).raise_for_status(); print(f"warmup {time.time()-t:.2f}s", flush=True)
for f in a.pdir.glob("*"): f.unlink()
res = {}
def go():
    t = time.time(); r = requests.post(url, timeout=900, **kw); r.raise_for_status(); res["s"] = time.time() - t
th = threading.Thread(target=go); th.start()
time.sleep(a.at); (a.pdir / "START").write_text(str(time.time()))
time.sleep(a.window); (a.pdir / "STOP").write_text(str(time.time()))
th.join(); print(f"profiled request {res['s']:.2f}s", flush=True)
time.sleep(30)  # let the workers export their traces
print(sorted(f.name for f in a.pdir.glob("*")))
