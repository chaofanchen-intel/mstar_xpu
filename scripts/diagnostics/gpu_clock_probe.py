"""Probe one XPU: big-GEMM throughput and a launch-bound small-op loop, and report the GPU clock seen
by xpu-smi while each phase runs. Usage: ZE_AFFINITY_MASK=<phys id> python gpu_clock_probe.py <phys id>"""
import subprocess, sys, threading, time
import torch

phys = sys.argv[1]
dev = torch.device("xpu", 0)
freqs = []; stop = False

def poll():
    while not stop:
        out = subprocess.run(["xpu-smi", "stats", "-d", phys], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "GPU Frequency" in line:
                freqs.append(int(line.split("|")[2].strip()))
        time.sleep(0.4)

def phase(name, fn):
    global stop
    freqs.clear(); stop = False
    t = threading.Thread(target=poll); t.start()
    res = fn()
    stop = True; t.join()
    print(f"xpu{phys} {name}: {res}  GPU clock seen (MHz) min/median/max = "
          f"{min(freqs or [-1])}/{sorted(freqs or [-1])[len(freqs)//2]}/{max(freqs or [-1])} (n={len(freqs)})", flush=True)

a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16); b = torch.randn_like(a)
for _ in range(3): a @ b
torch.xpu.synchronize()

def gemm():
    n = 1500; t = time.perf_counter()
    for _ in range(n): a @ b
    torch.xpu.synchronize(); dt = time.perf_counter() - t
    return f"{2 * 8192**3 * n / dt / 1e12:.1f} TFLOPS bf16"

x = torch.randn(1, 4096, device=dev, dtype=torch.bfloat16); w = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
def launch_bound():
    n = 40000; t = time.perf_counter()
    for _ in range(n):
        y = x @ w; y = torch.nn.functional.silu(y); y = y + x
    torch.xpu.synchronize(); dt = time.perf_counter() - t
    return f"{n * 3 / dt:.0f} small-ops/s"

phase("big GEMM   ", gemm)
phase("launch-bound", launch_bound)
phase("launch-bound (2nd, warm)", launch_bound)
