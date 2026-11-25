"""Diagnostic hooks for A/B runs; inactive unless an env var below is "1".
  BENCH_DISABLE_COMPILE=1   torch.compile becomes a no-op            (M* default compiles LLM/ViT/VAE)
  BENCH_FORCE_MATH_SDPA=1   F.scaled_dot_product_attention runs under sdpa_kernel(MATH)
Enabled through scripts/run_one.sh (NO_COMPILE=1, MATH_SDPA=1); the hook dir is put on PYTHONPATH so the
whole server process tree (spawned workers included) picks it up. Never used for the headline numbers.
"""
import os

import torch

if os.environ.get("BENCH_DISABLE_COMPILE") == "1":
    def _noop_compile(function=None, *args, **kwargs):
        return (lambda f: f) if function is None else function
    torch.compile = _noop_compile

if os.environ.get("BENCH_FORCE_MATH_SDPA") == "1":
    import torch.nn.functional as _F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _orig_sdpa = _F.scaled_dot_product_attention

    def _math_sdpa(*args, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            return _orig_sdpa(*args, **kwargs)
    _F.scaled_dot_product_attention = _math_sdpa

# BENCH_TORCH_PROFILE_DIR=<dir>: every process that has initialised an XPU device records a torch.profiler
# (CPU+XPU) window between the files <dir>/START and <dir>/STOP and writes <dir>/trace_<pid>.json.gz.
# Driven by scripts/profile_denoise.py; the same hook serves M* and vLLM-Omni so the traces are comparable.
_PDIR = os.environ.get("BENCH_TORCH_PROFILE_DIR")
if _PDIR:
    import threading
    import time

    def _profile_poller():
        from torch.profiler import ProfilerActivity, profile
        start, stop = os.path.join(_PDIR, "START"), os.path.join(_PDIR, "STOP")
        prof, t_start = None, 0.0
        while True:
            time.sleep(0.05)
            try:
                if prof is None and os.path.exists(start) and not os.path.exists(stop):
                    if not (hasattr(torch, "xpu") and torch.xpu.is_initialized()):
                        continue
                    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU])
                    prof.__enter__()
                    t_start = time.time()
                elif prof is not None and os.path.exists(stop):
                    prof.__exit__(None, None, None)
                    prof.export_chrome_trace(os.path.join(_PDIR, f"trace_{os.getpid()}.json.gz"))
                    prof, _ = None, open(os.path.join(_PDIR, f"done_{os.getpid()}"), "w").write(f"{t_start} {time.time()}")
                    return
            except Exception as e:  # never take the server down
                open(os.path.join(_PDIR, f"error_{os.getpid()}"), "w").write(repr(e))
                return
    threading.Thread(target=_profile_poller, daemon=True, name="bench-profile").start()
