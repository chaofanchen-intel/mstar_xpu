#!/usr/bin/env python3
"""Serial end-to-end BAGEL client that drives M* and vLLM-Omni with the same workload.

One invocation runs a list of cases against ONE running server and appends to
``<out>/timings.csv``; it also writes ``<out>/manifest.json`` (the exact request parameters),
one sample output per case and a ``summary.json``.

Workload definition (identical for both frameworks, see docs for the audit):

  t2i      512x512, 50 timesteps (49 denoising updates), timestep_shift 3.0,
           CFG text 4.0 / img 1.0, cfg_interval [0.4, 1.0], renorm "global"
           -> 1 branch outside the interval, 2 inside.
  i2t_*    one input image + question; greedy; N new tokens (N=1 isolates ViT+prefill, N=64 adds decode).
  i2i      same input image, 512x512 output, 50 timesteps, CFG text 4.0 / img 2.0, interval [0, 1],
           renorm "text_channel" -> 3 branches at every step.

Input-image preprocessing (checked against Omni's reported usage.prompt_tokens and the source):
  * vLLM-Omni main sizes the ViT input from the input image: 512x512 -> 518x518 = 1369 ViT patches
    (the same for I2T and I2I; the VAE side is 512x512 = 1024 latents). I2T prompt = 1386 tokens.
    M*'s *default* preprocessing gives the same 1369 patches and 1024 latents, so M* uses the default
    for both I2T and I2I (``--mstar-i2t-preprocess`` / ``--mstar-i2i-preprocess``, default "default").
  * vLLM-Omni v0.32.0 used a fixed 980x980 ViT (4900 patches) for both; M* ``image_preprocess:"vllm"``
    reproduces that and is only needed for v0.32 comparisons. Early I2I runs on main used "vllm" by mistake.
"""

from __future__ import annotations

import argparse
import base64
import io
import csv
import json
import statistics
import struct
import sys
import time
from pathlib import Path

import requests

WS = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = WS / "assets" / "bagel_input_512.png"
MODEL_ID = "ByteDance-Seed/BAGEL-7B-MoT"
TOKENIZER_DIR_GLOB = "models/cache/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/*"

T2I_PROMPT = "A red ceramic cup on a white table"
I2I_PROMPT = "Add a small cat sitting beside the bagel on the plate."
I2T_PROMPT_SHORT = "Please describe the food in this image in one concise sentence."
I2T_PROMPT_LONG = "Please describe this image in great detail, covering the food, the plate and the background."

CASES = {
    "t2i": dict(kind="t2i", prompt=T2I_PROMPT),
    "i2t_1tok": dict(kind="i2t", prompt=I2T_PROMPT_SHORT, new_tokens=1),
    "i2t_64tok": dict(kind="i2t", prompt=I2T_PROMPT_LONG, new_tokens=64),
    "i2i": dict(kind="i2i", prompt=I2I_PROMPT),
}

CFG = {
    "t2i": dict(cfg_text_scale=4.0, cfg_img_scale=1.0, cfg_interval=[0.4, 1.0],
                cfg_renorm_type="global", cfg_renorm_min=0.0),
    "i2i": dict(cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0.0, 1.0],
                cfg_renorm_type="text_channel", cfg_renorm_min=0.0),
}
# Fixed in M* (BagelModelConfig.num_timesteps / timestep_shift); sent explicitly to Omni.
MSTAR_STEPS, MSTAR_SHIFT = 50, 3.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framework", choices=("mstar", "omni"), required=True)
    p.add_argument("--mode", required=True, help="topology label stored in the results, e.g. mstar-tp2")
    p.add_argument("--url", required=True)
    p.add_argument("--cases", default="t2i,i2t_1tok,i2t_64tok,i2i")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--size", type=int, default=512, help="square output size for image generation")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--omni-prompt-style", choices=("bagel_raw", "chat"), default="bagel_raw",
                   help="bagel_raw: BAGEL's own <|im_start|> template (single-stage); "
                        "chat: plain user text, the server applies the chat template (two-stage Thinker)")
    p.add_argument("--omni-ignore-eos", action="store_true",
                   help="send ignore_eos for I2T (supported by the vLLM AR Thinker, not by single-stage)")
    p.add_argument("--mstar-i2t-preprocess", choices=("default", "vllm"), default="default",
                   help="M* ViT preprocessing for I2T; default = 1369 patches = what the Omni Thinker sees")
    p.add_argument("--mstar-i2i-preprocess", choices=("default", "vllm"), default="default",
                   help="M* ViT preprocessing for I2I; default = 1369 patches = vLLM-Omni main at 512x512")
    p.add_argument("--omni-i2i-size", type=int, default=0,
                   help="I2I only: upscale the input image to SIZE x SIZE before sending to vLLM-Omni. "
                        "At 980 the ViT gets 70x70 = 4900 patches while the VAE stays 512^2 (1024 latents), "
                        "the same token count as M*'s image_preprocess:'vllm'. 0 = send the 512^2 image.")
    p.add_argument("--i2i-prompt", default=None,
                   help="override the I2I edit instruction (same text for both frameworks); default: the original cat prompt")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.cases = [c for c in a.cases.split(",") if c]
    if a.i2i_prompt:
        CASES["i2i"] = dict(CASES["i2i"], prompt=a.i2i_prompt)
    for c in a.cases:
        if c not in CASES:
            p.error(f"unknown case {c}")
    if a.framework == "mstar" and a.steps != MSTAR_STEPS:
        p.error(f"M* BAGEL has a fixed {MSTAR_STEPS}-step schedule; use --steps {MSTAR_STEPS}")
    a.out = a.out.resolve()
    return a


# ----------------------------------------------------------------------------- request builders
def mstar_request(a, case: dict, name: str, image: bytes | None):
    kind = case["kind"]
    if kind == "i2t":
        kw = {"max_output_tokens": case["new_tokens"], "temperature": 0.0, "top_p": 1.0,
              "ignore_eos": True, "image_preprocess": a.mstar_i2t_preprocess}
    else:
        kw = {"width": a.size, "height": a.size, **CFG[kind]}
        if kind == "i2i":
            kw["image_preprocess"] = a.mstar_i2i_preprocess
    data = {"text": case["prompt"], "model_kwargs": json.dumps(kw)}
    if kind in ("t2i", "i2i"):
        data["output_modalities"] = "image"
    files = {"files": (a.image.name, image, "application/octet-stream")} if image else None
    return f"{a.url}/generate", {"data": data, "files": files}, {"model_kwargs": kw, "output_modalities": data.get("output_modalities", "text")}


def omni_text(a, case: dict) -> str:
    if a.omni_prompt_style == "chat":
        return case["prompt"]
    if case["kind"] == "i2t":
        return f"<|im_start|>user\n<|image_pad|>\n{case['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>{case['prompt']}<|im_end|>"


def omni_request(a, case: dict, name: str, image: bytes | None):
    kind = case["kind"]
    if image and kind == "i2i" and a.omni_i2i_size:
        from PIL import Image
        buf = io.BytesIO()
        Image.open(io.BytesIO(image)).convert("RGB").resize((a.omni_i2i_size,) * 2, Image.BICUBIC).save(buf, "PNG")
        image = buf.getvalue()
    content = [{"type": "text", "text": omni_text(a, case)}]
    if image:
        b64 = base64.b64encode(image).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": content}],
            "modalities": ["text" if kind == "i2t" else "image"]}
    if kind == "i2t":
        n = case["new_tokens"]
        body.update(max_tokens=n, temperature=0.0, top_p=1.0,
                    # single-stage BAGEL ignores max_tokens and uses max_think_tokens (BOS takes one slot)
                    extra_body={"max_think_tokens": n + 1})
        if a.omni_ignore_eos:
            body["ignore_eos"] = True
    else:
        body.update(height=a.size, width=a.size, num_inference_steps=a.steps, timestep_shift=MSTAR_SHIFT,
                    seed=42, **CFG[kind])
    shown = {k: v for k, v in body.items() if k != "messages"}
    shown["text_content"] = content[0]["text"]
    return f"{a.url}/v1/chat/completions", {"json": body}, shown


# ----------------------------------------------------------------------------- response decoding
def decode_mstar(raw: bytes):
    text, img, extra = [], [], {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("data"):
            (text if ev.get("modality") == "text" else img).append(base64.b64decode(ev["data"]))
    return b"".join(text).decode("utf-8", errors="replace"), b"".join(img), extra


def decode_omni(raw: bytes):
    payload = json.loads(raw)
    msg = payload["choices"][0]["message"]
    content = msg.get("content") or []
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    text, img = [], []
    for item in content:
        if item.get("type") == "text":
            text.append(item.get("text", ""))
        elif item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url", "")
            if url.startswith("data:") and "," in url:
                img.append(base64.b64decode(url.split(",", 1)[1]))
    metrics = payload.get("metrics") or {}
    text_s = "".join(text) or (metrics.get("text_output") or "")
    usage = payload.get("usage") or {}
    return text_s, b"".join(img), {"usage_completion_tokens": usage.get("completion_tokens"),
                                   "usage_prompt_tokens": usage.get("prompt_tokens")}


def png_size(data: bytes):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", data[16:24])  # (width, height)


_tok = None


def count_tokens(text: str) -> int | None:
    """Tokens in ``text`` with the BAGEL (Qwen2) tokenizer; None if the tokenizer is unavailable."""
    global _tok
    try:
        if _tok is None:
            from transformers import AutoTokenizer
            snap = next(WS.glob(TOKENIZER_DIR_GLOB))
            _tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        return len(_tok.encode(text, add_special_tokens=False))
    except Exception:
        return None


# ----------------------------------------------------------------------------- main loop
def run_case(a, name: str, writer, f) -> dict:
    case = CASES[name]
    kind = case["kind"]
    image = a.image.read_bytes() if kind in ("i2t", "i2i") else None
    build = mstar_request if a.framework == "mstar" else omni_request
    url, kwargs, shown = build(a, case, name, image)
    decode = decode_mstar if a.framework == "mstar" else decode_omni
    manifest = {"framework": a.framework, "mode": a.mode, "case": name, "kind": kind, "url": url,
                "prompt": case["prompt"], "input_image": str(a.image) if image else None,
                "input_image_bytes": len(image) if image else None, "request": shown,
                "warmups": a.warmups, "repeats": a.repeats}
    if a.dry_run:
        print(json.dumps(manifest, indent=2))
        return manifest
    samples = []
    for i in range(a.warmups + a.repeats):
        label = f"warmup{i + 1}" if i < a.warmups else str(i - a.warmups + 1)
        t0 = time.perf_counter()
        r = requests.post(url, timeout=a.timeout, **kwargs)
        raw = r.content
        dt = time.perf_counter() - t0
        r.raise_for_status()
        text, img, extra = decode(raw)
        row = {"framework": a.framework, "mode": a.mode, "case": name, "run": label, "seconds": f"{dt:.4f}",
               "http_status": r.status_code, "response_bytes": len(raw), "out_image": "", "out_tokens": "",
               "out_text_chars": len(text)}
        if kind in ("t2i", "i2i"):
            wh = png_size(img)
            if wh is None:
                raise RuntimeError(f"{name}: response has no valid PNG")
            row["out_image"] = f"{wh[0]}x{wh[1]}"
            if wh != (a.size, a.size):
                raise RuntimeError(f"{name}: output {wh} != requested {a.size}x{a.size}")
        else:
            if not text.strip():
                raise RuntimeError(f"{name}: empty text output")
            n_tok = extra.get("usage_completion_tokens") or count_tokens(text)
            row["out_tokens"] = n_tok if n_tok is not None else ""
        row["prompt_tokens"] = extra.get("usage_prompt_tokens") or ""   # Omni reports it; M* does not
        if i == a.warmups:  # keep one sample output per case
            if img:
                (a.out / f"sample_{a.mode}_{name}.png").write_bytes(img)
            if text:
                (a.out / f"sample_{a.mode}_{name}.txt").write_text(text)
        writer.writerow(row)
        f.flush()
        print(f"[{a.mode}] {name:10s} {label:8s} {dt:8.3f}s  {row['out_image'] or str(row['out_tokens']) + ' tok'}", flush=True)
        if i >= a.warmups:
            samples.append((dt, row))
    secs = [s for s, _ in samples]
    manifest["result"] = {"median_s": statistics.median(secs), "min_s": min(secs), "max_s": max(secs),
                          "stdev_s": statistics.stdev(secs) if len(secs) > 1 else 0.0, "n": len(secs),
                          "out_tokens": sorted({str(r["out_tokens"]) for _, r in samples})}
    print(f"[{a.mode}] {name} median {manifest['result']['median_s']:.3f}s "
          f"(min {min(secs):.3f}, max {max(secs):.3f})", flush=True)
    return manifest


def main() -> int:
    a = parse_args()
    if any(CASES[c]["kind"] != "t2i" for c in a.cases) and not a.image.is_file():
        sys.exit(f"input image missing: {a.image}")
    a.out.mkdir(parents=True, exist_ok=True)
    results = {}
    path = a.out / "timings.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["framework", "mode", "case", "run", "seconds", "http_status",
                                          "response_bytes", "out_image", "out_tokens", "out_text_chars", "prompt_tokens"])
        if new:
            w.writeheader()
        for name in a.cases:
            results[name] = run_case(a, name, w, f)
    if not a.dry_run:
        mpath = a.out / "manifest.json"
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        manifest.setdefault(a.mode, {}).update(results)
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
