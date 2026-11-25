#!/usr/bin/env python3
"""One-off derivation of the support-scope and model-integration chapters from the reference deck
(~/workspaces/share/omni_runtime_report_v2/chapters/05b_support.tex, 05c_model_integration.tex):
drop the SGLang-Omni column from every table and every SGLang-only frame. The derived .tex files are then
edited by hand (this script is kept for traceability only)."""
import re, sys
from pathlib import Path

REF = Path.home() / "workspaces/share/omni_runtime_report_v2/chapters"
OUT = Path(__file__).resolve().parents[1] / "chapters"

def split_frames(text):
    head_end = text.index("\\begin{frame}")
    head = text[:head_end]
    parts = re.split(r"(?=\\begin\{frame\})", text[head_end:])
    return head, [p for p in parts if p.strip()]

def drop_last_column(frame):
    lines = frame.split("\n")
    out, in_tab, strip = [], False, False
    for ln in lines:
        if ln.startswith("\\begin{tabularx}{\\textwidth}{") and not in_tab:
            # decide from the header row two lines below whether this table has a SGLang column
            idx = lines.index(ln)
            hdr = " ".join(lines[idx + 1: idx + 5])
            strip = "SGLang" in hdr.split("\\midrule")[0] and re.search(r"\sY\s*}$", ln.rstrip()) is not None
            if strip:
                ln = re.sub(r"\s*Y\s*}$", "}", ln.rstrip())
            in_tab = True
        elif ln.startswith("\\end{tabularx}"):
            in_tab, strip = False, False
        elif in_tab and strip and " & " in ln and ln.rstrip().endswith("\\\\"):
            body = ln.rstrip()[:-2]
            ln = body[: body.rfind(" & ")] + " \\\\"
        out.append(ln)
    return "\n".join(out)

for name, outname in [("05b_support.tex", "01_support.tex"), ("05c_model_integration.tex", "02_integration.tex")]:
    text = (REF / name).read_text()
    head, frames = split_frames(text)
    keep = []
    for f in frames:
        title = re.search(r"\\begin\{frame\}(?:\[[^\]]*\])?\{([^\n]*)\}", f).group(1)
        if title.startswith("SGLang-Omni"):
            continue
        keep.append(drop_last_column(f))
    (OUT / outname).write_text(head + "".join(keep))
    print(outname, len(keep), "frames kept of", len(frames))
