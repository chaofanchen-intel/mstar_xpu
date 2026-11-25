#!/usr/bin/env python3
"""Generate LaTeX macros and tables for report/ from results/*/<label>/timings.csv.

Writes report/generated/values.tex   \\val{<group>/<label>/<case>} -> median seconds (2-3 decimals)
                                     \\pct{<name>}                  -> derived percentage strings
       report/generated/tab_*.tex    ready-to-\\input tables (best cell per framework/case highlighted)
Invalid cells (wrong output) are never used for medians, they are printed as 'n/a'.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
RES = WS / "results"
OUT = WS / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
CASES = ["t2i", "i2t1", "i2t64", "i2i"]
CASE_CSV = {"t2i": "t2i", "i2t1": "i2t_1tok", "i2t64": "i2t_64tok", "i2i": "i2i"}
CASE_NAME = {"t2i": "T2I 512²·50 步", "i2t1": "I2T 1 token", "i2t64": "I2T 64 token", "i2i": "I2I 512²·50 步"}

# group -> (directory, software stack text)
GROUPS = {
    "final": (RES / "20261122_best_cfg" / "final", "vLLM-Omni main + vLLM 0.34"),
    "screen": (RES / "20261122_best_cfg", "vLLM-Omni main + vLLM 0.34"),
    "v032": (RES / "20261122_rerun", "vLLM-Omni 0.32.0 + vLLM 0.32"),
    "fact": (RES / "20261122_factors", "vLLM-Omni main + vLLM 0.34"),
    "g3": (RES / "20261123_graph", "三卡（卡 0、1、2），XPU graph 对比"),
    "tp2g": (RES / "20261124_tp2graph", "三卡（卡 0、1、2），M* TP2 graph（补丁）"),
}

# cells whose output was wrong (checked on the produced images / captions); never reported as a result
INVALID = {
    ("v032", "omni-single-tp2", "i2t1"): "描述不正确",
    ("v032", "omni-single-tp2", "i2t64"): "描述不正确",
    ("final", "omni-single-tp2", "i2t1"): "描述不正确",
    ("final", "omni-two-stage", "i2i"): "图像异常",
    ("fact", "omni-two-stage-shm", "i2i"): "图像异常",
    ("fact", "omni-two-stage-mooncake-tcp", "i2i"): "图像异常",
    ("fact", "omni-two-stage-nixl", "i2i"): "NIXL 传输失败",
    ("fact", "omni-two-stage-nixl", "t2i"): "NIXL 传输失败",
    ("screen", "omni-two-stage", "i2i"): "图像异常",
}
for _lab in ("mstar-tp2", "mstar-tp2-nocompile", "diag-mstar-mathsdpa", "diag-mstar-eager-mathsdpa"):
    for _c in ("i2t1", "i2t64"):
        INVALID[("v032", _lab, _c)] = "缓存不等"
for _lab in ("mstar-tp2", "mstar-tp2-graph", "mstar-cfgpar", "mstar-cfgpar-graph"):
    INVALID[("screen", _lab, "i2t64")] = "输入不等量"
EXTRA_INVALID_FILE = WS / "report" / "generated" / "invalid_extra.txt"  # "<group>/<label>/<case>" lines, optional
if EXTRA_INVALID_FILE.exists():
    for line in EXTRA_INVALID_FILE.read_text().split():
        g, l, c = line.split("/")
        INVALID[(g, l, c)] = "输出无效"


# cases that were measured in a run whose I2T prompt was later found invalid (hand-written image_pad prompt);
# T2I / I2I do not use that prompt, so these cells are read from the archived directory.
RAW = RES / "20261122_best_cfg" / "final" / "_invalid_bagel_raw_prompt"
BORROW = {("final", "omni-two-stage"): ("t2i", "i2i"), ("final", "omni-two-stage-graph"): ()}


# M* I2T cells of the first runs used the 4900-patch "vllm" preprocessing (3.6x more prefill than the Omni Thinker's
# 1386-token prompt). The equal-workload I2T numbers (default preprocessing, 1369 patches) come from queue8.
PARITY = RES / "20261122_i2t_parity"
PARITY_OVERRIDE = {("final", "mstar-cfgpar"), ("final", "mstar-cfgpar-nocompile"), ("final", "mstar-tp2"),
                   ("fact", "mstar-cfgpar-nocache")}
I2T_CASES = ("i2t_1tok", "i2t_64tok")


I2I_EQUAL = RES / "20261122_i2i_equal" / "omni-single-tp2-i2i980" / "timings.csv"   # Omni I2I, 512^2 input upscaled to 980^2
I2I_PARITY = RES / "20261122_i2i_parity"   # M* I2I with image_preprocess:"default" (1369 patches): edit not applied


# I2I with one instruction both frameworks follow at 1369 ViT patches (512^2 input, both sides)
I2I_EQ_DIR = RES / "20261122_i2i_diag"
I2I_EQ = {("final", "mstar-cfgpar"): "mstar-cfgpar-default-croissant",
          ("final", "mstar-tp2"): "mstar-tp2-default-croissant",
          ("final", "mstar-cfgpar-nocompile"): "mstar-cfgpar-nocompile-default-croissant",
          ("fact", "mstar-cfgpar-nocache"): "mstar-cfgpar-nocache-default-croissant",
          ("final", "omni-single-tp2"): "omni-single-tp2-croissant"}


def load(group, label, old4900=False):
    paths = [(GROUPS[group][0] / label / "timings.csv", None)]
    if (group, label) in I2I_EQ and not old4900:
        paths = [(GROUPS[group][0] / label / "timings.csv", ("t2i", "i2t1", "i2t64"))]
        paths.append((I2I_EQ_DIR / I2I_EQ[(group, label)] / "timings.csv", ("i2i",)))
    if (group, label) in PARITY_OVERRIDE and not old4900:
        paths[0] = (paths[0][0], ("t2i",))           # I2I comes from the equal set below
        paths.append((PARITY / label / "timings.csv", ("i2t1", "i2t64")))
    if (group, label) in BORROW:
        paths.append((RAW / label / "timings.csv", BORROW[(group, label)]))
    steady, warm = defaultdict(list), defaultdict(list)
    for path, only in paths:
        if not path.exists():
            continue
        for r in csv.DictReader(path.open()):
            if only is not None and r["case"] not in {CASE_CSV[c] for c in only}:
                continue
            (steady if r["run"].isdigit() else warm)[r["case"]].append(float(r["seconds"]))
    out = {}
    for short, csvname in CASE_CSV.items():
        if steady.get(csvname):
            v = steady[csvname]
            out[short] = dict(med=statistics.median(v), lo=min(v), hi=max(v), n=len(v),
                              warm=warm[csvname][0] if warm.get(csvname) else None)
    return out


DATA = {}


def get(group, label):
    key = (group, label)
    if key not in DATA:
        DATA[key] = load(group, label)
    return DATA[key]


def old4900(group, label):
    return load(group, label, old4900=True)


def med(group, label, case):
    d = get(group, label).get(case)
    if d is None or (group, label, case) in INVALID:
        return None
    return d["med"]


def medany(group, label, case):
    """median even when the cell is flagged grey-image (timing still meaningful for transfer comparisons)"""
    d = get(group, label).get(case)
    if d is None or INVALID.get((group, label, case), "图像异常") in ("NIXL 传输失败", "描述不正确", "输出无效"):
        return None
    return d["med"]


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


# ------------------------------------------------------------------ values.tex
vals = []
for group in GROUPS:
    gdir = GROUPS[group][0]
    labels = [p.parent.name for p in sorted(gdir.glob("*/timings.csv"))]
    for label in labels:
        for case, d in get(group, label).items():
            vals.append(f"\\expandafter\\def\\csname any@{group}/{label}/{case}\\endcsname{{{d['med']:.3f}}}")
            if (group, label, case) in INVALID:
                continue
            vals.append(f"\\expandafter\\def\\csname val@{group}/{label}/{case}\\endcsname{{{d['med']:.2f}}}")
            vals.append(f"\\expandafter\\def\\csname raw@{group}/{label}/{case}\\endcsname{{{d['med']:.3f}}}")
            if d["warm"] is not None:
                vals.append(f"\\expandafter\\def\\csname warm@{group}/{label}/{case}\\endcsname{{{d['warm']:.1f}}}")


def pct(a, b):
    """percentage change of a relative to b, as signed string like -41\\%"""
    if a is None or b is None:
        return "n/a"
    v = (a / b - 1) * 100
    return f"{v:+.0f}\\%".replace("-", "$-$").replace("+", "$+$")


def ratio(a, b):
    return "n/a" if a is None or b is None else f"{a / b:.2f}$\\times$"


def M(label, case, group="final"):
    return med(group, label, case)


derived = {
    # topology: CFG parallel vs TP2
    "topoMstarTti": pct(M("mstar-cfgpar", "t2i"), M("mstar-tp2", "t2i")),
    "topoMstarIti": pct(M("mstar-cfgpar", "i2i"), M("mstar-tp2", "i2i")),
    "topoMstarDec": pct(M("mstar-cfgpar", "i2t64"), M("mstar-tp2", "i2t64")),
    "topoMstarPre": pct(M("mstar-cfgpar", "i2t1"), M("mstar-tp2", "i2t1")),
    "topoOmniTti": pct(M("omni-single-cfgpar2", "t2i"), M("omni-single-tp2", "t2i")),
    # compile on vs off (CFG parallel)
    "cmpCfgTti": pct(M("mstar-cfgpar", "t2i"), M("mstar-cfgpar-nocompile", "t2i")),
    "cmpCfgIti": pct(M("mstar-cfgpar", "i2i"), M("mstar-cfgpar-nocompile", "i2i")),
    "cmpCfgPre": pct(M("mstar-cfgpar", "i2t1"), M("mstar-cfgpar-nocompile", "i2t1")),
    "cmpCfgDec": pct(M("mstar-cfgpar", "i2t64"), M("mstar-cfgpar-nocompile", "i2t64")),
    # best-vs-best ratios (Omni / M*)
    "bestTti": ratio(M("omni-single-cfgpar2", "t2i"), M("mstar-cfgpar", "t2i")),
    "bestIti": ratio(M("omni-single-tp2", "i2i"), M("mstar-cfgpar", "i2i")),
    "bestDecG": ratio(M("omni-two-stage-graph", "i2t64"), M("mstar-cfgpar", "i2t64")),
    "bestDecN": ratio(M("omni-two-stage-graph", "i2t64"), M("mstar-cfgpar-nocompile", "i2t64")),
    "omniGraphDec": pct(M("omni-two-stage-graph", "i2t64"), M("omni-two-stage", "i2t64")),
    # Omni version update (single-stage TP2, same config): 0.32 -> main
    "verOmniTti": pct(med("final", "omni-single-tp2", "t2i"), med("v032", "omni-single-tp2", "t2i")),
    "verOmniIti": pct(med("final", "omni-single-tp2", "i2i"), med("v032", "omni-single-tp2", "i2i")),
    "verOmni2Dec": pct(med("final", "omni-two-stage", "i2t64"), med("v032", "omni-two-stage-2gpu", "i2t64")),
}
# connector & cache factors (group fact), only valid cells
for lab, tag in [("omni-two-stage-mooncake-tcp", "Moon"), ("omni-two-stage-nixl", "Nixl")]:
    for case, ct in [("t2i", "Tti"), ("i2i", "Iti")]:
        derived[f"conn{tag}{ct}"] = pct(medany("fact", lab, case), medany("fact", "omni-two-stage-shm", case))
for k in ("connNixlTti", "connNixlIti"):
    derived[k] = "失败\\textsuperscript{‡}"
derived["cacheOmni1"] = pct(med("fact", "omni-two-stage-prefixcache", "i2t1"), M("omni-two-stage", "i2t1"))
derived["cacheOmni64"] = pct(med("fact", "omni-two-stage-prefixcache", "i2t64"), M("omni-two-stage", "i2t64"))
derived["cacheMstar1"] = pct(M("mstar-cfgpar", "i2t1"), med("fact", "mstar-cfgpar-nocache", "i2t1"))
derived["cacheMstar64"] = pct(M("mstar-cfgpar", "i2t64"), med("fact", "mstar-cfgpar-nocache", "i2t64"))
derived["cacheMstarIti"] = pct(M("mstar-cfgpar", "i2i"), med("fact", "mstar-cfgpar-nocache", "i2i"))
# I2T, equal workload (M* 1369 patches vs Omni Thinker 1386 prompt tokens): caches off on both / on on both
o_off1, o_off64 = med("final", "omni-two-stage", "i2t1"), med("final", "omni-two-stage", "i2t64")  # eager, no graph
m_off1, m_off64 = med("fact", "mstar-cfgpar-nocache", "i2t1"), med("fact", "mstar-cfgpar-nocache", "i2t64")
o_on1, o_on64 = med("fact", "omni-two-stage-prefixcache", "i2t1"), med("fact", "omni-two-stage-prefixcache", "i2t64")
m_on1, m_on64 = M("mstar-cfgpar", "i2t1"), M("mstar-cfgpar", "i2t64")
derived.update({
    "i2tOff1": pct(o_off1, m_off1), "i2tOff64": pct(o_off64, m_off64),
    "i2tOn1": pct(o_on1, m_on1), "i2tOn64": pct(o_on64, m_on64),
    "i2tOff1x": ratio(m_off1, o_off1), "i2tOn1x": ratio(m_on1, o_on1),
    "i2tOff64x": ratio(m_off64, o_off64), "i2tOn64x": ratio(m_on64, o_on64),
})
def per_tok(t64, t1):
    return "n/a" if t64 is None or t1 is None else f"{(t64 - t1) / 63 * 1000:.1f}"
derived.update({"msTokM": per_tok(m_on64, m_on1), "msTokMnc": per_tok(M("mstar-cfgpar-nocompile", "i2t64"), M("mstar-cfgpar-nocompile", "i2t1")),
                "msTokO": per_tok(o_on64, o_on1), "msTokMoff": per_tok(m_off64, m_off1), "msTokOg": per_tok(o_off64, o_off1)})
# summary table: vLLM-Omni relative to M* (positive = Omni slower), each with its own best configuration
_m_t2i, _o_t2i = med("final", "mstar-cfgpar", "t2i"), med("final", "omni-single-cfgpar2", "t2i")
_m_i2i, _o_i2i = med("final", "mstar-cfgpar", "i2i"), med("final", "omni-single-tp2", "i2i")
derived.update({
    "sumT2I": pct(_o_t2i, _m_t2i), "sumI2I": pct(_o_i2i, _m_i2i),
    "sumI1off": pct(o_off1, m_off1), "sumI64off": pct(o_off64, m_off64),
    "sumI1on": pct(o_on1, m_on1), "sumI64on": pct(o_on64, m_on64),
    # "faster by" wording for the summary text: M* faster = 1 - M*/Omni; Omni faster = 1 - Omni/M*
    "mFastT2I": f"{(1 - _m_t2i / _o_t2i) * 100:.0f}\\%", "mFastI2I": f"{(1 - _m_i2i / _o_i2i) * 100:.0f}\\%",
    "oFastI1off": f"{(1 - o_off1 / m_off1) * 100:.0f}\\%", "oFastI64off": f"{(1 - o_off64 / m_off64) * 100:.0f}\\%",
    "oFastI1on": f"{(1 - o_on1 / m_on1) * 100:.0f}\\%", "oFastI64on": f"{(1 - o_on64 / m_on64) * 100:.0f}\\%",
})
# XPU graph (three-card runs): absolute percentages for running text, signed for tables
def pctabs(a, b):
    return "n/a" if a is None or b is None else f"{abs(a / b - 1) * 100:.0f}\\%"


_gc_e, _gc_g = med("g3", "mstar-cfgpar3-eager", "i2t64"), med("g3", "mstar-cfgpar3-graph", "i2t64")
_tp_e, _tp_g = med("tp2g", "mstar-tp2-eager-rerun", "i2t64"), med("tp2g", "mstar-tp2-graph-fixed", "i2t64")
_om_g = med("g3", "omni-tp2-graph-chat", "i2t64")
derived.update({"gCfgOn64": pctabs(_gc_g, _gc_e), "gTp2G64": pctabs(_tp_g, _tp_e), "gTp2Gap": pctabs(_tp_g, _om_g),
                "gDiffT": pct(med("g3", "mstar-cfgpar3-graph", "t2i"), med("g3", "mstar-cfgpar3-eager", "t2i")),
                "gDiffI": pct(med("g3", "mstar-cfgpar3-graph", "i2i"), med("g3", "mstar-cfgpar3-eager", "i2i"))})
for k, v in derived.items():
    vals.append(f"\\expandafter\\def\\csname pct@{k}\\endcsname{{{v}}}")
(OUT / "values.tex").write_text(
    "% generated by scripts/make_tex_tables.py -- do not edit\n"
    "\\makeatletter\n"
    "\\newcommand{\\val}[1]{\\ifcsname val@#1\\endcsname\\csname val@#1\\endcsname\\else--\\fi}\n"
    "\\newcommand{\\raw}[1]{\\ifcsname raw@#1\\endcsname\\csname raw@#1\\endcsname\\else--\\fi}\n"
    "\\newcommand{\\anyval}[1]{\\ifcsname any@#1\\endcsname\\csname any@#1\\endcsname\\else--\\fi}\n"
    "\\newcommand{\\warm}[1]{\\ifcsname warm@#1\\endcsname\\csname warm@#1\\endcsname\\else--\\fi}\n"
    "\\newcommand{\\pct}[1]{\\ifcsname pct@#1\\endcsname\\csname pct@#1\\endcsname\\else n/a\\fi}\n"
    + "\n".join(vals) + "\n\\makeatother\n")


# ------------------------------------------------------------------ tables
def cell(group, label, case, best_set, rng=False):
    d = get(group, label).get(case)
    if d is None:
        return "--"
    if (group, label, case) in INVALID:
        if INVALID[(group, label, case)].startswith("NIXL"):
            return "\\textcolor{Red}{失败\\textsuperscript{‡}}"
        if INVALID[(group, label, case)] in ("输入不等量", "缓存不等"):
            return f"\\textcolor{{Muted}}{{{d['med']:.2f}\\textsuperscript{{§}}}}"
        if INVALID[(group, label, case)] == "图像异常":   # timing is informative (same work), content is not
            return f"\\textcolor{{Muted}}{{{d['med']:.2f}\\textsuperscript{{†}}}}"
        return f"\\textcolor{{Muted}}{{{INVALID[(group, label, case)]}\\textsuperscript{{†}}}}"
    nd = 3 if (d["med"] < 1 or rng) else 2
    s = f"{d['med']:.{nd}f}" if not rng else f"{d['med']:.3f}\\,\\textcolor{{Muted}}{{({d['lo']:.{nd}f}--{d['hi']:.{nd}f})}}"
    if (group, label, case) in best_set:
        s = f"\\cellcolor{{GoldBg}}\\textbf{{\\textcolor{{Teal}}{{{s}}}}}\\,$\\star$"
    return s


def split_row(group, row):
    """rows are 'label' (default group) or 'group:label'"""
    return tuple(row.split(":", 1)) if ":" in row else (group, row)


def best_cells(group, rows, fw_of):
    """best valid cell per (framework, case) among rows"""
    best = {}
    for row in rows:
        g, label = split_row(group, row)
        for case in CASES:
            m = med(g, label, case)
            if m is None:
                continue
            k = (fw_of[row], case)
            if k not in best or m < best[k][0]:
                best[k] = (m, g, label)
    out = set()
    for (fw, case), (m0, g0, lab0) in best.items():     # ties within 1% are all marked
        for row in rows:
            g, lab = split_row(group, row)
            m = med(g, lab, case)
            if fw_of[row] == fw and m is not None and m <= m0 * 1.01:
                out.add((g, lab, case))
    return out


def write_table(name, group, rows, fw_of, desc, rng=False, cases=CASES, lines_after=()):
    best = best_cells(group, [r for r in rows if r in fw_of], fw_of)
    L = ["\\begin{tabularx}{\\textwidth}{>{\\bfseries}p{5.9cm} p{0.9cm} " + " ".join("Y" for _ in cases) + "}",
         "\\toprule", "\\rowcolor{Mist}",
         "配置 & 框架 & " + " & ".join(CASE_NAME[c] for c in cases) + " \\\\", "\\midrule"]
    for row in rows:
        if row is None:
            L.append("\\midrule")
            continue
        g, label = split_row(group, row)
        fw = {"mstar": "\\textcolor{Teal}{M*}", "omni": "\\textcolor{Orange}{Omni}"}[fw_of[row]]
        L.append(desc[row] + " & " + fw + " & " + " & ".join(cell(g, label, c, best, rng) for c in cases) + " \\\\")
    L += ["\\bottomrule", "\\end{tabularx}"]
    (OUT / f"tab_{name}.tex").write_text("\n".join(L) + "\n")


FW = {}
DESC = {}


def reg(label, fw, text):
    FW[label] = fw
    DESC[label] = text


# ---- final matrix (best-configuration round)
reg("mstar-cfgpar", "mstar", "CFG 并行 (TP1×2)，compile 开")
reg("mstar-cfgpar-nocompile", "mstar", "CFG 并行，compile 关（诊断）")
reg("mstar-tp2", "mstar", "TP2，compile 开")
reg("omni-single-cfgpar2", "omni", "单 stage，CFG 并行 (TP1×2)")
reg("omni-single-tp2", "omni", "单 stage，TP2")
reg("omni-two-stage", "omni", "two-stage，eager")
reg("omni-two-stage-graph", "omni", "two-stage，Thinker graph")
reg("fact:mstar-cfgpar-nocache", "mstar", "CFG 并行，缓存关")
reg("fact:omni-two-stage-prefixcache", "omni", "two-stage，缓存开")
# full configuration text for the matrix rows (cards 1, 2 unless stated); see the configuration table in the deck
DESC["mstar-cfgpar"] = "M* CFG 并行 · compile 开 · 缓存开 · 无 graph"
DESC["mstar-cfgpar-nocompile"] = "M* CFG 并行 · compile 关（诊断）· 缓存开"
DESC["mstar-tp2"] = "M* TP2 · compile 开 · 缓存开 · 无 graph"
DESC["fact:mstar-cfgpar-nocache"] = "M* CFG 并行 · compile 开 · 缓存关"
DESC["omni-single-cfgpar2"] = "Omni 单 stage CFG 并行 · 无 compile"
DESC["omni-single-tp2"] = "Omni 单 stage TP2 · 无 graph"
DESC["omni-two-stage"] = "Omni two-stage · 缓存关 · eager"
DESC["omni-two-stage-graph"] = "Omni two-stage · 缓存关 · Thinker graph"
DESC["fact:omni-two-stage-prefixcache"] = "Omni two-stage · 缓存开 · eager"
final_rows = ["mstar-cfgpar", "mstar-cfgpar-nocompile", "mstar-tp2", "fact:mstar-cfgpar-nocache", None,
              "omni-single-cfgpar2", "omni-single-tp2", "omni-two-stage", "omni-two-stage-graph",
              "fact:omni-two-stage-prefixcache"]
write_table("final", "final", final_rows, FW, DESC)
write_table("final_range", "final", final_rows, FW, DESC, rng=True)

# ---- screening (3 repeats) incl. failed graph variants
reg("mstar-tp2-graph", "mstar", "TP2 + XPU graph（捕获失败→回退 eager）")
reg("mstar-cfgpar-graph", "mstar", "CFG 并行 + XPU graph（显存耗尽，起不来）")
screen_rows = ["mstar-tp2", "mstar-tp2-graph", "mstar-cfgpar", "mstar-cfgpar-graph", None, "omni-single-tp2",
               "omni-single-cfgpar2", "omni-two-stage", "omni-two-stage-graph"]
write_table("screen", "screen", screen_rows, FW, DESC, rng=True, cases=["t2i", "i2t64"])

# ---- v0.32 default-config round
V = {}
for lab, fw, text in [("mstar-tp2", "mstar", "TP2，compile 开（默认）"),
                      ("mstar-tp2-nocompile", "mstar", "TP2，compile 关（hook）"),
                      ("omni-single-tp2", "omni", "单 stage TP2"),
                      ("omni-two-stage-2gpu", "omni", "two-stage（Thinker 卡1 + DiT 卡2）")]:
    V[lab] = (fw, text)
write_table("v032", "v032", list(V), {k: v[0] for k, v in V.items()}, {k: v[1] for k, v in V.items()}, rng=True)

# ---- diagnostics (SDPA / OpenCL)
D = {"diag-omni-single-no-ocl-fix": ("omni", "单 stage TP2，不修 OpenCL"),
     "diag-omni-single-mathsdpa": ("omni", "单 stage TP2，强制 MATH SDPA"),
     "diag-mstar-mathsdpa": ("mstar", "TP2，强制 MATH SDPA"),
     "diag-mstar-eager-mathsdpa": ("mstar", "TP2，关 compile + MATH SDPA（早期 workaround）")}
write_table("diag", "v032", list(D), {k: v[0] for k, v in D.items()}, {k: v[1] for k, v in D.items()}, rng=True,
            cases=["t2i", "i2t1"])

# ---- factors: connector (two-stage T2I / I2I) and prefix cache (I2T / I2I)
F = {"omni-two-stage-shm": ("omni", "SharedMemory（默认）"),
     "omni-two-stage-mooncake-tcp": ("omni", "Mooncake TransferEngine（TCP）"),
     "omni-two-stage-nixl": ("omni", "NIXL（UCX）")}
write_table("conn", "fact", list(F), {k: v[0] for k, v in F.items()}, {k: v[1] for k, v in F.items()}, rng=True,
            cases=["t2i", "i2i"])
G = {"omni-two-stage-prefixcache": ("omni", "Omni two-stage，Thinker 前缀缓存开"),
     "mstar-cfgpar-nocache": ("mstar", "M* CFG 并行，KV 前缀缓存关")}
write_table("cache", "fact", list(G), {k: v[0] for k, v in G.items()}, {k: v[1] for k, v in G.items()}, rng=True,
            cases=["i2t1", "i2t64", "i2i"])

# ---- I2T workload parity: M* ViT preprocessing 4900 patches (first runs, wrong) vs 1369 patches (= Omni Thinker)
PROWS = [("CFG 并行（compile 开）", "final", "mstar-cfgpar"), ("CFG 并行（compile 关）", "final", "mstar-cfgpar-nocompile"),
         ("TP2（compile 开）", "final", "mstar-tp2"), ("CFG 并行，KV 前缀缓存关", "fact", "mstar-cfgpar-nocache")]
L = ["\\begin{tabularx}{\\textwidth}{>{\\bfseries}p{4.6cm} Y Y Y Y}", "\\toprule", "\\rowcolor{Mist}",
     "M* 配置 & 1 token，4900 patch（早期） & 1 token，1369 patch（本次） & 64 token，4900 patch（早期） & 64 token，1369 patch（本次） \\\\", "\\midrule"]
for text, g, lab in PROWS:
    o, n = old4900(g, lab), get(g, lab)
    def c(d, k, bold=False):
        if k not in d:
            return "--"
        v = f"{d[k]['med']:.{3 if d[k]['med'] < 1 else 2}f}"
        return f"\\textbf{{\\textcolor{{Teal}}{{{v}}}}}" if bold else f"\\textcolor{{Muted}}{{{v}}}"
    L.append(f"{text} & {c(o,'i2t1')} & {c(n,'i2t1',True)} & {c(o,'i2t64')} & {c(n,'i2t64',True)} \\\\")
    for k in ("i2t1", "i2t64"):
        if k in o and k in n:
            derived[f"par_{lab}_{k}"] = pct(n[k]["med"], o[k]["med"])
L.append("\\midrule")
for text, g, lab in [("Omni two-stage（1386 token 提示），缓存关", "final", "omni-two-stage"),
                     ("Omni two-stage，Thinker 前缀缓存开", "fact", "omni-two-stage-prefixcache")]:
    d = get(g, lab)
    L.append(f"\\textcolor{{Orange}}{{{text}}} & -- & \\textbf{{\\textcolor{{Orange}}{{{d['i2t1']['med']:.3f}}}}} & -- & "
             f"\\textbf{{\\textcolor{{Orange}}{{{d['i2t64']['med']:.3f}}}}} \\\\")
L += ["\\bottomrule", "\\end{tabularx}"]
(OUT / "tab_parity.tex").write_text("\n".join(L) + "\n")

# ---- I2I: the equal-workload set (one instruction both frameworks follow) and the earlier cat-instruction runs
def raw_steady(path, case):
    vals = [float(r["seconds"]) for r in csv.DictReader(path.open()) if r["case"] == CASE_CSV[case] and r["run"].isdigit()]
    return dict(med=statistics.median(vals), lo=min(vals), hi=max(vals)) if vals else None


def i2i_fmt(d, bold=False):
    if d is None:
        return "--"
    s = f"{d['med']:.2f}\\,\\textcolor{{Muted}}{{({d['lo']:.2f}--{d['hi']:.2f})}}"
    return f"\\textbf{{{s}}}" if bold else s


L = ["\\begin{tabularx}{\\textwidth}{>{\\bfseries}p{4.6cm} p{2.5cm} Y}", "\\toprule", "\\rowcolor{Mist}",
     "配置 & 图像 token（ViT / VAE） & I2I 中位数（min--max，秒） \\\\", "\\midrule"]
for text, g, lab in [("M* CFG 并行（compile 开）", "final", "mstar-cfgpar"), ("M* CFG 并行（compile 关）", "final", "mstar-cfgpar-nocompile"),
                     ("M* TP2（compile 开）", "final", "mstar-tp2"), ("M* CFG 并行，KV 前缀缓存关", "fact", "mstar-cfgpar-nocache")]:
    L.append(f"\\textcolor{{Teal}}{{{text}}} & 1369 / 1024 & {i2i_fmt(get(g, lab).get('i2i'), bold=lab == 'mstar-cfgpar')} \\\\")
L.append(f"\\textcolor{{Orange}}{{vLLM-Omni 单 stage TP2（512$^2$ 原生输入）}} & 1369 / 1024 & "
         f"{i2i_fmt(get('final', 'omni-single-tp2').get('i2i'), bold=True)} \\\\")
L += ["\\midrule", "\\multicolumn{3}{l}{\\textcolor{Muted}{早期指令（“Add a small cat sitting beside the bagel on the plate.”）的运行}} \\\\"]
for text, path, tok in [
        ("M* CFG 并行，\\code{vllm}（4900 patch）", RES / "20261122_best_cfg/final/mstar-cfgpar/timings.csv", "4900 / 1024"),
        ("M* CFG 并行，\\code{default}（输出未加入猫）", I2I_PARITY / "mstar-cfgpar/timings.csv", "1369 / 1024"),
        ("vLLM-Omni 单 stage TP2（512$^2$ 原生）", RES / "20261122_best_cfg/final/omni-single-tp2/timings.csv", "1369 / 1024"),
        ("vLLM-Omni 单 stage TP2（输入放大到 980$^2$）", I2I_EQUAL, "4900 / 3721")]:
    L.append(f"\\textcolor{{Muted}}{{{text}}} & {tok} & {i2i_fmt(raw_steady(path, 'i2i'))} \\\\")
L += ["\\bottomrule", "\\end{tabularx}"]
(OUT / "tab_i2i_parity.tex").write_text("\n".join(L) + "\n")

# ---- warmup (first request) table for the final matrix
L = ["\\begin{tabularx}{\\textwidth}{>{\\bfseries}p{5.2cm} Y Y Y Y}", "\\toprule", "\\rowcolor{Mist}",
     "配置 & T2I & I2T 1 token & I2T 64 token & I2I \\\\", "\\midrule"]
for row in [r for r in final_rows if r]:
    g, lab = split_row("final", row)
    d = get(g, lab)
    cells = [(f"{d[c]['warm']:.1f}" if c in d and d[c]["warm"] is not None and (g, lab, c) not in INVALID else "--")
             for c in CASES]
    L.append(DESC[row] + " & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabularx}"]
(OUT / "tab_warm.tex").write_text("\n".join(L) + "\n")

# values.tex needs the parity-derived percentages computed above: append them
extra = [f"\\expandafter\\def\\csname pct@{k}\\endcsname{{{v}}}" for k, v in derived.items() if k.startswith("par_")]
txt = (OUT / "values.tex").read_text().replace("\\makeatother\n", "\n".join(extra) + "\n\\makeatother\n")
(OUT / "values.tex").write_text(txt)
# ---- XPU graph (three-card runs, cards 0, 1, 2): eager vs graph, I2T and diffusion
GRAPH_I2T = [  # label, cache, (group, eager label), (group, graph label)
    ("M* CFG 并行（三卡）", "开", ("g3", "mstar-cfgpar3-eager"), ("g3", "mstar-cfgpar3-graph")),
    ("M* CFG 并行（三卡）", "关", ("g3", "mstar-cfgpar3-eager-nocache"), ("g3", "mstar-cfgpar3-graph-nocache")),
    ("M* TP2（三卡，LLM 卡 1、2）", "开", ("tp2g", "mstar-tp2-eager-rerun"), ("tp2g", "mstar-tp2-graph-fixed")),
    ("vLLM-Omni two-stage（三卡，Thinker TP2 卡 0、1）", "关", ("g3", "omni-tp2-eager-chat"), ("g3", "omni-tp2-graph-chat")),
    ("vLLM-Omni two-stage（三卡，Thinker TP2 卡 0、1）", "开", ("g3", "omni-tp2-eager-chat-cache"), ("g3", "omni-tp2-graph-chat-cache")),
]
GRAPH_DIFF = [  # label, T2I (eager, graph), I2I (eager, graph); None = not measured
    ("M* CFG 并行（三卡）", (("g3", "mstar-cfgpar3-eager"), ("g3", "mstar-cfgpar3-graph")),
     (("g3", "mstar-cfgpar3-eager"), ("g3", "mstar-cfgpar3-graph"))),
    ("M* TP2（三卡，LLM 卡 1、2）", (("tp2g", "mstar-tp2-eager-rerun"), ("tp2g", "mstar-tp2-graph-fixed")), None),
    ("vLLM-Omni 单 stage CFG 并行（三卡）", None, (("g3", "omni-single-cfgpar3-i2i"), None)),
    ("vLLM-Omni two-stage（三卡，Thinker TP2 卡 0、1，DiT 卡 2）", (("g3", "omni-two-stage-3gpu-tp2-eager"), ("g3", "omni-two-stage-3gpu-tp2-graph")), None),
]


def gval(ref, case):
    return None if ref is None else get(*ref).get(case, {}).get("med")


L = ["% generated by scripts/make_tex_tables.py -- do not edit",
     "\\scriptsize",
     "\\begin{tabularx}{\\textwidth}{>{\\bfseries}p{4.6cm} c r r r r r}",
     "\\toprule", "\\rowcolor{Mist}",
     "配置 & 缓存 & I2T 1 token eager & graph & I2T 64 token eager & graph & 64 token 变化 \\\\", "\\midrule"]
for name, cache, (ge, le), (gg, lg) in GRAPH_I2T:
    e1, g1 = gval((ge, le), "i2t1"), gval((gg, lg), "i2t1")
    e64, g64 = gval((ge, le), "i2t64"), gval((gg, lg), "i2t64")
    L.append(f"{name} & {cache} & {fmt(e1, 3)} & {fmt(g1, 3)} & {fmt(e64)} & {fmt(g64)} & {pct(g64, e64)} \\\\")
L += ["\\bottomrule", "\\end{tabularx}"]
(OUT / "tab_graph_i2t.tex").write_text("\n".join(L) + "\n")

L = ["% generated by scripts/make_tex_tables.py -- do not edit",
     "\\scriptsize",
     "\\begin{tabularx}{\\textwidth}{>{\\bfseries}p{5.6cm} r r r r r}",
     "\\toprule", "\\rowcolor{Mist}",
     "配置 & T2I eager & T2I graph & I2I eager & I2I graph & I2I 变化 \\\\", "\\midrule"]
for name, t, i in GRAPH_DIFF:
    te, tg = (gval(t[0], "t2i"), gval(t[1], "t2i")) if t else (None, None)
    if i is None:
        ie, ig = None, None
    elif i[1] is None:
        ie, ig = gval(i[0], "i2i"), None
    else:
        ie, ig = gval(i[0], "i2i"), gval(i[1], "i2i")
    L.append(f"{name} & {fmt(te)} & {fmt(tg)} & {fmt(ie)} & {fmt(ig)} & {pct(ig, ie)} \\\\")
L += ["\\bottomrule", "\\end{tabularx}"]
(OUT / "tab_graph_diff.tex").write_text("\n".join(L) + "\n")

print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
for k, v in derived.items():
    print(f"  {k:14s} {v}")
