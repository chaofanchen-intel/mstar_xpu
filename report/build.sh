#!/usr/bin/env bash
# 先从 results/ 生成表格，再用 Tectonic（或 latexmk + XeLaTeX）编译 main.pdf。
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -I ../scripts/make_tex_tables.py
"$PY" -I ../scripts/make_kernel_table.py
TECTONIC_BIN="${TECTONIC:-$(command -v tectonic || true)}"
[[ -z "$TECTONIC_BIN" && -x "$HOME/.local/bin/tectonic" ]] && TECTONIC_BIN="$HOME/.local/bin/tectonic"
if [[ -n "$TECTONIC_BIN" ]]; then "$TECTONIC_BIN" main.tex
elif command -v latexmk >/dev/null 2>&1; then latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
else echo "需要 Tectonic 或 XeLaTeX + latexmk" >&2; exit 127; fi
