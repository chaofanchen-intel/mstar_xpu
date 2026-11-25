# 汇报文档：M* 与 vLLM-Omni —— 支持范围、接入工作量、BAGEL 在 Intel B70 上的实测

16:9 幻灯片（Beamer + XeLaTeX），版式与配色沿用 `share/omni_runtime_report_v2`（`common.tex`、`deck.tex`）。
内容：1）框架支持范围；2）模型接入工作量；3）BAGEL-7B-MoT 在 2 × Arc Pro B70 上的全部实测结果（最佳配置标 ★）、各因素影响、复现方法。
SGLang-Omni 的内容已从两章参考稿中删去（`tools/derive_from_reference.py` 记录了派生方式）。

```
report/
├── main.tex                封面 + 章节顺序
├── common.tex / deck.tex   共用的字体、配色、宏 / 幻灯片版式（取自参考稿；封面日期在 deck.tex）
├── chapters/
│   ├── 00_summary.tex      结论、最优配置对比、阅读指南
│   ├── 01_support.tex      支持范围（派生自参考稿 05b）
│   ├── 02_integration.tex  模型接入工作量（派生自参考稿 05c）
│   ├── 03_bagel_setup.tex  环境、工作量定义、对初版的审计、OpenCL 修复、候选配置
│   ├── 04_bagel_results.tex 全部结果总表、最优对比图、I2T 工作量核对、首请求、筛选轮、上一轮
│   ├── 05_bagel_factors.tex 各因素影响：拓扑 / compile / graph / connector / 前缀缓存
│   └── 06_appendix.tex     min–max 数据、排除的运行及原因、复现命令
├── generated/              由 scripts/make_tex_tables.py 生成（不要手改）：values.tex（\val{}、\pct{} 宏）、tab_*.tex
└── build.sh                先生成表格，再用 Tectonic（或 latexmk + XeLaTeX）编译 main.pdf
```

## 编译
```bash
cd ~/workspaces/mstar_xpu/report
bash build.sh            # 需要 ../.venv 或系统 python3（make_tex_tables.py 只用标准库）和 tectonic（~/.local/bin/tectonic）
```

## 数字从哪来
所有性能数字都由 `scripts/make_tex_tables.py` 读取 `results/*/<label>/timings.csv`（稳态 = `run` 为数字的行，取中位数）生成，
正文里写 `\val{final/mstar-cfgpar/t2i}` 之类的宏，不手抄。该脚本里还记录了：
- `INVALID`：已测但不纳入比较的格子。标记文字：`图像异常`（灰图，†，保留耗时）、`描述不正确`（†，不显示耗时）、`NIXL 传输失败`（‡）、`输入不等量` / `缓存不等`（§，仅供参考）。不参与星标与引用；
- `PARITY_OVERRIDE`：M* 的图生文用输入对齐的新数据（`results/20261122_i2t_parity`）替换早期的 4900 patch 数据；
- I2I 指令 `Replace the bagel with a croissant.`：两边均为 1369 patch，等量比较；早期“加猫”指令的数据见 `tab_i2i_parity` 下半部分；`run_diag_i2i.sh` / `run_queue11.sh` 为对应运行。
- `BORROW`：Omni two-stage 的 T2I/I2I 取自归档目录 `_invalid_bagel_raw_prompt`（该运行的手写图生文提示不正确，T2I/I2I 不受影响）。
