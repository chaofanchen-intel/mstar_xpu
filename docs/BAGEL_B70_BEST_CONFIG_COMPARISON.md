# M* 与 vLLM-Omni：BAGEL-7B-MoT 在 Intel Arc Pro B70 上的对比

更新日期：2026-11-25。汇报版见 [report/main.pdf](../report/main.pdf)，数据与生成脚本见下文第 8 节。

## 1. 结论

两卡（卡 1、2）上，各框架取最优配置，等工作量比较（同一请求、同一输入图、相同步数与 CFG）：

| 测试项 | M* 结果（秒） | M* 配置 | vLLM-Omni 结果（秒） | vLLM-Omni 配置 | Omni 相对 M* |
| --- | ---: | --- | ---: | --- | ---: |
| 文生图 512²，50 步 | 7.68 | CFG 并行；compile 开；缓存开；无 graph | 9.98 | 单 stage CFG 并行；无 compile | +30% |
| 图像编辑 512²，50 步 | 16.79 | CFG 并行；compile 开；无 graph | 28.08 | 单 stage TP2 | +67% |
| 图生文 1 token，缓存关 | 0.293 | CFG 并行；compile 开；缓存关 | 0.193 | two-stage；缓存关；eager | −34% |
| 图生文 1 token，缓存开 | 0.085 | CFG 并行；compile 开；缓存开 | 0.053 | two-stage；缓存开；eager | −38% |
| 图生文 64 token，缓存关 | 2.66 | 同上（缓存关） | 1.88 | 同上（缓存关） | −29% |
| 图生文 64 token，缓存开 | 2.46 | 同上（缓存开） | 1.74 | 同上（缓存开） | −29% |

- 图像生成 M* 更快：文生图快 23%，图像编辑快 40%（以 M* 为基准，即 1 − M*/Omni）。
- 图生文 vLLM-Omni 更快。M* 的 I2T 解码受主机端开销限制；graph 能缩小差距，但三卡 TP2 下 M* 仍慢约 38%（第 6 节）。
- 两卡上两框架均未使用 XPU graph。三卡 TP2 下 M* 的 graph 可用，但需要对 M* 打补丁（第 6 节）。
- compile：扩散快 7–9%，64 token 解码慢约 13%（两卡 M* 诊断数据）。

## 2. 测试环境

| | M* | vLLM-Omni |
| --- | --- | --- |
| 源码 | `external/mstar` main 2c8ffde8f | `external/vllm-omni` main a36de509e（2026-11-21）；对照 v0.32.0 |
| PyTorch / 内核 | torch 2.13.0+xpu；vllm-xpu-kernels 0.1.15.4 | torch 2.15.0+xpu；vllm-xpu-kernels 0.1.17（vLLM 0.34.0+xpu） |
| 环境 | `.venv` | `~/workspaces/.venvs/vllm-omni-main`（v0.32 环境：`vllm-omni-v0.32.0`） |
| 模型 | BAGEL-7B-MoT，BF16，约 27 GiB；不做量化、缓存加速或减步 | 同左 |
| 硬件 | Arc Pro B70 × 3（每张 32 GiB），单 NUMA | 同左 |

- 两卡结果使用卡 1、2；三卡研究使用卡 0、1、2。运行前检查卡占用，只在空闲卡上运行。
- 两边都使用 eager 执行（不启用 XPU graph），除非配置名中注明 graph。
- 两框架各用各自源码要求的 torch 与内核版本，没有强行统一。

## 3. 配置定义

每个结果都对应下列命名配置。表中“缓存”为 KV / Thinker 前缀缓存。

### 3.1 两卡（卡 1、2）

| 配置 | 框架 | 卡与分工 | 说明 | compile | 缓存 | graph |
| --- | --- | --- | --- | --- | --- | --- |
| M* CFG 并行 | M* | 卡 1、2 各一份完整 LLM | 卡 1 主分支，卡 2 引导分支，无 TP 通信 | 开 | 开 | 无 |
| M* CFG 并行，缓存关 | M* | 同上 | 仅关闭前缀缓存 | 开 | 关 | 无 |
| M* CFG 并行，compile 关 | M* | 同上 | hook 关闭 `torch.compile`，非官方设置，仅诊断 | 关 | 开 | 无 |
| M* TP2 | M* | 卡 1、2 切分 LLM；ViT、VAE 在卡 1 | 每层 all-reduce；官方 `bagel_xpu_tp2` 的两卡版本 | 开 | 开 | 无 |
| Omni 单 stage CFG 并行 | Omni | 整个模型一个 stage，`cfg_parallel_size: 2` | 两路分支各占一卡；I2I 需要 3 路 | 无 | — | 无 |
| Omni 单 stage TP2 | Omni | 整个模型一个 stage，TP2 | 两卡图像编辑的最优方案 | 无 | — | 无 |
| Omni two-stage | Omni | Thinker 在卡 1，DiT 在卡 2 | KV 经 connector 传给 DiT；Thinker 的 `max_model_len` 为 6144 | 无 | 关 | 无 |
| Omni two-stage，缓存开 | Omni | 同上 | 仅开启 Thinker 前缀缓存 | 无 | 开 | 无 |
| Omni two-stage，graph | Omni | 同上 | Thinker 启用 graph；扩散仍为 eager | 无 | 关 | 开 |

### 3.2 三卡（卡 0、1、2）

| 配置 | 框架 | 卡与分工 | 说明 | compile | 缓存 | graph |
| --- | --- | --- | --- | --- | --- | --- |
| M* CFG 并行（三卡） | M* | 卡 0 主分支；卡 1、2 各一路引导分支 | 官方三 rank 的 CFG 方案 | 开 | 开 | 无 |
| M* CFG 并行（三卡），缓存关 | M* | 同上 | 仅关闭前缀缓存 | 开 | 关 | 无 |
| M* CFG 并行（三卡），graph | M* | 同上 | graph 开启；未打补丁 | 开 | 开 | 开 |
| M* TP2（三卡） | M* | 卡 0 为 ViT、VAE；卡 1、2 切分 LLM | LLM 切分与两卡 TP2 相同 | 开 | 开 | 无 |
| M* TP2（三卡），graph | M* | 同上 | 需要对 M* 打补丁（第 6.2 节），未合入上游 | 开 | 开 | 开 |
| Omni two-stage（三卡） | Omni | Thinker TP2 在卡 0、1；DiT 在卡 2 | 图生文不使用 DiT | 无 | 关 | 无 |
| Omni two-stage（三卡），graph | Omni | 同上 | Thinker 启用 graph；扩散仍为 eager | 无 | 关 | 开 |
| Omni 单 stage CFG 并行（三卡） | Omni | 卡 0、1、2 各一路 CFG 分支 | I2I 的三路 CFG 方案 | 无 | — | 无 |

## 4. 工作量与测量

| case | 定义（`scripts/bench_bagel.py`，两框架共用） |
| --- | --- |
| T2I 文生图 | prompt `A red ceramic cup on a white table`，512×512，50 步，shift 3.0，CFG text 4.0 / img 1.0，区间 [0.4, 1.0]，renorm `global` |
| I2T 1 token | 512×512 输入图 + 描述问题，贪心解码，生成 1 个 token（图像编码 + 预填充） |
| I2T 64 token | 同一输入图 + 详细描述问题，贪心解码，`ignore_eos` 生成精确 64 个 token（usage 与 tokenizer 校验） |
| I2I 图像编辑 | 同一输入图，指令 `Replace the bagel with a croissant.`，输出 512×512，50 步，CFG text 4.0 / img 2.0，区间 [0, 1]，renorm `text_channel` |

- 输入图像 token：I2T 为 1386（Omni 的 `usage.prompt_tokens`，其中 1369 个图像 patch + 17 个文本 token）；M* 默认预处理同为 1369 patch。I2I 两边均为 1369 个图像 patch（Omni 由源码确认）。
- 测量：2 次 warmup + 5 次稳态，串行 HTTP，端到端计时，取中位数。输出检查：T2I、I2I 为 512×512 有效图像；I2T 检查 caption 与 token 数。
- 只做不改变计算量的优化：并行拓扑、compile、graph、connector、前缀缓存。不使用 TeaCache、Cache-DiT、量化或减步。
- 前缀缓存：M* 默认开，Omni 默认关。I2T 分缓存关、开两组比较。

## 5. 两卡结果

### 5.1 全部结果（卡 1、2，稳态中位数，秒）

| 配置 | T2I | I2T 1 token | I2T 64 token | I2I |
| --- | ---: | ---: | ---: | ---: |
| M* CFG 并行，compile 开，缓存开 | **7.68** | 0.085 | 2.46 | **16.79** |
| M* CFG 并行，compile 关（诊断），缓存开 | 8.48 | **0.083** | **2.17** | 18.25 |
| M* TP2，compile 开，缓存开 | 12.96 | 0.330 | 3.16 | 21.21 |
| M* CFG 并行，compile 开，缓存关 | — | 0.293 | 2.66 | 16.82 |
| Omni 单 stage CFG 并行 | **9.98** | — | — | — |
| Omni 单 stage TP2 | 17.04 | — | — | **28.08** |
| Omni two-stage，缓存关，eager | 17.99 | 0.193 | 1.88 | 30.76†（灰图） |
| Omni two-stage，缓存关，Thinker graph | — | 0.191 | 1.86 | — |
| Omni two-stage，缓存开，eager | — | **0.053** | **1.74** | — |

- 加粗为同一框架内该 case 的最佳配置（1% 以内并列）。n = 5，min–max 见附录（report 附录 A）。
- †：Omni two-stage 的 I2I 在 main 版本输出灰图，耗时仅供参考，不参与比较。
- I2I 的 M* 结果使用默认预处理（1369 patch）和可颂指令，与 Omni 等量。

### 5.2 图生文等量比较

- 输入规模：M* 1369 patch，Omni 1386 token，两边一致。
- 缓存：缓存关对缓存关，缓存开对缓存开，各自比较。
- 缓存关时 Omni 快 34%（1 token）、29%（64 token）；缓存开时快 38%、29%。
- 每 token 解码（缓存关，由 64 token 与 1 token 差值估算）：M* 37.5 ms（compile 开），Omni 26.8 ms。

### 5.3 首请求

首请求指服务启动后该 case 的第一次请求（含 JIT、编译、图捕获）。M* 默认开启 compile，T2I、I2I、I2T 首请求均多出约 6–7 s；关闭 compile 后首请求更短，但扩散稳态慢 7–9%。vLLM-Omni 无 compile，首请求只多出内核懒加载的时间。

## 6. 各因素

| 因素 | A → B | T2I | I2I | I2T 1 token | I2T 64 token | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| M* 并行拓扑 | TP2 → CFG 并行 | −41% | −21% | −74% | −22% | 影响最大 |
| Omni 并行拓扑 | 单 stage TP2 → CFG 并行 | −41% | 不支持 | — | — | I2I 需要 3 卡 |
| M* compile | 关 → 开（CFG 并行） | −9% | −8% | +3% | +13% | 扩散快，解码慢 |
| Omni two-stage connector | SHM → Mooncake-TCP | +0% | −3% | — | — | 见 5.4 |
| Omni two-stage connector | SHM → NIXL | 失败 | 失败 | — | — | XPU 上不可用 |
| M* 前缀缓存 | 关 → 开（默认） | — | −0% | −71% | −7% | 同图重复提问时命中 |
| Omni 前缀缓存 | 关 → 开 | — | — | −73% | −7% | 同上 |
| Omni 软件版本 | v0.32 → main（单 stage TP2） | −3% | −9% | — | — | 两阶段解码 −44% |

### 5.1 并行拓扑

- TP2 每次文生图约 5000 次跨卡 all-reduce（50 步，两路分支，每步约 100 次）。在两卡上这些通信占了一半时间。
- CFG 并行没有 TP 通信：卡 1 算主分支，卡 2 算引导分支，每步只交换一次引导结果。每卡都是完整模型（约 27 GiB），显存几乎占满。
- 图像编辑的三路分支只有 CFG 并行或三卡能放下；两卡 Omni 只能用 TP2。

### 5.2 compile

- Inductor / Triton-XPU 的算子融合在扩散步上带来 7–9% 的加速。
- 小 batch 的解码变慢（M* 64 token 为 +13%，TP2 下更明显）。M* 没有官方开关，“关”是通过 hook 让 `torch.compile` 失效实现的，仅作诊断。
- CUDA 上 compile 的主要收益来自 CUDA graph；B70 上 GPU 利用率采样的最高值只有 14–22%，推测 eager 路径受内核启动限制。XPU graph 的结果见第 6 节。

### 5.3 前缀缓存

- 缓存只影响图生文。T2I 没有共享前缀；M* I2I 缓存关、开基本无差别（16.82 与 16.79）。
- 缓存对 1 token（即预填充）影响最大（−70% 左右），对 64 token 只省去预填充部分（约 −7%）。
- 实际服务中，同一张图被连续提问时才会命中缓存。两框架的默认设置不同，比较时必须分别给出缓存关、开的结果。

### 5.4 两阶段 connector

同一 two-stage 拓扑（Thinker 卡 1，DiT 卡 2，eager），只改变 KV 传输方式：

| connector | T2I | I2I | 备注 |
| --- | ---: | ---: | --- |
| SharedMemory（默认） | 17.96 | 30.74 | |
| Mooncake TransferEngine（TCP） | 17.98 | 29.92 | I2I 快 3%，T2I 无差别 |
| NIXL（UCX） | 失败路径 48.3 | 失败路径 121.6 | UCX 无法注册 XPU 显存，KV 传输重试 3 次后失败；耗时不代表传输速度 |

NIXL 失败时 I2I 反而输出正常图，而 SharedMemory 与 Mooncake 都输出灰图，说明 I2I 灰图与 KV 传输、接收路径有关（未定位）。

### 5.5 软件版本与 OpenCL 修复

- vLLM-Omni v0.32.0 与 main 的单 stage TP2 对比：T2I 17.66 → 17.04 s，I2I 30.74 → 28.08 s；两阶段 64 token 解码 3.35 → 1.88 s。
- M* 默认 SDPA 的 OpenCL 报错来自 conda 的 FFmpeg 自带 `libOpenCL.so.1` 遮蔽了系统 ICD。设置 `OCL_ICD_VENDORS=/etc/OpenCL/vendors` 即可修复（`scripts/env.sh` 已写入），修复本身无性能代价。

## 7. 三卡：XPU graph 与去噪步 kernel 分布

### 7.1 图生文解码（三卡，中位数，秒）

| 配置 | 缓存 | 1 token eager | 1 token graph | 64 token eager | 64 token graph | 64 token 变化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M* CFG 并行（三卡） | 开 | 0.089 | 0.085 | 2.42 | 1.95 | −20% |
| M* CFG 并行（三卡） | 关 | 0.304 | 0.300 | 2.68 | 2.15 | −20% |
| M* TP2（三卡） | 开 | 0.258 | 0.226 | 3.05 | 1.52 | −50% |
| Omni two-stage（三卡） | 关 | 0.127 | 0.125 | 1.36 | 1.10 | −19% |
| Omni two-stage（三卡） | 开 | 0.054 | 0.051 | 1.25 | 1.02 | −19% |

- M* TP2 的 eager 为同批复测（3.049 s）；早期单次运行为 2.764 s（`results/20261123_graph/mstar-tp2-3gpu-eager`）。
- graph 只作用于 LLM 解码步。M* TP2 graph 后为 1.52 s，Omni two-stage graph 为 1.10 s，M* 仍慢 38%。
- 三卡 M* CFG 并行的 graph 无需补丁（运行时间早于补丁）。

### 7.2 扩散步与 TP2 图捕获

| 配置 | T2I eager | T2I graph | I2I eager | I2I graph |
| --- | ---: | ---: | ---: | ---: |
| M* CFG 并行（三卡） | 7.73 | 7.75 | 9.28 | 9.26 |
| M* TP2（三卡） | 7.68 | 7.64 | — | — |
| Omni 单 stage CFG 并行（三卡） | — | — | 11.19 | — |
| Omni two-stage（三卡） | 17.91 | 18.01 | — | — |

- 扩散步两边都是 eager，graph 不作用于扩散。
- M* TP2 图捕获失败的三个原因：
  1. XCCL 的 `all_reduce` 消息超过约 4 MiB 时无法进入 graph。
  2. inductor 生成的 functional `all_gather`（lm_head 汇聚 logits）在 XPU graph 中不可用；普通 `dist.all_gather_into_tensor` 在约 4 行 logits 以内可用。
  3. 最大的 prefill（2048 token）捕获失败后，捕获状态残留，之后的 decode 也全部失败。
- 补丁 `scripts/patches/mstar_xpu_tp2_graph.patch` 作用于已安装的 M* 包（`.venv/lib/python3.12/site-packages/mstar`）：`all_gather` 不参与 compile；prefill 捕获 token 上限 512；decode 与 prefill 的捕获 batch size 限为 [1, 2, 4]。补丁前的原文件保存在 `results/20261124_tp2graph/*.orig`。当前 `.venv` 中的 M* 已还原为原文件；要重新运行 TP2 graph，需要先应用该补丁。
- 两卡 CFG 并行的 graph 因显存不足无法启动（每卡约 27 GiB 的完整模型）。

### 7.3 去噪步 kernel 分布（torch.profiler，XPU）

| kernel 类别 | Omni 时间（ms） | Omni 占比 | Omni 次数 | M* 占比（片段） | M* 次数（片段） |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemm | 119.7 | 61.8% | 229 | 84.1% | 78 |
| attention（FMHA） | 5.9 | 3.1% | 28 | 4.5% | 18 |
| 融合 kernel（Triton / vLLM） | 8.8 | 4.5% | 56 | 10.5% | 351 |
| index、gather、scatter | 27.1 | 14.0% | 1017 | 0.8% | 18 |
| copy、cat | 15.4 | 8.0% | 1021 | 0.0% | 0 |
| RMSNorm 拆散部分（pow、mean、reduce） | 5.7 | 2.9% | 452 | 0.0% | 0 |
| 其他 elementwise | 11.2 | 5.8% | 1396 | 0.1% | 18 |
| 合计（忙碌时间） | 193.7 | 100% | 4199 | 100% | 483 |
| 窗口时长 / 空闲占比 | 201.7 / 4.0% | | | 57.6 / 11.2% | |
| gemm 平均耗时（ms） | 0.52 | | | 0.55 | |

- 配置：Omni 单 stage CFG 并行（卡 0、1，T2I）；M* CFG 并行（卡 0、1、2，T2I，只用两路分支）。两边每张卡的分支工作量相同。
- Omni 为一个完整去噪步（rank 0），已剔除 profiler 引起的 oneCCL 等待（约 5 s，`results/20261124_kprof2`）。M* 为 rank 0 的 0.45 s 窗口（`results/20261125_kprof3`），只覆盖一段，只比较占比。
- 结论：Omni 去噪步中，非矩阵乘、非 attention 的小 kernel 约 68 ms（占 35%）；M* 片段中这类 kernel 很少，由融合 kernel 代替。两边单个 gemm 的平均耗时相同（0.52 与 0.55 ms）。
- 限制：M* 未取得完整一步，因此两边整步差距（约 51 ms）**没有被完全分配**。之前关于“约 80% 差距来自小 kernel”的说法超出了数据范围，已删去。

## 8. 早期结果与排除的运行

以下数据不进入第 5 节的比较：

- 早期测试（vLLM-Omni v0.32.0，两框架均为 TP2）：T2I 上 M* 快约 27%；改用 CFG 并行后两边都快 40% 以上，差距缩小为 23%。版本不同，不直接比较。
- 早期 M* 图生文使用 `image_preprocess:"vllm"`（4900 patch），与 Omni 的 1386 token 不等量，且缓存默认开。该批 I2T 数据作废，仅为解释早期结论保留。
- 早期 I2I 指令 “Add a small cat” 在 M* 默认预处理下未生效，改用可颂指令后生效。
- vLLM-Omni 单 stage 图生文描述不正确（额外做了 VAE 编码，且不支持 `ignore_eos`），不作为图生文结果。
- 手写 `<|image_pad|>` 提示词时 Thinker 将图描述为披萨，该批数据不纳入。
- vLLM-Omni two-stage，Thinker 显存利用率 0.90 / 0.95 时单卡 KV 空间不足，无法启动；0.98 且 `max_model_len` 6144 时可以运行。
- 候选配置筛选（3 次稳态，T2I 与 64 token）：`results/20261122_best_cfg/`。

## 9. 复现

```bash
cd ~/workspaces/mstar_xpu
# 两卡配置（自动选择空闲卡；卡数不足时不运行）
scripts/run_one.sh results/<run> mstar-cfgpar mstar configs/mstar/bagel_cfgpar_2gpu.yaml 2 t2i,i2t_1tok,i2t_64tok,i2i
# 三卡 graph 研究
scripts/run_queue12.sh   # M* eager/graph，CFG 并行与 TP2（三卡）
scripts/run_queue20.sh   # M* TP2 graph（需先应用 scripts/patches/mstar_xpu_tp2_graph.patch）
scripts/run_queue21.sh   # kernel profile
# 生成报告表格与 PDF（生成的 values 与表格来自 results/，不手抄）
bash report/build.sh     # 先运行 make_tex_tables.py 与 make_kernel_table.py，再编译 report/main.pdf
```

数据目录：

- 两卡最终矩阵：`results/20261122_best_cfg/final/<label>/timings.csv`；I2T 等量：`results/20261122_i2t_parity/`；I2I 等量：`results/20261122_i2i_diag/`；缓存与 connector：`results/20261122_factors/`。
- 三卡 graph：`results/20261123_graph/`（M* CFG 并行 eager/graph；Omni two-stage TP2 eager/graph；Omni I2I 三路 CFG）；`results/20261124_tp2graph/`（M* TP2 eager 复测与 graph 补丁后的结果）。
- kernel：`results/20261124_kprof2/`（Omni）、`results/20261125_kprof3/`（M* 片段）。
- 配置：`configs/mstar/bagel_{tp2,cfgpar}_{2,3}gpu{,_graph}.yaml`，`configs/vllm_omni/bagel_*.yaml`。
