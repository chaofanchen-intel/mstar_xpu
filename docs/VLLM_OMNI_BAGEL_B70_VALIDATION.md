# M* vs vLLM-Omni：BAGEL-7B-MoT 在 Intel Arc Pro B70 上的性能对比（2026-11-22 复核版）

> **2026-11-22 晚些时候更新：** 本文是"同默认配置（2 卡 TP2）"的对比，Omni 为 v0.32.0（旧环境 `vllm-omni-v0.32.0`）。源码随后已更新到 vLLM-Omni main / vLLM 0.34，并做了"各自最优配置"的对比，**结论以 [BAGEL_B70_BEST_CONFIG_COMPARISON.md](BAGEL_B70_BEST_CONFIG_COMPARISON.md) 为准**——其中发现本文 TP2 的结论强烈依赖拓扑（CFG 并行比 TP2 快 40% 以上）。本文中的 OpenCL 根因/修复、初版审计（§2、§3）仍然有效。**另：本文及 §1、§5 中 M* 的 I2T 数字用了 `image_preprocess:"vllm"`（4900 patch），而 Omni Thinker 的 I2T 只有 1386 token，I2T 部分不等量、作废；等量的 I2T 数据见 best-config 文档 §9 与 `report/`。**

**取代** 2026-11-18 初版（已归档到 `results/archive/20261118_initial/`）。本版重新验证、统一了两个框架的工作量，并修复了 M* 的 OpenCL 问题。

## 1. 结论

1. **初版结果不能直接采信。** 它的 I2T / I2I 工作量在两个框架间并不对等（见 §2），T2I 部分的参数是对的，但 M* 的数字来自强制 MATH SDPA + 关闭 `torch.compile` 的 workaround，且与本次在同一环境下的复测不一致。
2. **OpenCL 报错已查明并修复，不需要安装任何新东西**：原因是 conda 版 FFmpeg 带进来的 `libOpenCL.so.1` 抢在系统 ICD loader 之前被加载。`export OCL_ICD_VENDORS=/etc/OpenCL/vendors` 即可（已写入 `scripts/env.sh`）。对性能没有负面影响（§3）。
3. **统一工作量后的对比（2 张卡、串行单请求、eager，M* 为默认配置）：**
   - **图像生成（T2I / I2I）：M* 延迟更低**——T2I 低约 27%（vs Omni 单 stage）/ 30%（vs Omni two-stage），I2I 低约 26% / 43%。
   - **I2T 前缀（1 token，ViT + prefill）：M* 低约 30% / 37%。**
   - **I2T 长解码（64 token）：Omni two-stage 比 M* 默认快约 11%**（3.35 s vs 3.75 s）；M* 关闭 `torch.compile` 后（3.04 s）反过来比 Omni 快约 9%。
4. **只有 2 张卡可用**（卡 0 不可用），所以 3 卡配置（M* 官方 `bagel_xpu_tp2.yaml`、Omni Thinker-TP2 + DiT）本次**没有运行**，初版里的 3 卡数字不再引用。
5. **未解决的疑点：** 本次在卡 1、2 上测得的 T2I，两个框架相对初版（卡 0、1）都多了约 5.3 s（M* MATH+eager 13.90 vs 8.59 s；Omni 17.66 vs ~12.2 s）。已排除 SDPA 后端、OpenCL、`torch.compile`、卡间时钟与算力差异，原因未能确定（§6）。因此**绝对数值只对"本次、卡 1+2"成立**，框架间的相对关系是同会话内对比得到的。

## 2. 对初版的审计

| # | 问题 | 影响 | 本次处理 |
| --- | --- | --- | --- |
| 1 | M* 的 I2T/I2I 没有传 `image_preprocess:"vllm"`。Omni 把 ViT 输入固定缩放到 980×980（**4900 个 patch**），M* 默认只给这张 464×427 的图约 1400 个 patch，VAE 最小边也不同（512 vs 256） | 两边前缀/prefill 的 token 量不同（约 3.5×），I2T/I2I 的差距被夸大或扭曲 | M* 所有带图请求都发 `image_preprocess:"vllm"`（该模式与 Omni 的预处理公式一致，已对照源码） |
| 2 | M* 的 I2I 输出尺寸取自输入图（初版是 512×464），Omni 按请求给 512×512 | 去噪 token 数 928 vs 1024，I2I 不对等 | 输入图统一为 512×512 方图（`assets/bagel_input_512.png`），两边预处理和输出尺寸完全一致；client 校验输出尺寸 |
| 3 | Omni 单 stage 的 I2T 同时对输入图做 VAE + ViT 预填充，且答案是错的（把百吉饼描述成"碗里的奶油汤"，本次复测是"炸鸡"并带 `<think>`）；还会在 25~30 token 提前结束 | 它既做了多余的 VAE 编码，又不是 M* 那样的纯理解路径；初版的 I2T 对比（0.275 vs 1.639 s）没有意义 | I2T 改用 Omni **two-stage** 的 Thinker（vLLM 原生 AR，ViT-only，caption 正确，`ignore_eos` 下精确 64 token）。单 stage 的 I2T 只列出，标注为无效 |
| 4 | 整套 workaround（MATH SDPA + 把 `torch.compile` 变成空操作）改变了 M* 的执行方式：M* 默认会对 LLM/ViT/VAE 做 `torch.compile` | M* 数字不是默认配置结果 | 去掉 workaround，用修复后的默认路径；workaround 只作为 A/B 诊断单独标注 |
| 5 | M* 的步数/shift 不在请求里，靠默认值（50 / 3.0） | 隐式 | 已对照源码确认（`BagelModelConfig.num_timesteps=50`、`timestep_shift` 固定 3.0）；client 对 M* 拒绝非 50 步，Omni 显式传 50 / 3.0 |
| 6 | 早期 Omni two-stage 日志混入了失败的尝试、请求逐次响应文件 100+ 个 | 难以复核 | 重组目录；每次运行只保留 `timings.csv`、`manifest.json`、样例输出、服务日志、GPU 时间线 |

初版里**没有问题**的部分：T2I 的 CFG 设置（text 4.0 / img 1.0 / 区间 [0.4,1.0] / renorm global）、512×512、50 步、warmup 与稳态分开、串行计时，这些本次沿用。

## 3. OpenCL 问题：根因、修复、性能影响

**现象：** M* 默认 worker 的 SDPA 报 `Intel(R) Arc(TM) Pro B70 Graphics OpenCL device not found … could not create a primitive`（oneDNN `src/gpu/intel/sycl/utils.cpp:266`）。

**根因：** oneDNN 的 SYCL(Level Zero) 运行时要通过 OpenCL 找到同一块设备。`clinfo` 显示 `Number of platforms 0`，`OCL_ICD_DEBUG=7` 显示 ICD loader 读取的是 `~/workspaces/.native/ffmpeg/etc/OpenCL/vendors`（空目录）而不是 `/etc/OpenCL/vendors`。原因：用 micromamba 装 FFmpeg 时带入了 conda 的 `ocl-icd`（`ffmpeg/lib/libOpenCL.so.1`），而 `.bashrc` 把 `ffmpeg/lib` 加进了 `LD_LIBRARY_PATH`，它遮蔽了系统的 `libOpenCL.so.1`。去掉 `LD_LIBRARY_PATH` 后 `clinfo -l` 能看到 3 张 B70。系统的 `intel-opencl-icd 26.18`、L0 driver、IGC 都是好的，**不是驱动/包缺失问题，不需要 `apt install`**。

**修复（任选其一，已验证）：**
- 进程级：`export OCL_ICD_VENDORS=/etc/OpenCL/vendors`（`scripts/env.sh` 已设置，所有脚本自动生效；最小复现见 `scripts/diagnostics/ocl_sdpa_smoke.py`：修复前 ViT/VAE 形状的 SDPA 都 `could not create a primitive`，修复后都 OK）。
- 一劳永逸（尚未执行，需要你决定）：把 `ffmpeg/lib/libOpenCL.so*` 挪走（系统已有 `ocl-icd-libopencl1`，FFmpeg 会回落到它），或在 `.bashrc` 里加上面那行 `export`。

**会不会影响性能：** 修复本身不引入开销。数据（卡 1+2，中位数）：

| 场景 | 修复前（workaround：强制 MATH SDPA） | 修复后（默认 oneDNN SDPA） |
| --- | ---: | ---: |
| M* T2I | 12.957 s | 12.954 s |
| M* I2T 1 token（含 4900-patch ViT） | 1.445 s | **0.962 s** |
| Omni 单 stage T2I | 17.660 s（MATH） | 17.656 s（修复），17.651 s（不修复） |
| Omni 单 stage I2T 1 token | 2.041 s（MATH） | 1.376 s（修复），1.376 s（不修复） |

结论：T2I 与 SDPA 后端无关；带 ViT 的路径上，修复后的默认 SDPA 反而比 MATH 快约 1.5×。Omni 在不修复时也能运行（它没有走受影响的 oneDNN 路径），所以这个问题只让 M* 默认配置跑不起来，并不使 Omni 占便宜。**初版里"为一致性同时给 Omni 也强制 MATH"的做法实际上让 Omni 的 I2T 变慢了约 1.5×。**

## 4. 测试方法

**硬件/环境：** 3 × Arc Pro B70；卡 0 不可用，只用空闲的**卡 1、2**。运行前由 `scripts/gpu_guard.sh` 检查并只选择空闲卡；卡 1、2 的显存峰值与各框架自身占用一致，**没有外来负载污染**（`gpu_timeline.csv`）。三卡拓扑对称（单 NUMA，两两 NODE）。PyTorch 2.14.0+xpu / vLLM-Omni 0.32.0（commit 5917730）；M* 为 `external/mstar` main（2c8ffde8f，PyTorch 2.13.0+xpu）。BF16，权重相同（BAGEL-7B-MoT 快照）。

**拓扑（均为 2 张卡）：**

| 标签 | 拓扑 | 备注 |
| --- | --- | --- |
| `mstar-tp2` | LLM TP2（卡 1、2），ViT/VAE 与 rank 0 同卡 | M* 官方 XPU 配置的 2 卡版（官方是 ViT/VAE 单独占第 3 张卡），eager（`accelerator_graph:false`），**默认 `torch.compile` 开启**（M* 无官方开关） |
| `mstar-tp2-nocompile` | 同上，`torch.compile` 空操作 | 仅作诊断 |
| `omni-single-tp2` | 单 stage（整模型）TP2，卡 1、2 | eager（`enforce_eager`） |
| `omni-two-stage-2gpu` | Stage 0 Thinker（卡 1）+ Stage 1 DiT（卡 2），无 TP | eager；Thinker 单卡仅 1.03 GiB KV 空间（`gpu_memory_utilization` 0.95），两卡都占约 30.7 GiB，非常紧；初版的 Thinker TP2 需要第 3 张卡，本次无法运行 |

**统一工作量（`scripts/bench_bagel.py`，同一个 client 两个框架共用）：**

| case | 定义 |
| --- | --- |
| `t2i` | prompt `A red ceramic cup on a white table`，512×512，50 timestep（49 次更新），shift 3.0，CFG text 4.0 / img 1.0，区间 [0.4,1.0]，renorm `global`（区间外 1 分支、区间内 2 分支） |
| `i2t_1tok` | 512×512 输入图 + 一句话描述问题，贪心，新生成 1 token（ViT + prefill） |
| `i2t_64tok` | 同一输入图 + 详细描述问题，贪心，新生成 64 token；M* 与 Omni two-stage 用 `ignore_eos`，token 数用 usage / tokenizer 计数并校验 |
| `i2i` | 同一输入图，输出 512×512，50 步，CFG text 4.0 / img 2.0，区间 [0,1]，renorm `text_channel`（每步 3 分支） |

每个 case：2 次 warmup + 5 次稳态，串行 HTTP 端到端计时；client 校验响应是合法 PNG 且尺寸等于请求，文本 case 记录输出 token 数。输入预处理对齐方式见 §2。M* `/generate` 不暴露随机 seed，因此只比较延迟，不比较图像内容。

## 5. 结果（卡 1+2，稳态中位数，秒；括号为 min–max，n=5）

| case | M* TP2（默认） | Omni 单 stage TP2 | Omni two-stage（1+1 卡） | *M* TP2 关 compile（诊断）* |
| --- | ---: | ---: | ---: | ---: |
| T2I 512² 50 步 | **12.954** (12.944–12.959) | 17.656 (17.653–17.659) | 18.528 (18.411–18.558) | 13.905 |
| I2T 1 token | **0.962** (0.960–0.971) | 1.376 ¹ | 1.523 (1.517–1.525) | 0.947 |
| I2T 64 token | 3.754 (3.749–3.768) | 3.423 ¹ ² | **3.347** (3.338–3.349) | 3.040 |
| I2I 512² 50 步 | **22.817** (22.802–22.819) | 30.740 (30.736–30.742) | 40.125 (40.114–40.133) | 24.666 |

¹ Omni 单 stage 的 I2T 输出是错的（"炸鸡"、`<think>` 前缀）且先对图做 VAE，**不作为有效对比**。² 只生成了 25/27/30 token（单 stage 不支持 `ignore_eos`）。

M* 相对 Omni 的延迟（M* 更低为正）：

| case | vs Omni 单 stage | vs Omni two-stage |
| --- | ---: | ---: |
| T2I | 低 26.6%（Omni 慢 1.36×） | 低 30.1%（1.43×） |
| I2I | 低 25.8%（1.35×） | 低 43.1%（1.76×） |
| I2T 1 token | （无效） | 低 36.8%（1.58×） |
| I2T 64 token | （无效） | **高 12.1%**（Omni two-stage 快 10.8%） |

首个请求（含 JIT/compile，单位 s）：M* T2I 24.7 / I2I 52.5 / I2T 4.2；Omni 单 stage 18.2 / 30.9 / 2.3；Omni two-stage 30.4 / 52.5 / 1.8。M* 默认的首请求最贵，因为要为各形状触发 `torch.compile`。

显存（benchmark 结束时 xpu-smi 读数）：M* 两卡 19657 / 19052 MiB（≈19.2 / 18.6 GiB）；Omni 单 stage 18528 / 18322 MiB（≈18.1 / 17.9 GiB）；Omni two-stage 31480 / 31500 MiB（≈30.7 GiB，接近 32 GiB 上限）。

**怎么理解：**
- 图像生成里 M* 更快，且 Omni two-stage 比单 stage 还慢（I2I 慢 30%），多出来的是跨 stage 的 KV 传输/序列化和只在一张卡上做 DiT；它的好处是 Thinker/DiT 解耦，这里的串行单请求测试看不到。
- 64 token 解码里 Omni two-stage 的 Thinker 是单卡 TP1，没有 all-reduce；M* 是 TP2，每个 token 要跨卡通信，且默认 `torch.compile` 在这个形状上不占优（关掉后 3.04 s）。因此这一项更像"TP 通信 vs 单卡"的差别，而不是框架实现的优劣。
- 由于 Omni two-stage 和 M* 使用卡的方式不同（1+1 vs TP2），**"2 张卡"只保证资源总量相同，不保证并行度相同**。

## 6. 诊断与未解决的疑点

1. **SDPA 后端、OpenCL、`torch.compile` 都不是 T2I 变慢的原因**：
   - Omni 单 stage T2I：默认 17.656 / 不修 OpenCL 17.651 / 强制 MATH 17.660。
   - M* T2I：默认 12.954 / 强制 MATH 12.957 / 关 compile 13.905 / 关 compile + MATH（与初版 workaround 完全相同）**13.901**，而初版同一设置在卡 0、1 上是 8.59 s。
2. 与初版相比两个框架都多出约 **5.3 s**（M*：13.90−8.59=5.31 s；Omni：17.66−12.2≈5.4 s），是常数偏移而不是倍数。
3. 已排除：卡 1、卡 2 单卡性能完全一致（bf16 GEMM 150 TFLOPS、小算子 48.8k op/s，两卡相同；负载下 GPU 时钟 2.5–2.8 GHz）；主机空闲（load≈1，governor=performance）；PCIe 链路各卡一致。
4. 线索：(a) 本次 TP2 的 all-reduce 微基准在卡 1↔2 之间 7 MB 要 1.30 ms（约 5.6 GB/s），而 T2I 每次约有 5000 次 all-reduce（≈ 6.5 s）；但 Omni two-stage（无 TP）反而更慢，所以不能把差异归于通信。(b) 运行期间 GPU 利用率采样最高仅 14–22%，说明 eager 路径主要受主机端 kernel 下发限制，对主机 CPU 状态敏感。
5. **没有验证卡 0 的对照**（卡 0 不可用）。结论：**初版的绝对数值与本次不可直接比较**；框架间的相对关系来自本次同会话同卡对比，可信度更高，但最好等卡 0 空闲后在同一对卡 (0,1) 上重跑一次确认（`scripts/run_one.sh` 会自动选空闲卡，原样重跑即可）。

## 7. 本次没做的事 / 限制

- **3 卡配置未运行**（M* 官方三-rank、Omni Thinker-TP2+DiT）：只有 2 张空闲卡，按约定不运行。
- Omni two-stage 开 XPU graph 的失败（`CachingHostAllocator.h:921` 断言）是初版在 Thinker TP2 拓扑下观察到的，本次没有复测，全部用 eager。
- M* 单卡 / 两卡 CFG 并行（`bagel_cfg_parallel_2gpu.yaml`）：整模型权重约 26 GiB，加 KV 与 ViT/VAE 在 32 GiB 卡上太紧，没有尝试。
- 无并发吞吐、图像质量、其他 prompt 的评测；Omni 的 pytest/XPU CI 没跑。
- 以上结论只适用于这台主机、本模型、本次软件版本。

## 8. 文件位置

见 `README.md`。本次数据：`results/20261122_rerun/<label>/{timings.csv,manifest.json,sample_*,server.log,gpu_timeline.csv,effective_config.yaml}`；诊断输出在 `_diagnostics/`、`diag-*`；失败的第一次尝试（相对路径错误、Thinker 0.9 显存预算 KV 为负）保留在 `_failed_first_attempt/`。表格可用 `scripts/make_report.py` 重新生成。
