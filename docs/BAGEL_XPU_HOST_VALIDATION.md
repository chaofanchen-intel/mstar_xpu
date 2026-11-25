# M* BAGEL Intel XPU Host 验证报告

> **2026-11-22 更新（路径与结论修订）：** 本文记录的是 2026-11-18 的 M* Host 环境搭建，搭建与功能验证部分仍然有效。
> 其中“默认 SDPA 路径报 oneDNN `OpenCL device not found`”的原因已查明并修复：conda 版 FFmpeg 目录（`~/workspaces/.native/ffmpeg/lib`，经 `.bashrc` 进入 `LD_LIBRARY_PATH`）自带的 `libOpenCL.so.1` 遮蔽了系统 ICD loader；
> 设置 `OCL_ICD_VENDORS=/etc/OpenCL/vendors`（已写入 `scripts/env.sh`）即可，不再需要 MATH SDPA / 关闭 `torch.compile` 的 workaround。
> 下文第 70 行及“官方三-rank 性能复测”一节的数字使用了该 workaround，已被 [vLLM-Omni BAGEL B70 对比报告](VLLM_OMNI_BAGEL_B70_VALIDATION.md) 取代，仅作历史记录；
> 旧数据与脚本已移到 `results/archive/20261118_initial/`。

> 本报告记录 2026-11-18 在三张 Intel Arc Pro B70 Host 上的验证结果。

## 当前 Host：3 张 Intel Arc Pro B70（2026-11-18）

**源码：** `external/mstar`，commit `2c8ffde8f`（`main`，验证后 checkout 干净）  
**结果：** Host 原生环境搭建成功；按仓库官方三-rank TP2 配置加载 BAGEL，并通过文本生成、图像理解和图像生成请求。当前 Host 未出现显存或主机内存耗尽。

### Host 与模型

| 项目 | 实测 |
| --- | --- |
| OS / kernel | Ubuntu 24.04.4 LTS，Linux `6.17.0-1009-intel`，x86_64 |
| XPU | 3 × Intel(R) Arc(TM) Pro B70 Graphics，每张总内存 31.89 GiB |
| PCI BDF | `0000:18:00.0`、`0000:54:00.0`、`0000:cc:00.0` |
| 主机内存 | 251.3 GiB；验证前可用约 243.6 GiB，推理后约 229.4 GiB |
| Swap | 8 GiB；推理期间仍全部空闲 |
| 模型快照 | `models/cache/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b` |
| `ema.safetensors` | 29,214,685,336 bytes，约 27.21 GiB |
| `ae.safetensors` | 335,304,388 bytes，约 0.312 GiB |
| LLM BF16 权重 | safetensors 元数据约 26.341 GiB；TP2 后每个 LLM rank 约 13.17 GiB |

官方配置 `external/mstar/configs/bagel_xpu_tp2.yaml` 把 VAE encoder/decoder 与 ViT 放在 rank 0，LLM tensor parallel 放在 rank 1、2；实际日志确认三个 worker 分别运行于 `xpu:0`、`xpu:1`、`xpu:2`，attention backend 为 `xpu_paged`。每个 LLM rank 为 1024 页 × 64 slots 的 KV cache 分配约 1.75 GiB。

### 环境与验证

环境位于 `~/workspaces/mstar_xpu/.venv`，没有使用 Docker。软件版本：

- Python 3.12.3，uv 0.13.9，Rust 1.99.1
- PyTorch 2.13.0+xpu，torchvision 0.28.0+xpu，torchaudio 2.11.0+xpu
- torchcodec 0.16.0+cpu，torchcodec-xpu 0.16.0
- vllm-xpu-kernels 0.1.15.4，Transformers 5.23.0
- mstar-ai 0.2.1（本地源码），mstar-rust 0.1.1；`mstar-server` 从源码构建

环境搭建命令（Rustup 使用仓库固定的 1.99.1 工具链；`.bashrc` 已把下载缓存放到工作区）：

```bash
cd ~/workspaces/mstar_xpu/external/mstar
uv venv --python 3.12 --seed ~/workspaces/mstar_xpu/.venv
uv pip install --python ~/workspaces/mstar_xpu/.venv/bin/python --torch-backend=xpu \
  --overrides <(printf 'torch==%s\n' '2.13.0+xpu') \
  'torch==2.13.0+xpu' 'torchvision==0.28.0+xpu' 'torchaudio==2.11.0+xpu' \
  'torchcodec==0.16.0+cpu' 'torchcodec-xpu==0.16.0' '.[bagel_xpu]' ./rust \
  msgspec blake3 py-cpuinfo compressed_tensors gguf partial_json_parser tabulate
cargo build --release --locked --manifest-path rust/server/Cargo.toml
install -m 755 rust/server/target/release/mstar-server \
  ~/workspaces/mstar_xpu/.venv/bin/mstar-server
MAMBA_ROOT_PREFIX=~/workspaces/.mamba ~/workspaces/.local/bin/micromamba create \
  -y -p ~/workspaces/.native/ffmpeg -c conda-forge 'ffmpeg=6.1'
```

已通过的检查：

1. `torch.xpu.is_available() == True`，设备数为 3，设备均识别为 Intel Arc Pro B70。
2. XPU 0 上 256×256 FP32 矩阵乘，以及 XPU 1 上 256×256 BF16 矩阵乘均成功，结果为 `256.0`。
3. M*、`mstar_rust`、BAGEL、`vllm_xpu_kernels._C`、`vllm_xpu_kernels._xpu_C` 导入成功。
4. 官方 TP2 服务完成全部 worker 初始化，日志显示 `All workers ready`。
5. OpenAI-compatible 文本请求返回 `XPU_OK`，生成 4 tokens。
6. `test/manual/bagel/image_understanding_request.py` 成功理解 `test/manual/bagel/bagel.png`，回答识别出带奶油奶酪的面包圈、种子和蓝盘。
7. `test/manual/bagel/image_gen_request.py` 成功生成并打开验证 1024×1024 PNG；临时图片已删除。
8. 安装 FFmpeg 后，`torchcodec_xpu` 默认自动加载成功；不禁用 TorchCodec 的 M* 服务完成三卡初始化，并通过 `FFMPEG_OK` 文本推理请求。

### 官方三-rank T2I 性能复测（2026-11-18）

使用仓库官方配置 `external/mstar/configs/bagel_xpu_tp2.yaml`，三张 B70 全部可见：VAE/ViT 在 rank 0，LLM tensor parallel 仍为 TP2、使用 rank 1 和 2。这个“三-rank”表示 3 个 M* worker/rank，不表示 3 个 CFG 分支；本次 T2I 在 CFG 区间内仍是主分支 + text-CFG 两分支，区间外只有主分支。

同一 `A red ceramic cup on a white table` prompt、512×512、50 timesteps（49 次更新），warmup 13.865 s，稳态 13.867、13.866、13.876 s，中位数 **13.867 s**。同一轮两卡 M* eager/MATH SDPA TP2 为 8.540、8.587、8.615 s，中位数 **8.587 s**。此前已报告的三-rank结果保留为历史记录：12.878 s，样本 12.876、12.871、12.882 s；本轮没有在默认 SDPA 后端复现这组数值。

本轮默认 M* 服务路径在 VAE/ViT worker 的 SDPA 调用报 oneDNN `OpenCL device not found` / `could not create a primitive`。为完成端到端测量，使用 `results/archive/20261118_initial/data/eager_compile_bypass/sitecustomize.py` 在 benchmark 子进程树中关闭 `torch.compile` 并强制 PyTorch MATH SDPA；这是诊断用 workaround，不是默认 M* 性能结果，也没有修改 M* 源码或驱动。两卡/三卡拓扑比较及三类任务的完整样本见 [vLLM-Omni BAGEL B70 验证报告](VLLM_OMNI_BAGEL_B70_VALIDATION.md)。

### 内存与运行状态

| 时点 | XPU 0 free / total | XPU 1 free / total | XPU 2 free / total | 主机 MemAvailable | SwapFree |
| --- | ---: | ---: | ---: | ---: | ---: |
| 加载前 | 31.67 / 31.89 GiB | 31.85 / 31.89 GiB | 31.85 / 31.89 GiB | 约 243.6 GiB | 约 8 GiB |
| worker 就绪并完成文本/图像理解后 | 28.76 / 31.89 GiB | 13.39 / 31.89 GiB | 13.39 / 31.89 GiB | 约 229.4 GiB | 约 8 GiB |
| 服务停止后 | 31.67 / 31.89 GiB | 31.85 / 31.89 GiB | 31.85 / 31.89 GiB | - | - |

加载后两个 LLM rank 各使用约 18.5 GiB，其中约 13.17 GiB 是分片权重、约 1.75 GiB 是 KV cache，其余为运行时/工作区。测试后关闭服务，三张卡的空闲显存恢复至基线。没有观测到显存不足、主机内存耗尽或 swap 增长。

### 其他 Host 限制

- 当前 VS Code Server 进程是在加入 `render` 组前启动的；`id -Gn` 在现有会话中仍不含 `render`。通过 `sg render -c ...` 可访问设备，但 `sg` 会过滤继承的 `LD_LIBRARY_PATH`，因此需在 `sg -c` 的命令字符串内显式设置 FFmpeg 的 `PATH` 和 `LD_LIBRARY_PATH`。重新连接 SSH/VS Code 后，新会话会直接读取 `.bashrc`。
- 已在 `~/workspaces/.native/ffmpeg` 安装 FFmpeg 6.1.2 和 `libav*` shared libraries。`torchcodec_xpu` 默认自动加载已通过；不再需要 `TORCHCODEC_DEVICE_BACKEND_AUTOLOAD=0`。实际音视频解码/编码请求未测试。
- 上游 `flash-attn` 面向 CUDA/ROCm，不兼容当前 `torch 2.13.0+xpu`（`torch.version.cuda is None`），因此没有把不兼容扩展装进 venv。BAGEL ViT 使用 PyTorch SDPA 回退，功能正常但会较慢；这是可选性能优化，不是 XPU 运行依赖。
- oneCCL 输出 MPI launcher 环境变量缺失及检测到 PCIe 拓扑的 warning；三卡 TP2 初始化和请求均成功，这些 warning 未阻塞验证。
- Rust HTTP 前端构建成功；有两条 `unused_mut` 编译 warning。未运行 pytest，本次以实际服务启动及三类请求作为集成验证。

### 工作区存储规则

`~/workspaces` 是指向 `/mnt/models/chaofan` 的软链接，所在文件系统总量 7.2 TiB、剩余约 5.4 TiB。`.bashrc` 已将 uv/pip、HF、ModelScope、Torch、Triton、TorchInductor、npm、Cargo/Rustup、micromamba 缓存及临时目录设置到 `~/workspaces`；PATH 包含 FFmpeg、`~/workspaces/.local/bin` 和 `~/workspaces/.cargo/bin`，`LD_LIBRARY_PATH` 包含 FFmpeg shared-library 目录。此外，`~/.local`、`~/.cache`、`~/.vscode-server` 与 `~/Downloads` 均链接到 `~/workspaces` 对应目录。迁移后根分区剩余约 9.2 GiB。

### 复现命令

```bash
cd ~/workspaces/mstar_xpu/external/mstar
sg render -c 'PATH=/home/chaofan/workspaces/.native/ffmpeg/bin:$PATH LD_LIBRARY_PATH=/home/chaofan/workspaces/.native/ffmpeg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} HF_HUB_OFFLINE=1 ~/workspaces/mstar_xpu/.venv/bin/mstar serve bagel --config ~/workspaces/mstar_xpu/external/mstar/configs/bagel_xpu_tp2.yaml --cache-dir ~/workspaces/mstar_xpu/models/cache --host 127.0.0.1 --port 18765 -O0 --log-level INFO'
```

服务就绪后，可在另一个终端运行仓库的图像理解脚本：

```bash
cd ~/workspaces/mstar_xpu/external/mstar
HOST=127.0.0.1 PORT=18765 ~/workspaces/mstar_xpu/.venv/bin/python test/manual/bagel/image_understanding_request.py
```

