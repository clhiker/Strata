# 本地实测与调优记录 —— Strata @ 2×RTX 2080 Ti / Xeon E5-2696 v3

2026-10-06 归档。记录这台机器上跑 Strata 的全部实测数字、调优结论、试过但无效的方向，
以及为什么**某类量化（4-bit）在这台机器上不合适**。所有数字都注明测的是什么配置；
没测过的写"未测"，不做推测性结论。

## 0. 这台机器

| | |
|---|---|
| GPU | 2× RTX 2080 Ti 22 GB（Turing, sm_75），功耗墙 250 W（max 280），PCIe 3.0 x16 各一条，NVLink NV2 存在但本模型用不到 |
| CPU | 双路 Intel Xeon E5-2696 v3（Haswell-EP，18 核 ×2 = 36 物理核 / 72 线程，**只有 AVX2，无 AVX-512**） |
| 内存 | 125 GB；两个 NUMA 节点（node0 64 GB / node1 64 GB） |
| 存储 | `/` = NVMe（归档时约 96 GB 可用）；**`/D` = 机械硬盘 1.8 TB**（rotational=1） |
| 软件 | CUDA 13.0；Strata v0.1.38 自建（`-DCMAKE_CUDA_ARCHITECTURES=75`，`STRATA_GGML_DIR=third_party/llama.cpp`，钉住 commit `3cf0325`） |
| 构建产物 | `build/strata`（40 MB，含 IQ3_XXS/IQ4_NL/IQ2_XS MMQ 与 Q4_K/Q5_1 路径）、`build-vision/bin/strata-vision`（74 MB，llama.cpp mtmd） |

## 1. 结论摘要

1. **日常用 OrcaRouter IQ3_XXS + 256K 原生窗口**：实测 decode 50–73 tok/s（取决于采样与主机负载）、prefill ~1,600 tok/s（用户实测，256K 档），两卡分层、97% 以上路由命中。
2. **Unsloth UD-Q4_K_XL（4-bit）不适合这台机器**：单卡 **5.1 tok/s**，双卡分层 **10 tok/s**（用户实测）——即使把 9,720 个专家槽位、97.3% 路由质量都拿到手，速度仍不到 3-bit 档的五分之一。原因见 §2.2。
3. **本机可用的 +10% 免费提升**（`tools/calibrate.py` 单次加载内交错复测）：`--pcie-frac 0.20`、`--spec-min-p 0.70`。已写进两个 Orca 档。
4. **这台机器上 decode 最大的变量是"主机争用"，不是显存、不是功耗、不是 NUMA**：同一配置实测 **43 ↔ 73 tok/s**，差异全在引擎自报的 `pool+plan`（16.1 ↔ 4.0 ms/window）。
5. **44 GB 显存对 IQ3_XXS 已经饱和**：引擎自报 `PCIe 0.00`、每层每窗口只有 0.2–0.45 个专家落到 CPU、19–31 个来自显存。

## 2. 两个模型、两套实测

### 2.1 OrcaRouter Qwen3.8-Flash-Next-Uncensored IQ3_XXS（推荐）

**模型**：`orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` 的 IQ3_XXS（gated 仓库，需 HF_TOKEN）
- 分片1 41.57 GiB + 分片2 37.78 GiB = 79.35 GiB；MTP draft 3.85 GiB；mmproj 866 MiB（视觉）；落在 `~/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS/`
- 打包：`tools/iq_pack.py --compat-bf16` → `packs/orca-iq3_xxs`（1.43 GiB arena；1079 张量、302 原生、**460 转换**（264 精确 / 196 舍入，max |err| 0.0144）；48 层专家与 28.8 GB PLE 表字节未动）
- MTP：`mtp_fetch`（31 张量，SHA256 全过）→ `mtp_pack --experts q2_0` → `mtp_rt` → `mtp/rt`（787 MB）

**实测（本机）**

| 配置 | decode | 备注 |
|---|---:|---|
| 256K 原生，贪心，安静主机 | **66.8 / 70.1 / 73.2 tok/s** | 三次独立加载，中间那次含探针噪声 |
| 256K 原生，采样 T=0.7 | **59.0 / 61.5 / 65.9 tok/s** | 同加载内对比：采样比贪心低 ~9% |
| 256K 原生，主机忙时 | 43.2 tok/s | window 47 ms、pool+plan 16.1 ms |
| 256K 原生，安静 | 73.2 tok/s | window 29 ms、pool+plan 4.0 ms |
| prefill（256K 档，用户实测） | **~1,600 tok/s** | 用户自测 |
| 用户日常（60K 输入） | 50–62 tok/s，草稿接受 72–89% | GUI/DSH 采样路径 |

**512K / 其它窗口**：`--context` 可配（见 §4）；512K = yarn factor 2，引擎自报 `freq_scale 0.5, mscale 1.069315`；运行正常（用户实测）。1M 未测。

### 2.2 Unsloth UD-Q4_K_XL（不适合本机）

**模型**：`/D/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/`（4 分片，111,334,654,784 B，与 `docs/UNSLOTH_Q4.md` 表格逐个吻合）
- 打包：`iq_pack --compat-bf16` → `packs/ud-q4_k_xl`（1.38 GiB arena；303 原生、460 转换（264 精确 / 196 舍入，max |err| 0.0144）；`native_experts.txt` v4，第 11 层跨分片）

| 配置 | 专家槽位 | 路由质量 | decode | 加载耗时 | 内存 |
|---|---:|---:|---:|---:|---|
| **单卡**（卡1）+ `--resident-budget-gib 71` | 3,622 | **54.3%**（CPU 池吃 45.7%） | **5.1 tok/s** | ~35 min（读盘 61 GiB 补齐集） | 61.17 GiB 专家全部 page-locked 在 RAM（文件只读 109 个 blob） |
| **双卡分层**（K=24）+ `--kv-resident 131072`，无预算 | **9,720** | **97.3%** | **10 tok/s**（用户实测） | ~13 min（读盘 65.6 GB） | 81 GB 占用 / 可用 37 GB |

**为什么不适合**（三条，都是结构性的）：
1. **CPU 池没有 AVX-512**：4-bit 专家（Q4_K / Q5_1）落到 CPU 就很贵。单卡时 45.7% 的路由质量在 CPU 上，直接把速度压到 5 tok/s；即使双卡把 CPU 份额压到 ~2.7%，也只到 10 tok/s。
2. **Turing + 22 GB/卡，对 4-bit 模型太小**：这个文件的 dense 权重很大（Q8_0 注意力/共享专家 + 675 MB 的 Q5_K 输出头），单卡 22.5 GB 里只有 10.55 GiB 留给专家缓存；256K 的 KV 再吃掉 3.3 GB。
3. **111 GB 模型放在机械盘 `/D`**：`/` 的 NVMe 只有 96 GB 空闲，装不下；于是每次启动都要按专家随机读盘（实测 ~17 MB/s，而预算路径按层顺序读约 25 MB/s）→ 加载 13–35 分钟。
   
   （对照：`docs/UNSLOTH_Q4.md` 在 64 GB + RTX 5070 + NVMe + AVX-512 的机器上是 7–8.5 tok/s，主导因素正是"SSD 读取"。我们把它换成了"CPU 池"，结论同样是：这档在这台机器上没有优势。）

## 3. 调优：测出来的、和没用的

### 3.1 本机校准（+10%，已生效）

`tools/calibrate.py strata-orca-256k.json`（658 s）。可信的是"交错复测"那一段：

| 设置 | 三次实测 | 中位 |
|---|---|---:|
| 默认（pcie 0.28 / min-p 0.50） | 60.7 / 60.0 / 57.5 | ~60 |
| **调优（pcie 0.20 / min-p 0.70）** | 65.9 / 66.2 / 64.4 | **~65.9（+10%）** |

- `--pcie-frac` 曲线很陡：0.55（默认）→ 55.7，0.28（引擎自动探测）→ 63.4，**0.20 → 64.4**，0.0 → 62.9，0.75 → 54.5
- `--spec-min-p`：0.30 → 62.2，0.50 → 62.8，**0.70 → 67.1**（贪心）
- `--pool-workers`：35（默认）→ 33.6/42.5，23 → 40.3/40.0，18 → 29.2/46.1 → **噪声最大的一项**，只是"略优"
- 采样（T=0.7）下复测 draft floor：0.50 → 62.6，**0.70 → 65.8**，0.85 → 64.5，0.95 → 63.5 → 0.70 在两套采样下都最优

已固化进 `strata-orca-256k.json` 与 `strata-orca-512k.json`：`--pcie-frac 0.20 --spec-min-p 0.70 --pool-workers 23`（外加 `--prefill auto`）。

### 3.2 方法学（比结论更重要）

**逐次加载之间的波动达 ±25%**：同一个配置三次独立加载测出 **43.2 / 59.8 / 66.8 tok/s**（缓存大小都是 16.64 GiB）。
因此**只有"单次加载内、多轮交错"的比较才可信**；跨加载比较会把 10% 级别的差异完全淹没。
以下结论都是这样被判定的：`STRATA_ADAPT_NOWAIT=1`（+9.6%？）和 `--spec 6`（+4.2%）**都在噪声里，未采用**。

### 3.3 decode 的时间都去哪了（引擎自报，单次加载内）

```
66 windows, 1.94 tokens/window, 47.06 ms/window
 = verify 42.79 (GPU-reach wait 9.85 + per-layer host 7.09 [plan 0.89 actq 1.42 jobs 1.23 CPU 6.83] + stage 0.34)
 + commit/emit 1.31 + draft 2.39
 per layer-window: CPU experts 0.24 (0.25 entries), VRAM hits 22.78, PCIe 0.00
```

安静时同一结构：`window 29 ms`、`GPU-reach wait ~10.5 ms/stage`、`pool+plan 4.0 ms/stage`、`PCIe 0.00`。
**读法**：`pool+plan` 一变大就是主机被抢（43 tok/s 那次是 16.1 ms）；`PCIe 0.00` 说明显存驻留早已饱和。
开启方式：配置里加 `"env": {"STRATA_DECODE_TIMING": "1", "STRATA_SPLIT_TIMING": "1"}`。

GPU 采样（同一次运行、每秒）：SM 300/1350/1890 MHz、功耗中位 **64–73 W**、峰值 180–221 W（墙 250 W）、利用率中位 **0%**
→ **不是功耗/频率受限**，卡大部分时间在等主机。

### 3.4 试过但无效或更差（都测过，别再试）

| 方向 | 实测结果 |
|---|---|
| `--coupled-draft`（采样下的耦合草稿） | **−3%**（57.0 vs 59.0，同加载同采样） |
| `STRATA_ADAPT_NOWAIT=1` | 落在 ±25% 噪声内，未采用（且它会引入时序相关的舍入差异） |
| `--ple-io ram`（28.8 GB PLE 表锁进 RAM） | 无正面证据；那次"更慢 15%"是 ollama 抢显存造成的假象 |
| `numactl --cpunodebind=1`（把池绑到 arena 所在节点） | **无效**（72.8 vs 基线 70.1/73.2） |
| GPU 功耗墙 250→280 W | **无意义**（功耗中位仅 64–73 W） |
| 张量核做 sm_75 decode GEMV | 仓库自带 bench：**慢 1.3–2.2×**（带宽受限） |
| `--peer-device 1`（NVLink 专家层） | **prefill 1,600 → 1,100 tok/s**，decode 不变 → 分层更优（用户实测） |
| q4_0 草稿头（提高 MTP 接受率） | **走不通**：`tools/mtp_rt.py` 只重排 q2_0（断言 720/180 列） |
| 加显存/加缓存 | `PCIe 0.00`、CPU 专家 0.2–0.45/层窗口 → 已饱和 |

## 4. 文件与命令

**启动脚本** [run-orca-iq3_xxs.sh](../run-orca-iq3_xxs.sh)（名字带 orca，但功能通用：任何档都是 `CONFIG=<档>`）

```bash
./run-orca-iq3_xxs.sh                     # 默认档 = 256K 原生（strata-orca-256k.json）
./run-orca-iq3_xxs.sh --context 512k      # 512K（>256K 自动 yarn，factor = W/262144）
./run-orca-iq3_xxs.sh --no-think          # 服务端默认不思考（写 <档>.shared-settings.json）
./run-orca-iq3_xxs.sh --think high|clear  # 默认思考档 / 清除
./run-orca-iq3_xxs.sh --status | --stop | --restart   # 直接针对"当前在跑的那一个"，不用带 CONFIG
CONFIG=strata-orca-512k.json ./run-orca-iq3_xxs.sh    # 换档启动
```

**配置档**（仓库根，`strata-*.json`）

| 档 | 窗口 | 说明 |
|---|---|---|
| `strata-orca-256k.json` | 262144 原生 | **默认**，含校准参数（`--pcie-frac 0.20 --spec-min-p 0.70 --pool-workers 23`） |
| `strata-orca-512k.json` | 524288 yarn/2 | 同参数，超训练长度自动 yarn factor 2 |
| `strata-run-<tokens>.json` | 任意 | `--context W` 自动生成/维护（也可手写别的档） |

**归档（实测不适合本机，已挪进 `archive/`）**

| 路径 | 内容 |
|---|---|
| `archive/ud-q4_k_xl/strata-ud-q4_k_xl.json` + `.log` | 单卡 + `--resident-budget-gib 71`：3,622 槽位 / 54.3% 命中 / **5.1 tok/s** / 加载 ~35 min |
| `archive/ud-q4_k_xl/strata-ud-q4_k_xl-2gpu.json` + `.log` | 双卡分层 K=24：9,720 槽位 / 97.3% 路由质量 / **10 tok/s**（用户实测）/ 加载 ~13 min |
| `archive/strata-orca-iq3_xxs.layersplit.bak.json` | 早期 128K 分层档备份 |

（用它们只需把路径写回去：`CONFIG=archive/ud-q4_k_xl/strata-ud-q4_k_xl-2gpu.json ./run-orca-iq3_xxs.sh`；
不建议——见 §2.2。`packs/ud-q4_k_xl`（1.4 GiB）保留在原处，重试时不用重新打包。）

## 5. 从零复现（Orca IQ3_XXS）

```bash
# 1) 依赖与引擎
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# llama.cpp 钉住 commit 3cf0325 → third_party/llama.cpp（gguf-py + ggml）
.venv/bin/cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 32
.venv/bin/cmake -G Ninja -S tools/vision -B build-vision -DSTRATA_VISION_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_DIR=$PWD/third_party/llama.cpp && \
  .venv/bin/cmake --build build-vision --target strata-vision -j 32

# 2) 模型（gated：先在 HF 页面接受条款，再给 token）
export HF_TOKEN=hf_xxx HF_ENDPOINT=https://hf-mirror.com
./fetchs/fetch_orca_iq3_xxs.sh --with-mtp

# 3) 打包 + MTP
.venv/bin/python tools/iq_pack.py --gguf ~/models/...-00001-of-00002.gguf \
  --out packs/orca-iq3_xxs --compat-bf16
.venv/bin/python tools/mtp_fetch.py inventory --out mtp && \
.venv/bin/python tools/mtp_fetch.py fetch --out mtp && \
.venv/bin/python tools/mtp_fetch.py verify --out mtp && \
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf && \
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt && \
cp data/draft_vocab.bin mtp/rt/

# 4) 校准（可选，+10%）与启动
.venv/bin/python tools/calibrate.py strata-orca-256k.json
./run-orca-iq3_xxs.sh
```

## 6. 未解决 / 下次可做

- **1M 窗口未实测**（`--context 1m` 会生成 yarn factor 4 的档；内存与质量都没量过）。
- **自定义专家画像**：`--dump-routing` + `tools/make_profile.py` 可按自己的提示词造画像，理论上把命中率从 95% 再往上推；未做。
- **`--draft-vocab en`**（40,525 ids，省 110 MiB、英文快 1–2%）**不建议**：文档明确 CJK 回答会几乎拿不到草稿。
- **读盘问题**：`/D` 是机械盘，任何"专家不全在 RAM"的模式（UD-Q4_K_XL 的预算/流式）都会被它拖死。若以后还要试 4-bit，先把模型放到 NVMe（需要先腾出 `/` 的空间）。
- **长 prompt 的 RAM 水位**：UD 双卡模式把 71.7 GiB 专家放进 RAM 后可用只剩 37 GB；文档提到的 #384 事故就是长 prompt 把 RAM 吃到 0，未在本机复现/验证。

---

# 附：UD-Q4_K_XL 调优实录（2026-10-09）

用户在把模型从机械盘 `/D` 搬到 NVMe 后重新提出目标：**1024pp 下 decode >60、prefill >2000**。
下面是逐项实测与最终判定。硬件同上（2×2080 Ti 22 GB / 双路 E5-2696 v3 无 AVX-512 / 125 GB）。

## 做了什么

| 改动 | 实测结果 |
|---|---|
| 模型在 NVMe（用户搬运） | 加载 **13 min → 90-110 s** ✓ 有效 |
| 重建 `-DSTRATA_MMQ_KQUANTS=ON` | Q4_K/Q5_K/Q5_1 的 prompt MMQ 核编入；prompt GEMM 从"dequant→FP16 GEMM"变为 MMQ，30K prompt 的 `gemm gate/up + down` 共 3.0 s、`dequant` 0.72 s ✓ 有效（且不再有"等拷贝"主导） |
| 窗口 256K → 32K → 10K | 专家槽位 9720 → 10529 → 10633（97.6% 路由质量）；速度几乎不变 |
| `--prefill auto:16384` | 引擎仍自选 8192 chunk（受 lend 区域限制） |
| 大页 `vm.nr_hugepages=36727`（2 MB） | arena 确实用上 `hugetlb 2 MB pages`，**速度无变化** → 排除 |
| `--ple-row-cache 4194304` + `--ple-inflight 512` | PLE 491 → 485 ms；prefill 无变化 → 排除 |
| `STRATA_STAGER_THREADS=32` + `STRATA_STAGER_RING=128` | 无变化（拷贝早已流水化）→ 排除 |
| `STRATA_PREFILL_FUSED=1` | 无变化 → 排除 |
| `--prefill 2048` + `--vram-reserve-mib 128` | **引擎崩溃**：`prefill gemm: cublasCreate: cuBLAS status 1`（workspace 拿不到显存）。默认 700 MiB 不能压 |
| `--prefill 2048` 想省 prompt 缓冲显存 | 无效：日志显示 prompt 路径是**借用缓存槽位**（`borrows 833 slots (2.44 GiB)`），没有常驻缓冲可省 |

## 实测数字（10K 窗口、双卡分层 K=24、32K/10K KV int8、MTP、校准参数）

| 指标 | 结果 |
|---|---|
| **decode @1024pp 稳态**（同一 prompt 连打 4 次） | **51.2 / 65.4 / 68.5 / 68.4 tok/s**，草稿接受 151-156/169-175（≈89%）→ **达标（>60）** |
| decode @1024pp 首次（自适应缓存未热） | 36-45 tok/s |
| **prefill @1.66K** | 283-305 tok/s（wall 5.5-5.9 s） |
| **prefill @8.0K** | 950-965 tok/s（wall 8.4 s） |
| **prefill @30.0K** | **1513 tok/s**（wall 19.0 s） |
| 拟合 | `wall ≈ 4.7 s/chunk + 0.486 ms/token` → 渐近 **≈2058 tok/s** |

## 判定：1024pp 的 prefill >2000 在本硬件上不可达

算术：MoE 的 prompt chunk 必须让该 chunk 用到的**全部**专家参与（1024 tokens × top-10 = 10240 次选择，几乎覆盖 48×512 个专家）。
两卡显存只驻留 **30.1 GiB / 71.7 GiB**，其余 **41.7 GiB 每 chunk 必过一次主机→GPU**：
- PCIe 3.0 x16 满速 15.75 GB/s → 41.7 GiB ≥ **2.8 s**；
- 实测 1.65K prompt 的 wall 5.5 s（含 GPU dense 阶段 2.5 s）；
- → **1024pp 的天花板 ≈ 250-400 tok/s**，与 >2000 相差 5-7 倍。

要把 1024pp 的 prefill 提上去，只有三条路，都不是"调参"能解决的：
1. 专家集变小（换更小的量化 → 那是另一个模型，如 IQ3_XXS 的 42.9 GiB，1024pp 约 500 tok/s，同样到不了 2000）；
2. 显存能装下全部专家（需要 ~72 GB VRAM，本机 44 GB）；
3. 主机→GPU 带宽提高 7 倍（PCIe 3.0 → 不存在）。

**仍然可做的**：让 1024pp 的**首次** decode 也 >60 —— 用 `--dump-routing` + `tools/make_profile.py` 按真实流量造专家画像，或 `--expert-profile-save` 保存自适应所学，使缓存一启动就是热的。

## Round 3（2026-10-09）UD-Q4_K_XL：唯一真正有效的开关，以及为什么 fused 用不上

### 有效的：`STRATA_PREFILL_STREAM_MIN=4096`（1024pp prefill +25%）

prompt chunk 的专家搬运有两种模式：
- **stream-all**（默认，chunk ≥ `stream_all_min`=1024）：按固定顺序把**该层全部 512 个专家**过一遍大 ring（384 槽），让拷贝引擎在注意力阶段也一直工作；
- **routed-only**（默认只用于 <1024 的 chunk）：只暂存该层**实际被路由到**的专家（STAGE=8 环）。

把这个门槛抬到 4096，让 1024-4096 token 的 chunk 走 routed-only：

| | 默认（1024） | STREAM_MIN=4096 | STREAM_MIN=1M（永远） |
|---|---:|---:|---:|
| prefill @1.66K | 287-305 | **354-385（+25%）** | 301-341 |
| prefill @8K | 950-965 | 984 | **715（退步）** |
| 稳态 decode | 59.5-66 | 52.6-67.4 | **29-40（退步）** |

**正确性**：两条路径在相同会话状态下输出**完全一致**（同一提示 256 token，828 字符全同）。
（此前一次 827 vs 807 的差异来自**专家缓存的会话状态**——引擎文档已注明缓存路径的舍入差异，与 staging 路径无关。）

### 为什么 fused（引擎最快的 prompt 核）对本模型无效

`src/prefill/moe_fused_iq.cu`：
```c
bool gu_covered(int t) { return t == T_IQ2_XXS || T_IQ2_XS || T_IQ2_S || T_IQ3_XXS || T_IQ3_S || T_IQ4_XS; }
bool d_covered(int t)  { return t == T_Q2_0 || T_IQ4_NL; }
```
UD-Q4_K_XL 的专家是 **Q4_K / Q5_K（gate/up）+ Q5_1 / Q8_0（down）——全部不在覆盖内**，
所以 `STRATA_PF_FUSED=1` 对本模型的专家层完全无效（需重启前先读代码确认，省掉一次无效实验）。
引擎文档里那些 "fused 512 槽 +19-22%" 的收益只属于 IQ 系模型。

### 其它排除项（都有实测）

| 试的 | 结果 |
|---|---|
| `STRATA_SPLIT_RING=384` | 无变化：分层时 `res_share`=42% < 75% 不触发 96 槽覆盖，pinned-share 规则本来就给 384 |
| 页错误假设 | 排除：一次 1656-token prompt 只产生 46,453 次 minor fault（不是千万级） |
| 主机侧开销 | 实测 **52 核·秒 / 一次 1656-token prompt**（5.7 s 窗口内 ≈9 核满载）→ 是真实的散块 gather/打包，不是缺页也不是纯 memcpy |

### Round 3 结论

- **decode @1024pp >60：达成**（稳态 60.6 / 64.7 / 67.4，草稿接受 ~86%）。
- **prefill @1024pp = 354-385 tok/s**（本轮 +25%），距 >2000 仍差 **5.4×**。
- 该差距是本机硬件的边界：每个 chunk 必须移动 **41.7 GiB** 非驻留专家（两卡显存只驻留 30.1/71.7 GiB），
  PCIe 3.0 x16 满速 15.75 GB/s → ≥2.8 s/chunk；而 1024 token 要达到 2000 tok/s 需要 0.5 s。
  即需要 ~7× 的带宽，或把 71.7 GiB 全放进 44 GB 显存——都不存在。

## Round 4-5（2026-10-09）目标改为 10K 输入 + NVLink：结构性结论

用户把口径改为 **10K 输入 prefill >2000**，并指出"核心是用上 NVLink"。本轮把两种并行结构、NVLink 能力、以及 chunk 流水线全部测清。

### 硬件能力（已确认）
```
nvidia-smi nvlink --status : 两卡各 2 链路 × 25.781 GB/s
nvidia-smi topo -p2p r     : GPU0<->GPU1 OK（P2P 可用）
```

### 10K 输入实测地形（分层 / 16K 窗口 / KV int8 / MTP / 校准参数）

| 配置 | prefill | wall | 说明 |
|---|---:|---:|---|
| 1 块（`--prefill 16384`） | 1005-1027 | 9.1-9.9 s | stage0 4.08 s + stage1 4.52 s **串行**（stage1 的 wall = stage0 wall + 自身 GPU） |
| **2 块（`--prefill 5120`）** | **1027-1060** | 9.5-9.8 s | 有重叠：stage1 的 wall = stage0 的**第一块** + 自身 GPU（两块轮换握手） |
| 3 块（4096） | 804-827 | 12.1 s | 每块都要重扫专家 |
| 4 块（3072） | 676-693 | 14.4 s | 同上，扫描次数线性变差 |
| own buffers + 5120 | 863-896 | 11.1-11.5 s | 每 stage 永久占 3.9 GiB → 驻留专家变少、每块 GPU 2.86→3.57 s |
| **peer 模式（`--peer-device 1`，走 P2P）** | 826-855 | 11.0 s | **wall == GPU timeline**（握手完全重叠），但 CUDA0 要串行跑全部 48 层 dense → GPU 11 s |

### 为什么 NVLink 不是这里的杠杆

- peer 模式（引擎里唯一真正用 P2P/NVLink 的专家通路，`rows over P2P`）prefill 反而更慢：它把 dense 全压在一张卡上（`hc read` 2058 ms、`qsa attn` 1004、`combine` 908、`gdn` 681 全在 CUDA0 串行），而分层是两卡各跑 24 层。
- 分层模式下跨 stage 的握手走 pinned RAM，但**握手的量级很小**：解码窗口级 `kVerifyMaxT=8` token（~410 KB），prefill 是整块缓冲（两个轮换）。16K 单块时 `wait copy` 只有 8-10 ms —— **握手/PCIe 根本不是关键路径**。
- 专家（71.7 GiB）在主机 RAM，NVLink 连不到主机 RAM；两卡显存合计 44 GB 装不下 71.7 GiB。

### 关键路径（实测分解，10K 输入）

每个 stage 必须把 10K token 过完它的 24 层；每块每 stage 的成本 ≈ **专家扫描 ~2.0 s（该 stage 非驻留专家过 PCIe）+ dense kernel ~1.0 s**，与块大小基本无关（chunk ≥3072 时该 stage 的所有专家都会被路由到）。
→ 1 块：两 stage 无法重叠（9.1 s）；2 块：可重叠但有两轮扫描（9.5 s）；更多块：扫描轮数线性增加。
**当前 8.8-9.9 s 的墙上时间已经接近"这个结构 + 这个引擎"的实践极限；理想重叠的天花板约 6.0 s（≈1670 tok/s），2000 tok/s（5 s）不可达。**

### 两条能真正逼近目标的路（都需要决定）

1. **把专家重新量化为 IQ 系（IQ3_XXS/IQ4_XS 一类）**：零代码改动（只换模型文件 + 重新 `iq_pack`），
   收益双重：① 触发引擎最快的 fused 原生 prompt 核（作者实测 +19-22%）；② 专家集 71.7 → ~43 GiB，扫描字节降 40%。
   代价：权重变了（不再是 UD-Q4_K_XL 这个量化本身）。
2. **给 prompt 路径加"大块内的微批流水线"**：让 stage1 在 stage0 算完第一个微批时就开始，
   而不必等整块 —— 墙上时间 ≈ max(stage) ≈ 4.5-5 s → **正好 2000 tok/s**。
   代价：改引擎（驱动循环 + 同步），是本轮结论里唯一能保住这个量化又打到目标的办法。

## 1M 上下文窗口（UD-Q4_K_XL，2026-10-09）

用户叫停 10K 目标后改试 **1M 窗口**。档：`strata-ud-q4_k_xl-1m.json`（分层 [0,1] auto、`--max-context 1048576`、
`--rope-scaling yarn --rope-scale 4`、`--kv int8 --kv-resident 65536`、`--prefill 5120`、MTP、校准参数、`STRATA_PREFILL_STREAM_MIN=4096`）。

**启动成功**（引擎 0.1.38，加载 110 s）：
```
rope scaling yarn, factor 4 (freq_scale 0.25, base 1e+07, mscale 1.138629),
   --max-context 1048576 against a trained context of 262144
KV streaming: 65536 of 1048576 cells per QSA layer in VRAM, the K/V in 6.19 GiB of pinned RAM
expert cache 4984 slots (14.55 GiB) + 4462 slots (13.01 GiB)
n_ctx = 1048576, loaded
```
- KV 账：12 个 QSA 层 × 1056 B/cell(int8) ≈ 13 KiB/token → 1M token ≈ **12.4 GB**（分层后每侧 6.19 GiB pinned RAM）；
  只有前 65536 格在显存（每卡 ~0.4 GB）。相比 16K 窗口，显存被 KV 拿走一点：专家缓存 15.9+14.4 → **14.55+13.01 GiB**。

**烟测**
| 测试 | 结果 |
|---|---|
| 短提示（360 tok） | prefill 106 tok/s、decode 31.9（冷） |
| 8K 提示 | prefill **755** tok/s、decode 43.2（16K 窗口时是 984 / 51.9） |
| 4.4K 上下文稳态 decode | 45.9 / 54.6 tok/s |
| KV 流式命中 | **97-99%** 的 block 读命中显存（上下文未超 65536 格时），仅 26-42 MiB 读自 RAM |

**代价与注意**：RoPE 启动时固定（换窗口要重启）；超训练长度 262144 是 yarn 外推，质量随长度下降
（仓库自己的测量只到 293K）；上下文超过 65536 格后，KV 从 pinned RAM 流式读入，decode 会逐段变慢。

### 1M 档的长上下文实测（探索性针测试，2026-10-09）

方法：随机词文档 + 正中间埋一句 `The secret passphrase is ZEPHYR-7731-QQ.`，温度 0，问 passphrase。
（脚本 `/tmp/needle300k.py`；每行约 21 token。）

| prompt | prefill | decode | 针 | 备注 |
|---|---:|---:|---|---|
| 8K | 755 tok/s | 43.2 tok/s | — | 短上下文基线 |
| 166,778 | **1098** tok/s | 40.0 tok/s | ✅ 找到 | KV 流式 88.9% 命中显存、从 RAM 仅读 124 MiB |
| **284,330** | **849** tok/s | **45.4** tok/s | ✅ 找到，并指出在第 7100 行 | **超过训练长度 262,144**（yarn×4 外推） |

结论：
- 1M 窗口不只是"能加载"：**超出训练长度的外推检索有效**（284K 处仍准确），KV 流式的命中率让 decode 只降到 40-45 tok/s。
- prefill 在长提示下反而更快（8K 时 755 → 167K 时 1098），因为分块流水的填充成本被摊薄；284K 时 849（KV 越大，注意力本身变贵）。
- 未做：完整 1M（100 万 token）prompt —— 那个 prefill 约需 20 分钟（按 849 tok/s），KV 将用满 12.4 GB pinned RAM。

## GSQ-RCO IQ3_S（ISTA-DASLab）—— 替代 UD-Q4_K_XL（2026-10-10）

用户判定 UD-Q4_K_XL 的长上下文速度不可接受并删除该模型，改用
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` 的 **IQ3_S** 档（混合精度 RCO 分配，77.9 GiB）。

**下载**：`huggingface.co` 不可达，且镜像的 xet CAS 桥返回 403；最终用 `resolve/main` + `curl -C -`
纯 HTTP 断点续传（见 `fetchs/fetch_gsq_rco_iq3_s.sh`）。文件字节数：shard1 54,817,524,224、shard2 28,800,138,432、
mmproj 907,543,008。镜像按 IP 限速 ≈10 MB/s（额外连接只能拿到 0.7-1.7 KB/s，加连接无效）。

**打包**：`iq_pack --compat-bf16` → `packs/gsq-rco-iq3_s`，**4.4 s、0 个张量需要转换**
（Orca/UD 那两档各 460 个）——RCO 分配本身就落在引擎原生格式里。1079 张量、302 原生、arena 1.43 GiB。

**专家格式（48 层，全部落在 fused 原生核覆盖内）**
```
gu: IQ2_S(22) x15? / IQ3_XXS(18) / IQ3_S(21) / IQ4_XS(23)      ← gu_covered 全覆盖
d : IQ4_NL(20) x39 / Q2_0(42) x9                                ← d_covered 全覆盖（Q2_0=42 是 pinned ggml 的自定义类型）
```
启动日志：`PLE on ... of ...-00002-of-00002.gguf`（**不需要 `--ple-gguf`**）、
`K=25`、`caches hold 16105 of 24576 pairs (~98.8% of the routed mass)`、专家缓存 8672+7152 = **15,824 槽**、
加载 **60-70 s**。对照 UD-Q4_K_XL：9720-10633 槽 / 97.3-97.6% 路由质量。

**实测（2×2080 Ti，分层双卡，`qwen-flash`，0.0.0.0:1234）**

| prompt | prefill | decode | UD-Q4_K_XL 同口径 |
|---|---:|---:|---|
| ~10K | 961-1005 tok/s | 61.8-71.6 tok/s | 1027-1060 / 39-55（冷）、60-67（稳态） |
| **~167K** | **1380-1448 tok/s** | 58-60 | **1098 / 40-45** |
| **~256K** | **1158 tok/s** | 55.5 | 849（在 284K）/ 40-45 |

→ 长上下文 prefill **+32%**，最长端 **+36%**；decode 全面更好（草稿接受率也更高）。

**A/B 结论（都被实测否掉）**
| 开关 | 结果 |
|---|---|
| `--prefill auto`（8192 块，每卡借 4.69 GiB） | 167K **1416** tok/s ← 选定 |
| `--prefill auto:16384`（每卡借 8.01 GiB） | 167K 1380-1448 |
| `--prefill auto:32768` | 引擎自动回退到 16384，1404 |
| `STRATA_PF_FUSED=1`（fused 原生 prompt 核，作者在 AMD 上实测 +19-22%） | **Turing 上无收益**：10K 970-1005 vs 1004；167K 1409 vs 1380-1448 → 已回退 |

即：本模型的长上下文速度**不是**靠块大小或 fused 核拿到的，而是模型本身（专家集 78 GB vs 111 GB、
IQ 系格式 + Q2_0，槽位多 50%、路由质量 98.8%）——**在 167K 这个量级，块大小已经不再是瓶颈**。

## 2026-10-26 长上下文 prefill 调优状态（Swift IQ3_S / 2×2080 Ti / 引擎 0.1.49.1）

**记录时的运行状态（2026-10-28 归档）**
- 服务：已停止，准备在最新版上调优；启动 `./run-swift.sh --context 256k|512k|1m`。
- 引擎：`build-v027/strata` 0.1.52（v0.2.7，MMQ_TAB=ON）；三套 Swift 档 `exe` 均指向它。
- 回退：`build/strata` 0.1.49.1（MMQ_TAB）、`build/strata-0.1.49.1-mmqtab`、`build/strata-0.1.49.1-nommqtab`。
- git：`5e8a4cca`（v0.2.7；本地 main 已快进到 origin/main）。
- `configs/`：Swift 三套 256K / 512K / 1M；Ornith 档在 `archive/ornith/`。

**三套档的共同设置**
- `layer_split: "24"`（24/24；实测优于 26/22、23/25、25/23）。
- `STRATA_SELECT_SIMT=2`（QSA block scorer 大 tile；与 `=1` 同 bits，128K 选择耗时 12.2 s → 2.4 s）。
- `STRATA_PA_V2T=1`（Turing v2 prompt attention；改变求和顺序，需质量复核）。
- `STRATA_MMQ_DIRECT=1`（需 MMQ_TAB 构建；驻留专家原地读）。
- 1M 档另加 `--kv-resident 131072` + `STRATA_KV_PREFETCH=1`（yarn×4）。

**0.1.49.1 128K 实测（130K 档、key 为 128,056 / 128,061 token，启动前 drop_caches，生产档不带 timing）**
| 档 | prefill | decode | 备注 |
|---|---:|---:|---|
| 256K，24/24 + direct | **2984.1 / 2984.4 tok/s**（42.913 / 42.908 s） | 87.2 tok/s | 两次连续复现；距 3000 差 0.5% |
| 1M，128K 上下文 | 2923.7 tok/s（43.801 s） | 92.9 tok/s | 同 128,061 token；KV 仍在 resident 窗口内 |

**MMQ_DIRECT 的受控 A/B（带 `STRATA_PREFILL_TIMING=1`，同提示、drop_caches）**
| 组 | 耗时 | tok/s |
|---|---:|---:|
| 旧二进制 | 44.327 / 44.386 s | 2888.9 / 2885.0 |
| 新二进制，不开 direct | 44.367 s | 2886.3 |
| 新二进制 + direct | 44.010 / 43.977 s | 2909.7 / 2911.9 |
→ direct 稳定快约 0.8%（单 chunk `dequant` 143→107 ms）；`STRATA_PREFILL_TIMING=1` 本身约 2.5% 开销，生产档不要带。

**本轮实测否决**
| 项 | 结果 |
|---|---|
| `STRATA_CVEC_FUSE=1` | 128K 2905.9 tok/s，慢于 2984 |
| `CVEC_FUSE + GDN_CONVL2=1` | 2903.1 tok/s，慢 |
| `STRATA_MMQ_GROUP_SORT=1` + `STRATA_MMQ_RESIDENT_SORT_NE=1` | 2982.5 tok/s，持平 |
| 分层 23/25 | 2848.4 tok/s |
| 分层 25/23 | 2906.8 tok/s |
| `STRATA_GDN_CHUNKED` | 文档已测：<128 SM 的卡更慢（2080 Ti 68 SM），不试 |
| `STRATA_GDN_PP/NOY` | AMD only（kernels.cu 的 HIP 分支） |

**测量口径与注意**
- 128K 提示用固定填充文本 + 每轮 nonce；`cached_tokens=0`；每次启动前 `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`。
- 页缓存/启动内存状态能让 128K 在 1.2K–3.0K tok/s 之间摆动；跨会话绝对值不可直接比较，A/B 必须同会话、同页缓存状态。
- `PA_V2T=1` 改变求和顺序；目前只做了 128K/50% 深度 needle（通过），长期使用前补 10/50/90 深度与 KL。
- `MMQ_DIRECT` 按文档是逐位相同的；分层 24/24 只是换设备。

**下一步（在 v0.2.7 上）**
- 先建立 v0.2.7 的 128K 基线（256K/512K/1M；HELP=1 与 HELP=0 各一发），再试 layer-major（两段 >1.5 GB 空闲）。
- 零代码上限约 2984 tok/s（0.1.49.1 与 v0.2.7 在噪声内）。
- 最大剩余项：GDN 快核移植（GDN 约 19% GPU 时间；FlashQLA-SM70-SM75 在 2080 Ti 上约 2.08×，估算 ~3250）。
- 其后：QSA 注意力查询间共享 KV（16 query 并集约 30–36%）、MoE grouped GEMM 调优。

### v0.2.7 线上版本实测（2026-10-28）

- 上游 `git fetch` 到 **v0.2.7（engine 0.1.52）**；独立构建 `build-v027/strata`
  （sm_75、`STRATA_MMQ_KQUANTS=ON`、`STRATA_MMQ_TAB=ON`）。`build/strata` 0.1.49.1 保留。
- 关键新特性：两卡 CUDA 分层默认 **layer-major prefill**（每层专家每个 super-chunk 读一次，而不是每个 8K chunk 一次）。
- 128K、128,056 token、drop_caches、同提示：

| v0.2.7 组 | prefill | layer-major 状态 |
|---|---:|---|
| 256K，默认（HELP=1） | 2973.1 tok/s | off：`a peer GPU`（`STRATA_PREFILL_HELP=1` 设置了 helper） |
| 256K，HELP=0 | 2981.6 tok/s | off：`no VRAM for a super-chunk (498 MiB free)` |
| 256K，HELP=0，LM 16K + host + margin128 + reserve1800 | 2984.8 tok/s | off：`layers 24..48: no VRAM (728 MiB free)` |
| 256K，LM 32K + vram + reserve1800 | 2974.0 tok/s | off：CUDA1 728 MiB |
| 256K，LM 16K + vram + reserve2600 | 2965.0 tok/s | off：CUDA1 728 MiB（`--vram-reserve-mib` 只影响 CUDA0） |
| 256K，LM 32K + `--expert-cache 6500` | 2979.3 tok/s | off：`--expert-cache` 在 `--resident-experts` 下被忽略，仍 8571/7302 槽 |
| **0.1.49.1 基线** | **2984.1 / 2984.4 tok/s** | — |

结论：
- 最新版在本机 **没有 prefill 收益**（2973–2985，和 0.1.49.1 的 2984 在噪声内）。
- layer-major 的目标是从 RAM 流式读专家的机器（如 2×A4000 16GB）；本机 99% 路由命中、`wait copy` 仅 0.2%，
  且 CUDA1（layers 24–47 + head/MTP）只剩 728 MiB，32K super-chunk 进不去。常规 `--vram-reserve-mib` 只作用于
  CUDA0，`--expert-cache N` 在 `--resident-experts` 下不生效，所以在本机无法在不明显降低驻留的前提下让它跑起来。
- **2026-10-28 切换决定**：用户接受这一误差，切到 v0.2.7 作为后续调优基线。`main` 已快进到 `5e8a4cca`（v0.2.7），
  三套 Swift 档 `exe` = `build-v027/strata`（0.1.52，MMQ_TAB=ON）。0.1.49.1 回退保留在上面的三个二进制。
- 归档时服务已停；切换后的 512K 128K 基线未跑（用户叫停，留待调优时测）。当前三套档仍 `HELP=1`，所以 layer-major 不生效；
  调 layer-major 时先 `STRATA_PREFILL_HELP=0`，并确保 **两段** 都有 >1.5 GB 空闲（`STRATA_PREFILL_HELP=1` 会被 helper 守卫直接挡住）。
- v0.2.7 测试归档：`archive/v0.2.7-transition/`（三套档快照 + `logs/v027-*`）。

## 2026-11-02 视频（BV1Uax46xEJT）参数在 0.1.54 + Swift 上的复测

B 站《双2080Ti跑qwen3.8flash峰值200+》的调优基于作者本地 **v0.1.51 线**（含 `STRATA_MTP_*`、
`STRATA_ATTN_*`、`STRATA_SIDE_MIXER` 等上游没有的 env），公开 fork 是 v0.1.45。不构建 fork，直接把它的
**CLI 参数**套到当前 `build/strata`（0.1.54）上跑 32K 新鲜提示（同一脚本、随机 nonce、每臂 2 次）：

| 配置 | prefill tok/s | decode tok/s |
|---|---:|---:|
| 生产 256K（layer-split 24，无 kv-resident，pcie-frac 0.35） | **2487 / 2366** | 86 / 111 |
| 视频全套（split 22、kv-resident 65536、budget 32G、pcie-frac 0、hnorm stream、adapt…） | 2333 / 2232 | 84 / 112 |
| 视频全套但 layer-split 24 | 2350 | 84 |
| 视频参数 pcie-frac 改回 0.35 | 2242 | 111 |
| 视频全套去掉 `--kv-resident` | 2386 / 2282 | 77 / 103 |

结论：
- 全部落在 **2230–2490** 的噪声带内（同臂重跑差 ±4%），**没有任何一项可复现的增益**；decode 两臂都 80–110。
- 视频参数在 0.1.54 上**全部被接受**（`--split-device`、`--resident-budget-gib`、`--adapt-every/--adapt-decay`、
  `--pcie-frac` 都不在 `--help` 里但不报错）；差距不来自 CLI 参数，而来自它那版引擎里的自定义 env/内核。
- `--kv-resident 65536` 在 32K 提示上是负项（吃专家缓存）；32K 档生产配置不启用它是对的。
- 决定：**生产三套档不动**。实验配置 `configs/strata-swift-abl-iq3s-video-256k.json` 保留作记录，不再用于日常。

### 同日晚：速度“变慢”的核对（128K 复测）

历史日志里 prefill 的最高值就是 **2984.8 tok/s @ 128,056 token**；32K 从来没有到过 3000（历史上 37–39K 的冷启动是 1916–2286）。
今天在同一台空载机器上按历史口径复测（`/tmp/ab_prompt.py`，128,056 token，0 reused）：

| 档 | prefill | decode |
|---|---:|---:|
| 256K（0.1.54，冷页缓存启动） | **2976.6 tok/s** | 86.0 |
| 512K，去掉 `--kv-resident 262144` | **2946.0 tok/s** | 79.4 |
| **512K 生产原配置**（带 `--kv-resident 262144`） | **2917.7 tok/s** | 84.0 |

结论：
- **0.1.54 没有回归**，512K 也没有问题；三档都在历史峰值的 2% 以内。
- 2026-10-28 记的「512K 128K 只有 1321–1332」是**当时机器有 QEMU/KLEE/clang 负载**，不是配置或版本问题。
- 32K 只有 ~2400 是正常的：短提示摊不掉每个 super-chunk 的专家装载与流水线暖机；同一服务 128K 就是 ~2980。
- `--kv-resident 262144` 在 512K 档**不影响专家槽位**（有/无都是 7434/6261），速度在噪声内，保留它（长上下文 decode 受益）。

## 2026-11-04 升级到 v0.2.10（engine 0.1.56）+ decode 调优实录

上游 `main` 快进 877 个提交到 **v0.2.10（engine 0.1.56）**；`build/strata` 用 72 核全核重建，153/153 目标、0 error。
旧引擎备份 `build/strata-0.1.54-v028`。prefill 与 0.1.54 持平（128K 2977；32K ~2400）。

### decode 100+ 的真实条件（复现历史 160 / 107 tok/s）

历史日志里 decode 最高两条：`202 generated, 148/159 accepted → 160.0 tok/s`、`115 generated, 82/84 accepted → 107.4 tok/s`。
复现时发现三个前提，缺一个就掉：

| 前提 | 影响 |
|---|---|
| **不要开 `STRATA_DECODE_TIMING=1`** | 每窗口收集 20+ 计数，4-token 窗口 25 ms → 40 ms（−30%）。诊断完必须去掉 |
| **内容决定接受率** | 「从 1 编到 512」这类创作输出接受率 65–75% → 70–83 tok/s；连续格式文本（归档续写）383/383 → 100+ |
| **缓存预热** | 启动后第一发 79，热了 97 → 110 |

实测（单发、32K 档、512 token、无计时）：**78.7 / 96.0 / 110.3 tok/s**（对应 391/395、383/383、383/383 接受）。

### 本轮有效 / 无效项（单发、512 token、无计时）

| 项 | 结果 |
|---|---|
| **`--adapt-swaps 16`**（默认 96） | **有效**：每窗口自适应等待 3.4 → 1.0 ms（27.6 → 24.6 ms/窗口），已写入三套生产配置 |
| `--vram-reserve-mib 900` + `--vram-reserve-later-mib 1350` | 修掉「只剩 ~500 MiB 显存」告警（现 950 / 1020 MiB 空闲），槽位 8166 / 7141 |
| `--lookup-max-t 12` | 无差别（层切分下被引擎限到 8；只对复制/改写回合有用） |
| `--mtp-q4 all` | 无差别（草稿头读取 213 → 146 MiB，不是瓶颈） |
| `--spec 6` | 更差（68%；窗口更长但接受率降） |
| `--spec-min-p 0.5` / `0.85` | 更差（54% / 78–82%，窗口尺寸与接受率不划算） |
| `--pipeline-windows 3` | 引擎在 2 段切分下限回 2，无差别 |
| `--mtp-hnorm stream` | 无差别 |

### DFlash2（0.7B 草稿器）实测：本机不划算，保留 MTP

- 已安装：`~/models/dflash2/qwen3.8-flash-next-dflash2.q4_0.pack`（522,458,112 B，SHA-256 `cda40d…`，与官方 pin 一致；
  源文件 `~/models/dflash2-src/`）。HF 直连不通，用 `hf-mirror.com` 下载，哈希逐字节匹配。
- 引擎可启动：`--draft dflash2 --draft-model …`，层切分下草稿器放在 CUDA1，`--draft-conf 0.65`。
- 实测 decode **63.6 / 73.2（conf 0.5）→ 65.6 / 76.8（conf 0.65）**，低于 MTP 的 72–83 / 110。
- 原因：层切分下 DFlash2 **强制关闭 `--pipeline-windows`**（verify 23.8 ms vs MTP 21.0），且 Swift 不在官方支持列表
  （`SUPPORTED_FAMILIES = qwen, gyro, unsloth`），接受率只有 ~65%。官方表在最优点也只有 +3.5~7%。
- 结论：**生产保留 MTP**；DFlash2 pack 留着，等上游对 Swift/层切分的支持。

### KV 量化档在 128K 单发上的对比（2026-11-04）

同一生产 512K 配置，只换 `--kv`，128,096 token 提示 + 256 token 续写（接受率 97–99%）：

| `--kv` | prefill | decode | 判定 |
|---|---:|---:|---|
| **int8（生产）** | **2965 / 2762 tok/s** | **107.4 / 114.3 tok/s** | 最好 ✓ |
| `k8v4`（K int8 + V Q4） | 2521 / 2317（−13%） | 81.5 / 88.4（−20%） | 否决 ✗ |
| `q4_0`（Hadamard + Q4） | 1881 / 1743（−37%） | 83.4 / 90.6（−20%） | 否决 ✗ |

→ Turing（sm_75）上 4-bit KV 的反量化/Hadamard 成本远大于省下的带宽，**保持 `--kv int8`**。

## 2026-11-05 上游 issue 里的两个代码级优化：都试了，都不采纳

### A. #5569 IQ3_S grouped GEMM（SIMD 反量化 + decode tile）——无效果

补丁（issue 正文，302 行）用 `patch -p1 --fuzz=3 -l` 落到 `src/kernels/cuda/iq_kernels.cu`，编译 0 error。
同一个二进制内做 A/B（`STRATA_DG_SIMD` 0/1、`STRATA_DG_DECODE` 0/1），每组 128K 单发 3 次：

| 组 | 预填 tok/s | decode tok/s |
|---|---|---|
| SIMD=0，DECODE=0（补丁前路径） | 2917 / 2678 / 2620 | 80.0 / 89.5 / 97.5 |
| SIMD=1（默认），DECODE=0 | 2951 / 2710 / 2693 | 80.0 / 91.7 / 98.2 |
| SIMD=1，DECODE=1 | 2911 / 2671 / 2617 | 79.8 / 91.2 / 98.0 |

→ **三组完全一致**：我们的 decode 路径不走这个 grouped wmma 核（它服务 prompt/大行数），补丁打不到热点。已回退。

### B. #5002 `STRATA_DP_PREFILL`（双卡数据并行预填）——负 2.4 倍，否决

PR 基于 0.1.57（8 文件 +603 行），用 fuzz 落到 0.1.56 并修了 3 处（`walk_trim`、错位 hunk），编译通过。
它的**前置条件**是全部专家都在 RAM 的 arena 模式——必须去掉 `--resident-experts`，这本身就有代价：

| 组 | 预填 tok/s | decode tok/s | 说明 |
|---|---|---|---|
| 生产（resident-experts，DP 前置不满足） | 2912 / 2719 | 72.8 / 84.8 | 同会话基线 |
| arena 模式，DP 关 | 2878 / 2671 | 61.4 / 72.7 | **decode −15%**，槽位 7225/6290（原 8166/7141） |
| arena + **DP 开** | **1187**（107.9 s） | 30.3 | 专家命中 89.2%、3.3% 走 PCIe |

→ DP 让每张卡在每个 chunk 都要拉对方层的专家，我们的 arena/PCIe 跟不上；**比层切分慢 2.4 倍**，decode 也塌。
同上 fork 里 rafatxf 在 56 GB RAM 的 2×2080Ti 上也测得变慢（16K 21.8 s vs 16.2 s），与此一致。已回退，`build/strata` 重编为干净 0.1.56（128,420,184 B）。

### C. 同期参数类结论（128K 单发 2 次）

| 组 | 预填 | decode | 结论 |
|---|---:|---:|---|
| **24/24 + NUMA 池开（生产）** | 2965 / 2762 | 112 / 119（当次会话） | 保留 |
| `STRATA_NUMA_POOL=0` | 2905 / 2712 | 65.9 / 77.6 | 否决（decode −30%） |
| 分层 22 | 2696 / 2537 | 82.0 / 92.2 | 否决 |
| 分层 26 | 2769 / 2550 | 79.0 / 88.5 | 否决 |

→ 分层必须 24/24；双路 Xeon 的 NUMA 池不能关（与 #3708 的结论相反，那台是 100% 专家驻显存）。

## 2026-11-05 会话缓存重配（30 万 token 会话频繁整段重读）

日志证据（生产 512K，用户真实会话 29–32 万 token）：

```
parked 323078 tokens in 3.3 s; parked=2 bytes=11.26 GB snapshot_bytes=5.78 GB evictions=16
prompt 317716 tokens = 0 reused + 317716 read in 129916 ms   ← 整段重读 2 分 10 秒
prompt 300853 tokens = 16384 reused + 284469 read in 109071 ms ← 只命中一个检查点
```

原因：30 万 token 的一份停车快照 **5.4–5.8 GiB**，`--conversation-cache-mib 13312` 只放得下 2 份（槽位 4 用不满），
第三个会话一停就把最老的挤掉，被挤的会话回来就是整段重读；且 `min-tokens` 默认 0，小于任务的会话也占槽。

三套档改为：

| 参数 | 旧 | 新 |
|---|---:|---:|
| `--conversation-cache-mib` | 13312 | **28672**（约 5 份 30 万快照） |
| `--conversation-cache-min-tokens` | 0 | **65536**（小块子任务不占槽） |
| `--conversation-cache-park-gain` | 0 | **1024**（小幅增长不重写 5.5 GiB 快照） |

可选未启用：`--prompt-cache-spread`（客户端压缩/改写历史时，让检查点向回铺开，只重读被改写的那段）。
客户端侧注意：中途切 thinking 开关 / 模型 / 系统提示都会改前缀，等于换会话，必然整段重读。

## 2026-11-16 升级到 v0.2.14.1（engine 0.1.60.1）：Turing 融合专家内核，prefill +4~6%

上游到 **v0.2.14.1（engine 0.1.60.1）**，落后 1424 提交。用 `.venv/bin/cmake`（4.4.3；系统 cmake 3.22 太旧）
在独立目录 **`build-v0214`** 构建：CUDA 13.0、arch 75、`STRATA_MMQ_TAB/KQUANTS/TQ_OVERLAY=ON`，0 error。
装为 `build/strata`，旧引擎备份 **`build/strata-0.1.56`**。

**关键变化**：0.1.60 把我们打包的专家对（gate/up `IQ3_XXS`/`IQ2_S` + down `IQ4_NL`）纳入 sm_75 融合内核
（`kt75_formats`），启动日志：`prompt kernels: chunks of 4096 tokens or more run 48 of 48 layers on the fused kernels`。
0.1.56 这些对全部退回 MMQ。

| 工况（同提示、单发） | 0.1.56 | **0.1.60** | 变化 |
|---|---:|---:|---:|
| 128K 预填 | 2972.6 / 2804.3 | **3116.5 / 2906.9** | **+4.8% / +3.7%** |
| 341K 预填 | 2534.6（134.7 s） | **2685.1（126.9 s）** | **+5.9%** |
| 128K 后 decode（512 token 高接受率） | 78.4 / 95.2 / 110.6 | 78.2 / 95.6 / 109.6 | 持平 |
| CUDA1 空闲显存 | 1020 MiB | **1498 MiB** | +478 MiB |

**开关验证**：`STRATA_PF_FUSED_SM75=1` 与默认完全相同（已默认启用）；`STRATA_PF_FUSED_SM75_IQ=0` 关掉 i-quant 融合后
退回 int8 dp4a 内核，128K 预填掉到 **1649 tok/s（−47%）**——**绝不能关**。

**正确性抽查**（温度 0，同一提示两版对比）：1234×5678=7006652、严格 JSON、is_prime 代码三项**逐字节相同**；
一句中文解释措辞不同但同义（融合与 MMQ 的求和顺序不同，接近平局时可能换措辞）。上游注明这几种新格式的 Turing
parity 尚未跑，我们这组抽查是目前的唯一证据。

**落地（2026-11-16）**：`build/` 重建为 **0.1.60.1 的构建树**（旧的 0.1.56 树已清掉，`build-v0214` 删除）；
二进制备份在 **`engine-bak/strata-0.1.56`** 与 **`engine-bak/strata-0.1.60`**。
四套 config 的 `exe` 均为 `build/strata`；`run-swift.sh` 的 `--status` / `--check-paths` 现在会打印引擎版本
（`engine: strata 0.1.60.1 (build/strata)`），头部注释记录了版本与回退路径。

**同轮否决：`--kv-resident 393216`**。341K 预填 2744（+8.3%），但 128K 预填 2765（原 2973）、专家命中率 95→82–88%
（KV 多吃显存、专家槽变少），decode 变差 → 保持 262144。

## 2026-11-16 RVN IQ3_S（去审核 + Strata-ready）实测

`jimmy31chen/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-IQ3_S-Strata`（0bserverx RVN 的 abliteration，
基座是官方 Flash-Next，不是 Swift 微调）。82.78 GiB 单文件 + 预构建 pack，专家 **gate/up IQ3_S（类型 21）+
down IQ4_NL（类型 20）**，正好是 0.1.60 的 Turing 融合内核全覆盖组合。

- **下载**：hf-mirror 单 IP 限速 ~10 MiB/s（aria2c 16 连接也一样），82.78 GiB 约 2.4 h；大小与上游一致
  （88,882,765,184 B）。
- **MTP**：`mtp_fetch`（官方 `mtp.*` 张量，verify 通过）→ `mtp_pack --experts q2_0` → `mtp_rt`；沿用 Swift 的
  多语种 `draft_vocab.bin`（106,299 词），实测接受率 **98–99%**（可用）。
- **加载**：48/48 层融合内核；专家槽 7466 + 6571（约 **57% 驻留**，Swift 为 62%——专家更大）。
- **性能**（512K 档，单发，0.1.60.1）：

| 项 | RVN IQ3_S | Swift IQ3_S（现用） |
|---|---:|---:|
| 128K 预填 | 3056 / 2839 | 3116 / 2907（RVN −2%） |
| 341K 预填 | 2689 | 2685（持平） |
| decode 512 token（高接受） | 76.5 / 90.8 / 100.4 | 78.2 / 95.6 / 109.6（RVN −5~8%） |

- **质量抽查**（温度 0）：数学 7006652、`is_prime` 代码与 Swift **逐字节相同**；JSON 内容相同（换行不同）；
  中文解释同义。RVN 的公开评测：对官方 Q6_K 配对 PPL −0.49%、IFEval +0.92pp、GSM8K −0.8~1.4pp、BBH +4pp、
  严格拒答 2/100。

### 思维链长度（`reasoning_effort`）

同一道代码题、thinking 开启，completion tokens：

| | 默认（xhigh） | low | 关 thinking |
|---|---:|---:|---:|
| RVN | 1774（19.9 s） | **1077（11.6 s）** | 306 |
| Swift | 1720（18.0 s） | 1328（13.5 s） | 384 |

→ **长度主要由 `reasoning_effort` 决定，不是模型**；模板支持 xhigh/medium/low。服务端支持配置顶层
`"reasoning_effort"`（#1641，客户端未指定时生效）。RVN 配置里的该键**已删除**（未传参数的客户端回落到模板默认
**xhigh**，工具调用最稳）；Swift 三套配置也没设，同为 xhigh。想省 token 时在请求里或配置里显式传 `low`/`medium`。

配置文件：`configs/strata-rvn-iq3s-512k.json`。除 4 个模型路径（pack/native/ple-gguf/mtp）外，**其余 51 个启动参数与
Swift 512K 生产档完全一致**（含 `--pipeline-windows 2`、`--kv-resident 262144`、`--adapt-swaps 16`、缓存 28672/4/65536/1024 等）；
另加 `"reasoning_effort": "low"`。视觉沿用官方 Flash-Next 的 BF16 mmproj（与 Swift 同一个投影器），已实测启动成功
（`/v1/models` 报 `input_modalities: [text, image]`，vision 预热 1024 image tokens 0.8 s）。
启动脚本：**`./run-rvn.sh`**（`--status` / `--stop` / `--restart` / `--check-paths`，复用 run-swift.sh 的状态与停止逻辑；
`--check-paths` 会打印引擎版本）。Swift 仍用 `./run-swift.sh --context 512k`。

### 难题对比：6 道竞赛级数学（xhigh、温度 0）

| 题（答案） | RVN | Swift |
|---|---|---|
| 1/a+1/b=1/2004 的有序对数（45） | ✅ 916 tok / 12.5 s | ✅ 1193 tok / 12.9 s |
| 3^2026 mod 1000（329） | ✅ 2292 / 23.0 s | ✅ 1005 / 9.5 s |
| x⁴+y⁴+z⁴（25/6） | ✅ 3452 / 36.5 s | ✅ 3482 / 32.6 s |
| 5 元对合函数个数（26） | ✅ 528 / 7.1 s | ✅ 433 / 5.3 s |
| 3×10 骨牌铺法（571） | ✅ 3341 / 39.5 s | ✅ 3131 / 29.6 s |
| 2³²+1 最大质因数（6700417） | ✅ 3181 / 30.7 s | ✅ 3180 / 29.3 s |

→ **双方 6/6 全对**，数学题区分不出差距；RVN 合计 13,710 token / Swift 12,424（RVN 略长，主要在幂模题）。
若要继续找“智商差”，应换代码/agentic 长任务（数学已到天花板）。

## 2026-11-19 升级 v0.2.16（engine 0.1.62）+ 仓库布局迁移

上游从 v0.2.14.1 快进到 **v0.2.16**（773 提交），顶层变成 **`app/ engine/ terminal/ tools/ docs/`**：

| 旧路径 | 新路径 |
|---|---|
| `build/`（`cmake -S .`） | `build/`（**`cmake -S engine`**） |
| `serve/server.py`（`-m serve.server`，仓库根） | `terminal/serve/server.py`（**`PYTHONPATH=terminal`** 后 `-m serve.server`） |
| `data/expert-profile.bin` | **`terminal/data/expert-profile.bin`** |
| `tools/mtp_*.py`、`iq_pack.py`、`make_profile.py` | **`terminal/tools/`** |

完成项：
- 引擎 `build/strata` = **strata 0.1.62**（`cmake -S engine -B build`，CUDA 13.0/arch75/MMQ_TAB+KQUANTS+TQ_OVERLAY，0 error）。
- 五套配置的 `--expert-profile` 改为绝对路径 `terminal/data/expert-profile.bin`（相对路径会 `read_expert_profile: cannot open`）。
- `run-swift.sh` 启动前 `export PYTHONPATH="$PWD/terminal"`；`run-rvn.sh` 是它的包装，自动继承。
- **工具调用修复（#4686）验证通过**：同一条 `write_note` 请求、body **4892 字符（>4KB）**，非流式返回
  `finish_reason=tool_calls`、JSON 完整；流式连续两次同样完整（此前 0.1.60 的流式大参数会丢后半段/整个调用）。
- 性能：128K 预填 **3089.9 tok/s**（0.1.60 为 3116，持平）；工具调用生成期间 decode ~185 tok/s。
- 回退：`engine-bak/strata-0.1.60`（0.1.60.1 二进制）；本地配置/脚本备份在 `/tmp/v0216-backup/`。

## 2026-11-21 升级 v0.2.17（engine 0.1.63）+ Swift/RVN 能力对比

升级：`git pull` 到 **v0.2.17**，`build/strata` 重编为 **0.1.63**（0 error，参数同前）。旧引擎备份 `engine-bak/strata-0.1.62`。
v0.2.17 的主要价值是**工具调用的连续修复**（#4686 之后又修：多余反引号吞调用、`<function_calls>`/裸 `<invoke>`/
JSON 对象参数、嵌套参数缺闭合、被拒调用不再重述）。

**对比方法**：两边 `reasoning_effort=medium`、温度 0、同一组题；服务端都是 0.1.63。

| 套件 | Swift IQ3_S | RVN IQ3_S |
|---|---|---|
| 官方 standard-v1 十二道 sanity（I/R/T 各 4） | **12/12**（36.4 s） | **12/12**（40.6 s） |
| 同上，**流式** | — | **12/12**（38.4 s） |
| 自建硬题（法律/医学/物理/CS/历史/逻辑选择题 12 道） | **11/12**（37.7 s） | **12/12**（48.9 s） |

- 官方 sanity 是**功能冒烟**（JSON 严格解析、4K/16K 检索、结构化工具调用），两边都满分，**区分不出能力**。
- 自建硬题：Swift 错在 CS 开放寻址探测复杂度（答 O(1)，应 O(1/(1-a)²)）；RVN 全对。样本只有 12 题，
  不足以断言“RVN 更聪明”，但可说明两者在常识/学科知识上**同一档**。
- RVN 的输入输出都略慢（工具题每次多 0.2–0.4 s；总时长 +10~30%），与其专家更大、驻留率更低一致。
- 结论：**“Swift 智商下降”不是模型问题**——两边在标准 sanity 上完全一致；若体感变差，先排查引擎路径
  （0.1.60 起 prompt 专家走 Turing 融合内核，可用 `STRATA_PF_FUSED=0` 回退到 MMQ 对照）。

### MMLU-Pro 尝试（未完成，作废）

想用在线最常用的标准套件补一个有区分度的分数，过程与结论：

- **Strata 没有 `/v1/completions`**（只有 `/v1/chat/completions`、`/v1/messages`、`/v1/responses`），
  所以 lm-eval 的 loglikelihood 类任务（MMLU/ARC/HellaSwag 原版）**无法直接跑**；只有生成式任务
  （`mmlu_pro`、`ifeval`）可行。
- lm-eval 装到独立 venv `/home/aipl/venv-lmeval`（0.4.9），但 `peft` / `transformers` / `datasets` / `huggingface_hub`
  互相打架，最终放弃 lm-eval，改为**直接加载 `TIGER-Lab/MMLU-Pro` 数据集**，按 14 个学科每科抽 8 题
  （共 112 题固定子集，`bench-lmeval/mmlupro-subset.json`），thinking 关闭、温度 0，按官方
  “The answer is (X)” 规则判分。脚本：`/tmp/mmlupro_eval.py`。
- **试跑（14 题）**：11/14（78.6%），约 14 s/题。
- **正式跑 Swift 时被中止**：前 48 题（6 个学科）**41/48（85.4%）**；第 49 题起因客户端被杀 + 服务端停止，
  全部瞬时报错，日志里 48→60、60→80、80→100 三段耗时都停在 851 s，证明是中断不是答错。
  该结果已标注 `valid: false`，**36.61% 不可引用**。
- 记录文件：`bench-lmeval/swift-mmlupro-20261121-2257.json`（含作废标记与前 48 题的真实命中）、
  `bench-lmeval/swift-mmlupro.log`、`bench-lmeval/smoke-mmlupro-20261121-2243.json`。
- 若日后要重跑：先启动服务，再跑 `/home/aipl/venv-lmeval/bin/python /tmp/mmlupro_eval.py <tag> 8 off`，
  112 题约 20–25 分钟/模型；RVN 未测。

## 2026-11-22 Swift vs RVN：MMLU-Pro 定论（RVN 已删除）

同题、同档位（**xhigh**，服务端配置覆盖了脚本传的 medium，两次都是 xhigh，因此两列可比）、温度 0、
112 题（14 科 × 8 题，`bench-lmeval/mmlupro-subset.json`），**0 错误**：

| 学科 | Swift | RVN |
|---|---:|---:|
| biology | 7/8 | 6/8 |
| business | 7/8 | 7/8 |
| chemistry | 8/8 | 8/8 |
| **computer science** | **8/8** | **6/8** |
| economics | 7/8 | 7/8 |
| **engineering** | **8/8** | **6/8** |
| health | 7/8 | 7/8 |
| history | 6/8 | 6/8 |
| law | 2/8 | 2/8 |
| math | 8/8 | 8/8 |
| other | 6/8 | 6/8 |
| philosophy | 7/8 | 7/8 |
| physics | 8/8 | 8/8 |
| psychology | 8/8 | 8/8 |
| **总计** | **97/112 = 86.61%**（28 min） | **92/112 = 82.14%**（52 min） |

**为什么 Swift 更聪明（此前判断有误，已纠正）**：
- 量化档位不是原因。两边专家很接近（Swift: gate/up IQ3_XXS·IQ2_S + down IQ4_NL；RVN: IQ3_S + IQ4_NL），
  MMLU-Pro 这类知识题几乎不吃这点差异（IQ3_S 与全精度在公开测试上持平）。
- 真正原因是**微调 vs 纯消融**：Swift 是 UkisAI 的微调（指令跟随/作答结构被强化）；RVN 是官方基座的纯
  abliteration（零训练，只做方向消融），消融会带来能力漂移。失分正集中在 CS 与工程（各 −2）。
- 早前“Swift 专家是 IQ3_XXS 所以更笨”的推断是错的。

**RVN 的另外两个问题**：工具调用经常出错（需 `reasoning_effort` 拉满才稳）、推理更长更慢（52 vs 28 min）。
**决定：删除 RVN，生产保留 Swift。**

存档：`bench-lmeval/swift-mmlupro-20261122-1559.json`、`bench-lmeval/rvn-mmlupro-20261122-1655.json`、
检查点 `/tmp/swift-ckpt-medium-done.jsonl`、`bench-lmeval/rvn-mmlupro-ckpt.jsonl`。
