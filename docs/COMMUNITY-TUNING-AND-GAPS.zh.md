# 社区调优方法 × 我们的实测 × 差距清单

2026-10-13 整理。目的：把三份社区资料的做法与我们这一路的实测对齐，列出**还没做到的优化点**。
社区数字都是**他们自测**（第三方来源，未经我们验证）；我们的数字全部本机实测。

## 0. 三份社区资料

| 来源 | 硬件 | 引擎 | 关键数字（自述） |
|---|---|---|---|
| [zyYuc/strata-dual-2080ti-22g](https://github.com/zyYuc/strata-dual-2080ti-22g)（Swift IQ3_XXS） | 2×2080Ti 22G（PCIe **x8+x8**），Ryzen 7 5700X（8C/16T），**48 GB** DDR4-3600 | **v0.1.42** | 20K prefill 1706 / 60K 2116 / 128K 2163 tok/s；decode 77-84；2K 665、5K 1010、8K 1095 |
| [ai-muninn 博客](https://ai-muninn.com/en/blog/strata-flashnext-one-2080ti)（**GSQ-RCO IQ3_S**，即我们下的那个） | 1×/2×2080Ti 22G，EPYC，128 GB | **0.1.39** | 代码 70.3（单卡）/91.4（双卡）tok/s；16K prefill 1325/1525；TTFT 12.2/10.6 s |
| [lcz.me 论坛 #2282](https://lcz.me/topic/2282)（IQ3_S） | 2×2080Ti 22G 有 NVLink，EPYC 7F52（16C/32T），128 GB 8ch（153 GB/s） | **v0.1.43** | 200K 上下文：prefill 1000-1100、decode 110-60；观察到 prefill 单线程、decode 时 32 线程跑满、GPU 仅 50% |

## 1. 他们方法里"按贡献排序"的六条（zyYuc）

1. **v0.1.42 双卡借槽 RAM 备份**：`--resident-experts` 下，prompt path 借走的专家槽在锁页内存留副本，启动日志
   `the split stages lend N slots: N keep their experts in RAM too`，24576 专家 100% 被"显存+锁页"覆盖，**推理期零专家文件读**。零参数、升级即得。
2. **`STRATA_SELECT_SIMT=1`**（QSA FP32 tiled 评分器，0.1.40.3 起 opt-in）：60K **+11%**、128K **+20%**、decode +1.7% —— 作者称"最容易漏的开关"。
3. **`STRATA_SPLIT_MTP_BATCH=1`**：双卡 MTP 批处理，prompt +2~5%。
4. **`STRATA_SPLIT_RING=384`**：借槽环 96→384，20K +3.5%、128K +2.1%。
5. **`--aux-cpus auto`**（0.1.42 新增）：decode +3.6%、prefill −1~4%。
6. **基础项**：`STRATA_BF16_TC=1`（Turing 必开，**prompt +20~34%**）、`--pipeline-windows 2`、`--layer-split 26`、`--pcie-frac 0.35`、`--spec-min-p 0.70`。
   另有 `STRATA_STAGE_TRIM=1`（2K prefill 承重墙，删则 −19%）、`STRATA_PREFILL_HELP=1`（CPU 协助 1-3.3K prompt MMQ）、`STRATA_SPLIT_OWN=0`（显式借用）。

**他们的"已测否决"**：`--prefill auto:16384`（48 GB 机器 20K prefill −19~23%）、`--kv-resident 32768`（decode −6~9%）、calibrate 推荐的 `--pool-workers 5`（prefill −7%）、`STRATA_ROUTE_RESIDENT/PRIOR=0.5`（中性但改输出）。

**他们的验收门槛**：启动日志出现 `... keep their experts in RAM too`、`mapped pinned cache complement ready: resident 16-17 GiB`、`layer split: 77% of the experts resident`；两卡显存 ~96% 占满；内存补集 16-17 GiB 锁页；功耗墙 200 W（他们测 250 W 再 +3~5%）。

**他们的测速口径**（`bench/run_context_ttft_fair.py`）：流式、`prefill = prompt_tokens / TTFT`（TTFT=首个思考字符）、`decode = completion_tokens/(total−TTFT)`、max_tokens 1024、每 run 换 seed 防前缀缓存、五档 ×2 + 20K 暖机。

## 2. 我们已实测的事实（本机，2×2080Ti 22G / 双路 E5-2696 v3 72 线程 / 125 GB / PCIe **x16+x16**）

**模型**：OrcaRouter IQ3_XXS（79 GiB，`--compat-bf16` 460 张量转换）、GSQ-RCO IQ3_S（78 GB，**0 张量转换**，专家 gu∈{IQ2_S,IQ3_XXS,IQ3_S,IQ4_XS}、down∈{IQ4_NL,Q2_0}）。
**引擎**：自建 **0.1.38**（`-DSTRATA_MMQ_KQUANTS=ON`，sm_75）。**与上游差距 1892 个提交**（上游 tag 已到 `v0.1.43.1`，HEAD `a0116b27`）。

| 模型 | 10K prefill / decode | 167K | 256K | 启动 |
|---|---|---|---|---|
| GSQ-RCO IQ3_S | 961-1005 / 62-72 | **1380-1448 / 58-60** | **1158 / 55.5** | 60-70 s，槽位 15,824-16,105，路由 98.8% |
| Orca IQ3_XXS | （用户测 256K 档 prefill ~1600） | — | decode 66-73（贪心/安静） | — |
| UD-Q4_K_XL（已删） | 1027-1060 / 39-55 | 1098 / 40-45 | 849（284K） | 首次 13 min（HDD）→ 90 s（SSD） |

**我们已做且有效的**：搬到 SSD、`-DSTRATA_MMQ_KQUANTS=ON`、`--pcie-frac 0.20` + `--spec-min-p 0.70`（Orca 档 decode **+10%**）、`--pool-workers 23`（噪声内略优）、`STRATA_PREFILL_STREAM_MIN=4096`（<4096 块的 routed-only staging，1.6K 提示 **+25%**）、GSQ-RCO 量化本身（长上下文 prefill +32% vs UD）。

**我们已实测且无效/更差**（别重复）：`STRATA_PF_FUSED=1`（Turing 无收益）、大页 2 MB、PLE row-cache 4M/inflight 512、stager 线程 32/ring 128-256、`--prefill` 各种块长（167K 上 8192/16384/32768 全等）、own prompt buffers（−30%）、`--peer-device`（NVLink，prefill 更慢）、`--adapt-every 1 --adapt-swaps 192`（稳态更差）、`--vram-reserve-mib 128`（cuBLAS 崩）。

## 3. 逐条对照：他们做了 / 我们做了吗

| 项 | 他们 | 我们 | 备注 |
|---|---|---|---|
| 引擎 ≥0.1.42 | ✓ v0.1.42/v0.1.43 | ✗ **0.1.38** | 下面 5 项都依赖它（已用 `git grep` 核对：上游有、本地无） |
| `STRATA_BF16_TC=1` | ✓（Turing 必开） | ✗ 代码里没有 | 博客：双卡 16K TTFT 12.6→10.6 s |
| `STRATA_SELECT_SIMT=1` | ✓ | ✗ | 长上下文最大单项（128K +20%） |
| 借槽 RAM 备份 | ✓ 0.1.42 新特性 | ✗ | 零文件读；我们 0.1.38 的 split 每次借槽仍可能回读 |
| `STRATA_SPLIT_MTP_BATCH=1` | ✓ | ✗ | |
| `STRATA_STAGE_TRIM` / `STRATA_PREFILL_HELP` | ✓ | ✗ | 短 prompt（1-3.3K）专用 |
| `--aux-cpus auto` | ✓ | ✗ | decode +3.6% |
| `--pipeline-windows 2` | ✓ | ✗ | 从没用过 |
| `--layer-split 26` | ✓ 固定 26 | auto（K=24/25） | 可显式扫 24/25/26/27 |
| `--pcie-frac` | 0.35 | **0.20**（在 Orca 档上校准过 decode） | 应按模型/口径重校，prefill 与 decode 取向可能不同 |
| `--ple-row-cache` / `--ple-inflight` | 8388608 / 1024 | 4194304 / 512（在 UD 上试过，无差异） | 0.1.42 里 PLE 逻辑改过，需重测 |
| `--conversation-cache-mib/slots` | 13312 / 4 | ✗ 没用过 | 对 agent 反复前缀的场景可能有用 |
| `--resident-experts` | ✓（48 GB 机器） | ✗（只在 UD 上试过 `--resident-budget-gib` 单卡，5 tok/s） | 我们 **125 GB**，值得在 0.1.42 上重测 |
| 思考默认 = xhigh | 建议改 low（3.4× 更快、分更高） | 我们未设（默认 xhigh） | 我们有 `--think low` 写 `shared-settings.json`，但没启用 |
| 测速口径 | TTFT 基准 + 换 seed | 我们自己的脚本（prompt_per_second） | 换成他们的口径才能直接对比 |
| 功耗墙 200 W | ✓（250 W 再 +3-5%） | 250 W（实测 median 64-73 W） | 我们不是功耗受限 |

## 4. 差距清单（还没做的，按预期收益排序）

**P0 — 升级引擎（唯一的"结构性"差距）**
- `git checkout v0.1.43.1`（或 HEAD）+ 重编译 sm_75。解锁 P1 全部开关。
- 顺带拿到 0.1.39-0.1.43 的一堆 prompt/借槽/调度修复（如 #2090 借槽放不下时"减半而不是结束引擎"）。

**P1 — 环境变量（升级后零成本、按他们排序）**
```
STRATA_BF16_TC=1          # Turing 必开（prompt +20~34%）
STRATA_SELECT_SIMT=1      # 长上下文 +11~20%
STRATA_SPLIT_MTP_BATCH=1  # prompt +2~5%
STRATA_SPLIT_RING=384
STRATA_STAGE_TRIM=1
STRATA_PREFILL_HELP=1
STRATA_SPLIT_OWN=0        # 显式借用，避免 auto 切 own
```

**P2 — 启动参数（需在本机重校，别照搬）**
1. `--pipeline-windows 2`、`--layer-split 26`（可扫 24-28）；
2. `--pcie-frac` 重校（他们 0.35，我们 0.20：按 prefill **和** decode 分别看）；
3. `--resident-experts`（我们 125 GB 有余量，配上 0.1.42 的借槽 RAM 备份可能再上一层）；
4. `--ple-row-cache 8388608 --ple-inflight 1024`；
5. `--conversation-cache-mib 13312 --conversation-cache-slots 4`（agent 场景）；
6. 保持 `--prefill auto`（他们实测 `auto:16384` 在 48 GB 上伤 prefill；我们 167K 测得三档等速，选省显存的 `auto`）；
7. 保持 `--spec-min-p 0.70`、`--kv int8`；`--kv-resident` 按内存余量定（我们 125 GB，不必学他们的 32768）。

**P3 — 口径与验收**
- 采用他们的 TTFT 口径跑 2K/5K/8K/20K/60K/128K 阶梯（每档换 seed、×2 次），两个模型各跑一遍，和他们的表逐档对比；
- 验收门槛照抄：启动日志的"kept in RAM too"、`resident 16-17 GiB`、`N% of the experts resident`、两卡显存 ~96%；
- 思考默认改 `low`（对 agent 用途是白送的 3× 延迟改善）。

**P4 — 我们已经有、别丢的**
`--mtp mtp/rt`、`STA_PREFILL_STREAM_MIN=4096`、`--pcie-frac 0.20`（模型相关，重校后决定）、`--pool-workers 23`、安静的测量纪律（主机争用会让同一配置 43↔73 tok/s）、SSD 放模型、分层双卡（不要 peer 模式）。

## 5. 风险与注意

1. **新引擎 + 旧 pack**：0.1.39+ 的 pack 格式可能与 0.1.38 产物不兼容，升级后需用新 `tools/iq_pack.py` 重打（~40 s/模型）。
2. **配置键变化**：0.1.39+ 有 `--aux-cpus`、`--pipeline-windows` 等新键，老档仍可跑但不带新收益；`parallel` 键在 0.1.39+ 的 server 里出现（我们 0.1.38 的 server 是单序列 FIFO，等价 parallel=1）。
3. **他们与我们的差异**：他们 48 GB、PCIe x8+x8、8 核 Ryzen；我们 125 GB、x16+x16、72 线程。他们的"否决项"里有些是**内存不足导致**（`auto:16384`、`--kv-resident` 变大），我们不必然复现——要在本机重测，不要照搬结论。
4. **第三方数字未经我们验证**；但他们与我们在若干独立结论上一致（`--peer-device` 更慢、`--ple-io ram` 无差异、`--prefill` 大块无收益、calibrate 的 pool-workers 伤 prefill），可信度较高。

## 6. 已实施：升级 0.1.38 → v0.1.43.1 + 套用社区方案（2026-10-13）

- 引擎：`git checkout v0.1.43.1`（= 上游 HEAD `a0116b27`），**独立目录编译** `build-0.1.43/strata`（旧 `build/strata` 保留可回退）；llama.cpp 钉版未变（`3cf0325`），`STRATA_MMQ_KQUANTS=ON`。
- 档：`strata-orca-256k.json` / `strata-orca-512k.json` 换成社区方案；旧档备份在 `archive/v0.1.38-decks/`。
- **AVX2 关键点**：`generate.cpp:2813` 只在**非 native pack**时才要求 AVX512（`cpu_require_expert_support`）。我们的 IQ 系 pack 是 native pack → 新引擎在双路 E5-2696 v3（Haswell，仅 AVX2）上可跑，启动会打印
  `this CPU has no AVX-512: the expert kernels run on AVX-2 (multi-token for the i-quant gate/up rows)`。
- 与社区的**有意偏差**：① 不加 `--kv-resident`（社区 256K 档也没有，且与 `--resident-experts` 同用会吃 RAM）；② 保留我们校准的 `--pool-workers 23`（社区只否掉了 calibrate 的 5）；③ 保留 `STRATA_PREFILL_STREAM_MIN=4096`（我们实测 +25%）；④ `--pcie-frac` 用社区的 0.35（我们旧引擎上是 0.20，待新引擎重校）。

## 7. 验证结果：0.1.43.1 + 社区方案（IQ3_XXS，2026-10-13，本机实测）

运行配置（用户启动的实例，实测核对过命令行）：`build-0.1.43/strata`、`layer_split 26`（日志 `CUDA0, then CUDA1 (split 26)` / `CUDA1 runs layers 26-47`）、
512K 档（`yarn/2`）、`--resident-experts`、`--pipeline-windows 2`、`--aux-cpus auto`、`--pcie-frac 0.35`、`--pool-workers 23`、
`--ple-row-cache 8388608 --ple-inflight 1024`、`--conversation-cache-mib 13312 --conversation-cache-slots 4`、
env 8 项（BF16_TC / SELECT_SIMT / SPLIT_MTP_BATCH / SPLIT_RING / STAGE_TRIM / PREFILL_HELP / SPLIT_OWN / PREFILL_STREAM_MIN）。

**启动验收门槛（全部命中）**
```
this CPU has no AVX-512: the expert kernels run on AVX-2 (multi-token for the i-quant gate/up rows)
R4 hit path ON - resident experts are computed on the GPU
layer split: CUDA1 runs layers 26-47, expert cache 6635 slots (13.45 GiB), 6635 of its 11264 profiled pairs
caches hold 15063 of 24576 profiled pairs (~98.6% of the routed mass)
strata serve: 478 MiB of VRAM free with everything loaded
```

**阶梯实测**（社区 TTFT 口径，暖机后每档 ×2；下表用引擎自己的 `read in` 报数，TTFT 口径与之±2%）

| 档 | prompt tokens | prefill tok/s（均值） | decode tok/s（均值） | 社区同档 prefill / decode | 我们 vs 社区 |
|---|---|---:|---:|---|---|
| 2K | 2,120 / 2,124 | **1001.8** | 78.1 | 665 / 80.0 | **+51%** |
| 5K | 5,216 / 5,229 | **1202.4** | 77.2 | 1010 / 80.3 | +19% |
| 8K | 8,324 / 8,355 | **1265.5** | 79.4 | 1095 / 83.4 | +16% |
| 20K | 20,844 / 20,849 | **1908.5** | 77.1 | 1706 / 75.7 | +12% |
| 60K | 62,356 / 62,420 | **2285.7** | 71.2 | 2116 / 76.5 | +8% |
| 128K | 132,903 / 133,080 | **2269.9** | 76.8 | 2163 / 70.7 | +5% |

→ **prefill 每一档都超过社区调优值（+5%~+51%），decode 基本持平**。TTFT 对照：128K 58.5-59.6 s（社区 61.5 s）、60K 27.2-28.1 s（社区 29.3-29.7 s）、20K 11.0-11.4 s（社区 12.2-12.3 s）、2K 2.07-2.17 s（社区 2.9-3.5 s）。

**注意点**
1. **冷启动**：进程起来后的第一条请求多花约 2.3 s（我第一轮未暖机时 2K 档只有 438 tok/s / TTFT 4.84 s；暖机后 980-1023 tok/s / 2.07-2.17 s）。测小 prompt 必须先暖机。
2. **前缀缓存有效**：暖机与阶梯用了同一 seed，20K 那次命中 `20844 reused + 5 read in 2543 ms`——即 2 万 token 的前缀复用只花 2.5 s，对 agent 反复前缀的场景是白送的大收益。
3. **新引擎自带耗时分解**（`strata prefill timing:`），58K 的一次分解：GPU timeline 33.1 s / wall 37.8 s，其中 `hc read 20.4%`、`qsa attn 11.9%`、`qsa select 10.5%`、`gdn 6.6%`、`qsa proj 4.2%`；host 侧 `PLE 4876 ms` 是大头。
4. `--vision` 被 server 自动补上（档里有 vision 段而 args 里没有 `--vision`）——符合 0.1.39+ 的新行为。
5. 旧引擎 0.1.38 上同一模型没有跑过同口径阶梯（需要一次重启），因此"升级带来多少"暂无同模型对同模型的差值；已有的是不同量化的间接对照（旧引擎 IQ3_S：167K 1380-1448、256K 1158）。

## 8. `--pcie-frac` 重校（IQ3_XXS + 0.1.43.1 + 社区方案，2026-10-13）

`--pcie-frac` 决定**专家未命中时走 PCIe 还是走 CPU 路径**的份额（`generate.cpp:575`：native pack 默认 0.55；自动精调 `pcie_frac_auto` 只在**单卡**生效，我们是双卡分层，必须手选）。

**扫描**（社区 TTFT 口径：20K/60K 各 2 次、max_tokens 256、暖机换 seed；`archive/pcie-sweep/` 有原始日志）

| `--pcie-frac` | 20K prefill | 60K prefill | decode 均值 tok/s | 专家缓存命中 |
|---|---:|---:|---:|---:|
| 0.00 | 1863 / 1823 | 2365 / 2193 | 76.4 | 99.7% |
| 0.05 | 1850 / — | 2313 / 2211 | ~74 | 99.7% |
| 0.10 | 1822 / 1806 | 2299 / 2160 | 77.2 | 99.1% |
| **0.15** | 1892 / 1827 | 2385 / 2234 | **86.5** | 99.6% |
| 0.20 | 1868 / 1888 | 2329 / 2234 | 78.2 | 99.3% |
| 0.30 | 1811 / 1866 | 2320 / 2226 | 74.5 | 99.7% |
| 0.35（社区值） | 1855 / 1864 | 2293 / 2217 | 68.3 | 99.1% |

**头对头（贪心 temp=0，同一提示 ×4，排除采样随机性）**：0.15 → decode 92.4 tok/s，0.35 → 91.8 tok/s → **持平**。

**结论：在 `--resident-experts` 下这个开关几乎失效。** 引擎日志给出的机制证据：
`decode expert cache hit rate 99.1-99.7%`，走 PCIe 的未命中只有 **13 次 / 17.5 万次路由（0.0%）@0.20** 与 **271 次（0.2%）@0.35**——
专家几乎全在"显存 + 锁页 RAM"里，PCIe 这条路本来就没什么流量可分配。所以 0.00-0.35 的差异基本落在采样噪声内（±10-15%）。

**定值 0.20**（历史校准值；PCIe 读次数比 0.35 少一个数量级；任何一次测量里都不差于 0.35）。已写回
`strata-orca-256k.json` / `strata-orca-512k.json` / `strata-run-524288.json`。

**顺带的收获**：`--resident-experts` 让专家命中率从旧配置的 92.6% 升到 **99.1-99.7%**——社区清单第 1 条（借槽 RAM 备份 / 零专家文件读）在我们机器上确实成立。

## 9. 项目更新到 v0.1.44.1（2026-10-17）

`git fetch` 后有 **282 个新提交**，最新 tag **v0.1.44.1**（llama.cpp 钉版未变，仍是 `3cf0325`）。已 checkout 并编译到
**`build-0.1.44/strata`**（独立目录，`build/strata`=0.1.38 与 `build-0.1.43/strata`=0.1.43.1 都保留可回退）。

**与我们相关的新支持**
1. **Responses API 上游已实现**：`/v1/responses` 现在是正式 dialect（`serve/server.py:3756`），并带 retrieve/delete/cancel/compact 子路径（此前我们只是"没实现"）。
2. **无 AVX2 的 CPU 修复（#2324）**：`setup.py` 的 `MIN_ENGINE` 已抬到 `(0,1,44,1)`，说明 0.1.44.1 修掉了"老 CPU 上启动即非法指令"的一类崩溃；启动错误的提示现在会报出 CPU 的 AVX/AVX2/FMA/F16C 标志。
3. 新 opt-in：`--adapt-async`（arena/单卡，我们是双卡分层故不适用）、`STRATA_PIN_BY_NEED`、`STRATA_PCIE_MIN1`、`STRATA_ADAPT_OVERLAP`（文档已注明"分层切分下从不运行"）。
4. setup 的家族 env 白名单（`STRATA_HIPBLASLT_TUNING` / `STRATA_RESIDENT_PIN` / `STRATA_NO_ARENA_THP` / `STRATA_MMQ_KQ` / `STRATA_HC_Q8` / `STRATA_ROUTE_TAIL_SKIP`）主要面向 Unsloth UD-* 家族；我们的 IQ 系 pack 不适用，**无需改动**。

**兼容性核对**
- pack 格式：引擎接受 `# strata native experts vN`，我们两个 pack 都是 **v3** → 直接可用，**不需要重打**。
- 我们的档（`strata-orca-256k/-512k`、`strata-run-524288`）已把 `exe` 指向 `build-0.1.44/strata`。

## 10. 0.1.44.1 vs 0.1.43.1 速度实测：持平（关键结论 + 两个坑）

**阶梯对比**（同档 `strata-run-524288`、同提示词种子；0.1.43.1 用 pcie-frac 0.35，0.1.44.1 用 0.20）

| 档 | 0.1.43.1 | 0.1.44.1 | 差异 |
|---|---:|---:|---|
| 2K | 1002 | 744 / 906 | −10~25% |
| 5K | 1202 | 1181 / 1184 | ≈ |
| 8K | 1266 | 1221 / 1235 | ≈ |
| 20K | 1908 | 1802 / 1871 | −2~5% |
| 60K | 2286 | 2205 / 2274 | −1~3% |
| 128K | 2270 | 1497 | **−34%（假象，见下）** |

**受控终局测试**（两个引擎都 pcie-frac 0.35，启动前 `drop_caches` 让页缓存状态一致，各一发 128K）：

| 引擎 | 128K prefill | decode |
|---|---:|---:|
| 0.1.43.1 | **2432.5 tok/s** | 71.1 |
| 0.1.44.1 | **2402.9 tok/s** | 67.4 |

→ **引擎版本不是性能变量（差 1.2%）**。

**坑 1：页缓存/锁页补集会主导长上下文速度。** 每次新启动都会打
`FileExpertSource: N GiB really free for a 30.41 GiB page-locked complement; the model files' cached pages were released`——
引擎为了锁页补集**主动释放模型文件的页缓存**，于是 128K prefill 变成从 NVMe 读专家，实测可在 1253-2433 tok/s 之间摆动（±50%）。
对比长上下文时**必须先统一页缓存状态**（`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` 或干脆每次冷启动），否则结论不可信。
另注意 `WARNING: only 296-500 MiB of VRAM free at the start of this request (below 512 MiB)`：显存余量过低时驱动会页化显存，同样拖慢。

**坑 2：`--pcie-frac` 在长上下文**不**是惰性的。** 我在 20K/60K 上测到 0.00-0.35 差异都在噪声内（贪心头对头 0.15 vs 0.35 持平），据此把值改成 0.20——**这在 128K 上是错的**：

| 配置 | 128K prefill |
|---|---:|
| 0.1.43.1 + **0.35** | **2245 / 2295 / 2432** |
| 0.1.43.1 + 0.20 | 1253 |
| 0.1.44.1 + **0.35** | **1597 / 2403** |
| 0.1.44.1 + 0.20 | 1497 |

→ **回到 0.35**（已写回三个档）。原因：长上下文 prefill 要靠 PCIe 传专家，20K/60K 时专家多半已在缓存里，这个份额就看不出来。

**结论**：升级到 0.1.44.1 不损失速度（拿到 Responses API + #2324 老 CPU 修复），`--pcie-frac` 保持社区值 0.35。

## 11. `--vram-reserve-later-mib`（0.1.44.1 新）与"内存状态决定长上下文速度"

0.1.44.1 启动时会给出明确建议（旧版没有）：

```
CUDA1: 414 MiB of VRAM free with everything loaded - LOW: its window graphs are captured on first use and need room too,
and a full card can stall a request; add --vram-reserve-later-mib 1054 to the config's args (or lower --max-context)
```

按建议在三个 orca 档加 `--vram-reserve-later-mib 1054` 后：CUDA1 余量 414 → **776 MiB**，启动期 LOW 警告消失。
但**每请求**仍会警告 `only 298-510 MiB of VRAM free at the start of this request (below 512 MiB)`——133K token 的请求自身缓冲就吃掉了余量。

**128K 实测的分布（同代码同配置，全部 pcie-frac 0.35）**

| 会话条件 | 128K prefill |
|---|---:|
| 启动前 `drop_caches`（空闲 114 GB），0.1.43.1 | **2432** |
| 启动前 `drop_caches`（空闲 114 GB），0.1.44.1 | **2403** |
| 页缓存已满（空闲 ~0）启动，512K 档 | 1481 / 1497 / 1597 |
| 页缓存已满启动，256K 原生档 | 1060 |

→ **长上下文的 prefill 由"启动时的内存状态"主导（2 倍差距）**，机型/引擎/档都不是主因。要快就保证启动时有充足空闲内存
（`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` 后再启动服务），让引擎的锁页补集（30.41 GiB）与借槽 RAM 副本（7.88 GiB）都在内存里、不被页缓存挤压。
