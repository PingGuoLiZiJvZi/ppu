# Qwen3.5-2B BF16 Kernel 融合实施方案

适用环境：Transformers 5.15.0、PyTorch 2.9.0、BF16 eager、单卡 PPU-ZW810E。  
实测数据：[推理计算图](qwen3_5_2b_inference_graph.svg)、[kernel profile](qwen3_5_bf16_kernel_profile.json)。

## 1. 实测结论

| 阶段 | 无 profiler 延迟 | Host launch API | Device kernel | Kernel 时间合计 | 平均 kernel 时长 |
|---|---:|---:|---:|---:|---:|
| Vision encoder | 13.01 ms | 706 | 823 | 5.74 ms | 6.98 µs |
| Language prefill，长度 340 | 102.10 ms | 11,244 | 11,296 | 51.25 ms | 4.54 µs |
| Decode，每 token | 22.27 ms | 1,947 | 1,985 | 7.99 ms | 4.02 µs |

Decode 的非 kernel 时间上界为：

```text
22.274 ms - 7.986 ms = 14.288 ms/token
```

其中包含 host launch、设备调度间隙、框架执行、同步和缓存管理。当前 decode 是 launch-bound 路径，CUDA Graph 与算子融合同时进入 P0。

当前精度：

| 数据 | dtype |
|---|---|
| 权重、隐藏态、logits | BF16 |
| GEMM/GEMV | BF16 输入输出，FP32 累加 |
| Full Attention KV cache | BF16 |
| DeltaNet conv state | BF16 |
| DeltaNet recurrent state | FP32，形状 `[1,16,128,128]` |
| DeltaNet fallback 临时计算 | FP32 |
| RMSNorm 归约 | FP32 |

## 2. 调整后的优先级

| 优先级 | 项目 | 目标 |
|---|---|---|
| P0 | DeltaNet prefill fused operator | 消除 Python chunk loop 和数千个细粒度 kernel |
| P0 | Decode CUDA Graph baseline/final | 先测 launch 收益，融合完成后重新 capture |
| P0 | DeltaNet decode fast path | packed projection、conv、单次 state 读写 |
| P0 | Residual-add + Text RMSNorm | 融合 decoder 中反复出现的 residual/norm 边界 |
| P0 | Gated RMSNorm | 通用路径单独融合，decode 纳入 DeltaNet operator |
| P0 | SwiGLU | packed gate/up + fused SiLU×mul |
| P1 | Full Attention 周边 | packed Q+gate/K/V、QK norm、M-RoPE、cache write |
| P1 | LM head + BF16-exact argmax | 省去 logits 落盘与独立 argmax，不改变权重流量 |
| P2 | CPU processor | 语言模型 P0 后处理约 48 ms 的 TTFT 开销 |
| P2 | Vision 局部融合 | 处理剩余 13.01 ms Vision GPU 路径 |

CUDA Graph baseline 与 P0 算子开发并行推进；final graph 在静态缓存和融合算子落地后重新捕获。

## 3. P0：DeltaNet Prefill

### 当前路径

```text
hidden
 ├─ qkv projection → depthwise causal conv1d → SiLU
 ├─ z projection
 ├─ a projection
 └─ b projection
       ↓
Q/K L2Norm，q /= sqrt(128)，β=sigmoid(b)
g=-exp(A_log)·softplus(a+dt_bias)
       ↓
chunk gated delta rule，chunk_size=64
       ↓
Gated RMSNorm → output projection
```

当前未安装 FLA 和 `causal_conv1d`。长度 340 的 fallback 实测包含：

- FP32 BMM：595 次；
- `sum`：1,173 次；
- `clone`：2,426 次；
- `copy_`：4,203 次；
- host launch API：11,244 次。

Transformers fallback 在长度 63 的 Python 递推中反复执行 `clone/copy/sum`，随后执行 chunk 矩阵计算和跨 chunk state propagation。

### 实现边界

实现一个专用 `qwen35_delta_prefill` operator，内部使用少量设备 kernel：

1. Q/K L2Norm、Q scaling、`β/g` 和 chunk 数据准备；
2. 分块完成三角递推与 chunk 内 delta rule；
3. 分块完成跨 chunk state propagation 和输出；
4. 将最终 recurrent state 保持为 FP32；
5. 输出 BF16，删除 fallback 的 `pad/clone/copy` 中间张量。

每层 recurrent state 为：

```text
16 × 128 × 128 × 4 B = 1 MiB
```

按 head 和 state tile 分块，避免把完整 state 放入单个 block。目标是每层个位数到十几个 device kernel，而不是每层一个巨型 kernel。

### KPI

- 删除 Python chunk loop；
- FP32 中间结果不落全尺寸显存；
- prefill host launch：`11,244 → <3,000`；
- language prefill：`102.10 ms → <80 ms`。

先替换 delta-rule fallback，再融合 projection 和 conv。

## 4. P0：DeltaNet Decode

### Operator A：Packed projection + conv

四套 projection 为：

```text
qkv: 2048 → 6144，进入 depthwise causal conv
z:   2048 → 2048，直接保留
a:   2048 → 16，直接保留
b:   2048 → 16，直接保留
```

执行语义：

```text
packed GEMV
 ├─ qkv → conv state update → depthwise conv → SiLU
 ├─ z   → direct
 ├─ a   → direct
 └─ b   → direct
```

直接 benchmark 三种 weight packing：

1. `2048 → 8224`；
2. `2048 → 8256/8320`，尾部 padding；
3. `2048 → 8192`，a/b 使用独立 tiny projection。

选择依据是 PPU GEMV tile utilization、总 kernel 数和 TPOT。

### Operator B：Recurrent update + Gated RMSNorm

先执行：

```text
q̂ = L2Norm(q, eps=1e-6) / sqrt(128)
k̂ = L2Norm(k, eps=1e-6)
α = exp(g)
S' = αS
```

读取旧 state 时同时计算：

```text
m_k = k̂ᵀS'
m_q = q̂ᵀS'
δ   = β(v - m_k)
```

输出和 state 更新改写为：

```text
o     = m_q + (q̂ᵀk̂)δ
S_new = S' + k̂δᵀ
```

这与 eager 的 `o=q̂ᵀS_new` 等价。目标访存是 recurrent state 一次读取、一次写回，避免 rank-1 update 后再次读取完整 state。

Operator B 继续完成：

1. 每个 128 维 value head 的 Gated RMSNorm；
2. 直接乘 Gated RMSNorm 的 `weight`；
3. 乘 `SiLU(z.float())`；
4. cast BF16；
5. 接独立 output projection。

Decode fast path 不再提交独立 Gated RMSNorm kernel。通用 decode 目标为每层 2～4 个核心 device kernel。

### 数值约束

- Q/K L2Norm 使用 `eps=1e-6`；
- Q 在 L2Norm 后乘 `1/sqrt(128)`；
- `g=-exp(A_log)·softplus(a+dt_bias)`；
- recurrent state 始终为 FP32；
- conv state 原位更新并保留最后 4 个位置；
- prefill 与 decode 生成相同的最终 state。

## 5. P0：Decode CUDA Graph

当前栈已完成 BF16 graph 冒烟测试：

```text
PyTorch 2.9.0 + CUDA-compatible 12.9 + PPU-ZW810E
BF16 [1,2048] GEMV + SiLU capture/replay: PASS
```

分两次落地。

### Baseline graph

在现有 eager decode 上：

1. 预分配 token input、position、cache position 和输出 buffer；
2. 固定 KV、conv、recurrent cache 地址；
3. 预热后 capture 单 token decode；
4. replay 100～1,000 token；
5. 测量 TPOT、host launch 和 graph replay 间隙。

记录可回收比例：

```text
R_graph = (22.274 ms - baseline_graph_TPOT) / 14.288 ms
```

### Final graph

融合算子完成后重新 capture：

- 按 batch size 和 cache bucket 建 graph；
- 动态值写入固定地址 tensor；
- argmax 纳入 graph；
- streamer 和文本处理留在 graph 外。

算子融合减少 graph node 和显存流量；CUDA Graph 消除剩余 host launch。

真武 810E 的 PPU SDK 已包含 CUDA API wrapper、graph management API、vLLM CUDA Graph 修复和 cooperative-kernel graph node 修复。参考 [PPU SDK v1.4 release note](https://help.aliyun.com/zh/document_detail/2878029.html)、[后续 SDK release note](https://help.aliyun.com/zh/document_detail/2972910.html)和[PPU SDK v2.1 release note](https://help.aliyun.com/zh/document_detail/3030339.html)。

## 6. P0：Residual-add 与 RMSNorm

每个 token 有 79 次 norm：

```text
49 × Text RMSNorm
18 × DeltaNet Gated RMSNorm
12 × Full Attention Q/K RMSNorm
```

### 普通 RMSNorm

精确语义：

```text
x32 = x.float()
n32 = x32 * rsqrt(mean(x32²) + 1e-6)
out = (n32 * (1 + weight.float())).to(BF16)
```

实现三级 fast path：

1. `fused_qwen35_rmsnorm`：单独 1 kernel；
2. `residual_add_rmsnorm`：同时输出更新后的 residual 和 normalized hidden；
3. packed projection prologue：norm 后直接进入 DeltaNet、Attention 或 MLP projection，不写 normalized hidden。

Decoder 可形成 48 个 residual-add/norm 融合边界：

- 24 个 mixer residual add + post-attention RMSNorm；
- 23 个 MLP residual add + 下一层 input RMSNorm；
- 最后一层 MLP residual add + final RMSNorm。

第一层 input RMSNorm 保留独立 kernel，或并入第一层 projection prologue。

### Gated RMSNorm

Gated RMSNorm 的 weight 语义与普通 RMSNorm 不同：

```text
n32  = RMSNorm(x.float())
n_bf = n32.to(BF16)
u_bf = n_bf * weight              # 直接 weight，不是 1+weight
out  = (u_bf * SiLU(gate.float())).to(BF16)
```

- 独立 `fused_qwen35_gated_rmsnorm` 服务 prefill 和 fallback；
- DeltaNet decode 将这段计算纳入 Operator B。

## 7. P0：SwiGLU

当前每层执行两次输入 projection：

```text
gate = Linear(x)
up   = Linear(x)
y    = SiLU(gate) × up
out  = down_proj(y)
```

三 kernel baseline：

```text
packed gate/up GEMV: 2048 → 12288
fused SiLU-and-mul: 12288 → 6144
down GEMV: 6144 → 2048
```

下一阶段将 SiLU-and-mul 放入 packed GEMV epilogue，并把 RMSNorm 放入 projection prologue，目标为每层两个核心 kernel。

packed gate/up 不减少权重读取量。收益来自：

- 两次 GEMV launch 合成一次；
- hidden input 只读取一次；
- packed GEMV tile 利用率提高；
- SiLU 和 multiply 不再单独提交。

## 8. P1：Full Attention 周边

6 个 full-attention 层的 FlashAttention 主 kernel 约为 `0.12 ms/token`，保持现有实现。

`q_proj` 同时输出 Q 和 attention output gate。正确的 fast path 为：

```text
packed Q+gate/K/V projection
→ Q/K RMSNorm + M-RoPE
→ K/V direct cache store
→ FlashAttention
→ sigmoid(gate) × attention output
→ O projection
```

M-RoPE 参数：

- head dim：256；
- 旋转前 64 维；
- section：`[11,11,10]`；
- T/H/W interleaved；
- 后 192 维保持不变。

实现顺序：

1. packed Q+gate/K/V projection；
2. fused Q/K RMSNorm + M-RoPE + K/V cache write；
3. fused sigmoid gate × attention output；
4. 接入 O projection。

## 9. P1：LM Head + BF16-exact Argmax

```text
[1,2048] × [2048,248320] → BF16 [1,248320] → argmax
```

LM head 每 token 读取约：

```text
2048 × 248320 × 2 B = 1.017 GB BF16 weight
```

融合不会减少这部分权重流量，只省去约 0.5 MB logits 写回、再次读取和独立 argmax launch。当前 profile 中 `aten::argmax` 为 `94.04 µs / 8 token = 11.76 µs/token`，因此该项排在 P1 后段。

实现分块 vocab GEMV + top-1 reduction，并保持两条精确语义：

1. FP32 accumulator 先按 BF16 rounding 得到比较值；
2. 相同 BF16 logit 选择最小 vocab index。

## 10. P2：Processor 与 Vision

语言模型 P0 完成后的 TTFT 优化顺序：

```text
CPU processor：约 48 ms
→ Vision GPU：13.01 ms
```

Vision 已使用高效主 kernel：

- 98 个 BF16 Linear：PPU/CUTLASS GEMM；
- 24 个 attention：FlashAttention；
- 49 个 LayerNorm：vectorized LayerNorm；
- PatchEmbed：implicit-GEMM Conv3D。

Vision 局部融合点：

- position interpolation + add；
- LayerNorm + QKV；
- bias + GELU；
- residual add + 下一层 norm；
- merger reshape/concat + FC1。

## 11. 实施顺序

### 并行起步

1. 捕获现有 eager decode graph，得到 baseline graph TPOT；
2. 建立 operator microbenchmark、逐 token 等价测试和 kernel-count 统计。

### Language P0

1. DeltaNet prefill operator；
2. DeltaNet decode Operator B；
3. benchmark Operator A 的三种 packing；
4. residual-add + RMSNorm；
5. 独立 Gated RMSNorm；
6. packed SwiGLU；
7. capture final decode graph。

### Language P1

1. packed Q+gate/K/V；
2. Q/K norm + M-RoPE + cache write；
3. attention output gate；
4. LM head + BF16-exact argmax。

### TTFT P2

1. CPU processor；
2. Vision 局部融合。

## 12. 性能门禁

### P0 门禁

- decode host launch：`1,947 → <700/token`；
- decode TPOT：`22.27 ms → <18 ms`；
- prefill host launch：`11,244 → <3,000`；
- language prefill：`102.10 ms → <80 ms`；
- graph baseline 单独报告回收的 14.288 ms 比例；
- greedy token ids 与 eager 一致。

### 最终门禁

- decode host launch：`<400/token`；
- decode TPOT：约 `15 ms/token`；
- language prefill：`60～70 ms`；
- 中英文完整评测集不回退。

性能使用无 profiler CUDA Event、固定频率、预热后多轮统计；kernel 数使用 profiler 单独采集。

## 13. 正确性清单

- [ ] DeltaNet Q/K 使用 L2Norm，`eps=1e-6`；
- [ ] DeltaNet Q 在 L2Norm 后乘 `1/sqrt(128)`；
- [ ] recurrent state 保持 FP32；
- [ ] state 代数变换与 eager 输出逐 token 一致；
- [ ] prefill 与 decode 最终 recurrent state 一致；
- [ ] 普通 RMSNorm 使用 `(1+weight)`；
- [ ] Gated RMSNorm 直接使用 `weight`；
- [ ] Gated RMSNorm 保留 eager 的 BF16 rounding 位置；
- [ ] Gated RMSNorm 顺序为 norm → weight → SiLU(gate) multiply；
- [ ] Attention packed projection 包含 Q gate；
- [ ] M-RoPE 只旋转前 64 维，T/H/W interleaving 一致；
- [ ] K/V cache 写入位置和有效长度一致；
- [ ] fused argmax 比较 BF16-rounded logits；
- [ ] fused argmax 相等时选择最小 vocab index；
- [ ] greedy token ids 一致；
- [ ] 中英文完整评测集通过门禁。

上游语义核验：[Transformers Qwen3.5 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。
