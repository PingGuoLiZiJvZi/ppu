# Qwen3.5-2B 全融合验证结果

环境：PPU-ZW810E、PyTorch 2.9.0、Transformers 5.15.0、BF16、batch=1。

## 阶段性能

| 阶段 | 基线 | 融合后 | 提升 |
|---|---:|---:|---:|
| Vision encoder | 13.01 ms | 8.34 ms | 1.56× |
| Language prefill，L=340 | 102.10 ms | 25.81 ms | 3.96× |
| Decode eager | 22.27 ms/token | 8.38 ms/token | 2.66× |
| Decode CUDA Graph，cache=512 | 22.27 ms/token | 3.97 ms/token | 5.61× |
| Decode CUDA Graph，短缓存 | 22.27 ms/token | 3.66 ms/token | 6.09× |

| 阶段 | 基线 device kernel | 融合后 device kernel | Host 提交 |
|---|---:|---:|---:|
| Vision encoder | 823 | 260 | 258 次 launch API |
| Language prefill | 11,296 | 409 | 393 次 launch API |
| Decode eager | 1,985/token | 290/token | 288 次 launch API/token |
| Decode Graph | 1,985/token | 300 graph node/token | 1 次 `cudaGraphLaunch`/token |

Graph 使用固定 cache 地址，包含模型 decode、LM-head top-1、token feedback 和 position increment。Q/K norm、M-RoPE 与 K/V 静态缓存写入由同一个 kernel 完成，省去 6 个 Full Attention 层的 `index_copy_`；相对上一版 graph，节点数从 324 降至 300/token。

## 融合项消融

同一 340-token 输入、动态 cache、无 Graph：

| 配置 | Prefill | Decode |
|---|---:|---:|
| Baseline | 102.24 ms | 21.91 ms/token |
| RMSNorm + residual/norm + SwiGLU | 98.65 ms | 20.07 ms/token |
| DeltaNet | 27.93 ms | 12.56 ms/token |
| Full Attention 周边 | 99.58 ms | 20.89 ms/token |
| 全部 Language 融合 | 24.77 ms | 8.35 ms/token |

DeltaNet 消除了 prefill fallback 的 Python/chunk 细粒度提交，是 prefill 收益的主要来源；CUDA Graph 继续消除 decode 剩余的 host launch 间隙。

## Vision 深度融合

24 层 Vision Attention 原本每层分别提交 Q/K cast、`rotate_half`、乘加、回写和布局操作。当前路径改成：

```text
packed QKV GEMM
→ QKV layout + Q/K FP32 RoPE 单 kernel
→ FlashAttention
→ output projection
→ residual-add/next LayerNorm
```

单图 SDPA 不再每层重复计算和回读 `cu_seqlens`。Interpolation indices、weights、position ids 和 cu-seqlens 在 processor CPU 阶段预计算，五种 grid shape 与 PPU 结果均逐位一致。最终 kernel 构成为 98 个 Linear GEMM、24 个 FlashAttention、24 个 QKV+RoPE、47 个 residual/norm、25 个 GELU、Conv3D、布局和尾部算子，总计 260 个。

## 正式 benchmark 入口

中英文各 20 题，`warmup_samples=1`：

| 数据集 | 路径 | TTFT | Throughput | Accuracy |
|---|---|---:|---:|---:|
| EN | 基线 | 180.80 ms | 57.19 token/s | 15/20 |
| EN | 融合 | 31.48 ms | 168.58 token/s | 16/20 |
| CN | 基线 | 139.12 ms | 43.32 token/s | 18/20 |
| CN | 融合 | 30.29 ms | 225.56 token/s | 18/20 |

对应收益：

- EN：TTFT 降低 82.59%，吞吐提升 2.95×；
- CN：TTFT 降低 78.23%，吞吐提升 5.21×。

## 正确性覆盖

- standalone kernel：RMSNorm、Gated RMSNorm、residual/norm、SwiGLU、conv、Vision norm/position 均 BF16 exact；
- Delta recurrent：本轮 BF16 输出逐位一致，FP32 state 最大误差 `5.96e-8`；
- Full Attention 直接缓存写入：V、写入区间和未写区间逐位一致；
- Vision Q/K RoPE 和 V layout：逐位一致；
- 完整 Vision hidden state 与 pooler output：逐位一致；
- LM-head top-1：BF16 rounding 和最小 index tie-break 一致；
- 16-token graph rollout：17 个 token（首 token + 16 次 replay）逐 token 一致；
- 最终 EN/CN 20 题答案与直接写缓存前逐题一致；
- CN 100 题：100/100 与完整基线答案一致；
- EN 100 题：99/100 与完整基线一致，唯一变化样本 `254` 从错误答案 A 变为正确答案 B；
- 公开集校验：全部通过。

## 实测决策

- Delta projection 选择 `8320` padding：prefill 最快，decode 与 `8256` 仅差约 `0.00024 ms/layer`；
- fused LM-head top-1：`0.4599→0.4503 ms`，默认启用；
- Vision 深度融合：device kernel `823→260`，延迟 `13.01→8.34 ms`，默认启用；
- CPU processor 不属于设备 kernel 融合，正式入口去掉了 streamer 线程和逐块文本拼接。

原始记录：

- [基线 kernel profile](qwen3_5_bf16_kernel_profile.json)
- [融合 kernel profile](qwen3_5_fused_kernel_profile.json)
- [算子正确性](qwen35_fused_kernel_validation.json)
- [最终短缓存 graph benchmark](qwen35_fused_direct_kv_benchmark.json)
- [最终 EN 20 题](fused_en20_direct_kv.json)
- [最终 CN 20 题](fused_cn20_direct_kv.json)
- [Vision 深度融合 EN 20 题](fused_en20_vision_deep_steady.json)
- [Vision 深度融合 CN 20 题](fused_cn20_vision_deep_steady.json)
- [EN 100 题](fused_en100.json)
- [CN 100 题](fused_cn100.json)
