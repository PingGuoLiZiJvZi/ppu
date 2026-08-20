# Qwen3.5-2B 计算调度与内存管理优化方案

## 1. 性能依据

| 路径 | Decode 延迟 |
|---|---:|
| Baseline eager | 22.27 ms/token |
| Fused eager | 8.38 ms/token |
| Fused CUDA Graph，cache=512 | 3.97 ms/token |
| Fused CUDA Graph，短 cache | 3.66 ms/token |

CUDA Graph 将融合路径从 8.38 ms/token 降至 3.66 ms/token，额外降低 56.3%。短 cache 相比 cache=512 再降低 7.8%。后续重点是跨 token、跨样本调度和显存生命周期。

## 2. 计算调度优化

### 2.1 持久化 Decode Graph 和 StaticCache

当前每个样本重新创建 StaticCache 和 GreedyDecodeGraph。改为固定地址的持久化运行资源：

```text
持久化 StaticCache
持久化 token/position/output buffer
按 cache 长度选择 graph bucket
每个 bucket 只 capture 一次
每个请求重置 state 后执行 prefill
```

建议建立 512、1024、2048 三档 cache bucket。请求选择能够容纳 prompt 和最大输出的最小 bucket。

实现内容：

- Full Attention K/V cache 清零并重置长度；
- Delta conv state 和 recurrent state 清零；
- Prefill 写入持久化 cache；
- graph 使用固定 token、position、cache 和输出地址；
- graph capture 结果按 bucket 缓存。

收益：消除逐样本 cache 分配、graph capture 和 graph memory pool 重建。

### 2.2 多 Token Decode Graph

当前每个 token 执行一次 graph replay 和一次 `token.item()`：

```text
graph replay
→ token.item()
→ CPU 检查 EOS
→ 下一次 replay
```

改为一次 graph replay 连续生成 4～8 个 token：

```text
一次 graph replay
→ 连续 N 个 decode step
→ token 写入设备端输出数组
→ CPU 读取 N 个 token
→ 定位第一个 EOS
```

收益：graph replay 和 CPU-GPU 同步次数降低 4～8 倍。

### 2.3 PPU Kernel 参数调优

对关键 Triton kernel 扫描以下参数：

| Kernel | 参数组合 |
|---|---|
| Delta recurrent | `block_v=4/8/16`，`num_warps=4/8`，`num_stages=1/2` |
| LM head top-1 | `block_n=8/16/32`，`block_k=64/128/256` |
| RMSNorm/LayerNorm | block size、`num_warps=4/8` |
| SwiGLU/Attention gate | block size、vector width、warp 数量 |

按模型实际 shape 离线选择最快配置，并随预编译 kernel 一起发布。

评价指标：

- kernel 延迟；
- occupancy；
- register 使用和 spill；
- 显存带宽利用率；
- L2/cache 命中率。

### 2.4 GEMV 前后处理融合

继续压缩以下边界：

```text
Delta:
packed GEMV → causal conv → recurrence/gated norm → out projection

MLP:
packed gate/up GEMV → SiLU×mul → down projection

Attention:
packed QGKV GEMV → norm/RoPE/cache → FlashAttention → gate → O projection
```

候选实现：

- packed GEMV epilogue 直接执行 Delta conv 输入变换；
- Delta recurrent 输出直接执行 gated RMSNorm；
- packed gate/up GEMV epilogue 直接执行 SiLU×mul；
- Attention gate 进入 O projection 的输入 prologue。

目标是减少 activation 落显存和再次读取。

## 3. 内存管理优化

### 3.1 消除 Packed Weight 重复存储

Packed weight 与原始 projection 权重同时驻留，额外显存为：

| Packed weight | 额外显存 |
|---|---:|
| 18 层 Delta projection | 585 MiB |
| 24 层 MLP gate/up | 1152 MiB |
| 6 层 Attention QGKV | 120 MiB |
| 合计 | 1857 MiB，约 1.81 GiB |

处理方式：

- 模型加载时直接构造 packed storage；或
- packing 完成后释放对应的原始设备权重；
- fallback 权重转移到 CPU。

### 3.2 临时内存预分配验证

对 Delta、MLP 和 Attention 的 packed projection output 进行持久化预分配。EN 100 题结果：

| 配置 | TTFT | Avg Throughput |
|---|---:|---:|
| 不预分配 | 26.264 ms | 245.956 tok/s |
| 预分配 | 26.250 ms | 244.669 tok/s |

TTFT 变化 −0.05%，吞吐下降 0.52%，因此未启用。

### 3.3 Cache 长度分桶

短 cache 实测比 cache=512 快 7.8%。采用最小可容纳 bucket，降低 Full Attention 的无效 cache 访问：

| 总长度 | Bucket |
|---:|---:|
| `≤512` | 512 |
| `513～1024` | 1024 |
| `1025～2048` | 2048 |

记录各 bucket 的请求占比、graph 命中率和 decode 延迟，再调整 bucket 边界。

### 3.4 跨请求流水

将下一样本的 CPU processor、图像解码和输入准备，与当前样本的设备 decode 重叠：

```text
CPU worker: sample N+1 processor/image decode
PPU stream: sample N decode
H2D stream: sample N+1 input transfer
```

使用双缓冲保存 processor 输出和设备输入，减少完整数据集的设备空闲时间。

## 4. 实施顺序

| 优先级 | 项目 | 主要收益 |
|---|---|---|
| P0 | 持久化 StaticCache 和 Decode Graph | 消除逐样本分配与 capture |
| P0 | 4～8 token graph | 减少 replay 和 `token.item()` 同步 |
| P0 | 释放 packed weight 对应的原始设备权重 | 回收约 1.81 GiB 显存 |
| P1 | Cache 长度分桶 | 降低 cache 访问量并复用 graph |
| P1 | PPU Triton tile/warp/stage 调优 | 提高设备执行效率 |
| P1 | GEMV epilogue/prologue 融合 | 减少 activation 显存流量 |
| P2 | 跨请求 CPU/PPU 流水 | 缩短完整数据集 wall time |

## 5. 验证指标

### 性能

- TTFT；
- decode token/s；
- 完整数据集 wall time；
- graph capture 次数与累计耗时；
- 每 token graph replay 次数；
- CPU-GPU 同步次数；
- peak allocated/reserved memory；
- 模型常驻显存；
- device kernel 时间；
- graph node 间隙。

### 正确性

- 单 kernel BF16/FP32 误差；
- greedy token 序列；
- EOS 截断位置；
- cache reset 后跨样本状态；
- 完整中英文数据集准确率。

## 6. 修改范围

`benchmark_public.py` 保持不变。修改范围：

- `evaluation_wrapper.py`：graph/cache 生命周期和 decode 循环；
- `qwen35_fused/graph.py`：graph capture、replay 和 token buffer；
- `qwen35_fused/integration.py`：packed weight 和 cache 接入；
- `qwen35_fused/kernels.py`：Triton kernel 和 PPU 参数；
- 独立运行入口：跨请求流水和性能统计。
