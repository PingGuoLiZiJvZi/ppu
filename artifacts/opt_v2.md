# Qwen3.5-2B 相对 commit 3a95c4f 的有效优化

对比基准：

```text
commit: 3a95c4fb3b686b64e56e7a042ce044a2c0ad4626
subject: naive opt
```

评测文件：

```text
基线 EN: /root/ppu/fused_full_en.json
基线 CN: /root/ppu/fused_full_cn.json
新结果 EN: /root/ppu/artifacts/schedule_memory_v2/fused_full_en.json
新结果 CN: /root/ppu/artifacts/schedule_memory_v2/fused_full_cn.json
```

每份数据集包含 4029 题。`benchmark_public.py` 保持不变。

## 1. 跨样本复用 Decode Graph 和 Cache

### 原路径

每道题分别执行：

```text
创建 StaticCache
→ prefill
→ eager decode
→ capture Decode Graph
→ graph replay
→ 销毁本题 graph/cache
```

Graph capture、cache 分配和首次 eager decode 在每道题重复发生。

### 新路径

按照所需 cache 长度向上取整到 128 的倍数，最小为 512。每个长度档位保存一套：

- StaticCache；
- Decode Graph；
- token buffer；
- position buffer。

每个长度档位第一次使用时创建并 capture，后续题目执行：

```text
重置 Cache 状态
→ prefill 写入同一 Cache
→ 更新 token/position buffer
→ replay 已有 Graph
```

同一长度档位后续题目不再重复 capture，也不再执行 capture 前的 eager decode。

涉及代码：

- `evaluation_wrapper.py`：cache/graph 分桶、生命周期和复用；
- `qwen35_fused/graph.py`：更新 graph 输入 buffer 后重复 replay。

## 2. 删除 Packed Weight 的重复设备存储

### 原路径

Delta、MLP 和 Attention projection 完成 weight packing 后，packed weight 和原始 weight 同时驻留设备。

### 新路径

```text
拼接 packed weight
→ detach，切断原始 weight 引用
→ 注册 packed buffer
→ 释放已被替代的原始 weight parameter
```

释放范围：

- 18 层 Delta：QKV、Z、A、B projection；
- 24 层 MLP：gate、up projection；
- 6 层 Full Attention：Q、K、V projection。

保留 packed padding 后，融合新增常驻显存仅为 7,077,888 bytes。

## 3. 最终完整数据集性能

### English，4029 题

| 指标 | commit 3a95c4f 基线 | 新优化 | 变化 |
|---|---:|---:|---:|
| Accuracy | 79.7965% | 79.7965% | 不变，3215/4029 |
| Avg TTFT | 30.074 ms | 28.149 ms | −6.40% |
| Avg Throughput | 168.718 tok/s | 240.818 tok/s | +42.73%，1.427× |
| 完整运行时间 | 668.718 s | 606.699 s | −9.27%，节省 62.019 s |

### Chinese，4029 题

| 指标 | commit 3a95c4f 基线 | 新优化 | 变化 |
|---|---:|---:|---:|
| Accuracy | 83.9662% | 83.9662% | 不变，3383/4029 |
| Avg TTFT | 30.330 ms | 28.523 ms | −5.96% |
| Avg Throughput | 222.365 tok/s | 240.501 tok/s | +8.16%，1.082× |
| 完整运行时间 | 1211.722 s | 1157.214 s | −4.50%，节省 54.508 s |

### EN + CN 合并，8058 题

| 指标 | commit 3a95c4f 基线 | 新优化 | 变化 |
|---|---:|---:|---:|
| Accuracy | 81.8814% | 81.8814% | 不变，6598/8058 |
| Avg TTFT | 30.202 ms | 28.336 ms | −6.18% |
| Avg Throughput | 195.542 tok/s | 240.660 tok/s | +23.07%，1.231× |
| 完整运行时间 | 1880.440 s | 1763.913 s | −6.20%，节省 116.527 s |

`Avg Throughput` 按公开评测公式计算：先计算每题吞吐，再对全部题目取算术平均。

## 4. 显存表现

| 指标 | commit 3a95c4f 基线 | 新优化 | 变化 |
|---|---:|---:|---:|
| 融合前模型 allocated memory | 4,426,506,752 B | 4,426,506,752 B | 不变 |
| 融合后 allocated memory | 6,373,712,384 B | 4,433,584,640 B | −1,940,127,744 B |
| 融合新增 allocated memory | 1,947,205,632 B | 7,077,888 B | −99.64% |

实际回收 1.940 GB，即 1.807 GiB；融合后模型 allocated memory 下降 30.44%。

## 5. 输出一致性

| 检查项 | EN | CN |
|---|---:|---:|
| 解析答案变化 | 0/4029 | 0/4029 |
| token 数变化 | 0/4029 | 0/4029 |
| 总生成 token，基线/新优化 | 50,404 / 50,404 | 183,388 / 183,388 |
| Accuracy 变化 | 0 | 0 |
| Public validation failed samples，基线/新优化 | 1 / 1 | 3 / 3 |

## 6. 最终结论

相对 commit `3a95c4f`，最终保留的有效优化为：

1. StaticCache 和 Decode Graph 按长度分桶并跨样本复用；
2. 复用 Graph 时直接更新 token 和 position buffer；
3. packed weight 切断原始引用并释放重复 projection weight。

最终结果：准确率和输出长度不变，公开公式下双语 Avg Throughput 提升 23.07%，Avg TTFT 降低 6.18%，完整运行时间降低 6.20%，设备显存减少 1.940 GB。
