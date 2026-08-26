
1. 自研 decode 注意力 kernel —— 吞吐最大单项，预计 −1.2 ms/token
已定位根因：decode 时 HF create_causal_mask 给 StaticCache 生成了显式 [1,1,1,512] mask，且 GQA 被提前 repeat_interleave 成 8 头 KV。实测 PPU 的 fmha_cutlassF 走带 mask 路径 215.9µs/层，去 mask 后仅 23.8µs（9×）。写一个静态 shape 的 GQA decode kernel（有效长度从设备端 cumulative_length 张量读，解决图内变长问题），6 层 1.26ms → ~0.1ms。我已验证原型 kernel 数值 diff 仅 1e-2 量级（bf16 级）。吞吐 241 → ~350 tok/s。

2. 撤换 ppu_swiglu_gemv —— −0.29 ms/token，几乎零风险
图内实测自研融合 GEMV 30.2µs，而 F.linear（库 gemvt）13.1µs + silu_and_mul ~5µs 更快——现有“融合”实际是净损失（probe 数据支撑）。改回两步或把 GEMV 重写到库级带宽。

3. 小 GEMV 换 torch.mv —— [2048,2048]/[8320,2048]/[5120,2048] 上 mv 比 linear 快 20–35%，图内 24 处调用约 −0.1 ms/token。

4. Vision 按补丁数分桶 pad + CUDA graph —— TTFT −3.9 ms
Vision 是 launch-bound 而非算力-bound。MMBench 有 ~70 种 grid（按 shape 直接建图不现实），但 pad 到 128 补丁的桶后只剩 ~7 种 shape；pad 段用 cu_seqlens 隔离、merger 前 slice 掉，对真实 token 逐位一致。工程量中等（2–3 天）。

5. DeltaNet chunk 并行 prefill —— TTFT −5 ms，需过精度验证
串行 scan 占 prefill ~35%（num_stages 调优无效，已实测）。换 chunked WY/UT 表述可 5–10×，但会改变浮点结合顺序（非逐位一致），需全量答案级验证（当前门限还有 ~1.9pp 余量）。

6. 加载时预热全部 5 个 bucket 的 cache/graph —— 消除首样本 129ms TTFT 尖峰（私有集 bucket 分布未知，廉价保险）。

7. 权重 INT8（需主办方书面确认）：BF16 下 GEMV 已贴 2.27 TB/s 硬件读带宽上限，唯一再翻倍的手段是减半字节数。规则“量化后使用非主办方认可的权重包”是否覆盖运行时派生量化存在歧义；且之前 A100/TorchAO 的“无加速”结论不适用于 PPU 上手写 Triton W8A16 GEMV。收益最大但需先问清 + 精度验证。

综合 1+2+3：decode 预计 3.97 → ~2.4 ms（吞吐 ~400 tok/s，+65%）；加 4+6：TTFT 25.7 → ~21 ms（−18%）；再加 5：~16 ms（−38%）。

不建议投入：多步 graph（EOS 截断的浪费与同步节省基本抵消）、按 grid shape 建 vision 图（shape 太散）、lm_head 优化（已 2.27 TB/s 贴顶，除非走 INT8）。

建议从 1+2 开始（合计 ~1.5 ms/token、改动集中在 _patch_attention/_patch_mlp，一天内可验证），要我直接开始实现吗？