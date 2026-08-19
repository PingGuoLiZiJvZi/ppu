# Qwen3.5-2B PPU 融合实现

正式入口仍为 `evaluation_wrapper.py`。加载 BF16 模型后，入口自动调用 `qwen35_fused.apply_fusions`，greedy decoding 使用静态混合 cache 和 CUDA Graph。

## 已实现

- DeltaNet packed `2048→8320` projection；
- causal depthwise conv + SiLU + conv-state update；
- prefill/decode gated-delta recurrent kernel，FP32 state 原位更新；
- Text RMSNorm、Gated RMSNorm、48 个 residual-add/RMSNorm 边界；
- packed SwiGLU gate/up + fused SiLU×mul；
- packed Q+gate/K/V、Q/K RMSNorm + partial M-RoPE、K/V 直接写静态缓存、attention output gate；
- BF16-exact LM-head top-1，包含最低 vocab index tie-break；
- 静态 KV/Delta cache、完整 decode CUDA Graph 和 token feedback；
- Vision position interpolation/add、LayerNorm、residual/LayerNorm；
- 24 层 Vision packed QKV layout + FP32 RoPE 单 kernel；
- 单图 SDPA 直连和 CPU position/grid/cu-seqlens 预计算。

Vision 深度融合默认启用，实测 device kernel `823→260`，延迟 `13.01→8.34 ms`。

## 运行

```bash
python benchmark_public.py \
  --dataset-path ./datasets/mmbench/mmbench_dev_en.tsv \
  --model-path ./Qwen3.5-2B \
  --backend transformers \
  --num-samples 20 \
  --warmup-samples 1 \
  --output result_fused_en.json
```

禁用全部融合并运行基线：

```bash
QWEN35_FUSIONS=0 python benchmark_public.py \
  --dataset-path ./datasets/mmbench/mmbench_dev_en.tsv \
  --model-path ./Qwen3.5-2B \
  --backend transformers \
  --num-samples 20 \
  --warmup-samples 1 \
  --output result_baseline_en.json
```

## 验证

```bash
python -m qwen35_fused.validate_kernels
python artifacts/benchmark_qwen35_fused.py \
  --variant all --static-cache --cuda-graph \
  --decode-steps 16 --repetitions 2
python artifacts/profile_qwen3_5_bf16_kernels.py
```

当前 PPU/Triton 编译缓存位于 `qwen35_fused/triton_cache`。PyTorch、Triton 或 PPU SDK 版本变化时，Triton 会重新生成不匹配的条目。

温度为 0 时启用 fused greedy path；采样解码继续使用 Transformers 原始路径。

实测结果与原始记录见 `artifacts/qwen3_5_fusion_results.md`。最终 profile 的
CUDA Graph decode 为 `3.97 ms/token`（512-token cache），短缓存专项 benchmark
为 `3.66 ms/token`；基线为 `22.27 ms/token`。
