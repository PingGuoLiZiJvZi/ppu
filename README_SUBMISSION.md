# Qwen3.5-2B PPU 优化提交说明

## 优化点

相对原始 Transformers BF16 eager baseline，本提交保留了以下优化：

- 投影合并：将 DeltaNet 的 QKV/Z/A/B、MLP 的 gate/up 以及 Full
  Attention 的 Q/gate/K/V 从多次独立 projection 合并为一次，减少 kernel
  launch 和 hidden-state 重复读取。
- 通用算子融合：使用 PPU Triton kernel 融合 RMSNorm、residual-add +
  RMSNorm/LayerNorm、gated RMSNorm、SiLU×mul、attention gate 和位置编码相加。
- DeltaNet：融合 causal conv1d、recurrent update 和状态原位更新；prefill
  预计算跨 value tile 重复使用的 Q/K reduction、缩放和指数因子，降低 TTFT。
- MLP：decode 阶段使用 PPU 专用 fused SwiGLU GEMV，合并 packed gate/up
  projection、SiLU 和乘法，减少中间 BF16 访存和 kernel 启动。
- Attention：融合 packed QGKV、Q/K norm 和 RoPE；单 token decode 使用
  maskless split-KV GQA Triton kernel，直接读取 2-head StaticCache，避免完整
  causal mask 和 KV head 展开。
- Vision：融合 QKV/RoPE、residual/norm 和位置插值，并按 patch 数缓存 Vision
  blocks 的 CUDA Graph，重复 shape 直接 replay。
- 推理框架：按长度 bucket 复用 StaticCache，使用静态地址捕获并复用 greedy
  decode graph；加载时预热 5 个 decode bucket 和常见 Vision shape；LM head
  直接执行分块 argmax，避免完整 logits 落盘。


## 运行要求

已验证环境为 PPU-ZW810E、Python 3.12、PyTorch 2.9.0、Transformers
5.15.0、Torchvision 0.24.0，以及 PPU SDK 提供的定制 Triton
`3.5.0+git4328cd8b`。不要用上游 PyPI Triton 覆盖 PPU 版本。

安装额外依赖：

```bash
python -m pip install -r requirements_extra.txt
```

`qwen35_fused/` 和 `triton/` 必须与 `evaluation_wrapper.py` 放在同一目录。
wrapper 加载模型后会自动安装优化路径并使用同级 `triton/` 预编译缓存，无需
设置环境变量；缓存目录不存在时会静默回退到 Triton 运行时编译。模型权重由
评测环境通过 `model_path` 提供，不包含在提交包中。

本地启动示例：

```bash
python benchmark_public.py \
  --model-path ./Qwen3.5-2B \
  --dataset-path ./datasets/mmbench/mmbench_dev_en.tsv \
  --backend transformers \
  --device cuda:0 \
  --output result.json
```
