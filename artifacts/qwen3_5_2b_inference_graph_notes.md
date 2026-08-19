# Qwen3.5-2B 推理计算图说明

主图：[qwen3_5_2b_inference_graph.svg](qwen3_5_2b_inference_graph.svg)

## 适用范围

这张图描述当前工作区 `/root/ppu` 的实际推理入口：`evaluation_wrapper.py` 使用本地 `Qwen3.5-2B` checkpoint，经 `AutoProcessor` 和 `AutoModelForImageTextToText`（实际类为 `Qwen3_5ForConditionalGeneration`）执行单图、多选题推理。运行时检查日期为 2026-08-19 UTC，已安装 Transformers 5.15.0。

图以模型的数学语义为主。具体注意力与 DeltaNet kernel 会按硬件和安装情况在 SDPA、FlashAttention、Hub kernel、`causal_conv1d`/FLA 或 PyTorch fallback 之间选择，但张量数据流与缓存语义不变。

## 本地证据快照

- checkpoint revision：`15852e8c16360a2fea060d615a32b45270f8a8fc`
- `config.json` SHA-256：`ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`
- 架构：`Qwen3_5ForConditionalGeneration`
- 活动参数：2,213,241,664（语言模型 1,881,825,088；视觉塔 331,416,576）
- checkpoint 另含 60,828,160 个 `mtp.*` 参数；当前 Transformers benchmark 不加载/不执行这条旁路。
- 语言主干：24 层，按 `linear, linear, linear, full` 重复 6 次。
- 视觉塔：24 层，hidden 1024，16 heads，patch `(2,16,16)`，空间合并 2×2，输出宽度 2048。
- 固定生成参数：`max_new_tokens=256, temperature=0.0, top_p=1.0, use_cache=True`，因此 token 选择为 greedy argmax。

## 在线资料

- [Qwen 官方 Qwen3.5-2B 模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)
- [官方 checkpoint config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)
- [Hugging Face Transformers：Qwen3.5 架构文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Hugging Face Transformers：Qwen3.5 当前建模源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

## 复现

在工作区根目录运行：

```bash
python artifacts/draw_qwen3_5_2b_graph.py
```

SVG 是自包含矢量文件，可直接用浏览器打开并任意缩放。
