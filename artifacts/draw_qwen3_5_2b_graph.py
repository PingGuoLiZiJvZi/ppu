#!/usr/bin/env python3
"""Generate a standalone SVG of the local Qwen3.5-2B inference graph."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).with_name("qwen3_5_2b_inference_graph.svg")
W, H = 2800, 5480


class SVG:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, value: str) -> None:
        self.parts.append(value)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: int = 26,
        weight: int = 400,
        fill: str = "#d9e4f2",
        anchor: str = "start",
        cls: str = "",
        opacity: float = 1.0,
    ) -> None:
        class_attr = f' class="{cls}"' if cls else ""
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" opacity="{opacity}"{class_attr}>{escape(value)}</text>'
        )

    def multiline(
        self,
        x: float,
        y: float,
        lines: list[str],
        *,
        size: int = 24,
        line_h: int = 34,
        weight: int = 400,
        fill: str = "#d9e4f2",
        anchor: str = "start",
        opacity: float = 1.0,
    ) -> None:
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" opacity="{opacity}">'
        )
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else line_h
            self.add(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
        self.add("</text>")

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "#122033",
        stroke: str = "#314a66",
        sw: float = 2,
        rx: float = 18,
        opacity: float = 1.0,
        dash: str | None = None,
        shadow: bool = False,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        filter_attr = ' filter="url(#shadow)"' if shadow else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{dash_attr}{filter_attr}/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "#7aa2cc",
        sw: float = 3,
        dash: str | None = None,
        arrow: bool = False,
        opacity: float = 1.0,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"{dash_attr}{marker}/>'
        )

    def path(
        self,
        d: str,
        *,
        stroke: str = "#7aa2cc",
        sw: float = 3,
        fill: str = "none",
        dash: str | None = None,
        arrow: bool = True,
        opacity: float = 1.0,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(
            f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}" '
            f'opacity="{opacity}"{dash_attr}{marker}/>'
        )


s = SVG()
s.add(
    f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
 viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
 aria-labelledby="title desc">
<title id="title">Qwen3.5-2B 多模态推理完整计算图</title>
<desc id="desc">基于本地 Qwen3.5-2B 配置、权重张量和 Transformers 5.15.0 实现绘制，覆盖预处理、视觉塔、融合、混合注意力语言主干、缓存、prefill 和逐 token decode。</desc>
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#07111f"/><stop offset="1" stop-color="#0b1929"/></linearGradient>
  <linearGradient id="cyan" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#133d50"/><stop offset="1" stop-color="#10263b"/></linearGradient>
  <linearGradient id="purple" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#33255c"/><stop offset="1" stop-color="#17213a"/></linearGradient>
  <linearGradient id="amber" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4b3618"/><stop offset="1" stop-color="#182235"/></linearGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#000" flood-opacity=".34"/></filter>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#7aa2cc"/></marker>
  <style>
    text {{ font-family: Inter, "Noto Sans CJK SC", "Microsoft YaHei", system-ui, sans-serif; }}
    .mono {{ font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace; }}
    a text {{ text-decoration: underline; }}
  </style>
</defs>
<rect width="2800" height="5480" fill="url(#bg)"/>
<circle cx="2510" cy="170" r="260" fill="#154a63" opacity=".10"/>
<circle cx="260" cy="4800" r="420" fill="#5b3aa0" opacity=".08"/>
'''
)


def section(y: int, h: int, num: str, title: str, subtitle: str, color: str) -> None:
    s.rect(55, y, 2690, h, fill="#0c1828", stroke="#263c55", sw=2, rx=26, opacity=0.96)
    s.rect(82, y + 24, 72, 42, fill=color, stroke=color, sw=0, rx=21)
    s.text(118, y + 54, num, size=22, weight=800, fill="#07111f", anchor="middle")
    s.text(176, y + 55, title, size=34, weight=750, fill="#f0f6ff")
    s.text(2720, y + 54, subtitle, size=20, fill="#8da4be", anchor="end")


def box(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    lines: list[str],
    *,
    accent: str = "#4cc9c0",
    fill: str = "#112238",
    stroke: str | None = None,
    title_size: int = 26,
    text_size: int = 21,
    mono: bool = False,
    dash: str | None = None,
    opacity: float = 1.0,
) -> None:
    stroke = stroke or accent
    s.rect(x, y, w, h, fill=fill, stroke=stroke, sw=2, rx=18, dash=dash, opacity=opacity, shadow=True)
    s.rect(x, y, 9, h, fill=accent, stroke=accent, sw=0, rx=5, opacity=opacity)
    s.text(x + 28, y + 40, title, size=title_size, weight=720, fill="#f4f8fd", opacity=opacity)
    s.multiline(
        x + 28,
        y + 76,
        lines,
        size=text_size,
        line_h=text_size + 11,
        fill="#c4d3e4",
        opacity=opacity,
    )


def pill(x: int, y: int, w: int, label: str, color: str = "#45d1c6") -> None:
    s.rect(x, y, w, 38, fill="#0c1b2b", stroke=color, sw=1.5, rx=19)
    s.text(x + w / 2, y + 27, label, size=19, weight=650, fill=color, anchor="middle")


# Header
s.text(82, 98, "Qwen3.5-2B 多模态推理完整计算图", size=54, weight=820, fill="#f5f8fc")
s.text(84, 146, "当前工作区实际 checkpoint × Transformers 语义图（prefill + autoregressive decode）", size=26, fill="#9fb4c9")
pill(84, 178, 255, "Qwen3_5ForConditionalGeneration")
pill(354, 178, 210, "Transformers 5.15.0", "#a88cff")
pill(579, 178, 180, "BF16 主权重", "#f5bd61")
pill(774, 178, 212, "2.213B 活动参数", "#76d394")
pill(1001, 178, 220, "262,144 上下文", "#64b5f6")
s.multiline(
    2718,
    92,
    ["本地 revision: 15852e8c1636…", "config SHA-256: ed1c1723241f…", "绘制日期: 2026-08-19 UTC"],
    size=19,
    line_h=28,
    fill="#829ab4",
    anchor="end",
)


# 01 Input and preprocessing
section(260, 700, "01", "输入与处理器", "实际 benchmark 路径：单图 + 单选题 prompt", "#45d1c6")
box(95, 345, 430, 190, "图像输入", ["PIL.Image / RGB", "任意宽高；单图样本", "图像、文本共同进入 chat template"], accent="#4cc9c0")
box(95, 615, 430, 190, "题干与 A/B/C/D", ["拼接任务指令、题干、选项", "role=user + add_generation_prompt", "当前上限 max_new_tokens = 256"], accent="#7ea6ff")

box(
    650,
    325,
    680,
    270,
    "Qwen2VLImageProcessor",
    [
        "smart resize：总像素约束 65,536 … 16,777,216",
        "尺寸对齐 patch × merge = 16 × 2 = 32",
        "归一化：mean=[.5,.5,.5]，std=[.5,.5,.5]",
        "单张图复制/打包成 temporal_patch=2 的 patch 向量",
        "pixel_values: [Σ(T·H·W), 3·2·16·16=1536]",
        "image_grid_thw: [n_image, 3]",
    ],
    accent="#4cc9c0",
    fill="url(#cyan)",
)
box(
    650,
    625,
    680,
    245,
    "Qwen3VLProcessor + Qwen2Tokenizer",
    [
        "chat_template：插入视觉起止与 image_pad 占位符",
        "BPE → input_ids [B,L]；vocab = 248,320",
        "attention_mask [B,L]；type ids：文本0 / 图像1 / 视频2",
        "占位符数量 = Σ(T·H·W / 2²)",
        "224² 实测→256²；grid=[1,16,16] → 64 图像 token",
    ],
    accent="#7ea6ff",
    fill="url(#purple)",
)

box(
    1490,
    345,
    540,
    210,
    "送入模型的张量包",
    [
        "input_ids / attention_mask",
        "mm_token_type_ids",
        "pixel_values / image_grid_thw",
        "全部转移到 model.device",
    ],
    accent="#f5bd61",
    fill="url(#amber)",
)
box(
    1490,
    635,
    540,
    190,
    "固定生成参数",
    [
        "temperature=0.0 → do_sample=False",
        "top_p=1.0（greedy 下不生效）",
        "use_cache=True；TextIteratorStreamer",
    ],
    accent="#f5bd61",
)
box(
    2180,
    420,
    470,
    300,
    "两条首次前向支路",
    [
        "图像张量 → 视觉塔",
        "input_ids → 共享词嵌入",
        "视觉特征在 image_pad 位置替换",
        "随后统一进入语言模型 prefill",
        "decode 时不再重复运行视觉塔",
    ],
    accent="#76d394",
)
s.path("M525 440 C575 440 595 430 650 430", stroke="#4cc9c0")
s.path("M525 700 C580 700 595 735 650 735", stroke="#7ea6ff")
s.path("M1330 470 C1390 470 1420 450 1490 450", stroke="#f5bd61")
s.path("M1330 745 C1400 745 1415 710 1490 710", stroke="#f5bd61")
s.path("M2030 455 C2080 455 2110 505 2180 505", stroke="#76d394")
s.path("M2030 720 C2100 720 2110 625 2180 625", stroke="#76d394")


# 02 Vision tower
section(995, 990, "02", "视觉塔（仅 prefill）", "Qwen3-VL 风格 ViT；每个图像/视频块内双向注意力", "#4cc9c0")
box(
    95,
    1085,
    430,
    230,
    "Patch Embedding",
    [
        "Conv3D: 3 → 1024",
        "kernel=stride=(2,16,16)",
        "权重 [1024,3,2,16,16]",
        "输出 Xᵥ: [Npatch,1024]",
    ],
    accent="#4cc9c0",
)
box(
    625,
    1070,
    520,
    260,
    "视觉位置编码",
    [
        "可学习绝对表 [2304=48²,1024]",
        "按每张图的 H×W 网格双线性插值后相加",
        "2D rotary position ids（H,W）",
        "16 heads；head_dim=64；RoPE 覆盖完整 64 维",
        "cu_seqlens 隔离打包的多图/多帧序列",
    ],
    accent="#65b7ff",
)
box(
    1260,
    1050,
    920,
    410,
    "Vision Transformer Block × 24",
    [
        "U = X + Attention(LayerNorm(X))",
        "QKV: 1024 → 3072 → [N,16,64] × 3",
        "Q,K ← 2D-RoPE(Q,K)",
        "A = softmax(QKᵀ / √64)；同一视觉块内非因果全注意力",
        "Attention 输出：A·V → concat → Linear 1024→1024",
        "Y = U + MLP(LayerNorm(U))",
        "MLP: 1024 → 4096 → GELU(tanh approx) → 1024",
        "两处残差；LayerNorm eps=1e-6；所有 24 块同构",
    ],
    accent="#a88cff",
    fill="url(#purple)",
)
box(
    2290,
    1075,
    380,
    340,
    "Patch Merger",
    [
        "先 LN(1024)",
        "空间 2×2 邻域拼接",
        "4×1024 → 4096",
        "FC1 4096→4096",
        "GELU",
        "FC2 4096→2048",
        "Nvis = T·H·W / 4",
    ],
    accent="#f5bd61",
    fill="url(#amber)",
)
s.path("M525 1200 L625 1200", stroke="#4cc9c0")
s.path("M1145 1200 L1260 1200", stroke="#65b7ff")
s.path("M2180 1250 L2290 1250", stroke="#f5bd61")

box(
    180,
    1530,
    1090,
    330,
    "视觉张量形状示例（来自本地 processor 实测）",
    [
        "原始 224×224 RGB → smart resize 256×256",
        "pixel_values [256,1536]；image_grid_thw [[1,16,16]]",
        "PatchEmbed → [256,1024]；24×ViT 后仍为 [256,1024]",
        "2×2 merger → image_embeds [64,2048]",
        "与 input_ids 中 64 个 image_pad 占位符严格一一对应",
    ],
    accent="#76d394",
)
box(
    1450,
    1530,
    1090,
    330,
    "视频沿用同一视觉图（本 benchmark 未触发）",
    [
        "pixel_values_videos + video_grid_thw → 同一个 visual.forward",
        "处理器按时间采样，temporal_patch_size=2；帧用时间戳 token 分隔",
        "视觉输出替换 video_token_id=248057；图像替换 image_token_id=248056",
        "vision_start=248053；vision_end=248054",
        "当前单图评测只激活左侧图像路径，保留此框用于完整模型语义",
    ],
    accent="#7893ab",
    dash="10 8",
    opacity=0.78,
)
s.path("M2480 1415 C2480 1490 2220 1490 2220 1530", stroke="#7893ab", dash="10 8", opacity=.8)


# 03 Fusion
section(2020, 650, "03", "多模态早期融合与 M-RoPE", "视觉 token 与文本 token 在进入 decoder 前已处于同一序列", "#76d394")
box(
    95,
    2110,
    500,
    215,
    "共享词嵌入",
    [
        "Embedding Wₑ: [248320,2048]",
        "input_ids → Xtext [B,L,2048]",
        "Wₑ 与最终 lm_head 严格权重共享",
    ],
    accent="#7ea6ff",
)
box(
    705,
    2090,
    610,
    260,
    "masked_scatter 视觉替换",
    [
        "mask = (input_ids == image_token_id)",
        "校验 token 数 × 2048 == image_features.numel()",
        "X[mask] ← image_embeds（按模板占位顺序）",
        "得到融合序列 X₀ [B,L,2048]",
        "非 cross-attention：视觉向量成为普通 decoder token",
    ],
    accent="#76d394",
)
box(
    1435,
    2080,
    660,
    280,
    "4 行 position_ids → 3D M-RoPE",
    [
        "row 0：文本/因果 mask 的一维序列位置",
        "row 1..3：temporal / height / width（文本段三行相同）",
        "head_dim=256，仅前 25%=64 维做旋转；其余 192 维直通",
        "64 维由 32 个频率复制 cos/sin；mrope_section=[11,11,10]",
        "T/H/W 频率交错排布；rope_theta=10,000,000",
    ],
    accent="#a88cff",
    fill="url(#purple)",
)
box(
    2210,
    2100,
    460,
    240,
    "两类 mask",
    [
        "full_attention：标准 causal mask",
        "linear_attention：recurrent mask",
        "padding 位清零/隔离",
        "rope_deltas 供后续 decode 续接位置",
    ],
    accent="#f5bd61",
)
s.path("M595 2215 L705 2215", stroke="#76d394")
s.path("M1315 2215 L1435 2215", stroke="#a88cff")
s.path("M2095 2215 L2210 2215", stroke="#f5bd61")
s.text(1380, 2520, "融合隐藏态 X₀ + position_embeddings + mask mapping → 24 层语言主干", size=28, weight=700, fill="#eaf2fc", anchor="middle")
s.line(1380, 2545, 1380, 2640, stroke="#76d394", sw=4, arrow=True)


# 04 Language backbone
section(2705, 1560, "04", "语言主干：6 × (3 个 Gated DeltaNet + 1 个 Gated Full Attention)", "24 layers · hidden 2048 · dense SwiGLU MLP", "#a88cff")

# explicit layer chips
s.text(110, 2805, "逐层顺序（完整列举）", size=25, weight=700, fill="#e6eef8")
chip_y = 2830
for i in range(24):
    col = i % 12
    row = i // 12
    x = 110 + col * 212
    y = chip_y + row * 60
    full = (i + 1) % 4 == 0
    color = "#f5bd61" if full else "#45d1c6"
    label = f"L{i}: " + ("Full Attn" if full else "DeltaNet")
    s.rect(x, y, 194, 42, fill="#101f32", stroke=color, sw=1.5, rx=10)
    s.text(x + 97, y + 29, label, size=17, weight=650, fill=color, anchor="middle")

box(
    95,
    2995,
    1280,
    780,
    "A. 线性注意力层（L0-2,4-6,…,20-22，共 18 层）",
    [
        "① Pre RMSNorm：x̄ = RMSNorm(x)，eps=1e-6；权重语义为 (1+w)",
        "② 投影：qkv=Linear(2048→6144)，z=Linear(2048→2048)",
        "   a=Linear(2048→16)，b=Linear(2048→16)",
        "③ depthwise causal Conv1D：6144 groups，kernel=4，SiLU；再拆 q,k,v",
        "   q,k,v → [B,L,16,128]；q,k 做 L2Norm",
        "④ β=sigmoid(b)；g=−exp(A_log)·softplus(a+dt_bias)",
        "⑤ Delta rule（逐 token 语义；prefill 用等价 chunk kernel）：",
        "   S′ₜ = exp(gₜ)·Sₜ₋₁",
        "   δₜ = βₜ·(vₜ − kₜᵀS′ₜ)",
        "   Sₜ = S′ₜ + kₜ δₜᵀ；oₜ=(qₜ/√128)ᵀSₜ",
        "⑥ per-head Gated RMSNorm：õ=RMSNorm(o) ⊙ SiLU(z)",
        "   concat 16 heads → Linear(2048→2048) → 与输入 residual add",
        "⑦ Post RMSNorm → SwiGLU MLP → 第二次 residual add",
        "   MLP(x)=down[ SiLU(gate(x)) ⊙ up(x) ]；2048→6144→2048",
        "",
        "Prefill：chunk_gated_delta_rule（chunk=64），生成最终 S 与 conv state",
        "Decode(L=1)：recurrent_gated_delta_rule，O(16·128²) 固定状态更新",
    ],
    accent="#45d1c6",
    fill="url(#cyan)",
    text_size=20,
)
box(
    1425,
    2995,
    1270,
    780,
    "B. 全注意力层（L3,7,11,15,19,23，共 6 层）",
    [
        "① Pre RMSNorm：x̄ = RMSNorm(x)",
        "② q_proj: 2048→4096，按每头拆成 query 与 output gate",
        "   Q [B,8,L,256]；gate [B,L,2048]",
        "   k_proj/v_proj: 2048→512 → K,V [B,2,L,256]（GQA 4:1）",
        "③ 每头 Q/K RMSNorm；对 Q/K 前 64 维应用 T/H/W M-RoPE",
        "④ K,V 追加到该层 DynamicCache；KV 头 repeat 4× 对齐 8 个 Q 头",
        "⑤ 因果注意力：",
        "   A = softmax((QKᵀ)/√256 + causal_mask)（softmax 用 FP32）",
        "   O = A·V → [B,L,8,256] → reshape [B,L,2048]",
        "⑥ 输出门控：Õ = O ⊙ sigmoid(gate)",
        "   o_proj: 2048→2048 → 与输入 residual add",
        "⑦ Post RMSNorm → 与左侧完全相同的 dense SwiGLU MLP",
        "   2048→6144 (gate/up) → elementwise product → down 6144→2048",
        "   再做 residual add",
        "",
        "Prefill：对整段 L 做 causal attention；开销随 L² 增长",
        "Decode(L=1)：新 Q 读取历史 K/V；每层缓存随已生成长度线性增长",
    ],
    accent="#f5bd61",
    fill="url(#amber)",
    text_size=20,
)

box(
    95,
    3835,
    830,
    280,
    "Hybrid DynamicCache：18 个线性层",
    [
        "conv_state [B,6144,4]，通常 BF16",
        "recurrent_state S [B,16,128,128]，FP32",
        "每层约 1.05 MiB（B=1）；与上下文长度无关",
        "18 层合计约 18.84 MiB（参考 PyTorch fallback dtype）",
    ],
    accent="#45d1c6",
)
box(
    985,
    3835,
    820,
    280,
    "Hybrid DynamicCache：6 个全注意力层",
    [
        "K,V 各 [B,2,Lcache,256]，BF16",
        "每层每 token：2(K/V)×2头×256×2B = 2048B",
        "6 层合计约 12 KiB / token / batch item",
        "不做 sliding window；最长位置配置为 262,144",
    ],
    accent="#f5bd61",
)
box(
    1865,
    3835,
    830,
    280,
    "主干输出",
    [
        "L23 之后：final RMSNorm [B,L,2048]",
        "generate 自动设置 logits_to_keep=1",
        "lm_head(h)=h·Wₑᵀ → logits [B,1,248320]",
        "lm_head 无独立权重：与 input embedding tied",
    ],
    accent="#76d394",
)
s.path("M925 3975 L985 3975", stroke="#90a5bb")
s.path("M1805 3975 L1865 3975", stroke="#90a5bb")


# 05 Generation loop
section(4300, 720, "05", "实际 generate() 时间线", "首次 prefill 计算图与后续单 token decode 图分离", "#f5bd61")
box(
    95,
    4405,
    530,
    360,
    "Step 0 · Prefill",
    [
        "运行视觉塔并替换 image token",
        "整段融合序列 [B,L,2048] 过 24 层",
        "初始化 18×(conv,S) + 6×KV cache",
        "仅最后位置过 lm_head（logits_to_keep=1）",
        "argmax 产生首个新 token",
        "首个非空 streamer chunk → TTFT",
    ],
    accent="#76d394",
)
box(
    770,
    4405,
    600,
    360,
    "Step t≥1 · Decode（循环）",
    [
        "仅新 token id → embedding [B,1,2048]",
        "position = text_position + rope_delta",
        "视觉塔跳过；pixel_values 不再参与 forward",
        "DeltaNet 原位更新固定 S/conv state",
        "Full Attn 把新 K/V 追加并读取历史 cache",
        "lm_head → next logits",
    ],
    accent="#f5bd61",
    fill="url(#amber)",
)
box(
    1530,
    4405,
    510,
    360,
    "Token 选择",
    [
        "当前 benchmark：do_sample=False",
        "next_token = argmax(logits)",
        "追加到 input_ids；attention_mask +1",
        "streamer 增量 decode 文本片段",
        "若未终止则回到 Step t≥1",
    ],
    accent="#7ea6ff",
)
box(
    2180,
    4405,
    490,
    360,
    "停止与返回",
    [
        "遇到 eos_token_id=248044",
        "或生成达到 256 tokens",
        "tokenizer decode（跳过特殊 token）",
        "返回 text / token_count",
        "记录 TTFT / elapsed / throughput",
    ],
    accent="#a88cff",
)
s.path("M625 4585 L770 4585", stroke="#76d394")
s.path("M1370 4585 L1530 4585", stroke="#f5bd61")
s.path("M2040 4585 L2180 4585", stroke="#7ea6ff")
s.path("M1785 4765 C1785 4920 1055 4920 1055 4765", stroke="#f5bd61", dash="11 8")
s.text(1420, 4905, "未达到停止条件：继续自回归循环", size=20, fill="#f5bd61", anchor="middle")


# 06 Evidence, parameters, inactive MTP
section(5055, 360, "06", "checkpoint 组成、边界与证据", "实线 = 当前 Transformers benchmark 激活路径；虚线 = checkpoint 存在但当前路径未激活", "#7893ab")
s.rect(90, 5145, 820, 200, fill="#101d2d", stroke="#76d394", sw=2, rx=18)
s.text(118, 5185, "活动参数与本地权重", size=25, weight=720, fill="#f4f8fd")
s.multiline(
    118,
    5220,
    [
        "语言模型 1,881,825,088 + 视觉塔 331,416,576",
        "= 2,213,241,664 个活动参数",
        "活动权重 4,426,488,512 bytes（绝大多数 BF16）",
        "其中 A_log/dt_bias 等共有 2,592 个 FP32 参数",
        "文件总计 2,274,069,824 参数 / 4,548,144,832 tensor bytes",
    ],
    size=18,
    line_h=28,
    fill="#c4d3e4",
)

s.rect(950, 5145, 760, 200, fill="#101d2d", stroke="#7893ab", sw=2, rx=18, dash="10 8", opacity=.8)
s.text(978, 5185, "MTP 旁路（当前 benchmark 不激活）", size=25, weight=720, fill="#d9e4f2", opacity=.85)
s.multiline(
    978,
    5220,
    [
        "checkpoint 含 mtp.*：60,828,160 参数（121.66 MB BF16）",
        "Qwen3_5ForConditionalGeneration 将其列为 ignored unexpected keys",
        "当前 generate() 无 assistant/speculative 配置",
        "需用 SGLang/vLLM 的 NEXTN speculative 路径另行接入",
    ],
    size=18,
    line_h=28,
    fill="#aebed0",
    opacity=.82,
)

s.rect(1750, 5125, 950, 235, fill="#101d2d", stroke="#65b7ff", sw=2, rx=18)
s.text(1778, 5165, "资料来源（点击可打开）", size=25, weight=720, fill="#f4f8fd")
sources = [
    ("官方 Qwen3.5-2B 模型卡", "https://huggingface.co/Qwen/Qwen3.5-2B"),
    ("官方 checkpoint config.json", "https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json"),
    ("Transformers Qwen3.5 架构文档", "https://huggingface.co/docs/transformers/model_doc/qwen3_5"),
    ("Transformers 当前建模源码", "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py"),
]
for i, (label, url) in enumerate(sources):
    yy = 5203 + i * 34
    s.add(f'<a xlink:href="{escape(url)}" target="_blank">')
    s.text(1780, yy, "• " + label, size=18, fill="#79bfff")
    s.add("</a>")

s.text(
    1400,
    5450,
    "语义基线：/root/ppu/evaluation_wrapper.py + 本地 checkpoint revision 15852e8c…；算子后端可按 GPU/安装情况选择 SDPA、FlashAttention、Hub kernel 或 PyTorch fallback，但不改变图中数学数据流。",
    size=17,
    fill="#748ca5",
    anchor="middle",
)
s.add("</svg>")

OUT.write_text("".join(s.parts), encoding="utf-8")
print(OUT)
