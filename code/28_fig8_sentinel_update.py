"""Figure 8 (更新版) — 分子哨兵锚定减重表型
四个配对/对照面板：
  A  GSE72158 post vs pre (n=42)
  B  GSE59034 post vs pre (n=16)
  C  GSE59034 participants with vs without obesity (n=16)
  D  GSE53376 post vs pre (n=16)  ← 新增第三减重队列
标签中的箭头表示"真实减重"（A/B/D）或"肥胖"（C）下的预期方向。

图内文字遵循 AE&M 规范：完整词 p-value、两位小数（第二位为 0 时写 < 0.01）；
people-first 措辞；不使用星级显著性符号。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

DN = "#378ADD"
UP = "#D85A30"

PANELS = [
    ("GSE72158  post vs pre  (n = 42 pairs)", [
        ("LEP \u2193", -1.53, 2.1e-12),
        ("CD68 \u2193", -1.47, 6.1e-12),
        ("SLC2A4 \u2191", 0.44, 6.5e-3),
        ("PLIN1 \u2191", 0.59, 4.0e-4),
        ("PPARG \u2191", 0.74, 2.1e-5),
        ("ADIPOQ \u2191", 0.89, 1.0e-6),
        ("FASN \u2191", 1.07, 2.1e-8),
    ]),
    ("GSE59034  post vs pre  (n = 16 pairs)", [
        ("CD68 \u2193", -2.48, 5.5e-8),
        ("LEP \u2193", -1.17, 3.0e-4),
        ("ADIPOQ \u2191", 1.57, 1.5e-5),
    ]),
    ("GSE59034  participants with vs without obesity  (n = 16)", [
        ("ADIPOQ \u2193", -1.98, 9.8e-7),
        ("LEP \u2191", 0.99, 1.3e-3),
        ("CD68 \u2191", 1.77, 3.8e-6),
    ]),
    ("GSE53376  post vs pre  (n = 16 pairs)", [
        ("CD68 \u2193", -1.53, 2.0e-5),
        ("LEP \u2193", -1.26, 1.5e-4),
        ("ADIPOQ \u2191", 0.65, 2.0e-2),
        ("PLIN1 \u2191", 1.03, 8.8e-4),
        ("PPARG \u2191", 1.20, 2.3e-4),
        ("SLC2A4 \u2191", 1.83, 2.5e-6),
        ("FASN \u2191", 1.97, 1.1e-6),
    ]),
]


def fmt_p(p):
    """AE&M wording rule: the complete word "p-value" and two decimals; a value
    that would round to 0.00 is reported as "p-value < 0.01"."""
    if p < 0.005:
        return "p-value < 0.01"
    return "p-value = {:.2f}".format(p)


fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
LIM = (-3.8, 2.6)

for k, (ax, (title, rows)) in enumerate(zip(axes.ravel(), PANELS)):
    rows = sorted(rows, key=lambda r: r[1])
    labels = [r[0] for r in rows]
    ds = [r[1] for r in rows]
    ps = [r[2] for r in rows]
    colors = [DN if d < 0 else UP for d in ds]
    y = np.arange(len(rows))

    ax.barh(y, ds, color=colors, height=0.62, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#5F5E5A", lw=0.8)

    for i, (d, p) in enumerate(zip(ds, ps)):
        off = 0.10 if d >= 0 else -0.10
        ha = "left" if d >= 0 else "right"
        ax.text(d + off, i + 0.17, "d = {:+.2f}".format(d), va="center", ha=ha,
                fontsize=7.4, color="#2C2C2A")
        ax.text(d + off, i - 0.21, fmt_p(p), va="center", ha=ha, fontsize=7.0,
                color="#5F5E5A")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(LIM)
    ax.set_xlabel("Cohen's d", fontsize=9)
    ax.set_title("{}.  {}".format("ABCD"[k], title), fontsize=9.5, loc="left", pad=10)
    ax.tick_params(axis="x", labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888780")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_ylim(-0.7, len(rows) - 0.3)

fig.text(0.008, 0.985,
         "Molecular sentinels: every observed change matches the direction "
         "expected under genuine weight loss",
         fontsize=10.5, fontweight="bold", va="top")
fig.text(0.008, 0.950,
         "Blue = falling transcript, orange = rising transcript; arrows denote the expected "
         "direction.  Panel D additionally reproduces 16/16 genes\nvalidated by qPCR in the "
         "original study of this cohort (exact binomial p-value < 0.01), "
         "providing an internal positive control for the pipeline.",
         fontsize=8.0, color="#5F5E5A", va="top")

fig.tight_layout(rect=[0, 0, 1, 0.915])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, "Fig8_sentinel_anchoring." + ext),
                dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved Fig8_sentinel_anchoring.png / .pdf")
