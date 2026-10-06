"""Figure 6 (更新版) — 跨队列重复 + 三队列减重 meta 分析
A  UFM1 与 ER 应激伙伴的偶联（新增 GSE53376 基线横断面，n=16）
B  三个减重队列术后 vs 术前的 Cohen's d（新增 GSE53376）
C  三队列合并效应的森林图（74 对，随机效应 DerSimonian-Laird）
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


def fmt_p(p):
    """AE&M wording rule: the complete word "p-value" and two decimals; a value
    that would round to 0.00 is reported as "p-value < 0.01"."""
    return "p-value < 0.01" if p < 0.005 else "p-value = {:.2f}".format(p)


def fmt_p_2line(p):
    """Same rule, wrapped onto two lines for the narrow heat-map cells."""
    return "p-value\n< 0.01" if p < 0.005 else "p-value\n= {:.2f}".format(p)

# ================= A 偶联热图（4 队列） =================
cohorts = ["GSE70353\nbulk · 770", "GSE174475\npurified · 43",
           "GSE245948\nRNA-seq · 76", "GSE53376\nbaseline · 16"]
genes = ["EIF2AK3", "ATF6", "SEL1L", "EIF2S1", "PLIN1"]
R = np.array([
    [np.nan, 0.440, 0.651, 0.590],
    [0.494, np.nan, 0.868, 0.502],
    [np.nan, np.nan, 0.876, 0.416],
    [np.nan, 0.741, 0.928, 0.696],
    [0.484, np.nan, 0.059, -0.427],
])
P = np.array([
    [np.nan, 0.0031, 2.0e-10, 0.0161],
    [1e-20, np.nan, 3.1e-24, 0.0473],
    [np.nan, np.nan, 3.7e-25, 0.1087],
    [np.nan, 1e-6, 1.8e-33, 0.0028],
    [1e-20, np.nan, 0.62, 0.0993],
])

fig = plt.figure(figsize=(13.6, 4.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.10, 0.92, 1.25], wspace=0.55)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(4))
ax.set_xticklabels(cohorts, fontsize=6.8, rotation=20, ha="right")
ax.set_yticks(range(5))
ax.set_yticklabels(genes, fontsize=9)
for i in range(5):
    for j in range(4):
        v, p = R[i, j], P[i, j]
        if np.isnan(v):
            ax.text(j, i, "n.t.", ha="center", va="center", fontsize=7, color="#888780")
        else:
            # No asterisk significance markers — AE&M discourages "statistically
            # significant" labelling; report the value and a two-decimal p-value.
            col = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}\n{fmt_p_2line(p)}", ha="center", va="center", fontsize=6.4,
                    color=col, linespacing=1.3)
ax.set_title("A  UFM1 – ER stress coupling", fontsize=10, loc="left", pad=8)
cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.015)
cb.set_label("Pearson r", fontsize=8)
cb.ax.tick_params(labelsize=7)

# ================= B 三队列干预重复 =================
ax2 = fig.add_subplot(gs[0, 1])
labels = ["GSE59034\n(n=16)", "GSE72158\n(n=42)", "GSE53376\n(n=16)"]
ds = [-1.164, -0.408, -0.134]
ps = [3.1e-4, 0.0115, 0.60]
dd = [-1.192, -0.619, -0.850]
pd_ = [2.5e-4, 2.5e-4, 4.0e-3]
x = np.arange(3)
w = 0.36
ax2.bar(x - w / 2, ds, w, color="#7F77DD", edgecolor="#534AB7", linewidth=0.6, label="UFM1 z")
ax2.bar(x + w / 2, dd, w, color="#1D9E75", edgecolor="#0F6E56", linewidth=0.6,
        label="Dysregulation index")
for i in range(3):
    ax2.text(x[i] - w / 2, ds[i] / 2, fmt_p(ps[i]), rotation=90, ha="center",
             va="center", fontsize=6.0, color="white", fontweight="bold")
    ax2.text(x[i] + w / 2, dd[i] / 2, fmt_p(pd_[i]), rotation=90, ha="center",
             va="center", fontsize=6.0, color="white", fontweight="bold")
    ax2.text(x[i] - w / 2, ds[i] - 0.07, f"{ds[i]:+.2f}", ha="center", va="top",
             fontsize=6.6, color="#3C3489")
    ax2.text(x[i] + w / 2, dd[i] - 0.07, f"{dd[i]:+.2f}", ha="center", va="top",
             fontsize=6.6, color="#0F6E56")
ax2.axhline(0, color="#5F5E5A", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Cohen's d (post vs pre)", fontsize=9)
ax2.set_ylim(-1.75, 0.40)
ax2.legend(frameon=False, fontsize=7.5, loc="upper right")
ax2.set_title("B  Reversal after bariatric surgery", fontsize=10, loc="left", pad=8)

# ================= C 森林图 =================
ax3 = fig.add_subplot(gs[0, 2])
meta = pd.read_csv(os.path.join(RES, "bariatric_three_cohort_meta.csv"))


rows = []
for feat in ["UFM1 z", "Dysregulation index"]:
    key = "UFM1 z-score" if feat == "UFM1 z" else "Dysregulation index"
    m = meta[meta.feature == key].iloc[0]
    raw = [("GSE59034 (16)", -1.164 if feat == "UFM1 z" else -1.192, 16),
           ("GSE72158 (42)", -0.408 if feat == "UFM1 z" else -0.619, 42),
           ("GSE53376 (16)", -0.134 if feat == "UFM1 z" else -0.850, 16)]
    studies = [(lab, d, float(np.sqrt(1.0 / n + d ** 2 / (2.0 * n)))) for lab, d, n in raw]
    for lab, d, se in studies:
        rows.append(dict(feat=feat, lab=lab, d=d, se=se, pooled=False))
    rows.append(dict(feat=feat, lab="Pooled (random)", d=m.d_random, se=m.se_random, pooled=True))
    rows.append(dict(feat=feat, lab=None, d=np.nan, se=np.nan, pooled=None,
                     note=f"I²={m.I2:.0f}%, {fmt_p(m.p_random)}"))
rows.append(dict(feat=None, lab=None, d=np.nan, se=np.nan, pooled=None, note=None))

y = 0
ytick, ylab = [], []
notes = []
for r in rows:
    if r["lab"] is None and r.get("note"):
        notes.append((y + 0.45, r["note"]))
        y -= 0.35
        continue
    if r["feat"] is None:
        y -= 0.45
        continue
    d, se = r["d"], r["se"]
    lo, hi = d - 1.96 * se, d + 1.96 * se
    if r["pooled"]:
        ax3.plot([lo, hi], [y, y], color="#0F6E56", lw=1.8)
        ax3.plot(d, y, marker="D", ms=6, color="#1D9E75")
    else:
        ax3.plot([lo, hi], [y, y], color="#7F77DD", lw=1.3)
        ax3.plot(d, y, marker="s", ms=5.2, color="#534AB7")
    ax3.text(1.03, y, f"{d:+.2f} ({lo:+.2f}, {hi:+.2f})", fontsize=6.8,
             color="#2C2C2A", va="center")
    ylab.append((y, r["lab"], r["pooled"]))
    y -= 1.0

for yy, txt in notes:
    ax3.text(1.32, yy, txt, fontsize=7.0, color="#5F5E5A", va="center", ha="right")

ax3.axvline(0, color="#5F5E5A", lw=0.8, ls="--")
ax3.set_yticks([a for a, _, _ in ylab][::-1])
ax3.set_yticklabels([b for _, b, _ in ylab][::-1], fontsize=7.4)
ax3.set_xlim(-2.1, 1.35)
ax3.set_ylim(min([a for a, _, _ in ylab]) - 0.7, 1.0)
ax3.set_xlabel("Cohen's d (post vs pre), 95% CI", fontsize=9)
ax3.tick_params(axis="x", labelsize=8)
ax3.set_title("C  Three-cohort pooled effect (74 pairs)", fontsize=10, loc="left", pad=8)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, "Fig6_cross_cohort_replication." + ext),
                dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved Fig6_cross_cohort_replication.png / .pdf")
