"""生成补强后的两张核心图：跨队列 ER 偶联 + 两队列干预重复"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


def fmt_p_2line(p):
    """AE&M wording rule: the complete word "p-value" and two decimals; a value
    that would round to 0.00 is reported as "p-value < 0.01". No asterisk
    significance markers."""
    return "p-value\n< 0.01" if p < 0.005 else "p-value\n= {:.2f}".format(p)

# ================= Fig7: 跨队列 ER/eIF2α 偶联 =================
cohorts = ["GSE70353\nbulk (n=770)", "GSE174475\npurified (n=43)", "GSE245948\nRNA-seq (n=76)"]
genes = ["EIF2AK3", "ATF6", "SEL1L", "EIF2S1", "PLIN1"]
# 行=基因, 列=队列；NaN 表示未测
R = np.array([
    [np.nan, 0.440, 0.651],   # EIF2AK3
    [0.494, np.nan, 0.868],   # ATF6
    [np.nan, np.nan, 0.876],  # SEL1L
    [np.nan, 0.741, 0.928],   # EIF2S1
    [0.484, np.nan, 0.059],   # PLIN1
])
P = np.array([
    [np.nan, 0.0031, 2.0e-10],
    [1e-20, np.nan, 3.1e-24],
    [np.nan, np.nan, 3.7e-25],
    [np.nan, 1e-6, 1.8e-33],
    [1e-20, np.nan, 0.62],
])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))

ax = axes[0]
im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(cohorts, fontsize=8)
ax.set_yticks(range(5)); ax.set_yticklabels(genes, fontsize=9)
for i in range(5):
    for j in range(3):
        v = R[i, j]; p = P[i, j]
        if np.isnan(v):
            ax.text(j, i, "n.t.", ha="center", va="center", fontsize=7, color="#888780")
        else:
            star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))
            col = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}\n{star}", ha="center", va="center",
                    fontsize=8, color=col, fontweight="500" if p < 0.05 else "400")
ax.set_title("A  UFM1 – ER stress coupling across cohorts", fontsize=10, loc="left", pad=8)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Pearson r", fontsize=8)
cb.ax.tick_params(labelsize=7)

# ================= Fig7B: 两队列干预重复 =================
ax2 = axes[1]
np.random.seed(0)
labels = ["GSE59034\n(n=16)", "GSE72158\n(n=42)"]
ds = [-1.164, -0.408]
ps = [3.1e-4, 0.0115]
dd = [-1.192, -0.619]
pd_ = [3.0e-4, 2.5e-4]
x = np.arange(2); w = 0.36
b1 = ax2.bar(x - w/2, ds, w, color="#7F77DD", edgecolor="#534AB7", linewidth=0.6,
             label="UFM1 z-score")
b2 = ax2.bar(x + w/2, dd, w, color="#1D9E75", edgecolor="#0F6E56", linewidth=0.6,
             label="Dysregulation index")
for i in range(2):
    ax2.text(x[i] - w/2, ds[i] - 0.09, f"p={ps[i]:.1e}", ha="center", fontsize=7, color="#444441")
    ax2.text(x[i] + w/2, dd[i] - 0.09, f"p={pd_[i]:.1e}", ha="center", fontsize=7, color="#444441")
ax2.axhline(0, color="#5F5E5A", linewidth=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Cohen's d (post vs pre)", fontsize=9)
ax2.set_ylim(-1.5, 0.35)
ax2.legend(frameon=False, fontsize=8, loc="upper right")
ax2.set_title("B  Reversal after bariatric surgery", fontsize=10, loc="left", pad=8)

plt.tight_layout()
plt.savefig("figures/Fig7_cross_cohort_replication.png", bbox_inches="tight")
plt.savefig("figures/Fig7_cross_cohort_replication.pdf", bbox_inches="tight")
print("saved Fig7")

# ================= Fig8: 纯度假象 =================
fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 3.9))

ax3 = axes2[0]
mk = ["ADIPOQ", "SLC2A4", "PLIN1", "CD68", "PTPRC", "LYZ", "CD14"]
val = [-0.699, -0.732, -0.630, 0.716, 0.658, 0.676, 0.590]
cols = ["#1D9E75" if v < 0 else "#D85A30" for v in val]
ax3.barh(mk, val, color=cols, edgecolor="none", height=0.65)
ax3.axvline(0, color="#5F5E5A", linewidth=0.8)
ax3.set_xlabel("Pearson r with BMI (purified adipocytes, n=43)", fontsize=9)
ax3.set_xlim(-0.85, 0.85)
ax3.invert_yaxis()
ax3.set_title("A  Purity of 'purified' adipocytes varies with BMI", fontsize=10, loc="left", pad=8)
for i, v in enumerate(val):
    ax3.text(v + (0.03 if v > 0 else -0.03), i, f"{v:+.2f}", va="center",
             ha="left" if v > 0 else "right", fontsize=8, color="#444441")

ax4 = axes2[1]
feats = ["UFM1 z", "Machinery z", "Dysregulation"]
crude = [0.326, -0.605, 0.495]
adj = [0.015, -0.256, 0.106]
pc = [0.033, 1.75e-5, 7.3e-4]
pa = [0.93, 0.097, 0.50]
x2 = np.arange(3)
ax4.bar(x2 - 0.19, crude, 0.36, color="#B5D4F4", edgecolor="#185FA5",
        linewidth=0.6, label="Crude")
ax4.bar(x2 + 0.19, adj, 0.36, color="#F7C1C1", edgecolor="#A32D2D",
        linewidth=0.6, label="Adjusted for contamination")
for i in range(3):
    for off, v, p in ((-0.19, crude[i], pc[i]), (0.19, adj[i], pa[i])):
        txt = fmt_p_2line(p)
        if v >= 0:
            ax4.text(x2[i] + off, v + 0.03, txt, ha="center", va="bottom",
                     fontsize=6.2, color="#444441", linespacing=1.2)
        else:
            ax4.text(x2[i] + off, v - 0.03, txt, ha="center", va="top",
                     fontsize=6.2, color="#444441", linespacing=1.2)
ax4.axhline(0, color="#5F5E5A", linewidth=0.8)
ax4.set_xticks(x2); ax4.set_xticklabels(feats, fontsize=9)
ax4.set_ylabel("Correlation with BMI", fontsize=9)
ax4.set_ylim(-0.98, 0.72)
ax4.legend(frameon=False, fontsize=8)
ax4.set_title("B  Adiposity associations are purity-driven", fontsize=10, loc="left", pad=8)

plt.tight_layout()
plt.savefig("figures/Fig8_purity_confound.png", bbox_inches="tight")
plt.savefig("figures/Fig8_purity_confound.pdf", bbox_inches="tight")
print("saved Fig8")
