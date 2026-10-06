#!/usr/bin/env python3
"""Fig 2 重绘：单核部分改为描述性（方案 B）。

背景：原 Fig2 用的是 metadata 的 donor_id 作供体键，而该字段并非真实供体
（见 scripts/29 的说明）。修正后 UFM1 的肥胖/瘦差异不再显著（P = 0.064，
9 肥胖 vs 3 瘦），且 T 细胞的供体层面相关反而更强（rho = +0.637）。

因此 Figure 2 不再主张"脂肪细胞特异（adipocyte-intrinsic）"，只做
UFMylation 轴的细胞类型表达谱描述，并如实标注修正后的统计量。

面板：
  a  UFM1 在各脂肪组织细胞类型中的表达（表达量，描述性）
  b  五种主要细胞群中 UFM1 的组间 log2FC 与 p-value（供体层面，修正后）
  c  成熟脂肪细胞供体层面伪 bulk UFM1：无肥胖（n=3）vs 有肥胖（n=9）

图内文字遵循 AE&M 规范：完整词 p-value、两位小数；people-first 措辞；不使用星级符号。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")

CT = pd.read_csv(os.path.join(RES, "sc_celltype_expression.csv"))
PB = pd.read_csv(os.path.join(RES, "sc_pseudobulk_donor_corrected.csv"))
OVL = pd.read_csv(os.path.join(RES, "sc_obese_vs_lean_corrected.csv"))

fig = plt.figure(figsize=(11.4, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.85, 0.72], wspace=0.42)

# ---------- a: UFM1 表达谱 ----------
ax = fig.add_subplot(gs[0, 0])
top = CT.sort_values("UFM1", ascending=True).tail(12)
ax.barh(range(len(top)), top["UFM1"], color="#7fb3d5", edgecolor="#2c5f8d", height=.7)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top["cell_type"], fontsize=8)
ax.set_xlabel("UFM1 expression (per 10k UMI)")
ax.set_title("a  UFM1 expression across adipose cell types\n(descriptive)", fontsize=9.5, loc="left")

# ---------- b: 五种细胞群 obese-vs-lean ----------
ax = fig.add_subplot(gs[0, 1])
KEY = ["fat cell", "preadipocyte", "macrophage", "endothelial cell", "T cell"]
u = OVL[OVL.gene == "UFM1"].set_index("cell_type").reindex(KEY)
SHORT = {"fat cell": "Mature adipocyte", "preadipocyte": "Preadipocyte",
         "macrophage": "Macrophage", "endothelial cell": "Endothelium", "T cell": "T cell"}
cols = ["#c0392b" if p < 0.05 else "#7f8c8d" for p in u.p]
ax.bar(range(len(u)), u.log2FC, color=cols, edgecolor="#34495e", width=.62)
for i, (fc, p) in enumerate(zip(u.log2FC, u.p)):
    ax.text(i, fc + 0.035, f"p-value\n= {p:.2f}", ha="center", va="bottom", fontsize=6.2,
            color="#2c2c2a", linespacing=1.15)
ax.axhline(0, color="#5F5E5A", lw=0.8)
ax.set_xticks(range(len(u)))
ax.set_xticklabels([SHORT[c] for c in u.index], fontsize=7.0, rotation=32, ha="right")
ax.set_ylabel("log$_2$FC  UFM1  (with vs without obesity)")
ax.set_ylim(-0.25, 1.02)
ax.set_title("b  UFM1 by population, with versus without obesity\n(donor level; none reached p-value < 0.05)",
             fontsize=9.5, loc="left")

# ---------- c: 脂肪细胞供体层面 ----------
ax = fig.add_subplot(gs[0, 2])
fc = PB[(PB.cell_type == "fat cell") & (PB.grp.isin(["Lean", "Obese"]))]
d = [fc.loc[fc.grp == g, "UFM1"].dropna().values for g in ["Lean", "Obese"]]
bp = ax.boxplot(d, widths=.5, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="#f5cba7", edgecolor="#b9770e"),
                medianprops=dict(color="#c0392b", lw=1.6),
                whiskerprops=dict(color="#b9770e"), capprops=dict(color="#b9770e"))
rng = np.random.default_rng(1)
for i, dd in enumerate(d):
    ax.scatter(rng.normal(i + 1, .05, len(dd)), dd, s=30, color="#b9770e", alpha=.75, zorder=3)
u_stat, p = stats.mannwhitneyu(d[1], d[0], alternative="two-sided")
ax.text(0.5, 0.965, f"Mann\u2013Whitney p-value = {p:.2f}", transform=ax.transAxes,
        ha="center", va="top", fontsize=8, color="#2c2c2a")
ax.set_xticks([1, 2])
ax.set_xticklabels([f"Without obesity\n(BMI < 25, n={len(d[0])})",
                    f"With obesity\n(BMI \u2265 30, n={len(d[1])})"],
                   fontsize=7.4)
ax.set_ylim(min(50, d[0].min() - 8), max(d[1].max() + 14, 125))
ax.set_ylabel("UFM1 (per 10k UMI, pseudobulk)")
ax.set_title("c  Mature adipocytes, donor level", fontsize=9.5, loc="left")

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "Fig2_singlecell.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig2_singlecell.pdf"), bbox_inches="tight")
plt.close(fig)

print("panel b log2FC / p:")
print(u[["log2FC", "p", "n_obese", "n_lean"]].to_string())
print(f"\npanel c  fat cell UFM1: Lean n={len(d[0])}, Obese n={len(d[1])}, p={p:.4f}")
print("saved Fig2_singlecell.png / .pdf")
