#!/usr/bin/env python3
"""生成 Fig1 (研究设计) 与 Fig4 (通路富集)"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})

# ================= Fig 1: 研究设计 =================
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

def box(x, y, w, h, txt, fc, fs=8.2, bold=False, tc="#1a1a1a"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                       fc=fc, ec="#4a5568", lw=.9, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=3, linespacing=1.35)

def arrow(x1, y1, x2, y2, c="#2d3748"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color=c, lw=1.1, zorder=1))

# 队列层
box(1, 34, 30, 9.5, "Discovery cohort\nGSE70353 (METSIM)\nn = 770 men, subcutaneous AT\nBMI / HOMA-IR / Matsuda",
    "#bee3f8", bold=True)
box(35, 34, 30, 9.5, "Single-nucleus atlas\nGSE176171\n137,684 nuclei / visceral+subQ\n13 donors (BMI 20-49)",
    "#c6f6d5", bold=True)
box(69, 34, 30, 9.5, "Intervention cohort\nGSE59034 (RYGB)\n16 women, paired pre/post\nbariatric surgery",
    "#fed7d7", bold=True)

arrow(16, 34, 16, 29.5)
arrow(50, 34, 50, 29.5)
arrow(84, 34, 84, 29.5)

box(1, 21, 30, 8.5, "Step 1  Bulk association\n- partial correlation\n- cell-composition adjustment\n- phenotype scan (n=770)", "#e2e8f0")
box(35, 21, 30, 8.5, "Step 2  Cellular localization\n- UFM1 across 15 cell types\n- pseudobulk by donor\n- adipocyte-intrinsic test", "#e2e8f0")
box(69, 21, 30, 8.5, "Step 3  Causal direction\n- paired pre/post comparison\n- within-individual control\n- reversibility test", "#e2e8f0")

arrow(16, 21, 16, 17)
arrow(50, 21, 50, 17)
arrow(84, 21, 84, 17)

box(16, 8.5, 68, 8.5, "Step 4  Mechanistic & translational triangulation\nco-expression module → pathway enrichment (ER processing / secretion / ERAD)\n"
     "| machine score ↔ insulin resistance (BMI-adjusted)  |  incremental value over clinical variables (ΔAUC)",
     "#faf089", fs=8.5, bold=True)

arrow(50, 8.5, 50, 5.5)
box(20, 1.2, 60, 4.3, "UFMylation-axis remodelling as an adipocyte-intrinsic,\nmetabolically responsive ER-stress signature",
    "#667eea", fs=9, bold=True, tc="white")

ax.text(50, 44.2, "Study design and analytical workflow", ha="center",
        fontsize=12.5, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/Fig1_design.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}/Fig1_design.pdf", bbox_inches="tight")
plt.close()

# ================= Fig 4: 富集 =================
enr = pd.read_csv("results/enrichr_POS_UFM1_correlated.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
sel = (enr[enr["Gene_set"].isin(["MSigDB_Hallmark_2020", "KEGG_2021_Human"])]
       .sort_values("Adjusted P-value").head(14).iloc[::-1])
terms = [t.replace(" (GO:", "\n(GO:") if len(t) > 42 else t for t in sel["Term"]]
terms = [t[:52] for t in sel["Term"]]
colors = ["#2b6cb0" if "Hallmark" in g else "#c05621"
          for g in sel["Gene_set"]]
ax.barh(range(len(sel)), -np.log10(sel["Adjusted P-value"].values), color=colors, height=.72)
ax.set_yticks(range(len(sel))); ax.set_yticklabels(terms, fontsize=7.6)
ax.set_xlabel("-log10(adjusted p-value)")
ax.set_title("a  Pathways enriched in the UFM1-correlated module\n(n = 1,164 genes, r > 0.30, q < 0.05)",
             fontsize=9.2, loc="left")
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#2b6cb0", "#c05621"]]
ax.legend(h, ["MSigDB Hallmark", "KEGG 2021"], fontsize=7.5, loc="lower right", frameon=False)

# 面板 b: 关键基因的散点 (ER stress genes)
ax = axes[1]
corr = pd.read_csv("results/GSE70353_UFM1_marker_corr.csv")
er = corr[corr["category"] == "ER_stress"].sort_values("r_UFM1")
selc = corr[corr["category"] == "adipocyte"].sort_values("r_UFM1")
allm = pd.concat([er, selc])
allm = allm.sort_values("r_UFM1")
ax.barh(range(len(allm)), allm["r_UFM1"],
        color=["#dd6b20" if c == "ER_stress" else "#2b6cb0" for c in allm["category"]], height=.72)
ax.set_yticks(range(len(allm))); ax.set_yticklabels(allm["gene"], fontsize=8)
ax.axvline(0, color="#333", lw=.8)
ax.set_xlabel("Pearson r with UFM1 (n = 770)")
ax.set_title("b  UFM1 couples to ER-stress and lipid-droplet markers\nbut not to macrophage content",
             fontsize=9.2, loc="left")
hh = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#dd6b20", "#2b6cb0"]]
ax.legend(hh, ["ER stress / UPR", "Adipocyte"], fontsize=7.5, loc="lower right", frameon=False)

# 标注 CD68 作为对照
cd = corr[(corr["category"] == "macrophage") & (corr["gene"] == "CD68")]
if len(cd):
    # AE&M wording rule: complete word "p-value", two decimals.
    _pv = cd["p_UFM1"].iloc[0]
    _ptxt = "p-value < 0.01" if _pv < 0.005 else f"p-value = {_pv:.2f}"
    ax.text(0.02, -1.4, f"CD68 (macrophage): r = {cd['r_UFM1'].iloc[0]:+.2f}, {_ptxt}",
            fontsize=7.8, style="italic", color="#555")
plt.tight_layout()
plt.savefig(f"{OUT}/Fig4_enrichment.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}/Fig4_enrichment.pdf", bbox_inches="tight")
plt.close()
print("Fig1, Fig4 已生成:", sorted(os.listdir(OUT)))
