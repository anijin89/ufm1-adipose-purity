#!/usr/bin/env python3
"""GSE59034 RYGB 干预: UFMylation 轴负荷 (= UFM1 - machine) 的动态变化 + 生成全部图表"""
import gzip
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
OUT = "figures"
import os
os.makedirs(OUT, exist_ok=True)

# ---------- 读 GSE59034 表达 ----------
E = pd.read_csv("results/GSE59034_UFM_expr.csv", index_col=0)
titles, gsms = None, None
with gzip.open("data/GSE59034_series_matrix.txt.gz", "rt", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if line.startswith("!Sample_title"):
            titles = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_geo_accession"):
            gsms = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!series_matrix_table_begin"):
            break
grp, pair = [], []
for t in titles:
    if "before bariatric" in t:
        grp.append("Obese_pre")
    elif "after bariatric" in t:
        grp.append("Post_RYGB")
    else:
        grp.append("Lean")
    pair.append(t.split("rep")[-1].strip())
meta = pd.DataFrame({"group": grp, "pair": pair}, index=gsms)
E = E.join(meta)
print(E.groupby("group").size().to_string())

CORE = ["UFM1", "UFC1", "UBA5", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]
MACHINE = ["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"]


def fmt_p(p):
    """AE&M wording rule: the complete word "p-value" and two decimals; a value
    that would round to 0.00 is reported as "p-value < 0.01"."""
    return "p-value < 0.01" if p < 0.005 else "p-value = {:.2f}".format(p)

# z 标准化 (基于全部 48 样本)
Z = E[CORE].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
E["UFM1_z"] = Z["UFM1"]
E["machine_z"] = Z[MACHINE].mean(axis=1)
E["dysreg"] = Z["UFM1"] - E["machine_z"]
E.to_csv("results/GSE59034_scores.csv")

ORDER = ["Lean", "Obese_pre", "Post_RYGB"]
# AE&M people-first wording: no "lean"/"obese" used as a label for people.
LABEL = {"Lean": "Control group\nwithout obesity",
         "Obese_pre": "Pre-surgical\n(with obesity)",
         "Post_RYGB": "Post-surgical\n(after weight loss)"}

# ---------- Fig 5: RYGB 动态 ----------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
for ax, col, ttl in zip(axes, ["UFM1_z", "machine_z", "dysreg"],
                        ["UFM1 (modified protein)", "Conjugation machinery", "UFMylation dysregulation index"]):
    data = [E.loc[E["group"] == g, col].dropna().values for g in ORDER]
    bp = ax.boxplot(data, widths=.55, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor="#cfe3f7", edgecolor="#2c5f8d"),
                    medianprops=dict(color="#c0392b", lw=1.6), whiskerprops=dict(color="#2c5f8d"))
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        xs = rng.normal(i + 1, .06, len(d))
        ax.scatter(xs, d, s=14, color="#2c5f8d", alpha=.55, zorder=3)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels([LABEL[g] for g in ORDER], fontsize=7.2)
    ax.set_ylabel("z-score" if "machine" not in col else "mean z-score")
    ax.set_title(ttl, fontsize=9.5)
    # 配对连线级别显著性
    def pv(a, b):
        x, y = [], []
        for p in E["pair"].unique():
            ga = E[(E["pair"] == p) & (E["group"] == a)][col]
            gb = E[(E["pair"] == p) & (E["group"] == b)][col]
            if len(ga) and len(gb):
                x.append(ga.iloc[0]); y.append(gb.iloc[0])
        return stats.ttest_rel(x, y)[1], np.mean(x) - np.mean(y)
    p1, d1 = pv("Obese_pre", "Lean")
    p2, d2 = pv("Post_RYGB", "Obese_pre")
    def st(p):
        # AE&M wording rule: complete word "p-value", two decimals; a value that
        # would round to 0.00 is written as "p-value < 0.01". No asterisk markers.
        return "p-value\n< 0.01" if p < 0.005 else "p-value\n= {:.2f}".format(p)
    ymax = max([d.max() for d in data]); ymin = min([d.min() for d in data])
    h = (ymax - ymin) * .07
    tops = []
    for (i, j, p, dd) in [(1, 2, p1, d1), (2, 3, p2, d2)]:
        yy = max(np.mean(data[i - 1]) + np.std(data[i - 1]), np.mean(data[j - 1]) + np.std(data[j - 1])) + h
        ax.plot([i, i, j, j], [yy, yy + h * .25, yy + h * .25, yy], color="#555", lw=.9)
        ax.text((i + j) / 2, yy + h * .30, st(p), ha="center", va="bottom",
                fontsize=6.4, color="#333", linespacing=1.2)
        tops.append(yy + h * .25)
    # head-room so the two-line p-value labels clear the panel title
    ax.set_ylim(top=max(tops) + h * 1.7)
    ax.axhline(0, color="#bbb", lw=.7, ls="--")
plt.tight_layout()
plt.savefig(f"{OUT}/Fig5_RYGB_dynamics.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}/Fig5_RYGB_dynamics.pdf", bbox_inches="tight")
plt.close()

print("\n=== RYGB 复合评分动态 (配对 t 检验) ===")
rows = []
for col in ["UFM1_z", "machine_z", "dysreg"]:
    for a, b in [("Obese_pre", "Lean"), ("Post_RYGB", "Obese_pre"), ("Post_RYGB", "Lean")]:
        x, y = [], []
        for p in E["pair"].unique():
            ga = E[(E["pair"] == p) & (E["group"] == a)][col]
            gb = E[(E["pair"] == p) & (E["group"] == b)][col]
            if len(ga) and len(gb):
                x.append(ga.iloc[0]); y.append(gb.iloc[0])
        x, y = np.array(x), np.array(y)
        d = x - y
        t, p = stats.ttest_rel(x, y)
        rows.append(dict(score=col, comparison=f"{a} vs {b}", n=len(x),
                         mean_a=x.mean(), mean_b=y.mean(), diff=d.mean(),
                         cohen_d=d.mean() / d.std(ddof=1), t=t, p=p,
                         wilcoxon=stats.wilcoxon(d)[1]))
INT = pd.DataFrame(rows)
INT.to_csv("results/GSE59034_intervention_scores.csv", index=False)
print(INT.round(4).to_string(index=False))

# ---------- Fig 2: GSE70353 相关分析 ----------
corr = pd.read_csv("results/GSE70353_UFM1_marker_corr.csv")
sc = pd.read_csv("results/GSE70353_UFM_scores.csv", index_col=0)
clinA = pd.read_csv("results/GSE70353_clinical_aligned.csv", index_col=0)
dF = sc.join(clinA, how="inner")
for c in dF.columns:
    dF[c] = pd.to_numeric(dF[c], errors="coerce")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
# (a) 标记基因相关森林图
sel = corr[corr["category"].isin(["adipocyte", "macrophage", "ASPC", "endothelial", "T_cell", "ER_stress"])].copy()
sel = sel.reindex(sel["r_UFM1"].sort_values().index)
colors = {"adipocyte": "#d97706", "macrophage": "#059669", "ASPC": "#2563eb",
          "endothelial": "#7c3aed", "T_cell": "#dc2626", "ER_stress": "#db2777"}
ax = axes[0]
ax.barh(range(len(sel)), sel["r_UFM1"], color=[colors[c] for c in sel["category"]], height=.72)
ax.set_yticks(range(len(sel)))
ax.set_yticklabels(sel["gene"], fontsize=7.5)
ax.axvline(0, color="#333", lw=.8)
ax.set_xlabel("Pearson r with UFM1")
ax.set_title("a  UFM1 vs cell-type / stress markers", fontsize=9.5, loc="left")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
ax.legend(handles, colors.keys(), fontsize=6.5, loc="lower right", frameon=False)

# (b) machine score vs HOMA-IR
ax = axes[1]
sub = dF.dropna(subset=["machine_z", "homair"])
s = ax.scatter(sub["machine_z"], sub["homair"], s=8, alpha=.35, color="#2563eb", edgecolors="none")
r, p = stats.pearsonr(sub["machine_z"], sub["homair"])
b1, b0 = np.polyfit(sub["machine_z"], sub["homair"], 1)
xs = np.linspace(sub["machine_z"].min(), sub["machine_z"].max(), 50)
ax.plot(xs, b0 + b1 * xs, color="#c0392b", lw=1.6)
ax.set_xlabel("Conjugation machinery z-score"); ax.set_ylabel("HOMA-IR")
ax.set_title(f"b  Machinery vs HOMA-IR\nr = {r:.2f}, {fmt_p(p)}", fontsize=9.5, loc="left")

# (c) dysreg vs waist
ax = axes[2]
sub = dF.dropna(subset=["dysreg", "waist_circumference"])
ax.scatter(sub["dysreg"], sub["waist_circumference"], s=8, alpha=.35, color="#db2777", edgecolors="none")
r, p = stats.pearsonr(sub["dysreg"], sub["waist_circumference"])
b1, b0 = np.polyfit(sub["dysreg"], sub["waist_circumference"], 1)
xs = np.linspace(sub["dysreg"].min(), sub["dysreg"].max(), 50)
ax.plot(xs, b0 + b1 * xs, color="#c0392b", lw=1.6)
ax.set_xlabel("UFMylation dysregulation index"); ax.set_ylabel("Waist circumference (cm)")
ax.set_title(f"c  Dysregulation vs waist\nr = {r:.2f}, {fmt_p(p)}", fontsize=9.5, loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/Fig2_bulk_associations.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}/Fig2_bulk_associations.pdf", bbox_inches="tight")
plt.close()

# ---------- [已废弃] 单细胞图 ----------
# 警告：本段使用 results/sc_pseudobulk_donor.csv，该文件的供体键取自 GEO 元数据的
# donor_id 字段，而该字段并非真实供体（详见 scripts/29 与 30 的说明）。据此画出的
# "Adipocyte-intrinsic UFM1, p=0.008" 结论已被撤回。
# 稿件中的 Figure 2 现在由 scripts/30_fig2_singlecell_update.py 生成（修正供体、
# 描述性框架）。本段仅作历史留存，输出文件名已加 legacy_ 前缀以免与正文图混淆。
CT = pd.read_csv("results/sc_celltype_expression.csv")
CTs = CT.sort_values("UFM1", ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
ax = axes[0]
top = CTs.tail(12)
ax.barh(range(len(top)), top["UFM1"], color="#7fb3d5", edgecolor="#2c5f8d", height=.7)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top["cell_type"], fontsize=8)
ax.set_xlabel("UFM1 expression (per 10k UMI)")
ax.set_title("a  UFM1 across adipose cell types", fontsize=9.5, loc="left")

ax = axes[1]
PB = pd.read_csv("results/sc_pseudobulk_donor.csv")
fc = PB[PB["cell_type"] == "fat cell"].copy()
fc["ob"] = np.where(fc["bmi"] >= 30, "Obese", np.where(fc["bmi"] < 25, "Lean", "OW"))
d = [fc.loc[fc["ob"] == g, "UFM1"].values for g in ["Lean", "Obese"]]
bp = ax.boxplot(d, widths=.5, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="#f5cba7", edgecolor="#b9770e"),
                medianprops=dict(color="#c0392b", lw=1.6), whiskerprops=dict(color="#b9770e"))
rng = np.random.default_rng(1)
for i, dd in enumerate(d):
    ax.scatter(rng.normal(i + 1, .05, len(dd)), dd, s=30, color="#b9770e", alpha=.7, zorder=3)
u, p = stats.mannwhitneyu(d[1], d[0])
ax.set_xticks([1, 2]); ax.set_xticklabels(["Lean\n(BMI<25)", "Obese\n(BMI>=30)"], fontsize=8.5)
ax.set_ylabel("UFM1 (per 10k UMI, pseudobulk)")
ax.set_title(f"b  [LEGACY — WRONG DONOR KEY]  UFM1\ndonor-level, p={p:.3f}", fontsize=9.5, loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/legacy_singlecell_WRONG_DONOR.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}/legacy_singlecell_WRONG_DONOR.pdf", bbox_inches="tight")
plt.close()
print("\n图表已生成至 figures/")
