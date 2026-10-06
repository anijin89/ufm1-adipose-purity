#!/usr/bin/env python3
"""GSE176171 单核 RNA-seq: UFMylation 轴的细胞类型定位与肥胖分层分析
采用伪 bulk (per donor x cell type) 策略避免伪重复。
"""
import gzip
import numpy as np
import pandas as pd
from scipy import stats

# ---------- 1. barcodes: cell_idx -> cell_id ----------
with gzip.open("data/GSE176171_barcodes.tsv.gz", "rt") as fh:
    barcodes = [l.strip() for l in fh if l.strip()]
print("barcodes:", len(barcodes))

# ---------- 2. metadata ----------
md = pd.read_csv("data/GSE176171_cell_metadata.tsv.gz", sep="\t", low_memory=False)
md = md[md["cell_id"].isin(set(barcodes))].copy()
md = md.set_index("cell_id").reindex(barcodes)
print("匹配细胞:", md.shape[0], "| cell_type 非缺失:", md["cell_type__ontology_label"].notna().sum())

# ---------- 3. 稀疏表达 ----------
ex = pd.read_csv("data/sc_extract.tsv", sep="\t", header=None,
                 names=["gene", "cell_idx", "count"])
genes = sorted(ex["gene"].unique())
gmap = {g: i for i, g in enumerate(genes)}
cmap = {c: i for i, c in enumerate(barcodes)}

M = np.zeros((len(genes), len(barcodes)), dtype=np.float32)
gi = ex["gene"].map(gmap).values
ci = (ex["cell_idx"].values - 1)
M[gi, ci] = ex["count"].values
print("稀疏矩阵:", M.shape, "| 非零:", (M > 0).sum())

# ---------- 4. 每细胞总 UMI 归一化 ----------
tot = md["n_umis"].astype(float).values.copy()
tot[tot <= 0] = np.nan
CPM = M / tot * 1e4   # per 10k

celltypes = md["cell_type__ontology_label"].fillna("unknown").values
bmi = pd.to_numeric(md["bmi"], errors="coerce").values
donor = md["donor_id"].astype(str).values
depot = md["depot__ontology_label"].fillna("unknown").values

# ---------- 5. 细胞类型表达谱 ----------
CORE = ["UFM1", "UFC1", "UBA5", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]
MARK = ["PLIN1", "FASN", "PPARG", "ADIPOQ", "LEP", "PDGFRA", "CD68", "PECAM1",
        "CD3E", "HSPA5", "ATF6", "DDIT3", "EIF2AK3"]
rows = []
cts = pd.Series(celltypes).value_counts()
for ct in cts.index:
    if cts[ct] < 200:
        continue
    sel = celltypes == ct
    r = {"cell_type": ct, "n_cells": int(sel.sum())}
    for g in CORE + MARK:
        if g in gmap:
            v = CPM[gmap[g], sel]
            r[g] = float(np.nanmean(v))
            r[f"{g}_pct"] = float(np.mean(M[gmap[g], sel] > 0) * 100)
    rows.append(r)
CT = pd.DataFrame(rows)
CT.to_csv("results/sc_celltype_expression.csv", index=False)
print("\n=== UFMylation 轴在各细胞类型的平均表达 (per 10k UMI) ===")
print(CT[["cell_type", "n_cells"] + [g for g in CORE if g in CT.columns]].round(3).to_string(index=False))

# ---------- 6. 伪 bulk: donor x cell type ----------
KEY_CT = ["fat cell", "preadipocyte", "macrophage", "endothelial cell", "T cell"]
rows = []
for ct in KEY_CT:
    sel_ct = celltypes == ct
    for d in pd.unique(donor[sel_ct]):
        sel = sel_ct & (donor == d)
        if sel.sum() < 30:
            continue
        bm = np.nanmedian(bmi[sel])
        if np.isnan(bm):
            continue
        r = {"cell_type": ct, "donor": d, "n_cells": int(sel.sum()), "bmi": bm,
             "depot": pd.Series(depot[sel]).mode().iloc[0] if sel.sum() else "NA"}
        raw_sum = M[:, sel].sum(axis=1)
        tot_d = raw_sum.sum()
        for g in CORE + MARK:
            if g in gmap:
                r[g] = raw_sum[gmap[g]] / tot_d * 1e4 if tot_d > 0 else np.nan
        rows.append(r)
PB = pd.DataFrame(rows)
PB.to_csv("results/sc_pseudobulk_donor.csv", index=False)
print("\n伪 bulk 样本数 (donor x cell type):", len(PB))
print(PB.groupby("cell_type").size().to_string())

# ---------- 7. 肥胖 vs 瘦 (donor 层面) ----------
PB["ob_grp"] = np.where(PB["bmi"] >= 30, "Obese", np.where(PB["bmi"] < 25, "Lean", "OW"))
print("\n=== 肥胖(BMI>=30) vs 瘦(BMI<25): donor 层面 Mann-Whitney ===")
out = []
for ct in KEY_CT:
    sub = PB[(PB["cell_type"] == ct) & (PB["ob_grp"].isin(["Obese", "Lean"]))]
    a = sub[sub["ob_grp"] == "Obese"]
    b = sub[sub["ob_grp"] == "Lean"]
    if len(a) < 3 or len(b) < 3:
        print(f"\n[{ct}] 样本不足 (Obese={len(a)}, Lean={len(b)})")
        continue
    print(f"\n[{ct}]  Obese donors n={len(a)} (BMI {a['bmi'].min():.0f}-{a['bmi'].max():.0f}) | "
          f"Lean n={len(b)} (BMI {b['bmi'].min():.0f}-{b['bmi'].max():.0f})")
    for g in CORE + MARK:
        if g not in sub.columns:
            continue
        x, y = a[g].dropna(), b[g].dropna()
        if len(x) < 3 or len(y) < 3:
            continue
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        fc = np.log2((x.mean() + .01) / (y.mean() + .01))
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
        print(f"   {g:<8} Obese={x.mean():7.3f}  Lean={y.mean():7.3f}  log2FC={fc:+.3f}  p={p:.4f} {star}")
        out.append(dict(cell_type=ct, gene=g, mean_obese=x.mean(), mean_lean=y.mean(),
                        log2FC=fc, p=p))
pd.DataFrame(out).to_csv("results/sc_obese_vs_lean.csv", index=False)

# ---------- 8. UFM1 与 BMI 的 donor 层面相关 ----------
print("\n=== donor 层面 UFM1 vs BMI 相关 (Spearman) ===")
for ct in KEY_CT:
    sub = PB[PB["cell_type"] == ct].dropna(subset=["UFM1", "bmi"])
    if len(sub) < 6:
        continue
    r, p = stats.spearmanr(sub["bmi"], sub["UFM1"])
    print(f"  {ct:<20} n={len(sub):3d}  rho={r:+.3f}  p={p:.4f}")
