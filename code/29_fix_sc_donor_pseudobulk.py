#!/usr/bin/env python3
"""GSE176171 单核：修正供体身份后重做伪 bulk。

问题：原分析（10b / 14 号脚本）用 metadata 的 donor_id 作为供体键，但该字段
并非真实供体。在 Hs10X 矩阵的 137,684 个细胞里：
  donor_id = "TP01"  实际混合了 5 个真实供体（01 / 253 / 254 / 255 / 256）
  donor_id = "EPI253/254/255/256" 是上述 4 个供体的 OAT 样本，另给了标签
即供体 253–256 被重复计入（一次在 TP01 内、一次单独），而供体 01 的 BMI
（42.7，属肥胖）被写成了 24.27（供体 255 的 BMI），被错分到"瘦"组。

正确的供体键可由 biosample_id 还原：Hs_{OAT|SAT}_{donor}-{batch}。

本脚本按真实供体重算伪 bulk 与肥胖/瘦比较，并输出新旧对照。
"""
import gzip
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

# ---------- 1. barcodes ----------
with gzip.open(os.path.join(DATA, "GSE176171_barcodes.tsv.gz"), "rt") as fh:
    barcodes = [l.split("\t")[0].strip() for l in fh if l.strip()]

# ---------- 2. metadata ----------
md = pd.read_csv(os.path.join(DATA, "GSE176171_cell_metadata.tsv.gz"),
                 sep="\t", low_memory=False)
md = md[md["cell_id"].isin(set(barcodes))].copy()
md = md.set_index("cell_id").reindex(barcodes)
print("匹配细胞:", md.shape[0])

PAT = re.compile(r"^Hs_(?:OAT|SAT)_([0-9]+)-([0-9]+)$")


def donor_from_biosample(s):
    m = PAT.match(str(s))
    if not m:
        return None
    return "D" + m.group(1).lstrip("0").rjust(2, "0")


md["true_donor"] = md["biosample_id"].map(donor_from_biosample)
print("未能解析 biosample:", md["true_donor"].isna().sum())

# ---------- 3. 供体映射对照 ----------
mp = (md.dropna(subset=["true_donor"])
        .groupby(["true_donor", "donor_id", "biosample_id", "bmi"])
        .size().rename("cells").reset_index())
mp.to_csv(os.path.join(RES, "sc_donor_mapping_corrected.csv"), index=False)
print("\n=== 真实供体的细胞数 / BMI / 原 donor_id 标签 ===")
summ = mp.groupby("true_donor").agg(
    cells=("cells", "sum"),
    bmi=("bmi", "first"),
    old_labels=("donor_id", lambda s: "/".join(sorted(set(s)))),
    n_biosample=("biosample_id", "nunique"),
).reset_index().sort_values("bmi")
print(summ.to_string(index=False))
print("\n真实供体数:", mp.true_donor.nunique(),
      "| 原 donor_id 标签数:", md.donor_id.nunique())
dup = md.groupby("donor_id")["true_donor"].nunique()
print("原 donor_id 中一个标签含多供体:", dict(dup[dup > 1]))
multi = md.groupby("true_donor")["donor_id"].nunique()
print("真实供体被拆到多个标签:", dict(multi[multi > 1]))

# ---------- 4. 表达矩阵 ----------
ex = pd.read_csv(os.path.join(DATA, "sc_extract.tsv"), sep="\t", header=None,
                 names=["gene", "cell_idx", "count"])
genes = sorted(ex["gene"].unique())
gmap = {g: i for i, g in enumerate(genes)}
M = np.zeros((len(genes), len(barcodes)), dtype=np.float32)
M[ex["gene"].map(gmap).values, ex["cell_idx"].values - 1] = ex["count"].values
print("\n稀疏矩阵:", M.shape, "| 非零:", int((M > 0).sum()))

celltypes = md["cell_type__ontology_label"].fillna("unknown").values
bmi_true = pd.to_numeric(md["bmi"], errors="coerce").values
depot = md["depot__ontology_label"].fillna("unknown").values
donor_true = md["true_donor"].values
donor_old = md["donor_id"].astype(str).values

CORE = ["UFM1", "UFC1", "UBA5", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]
MARK = ["PLIN1", "FASN", "PPARG", "ADIPOQ", "LEP", "PDGFRA", "CD68", "PECAM1",
        "CD3E", "HSPA5", "ATF6", "DDIT3", "EIF2AK3"]
KEY_CT = ["fat cell", "preadipocyte", "macrophage", "endothelial cell", "T cell"]


def pseudobulk(donor_arr, tag):
    rows = []
    for ct in KEY_CT:
        sel_ct = celltypes == ct
        for d in pd.unique(donor_arr[sel_ct]):
            if d is None or (isinstance(d, float) and np.isnan(d)):
                continue
            sel = sel_ct & (donor_arr == d)
            if sel.sum() < 30:
                continue
            bm = np.nanmedian(bmi_true[sel])
            if np.isnan(bm):
                continue
            raw = M[:, sel].sum(axis=1)
            tot = raw.sum()
            r = {"cell_type": ct, "donor": d, "n_cells": int(sel.sum()), "bmi": bm,
                 "depot": pd.Series(depot[sel]).mode().iloc[0]}
            for g in CORE + MARK:
                if g in gmap:
                    r[g] = raw[gmap[g]] / tot * 1e4 if tot > 0 else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    df["grp"] = np.where(df.bmi >= 30, "Obese", np.where(df.bmi < 25, "Lean", "OW"))
    print(f"\n[{tag}] 伪 bulk 样本数: {len(df)}")
    print(df.groupby("cell_type").size().to_string())
    print(f"[{tag}] 分组: {df[df.cell_type=='fat cell'].grp.value_counts().to_dict()}")
    return df


PB_old = pseudobulk(donor_old, "原 donor_id")
PB_new = pseudobulk(donor_true, "真实供体")
PB_new.to_csv(os.path.join(RES, "sc_pseudobulk_donor_corrected.csv"), index=False)


def ob_vs_lean(PB, tag):
    out = []
    for ct in KEY_CT:
        sub = PB[(PB.cell_type == ct) & (PB.grp.isin(["Obese", "Lean"]))]
        a = sub[sub.grp == "Obese"]
        b = sub[sub.grp == "Lean"]
        if len(a) < 3 or len(b) < 3:
            print(f"[{tag}] {ct}: 样本不足 Obese={len(a)} Lean={len(b)}")
            continue
        for g in CORE + MARK:
            if g not in sub.columns:
                continue
            x, y = a[g].dropna(), b[g].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            out.append(dict(cell_type=ct, gene=g, mean_obese=x.mean(), mean_lean=y.mean(),
                            log2FC=np.log2((x.mean() + .01) / (y.mean() + .01)), p=p,
                            n_obese=len(x), n_lean=len(y)))
    return pd.DataFrame(out)


old = ob_vs_lean(PB_old, "原")
new = ob_vs_lean(PB_new, "修正")
old.to_csv(os.path.join(RES, "sc_obese_vs_lean.csv"), index=False)
new.to_csv(os.path.join(RES, "sc_obese_vs_lean_corrected.csv"), index=False)

cmp = old.merge(new, on=["cell_type", "gene"], suffixes=("_old", "_new"))
cmp["d_log2FC"] = cmp.log2FC_new - cmp.log2FC_old
print("\n=== UFM1 (fat cell) 新旧对照 ===")
print(cmp[(cmp.cell_type == "fat cell") & (cmp.gene == "UFM1")].to_string(index=False))
print("\n=== 各细胞类型 UFM1 新结果 ===")
u = new[new.gene == "UFM1"].copy()
u["star"] = np.where(u.p < .001, "***", np.where(u.p < .01, "**",
                    np.where(u.p < .05, "*", "ns")))
print(u.to_string(index=False))
BH = (u.p.rank() / len(u) * len(u)).clip(upper=1)
q = (u.sort_values("p").p.values * len(u) / (np.arange(len(u)) + 1))
q = np.minimum.accumulate(q[::-1])[::-1]
u2 = u.sort_values("p").assign(q=q)
print("\nBH 校正（5 个细胞类型）:")
print(u2[["cell_type", "log2FC", "p", "q"]].to_string(index=False))
