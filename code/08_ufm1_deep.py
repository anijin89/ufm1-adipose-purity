#!/usr/bin/env python3
"""GSE70353 深度分析:
1. 正确对齐临床表 (GSM <-> sample_id 经 Sample_title)
2. UFM1 与细胞类型标记基因的相关 -> 判断细胞组成混杂
3. 全局技术混杂检查 (PC1 / 样本总表达)
4. 控制混杂后的偏相关
5. UFM1 高低分组差异表达 -> 富集分析输入
"""
import gzip
import numpy as np
import pandas as pd
from scipy import stats

DATA = "data/GSE70353_series_matrix.txt.gz"
ANN = "data/GPL13667_probe2gene.tsv"

# ---------- 1. 表达矩阵 ----------
rows = []
with gzip.open(DATA, "rt", encoding="utf-8", errors="ignore") as fh:
    in_t = False
    for line in fh:
        if line.startswith("!series_matrix_table_begin"):
            in_t = True
            continue
        if line.startswith("!series_matrix_table_end"):
            break
        if in_t:
            rows.append(line)
header = [x.strip('"') for x in rows[0].rstrip("\n").split("\t")]
samples = header[1:]
probes, mat = [], []
for line in rows[1:]:
    f = line.rstrip("\n").split("\t")
    probes.append(f[0].strip('"'))
    mat.append(np.array(f[1:], dtype=np.float32))
X = pd.DataFrame(np.vstack(mat), index=probes, columns=samples)
del rows, mat

ann = pd.read_csv(ANN, sep="\t", dtype=str)
p2g = dict(zip(ann["probe"], ann["symbol"]))
X.index = [p2g.get(p, "") for p in X.index]
X = X[X.index != ""]
X = X.groupby(level=0).mean()
print("表达矩阵(基因级):", X.shape)

# ---------- 2. 临床对齐 ----------
titles = None
with gzip.open(DATA, "rt", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if line.startswith("!Sample_title"):
            titles = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            break
clin = pd.read_csv("data/GSE70353_clinical.csv", index_col=0)
clin.index = clin.index.astype(str)
t2gsm = dict(zip(titles, samples))
clin = clin.reindex([t for t in titles])
clin.index = samples                      # 换成 GSM，与表达矩阵列一致
for c in clin.columns:
    if c not in ("tissue", "gender"):
        clin[c] = pd.to_numeric(clin[c], errors="coerce")
print("临床对齐后:", clin.shape, "| BMI 非缺失:", clin["bmi"].notna().sum())
clin.to_csv("results/GSE70353_clinical_aligned.csv")

# ---------- 3. 标记基因相关 ----------
u = X.loc["UFM1"].astype(float)
bmi = clin["bmi"]

MARKERS = {
    "adipocyte": ["PPARG", "FASN", "ADIPOQ", "PLIN1", "LEP", "LPL", "SREBF1", "CEBPA"],
    "macrophage": ["CD68", "PTPRC", "ITGAM", "CD14", "CSF1R", "MRC1"],
    "ASPC": ["PDGFRA", "CD34", "THY1", "PDGFRB", "DCN", "LUM"],
    "endothelial": ["PECAM1", "VWF", "CDH5"],
    "T_cell": ["CD3E", "CD2", "IL7R"],
    "ER_stress": ["HSPA5", "DDIT3", "ATF4", "ERN1", "EIF2AK3", "XBP1", "ATF6", "SEL1L"],
    "proteasome": ["PSMA1", "PSMB1", "PSMD1"],
    "housekeeping": ["ACTB", "GAPDH", "B2M", "TUBB"],
}
print("\n=== UFM1 与标记基因的相关 (Pearson) ===")
res = []
for ct, gs in MARKERS.items():
    for g in gs:
        if g in X.index:
            v = X.loc[g].astype(float)
            ok = u.notna() & v.notna()
            r, p = stats.pearsonr(u[ok], v[ok])
            rb, pb = (stats.pearsonr(bmi[ok], v[ok]) if ok.sum() > 10 else (np.nan, np.nan))
            res.append(dict(category=ct, gene=g, r_UFM1=r, p_UFM1=p, r_BMI=rb, p_BMI=pb))
M = pd.DataFrame(res)
print(M.round(3).to_string(index=False))
M.to_csv("results/GSE70353_UFM1_marker_corr.csv", index=False)

# ---------- 4. 技术混杂: PC1 ----------
Xc = X.loc[:, X.columns]
Z = Xc.sub(Xc.mean(axis=1), axis=0)
Z = Z.loc[Z.var(axis=1) > 0]
# 用随机 5000 基因做 PCA (省时)
sub = Z.sample(n=min(5000, len(Z)), random_state=0)
U, S, Vt = np.linalg.svd(sub.values - sub.values.mean(axis=0), full_matrices=False)
PC1 = pd.Series(Vt[0], index=X.columns)
var_exp = (S ** 2 / (S ** 2).sum())[:5]
print("\n=== PCA (5000 随机基因) 解释方差 ===")
print(np.round(var_exp, 4))
for i in range(3):
    pc = pd.Series(Vt[i], index=X.columns)
    ru = stats.pearsonr(u, pc)[0]
    rb = stats.pearsonr(bmi.dropna(), pc[bmi.dropna().index])[0] if bmi.notna().sum() > 10 else np.nan
    print(f"PC{i+1}: r(UFM1)={ru:+.3f}  r(BMI)={rb:+.3f}")

# ---------- 5. UFM1 高低分组差异表达 ----------
grp = pd.qcut(u, 4, labels=["Q1", "Q2", "Q3", "Q4"])
hi = u[grp == "Q4"].index
lo = u[grp == "Q1"].index
print(f"\nUFM1 高分组 n={len(hi)}, 低分组 n={len(lo)}")


def welch(sub_df, a, b):
    A = sub_df[a].values.astype(float)
    B = sub_df[b].values.astype(float)
    mu = A.mean(axis=1) - B.mean(axis=1)
    va = A.var(axis=1, ddof=1) / A.shape[1]
    vb = B.var(axis=1, ddof=1) / B.shape[1]
    se = np.sqrt(va + vb)
    se[se == 0] = np.nan
    t = mu / se
    p = 2 * stats.t.sf(np.abs(t), A.shape[1] + B.shape[1] - 2)
    return mu, t, p


lfc, t, p = welch(X, hi, lo)
DE = pd.DataFrame({"log2FC_Q4vsQ1": lfc / np.log(2) * 0 + lfc, "t": t, "p": p}, index=X.index)
pv = DE["p"].values
ok = ~np.isnan(pv)
order = np.argsort(pv[ok])
q = np.empty(ok.sum())
ranked = pv[ok][order]
qq = ranked * ok.sum() / (np.arange(ok.sum()) + 1)
qq = np.minimum.accumulate(qq[::-1])[::-1]
tmp = np.empty(ok.sum()); tmp[order] = np.clip(qq, 0, 1); q = tmp
DE.loc[ok, "q"] = q
DE = DE.sort_values("t")
DE.to_csv("results/GSE70353_UFM1_Q4vsQ1_DE.csv")
print("\n=== UFM1 高 vs 低: top15 上调 ===")
print(DE.dropna().tail(15)[["log2FC_Q4vsQ1", "q"]].round(3).iloc[::-1].to_string())
print("\n=== UFM1 高 vs 低: top15 下调 ===")
print(DE.dropna().head(15)[["log2FC_Q4vsQ1", "q"]].round(3).to_string())

print("\n两组 BMI 比较: 高组 %.2f vs 低组 %.2f, p=%.3g" % (
    bmi[hi].mean(), bmi[lo].mean(), stats.ttest_ind(bmi[hi], bmi[lo], equal_var=False)[1]))
