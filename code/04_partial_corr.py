#!/usr/bin/env python
"""细胞组成校正 + 非线性检验：区分真信号与脂肪细胞稀释效应"""
import os, numpy as np, pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results")
CLIN = os.path.join(BASE, "data", "GSE70353_clinical.csv")
EXPR = os.path.join(OUT, "GSE70353_UFMylation_expr.csv")

CORE = ["UFM1", "UFC1", "UBA5", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]
COMP = {"CD68": "CD68", "PLIN1": "PLIN1", "ADIPOQ": "ADIPOQ"}

expr = pd.read_csv(EXPR, index_col=0)
clin = pd.read_csv(CLIN, index_col=0)
for c in ["bmi", "matsuda", "homair", "totfa", "age"]:
    clin[c] = pd.to_numeric(clin[c], errors="coerce")

# 组成代理：取表达最高的探针
comp_vec = {}
for g in COMP:
    ps = [c for c in expr.columns if c.startswith("117") ]
    cand = [c for c in expr.columns if c in (
        "11755675_s_at", "11759285_x_at")]
    # 直接用已抽取矩阵中该基因的探针（由 03 脚本写入，探针名即列名）
comp_probe = {"CD68": None, "PLIN1": None, "ADIPOQ": None}
# 从注释重新定位组成基因探针
ann = pd.read_csv(os.path.join(BASE, "data", "GPL13667_probe2gene.tsv"), sep="\t", dtype=str)
for g in COMP:
    ps = [p for p, s in zip(ann["probe"], ann["symbol"]) if s == g and p in expr.columns]
    if ps:
        comp_probe[g] = max(ps, key=lambda p: expr[p].mean())
print("细胞组成代理探针:", comp_probe)

gene_probe = {"UFL1": "11743510_at"}  # 别名匹配得到，脚本 03 中信号最强
for g in CORE:
    if g in gene_probe:
        continue
    ps = [p for p, s in zip(ann["probe"], ann["symbol"]) if s == g and p in expr.columns]
    if ps:
        gene_probe[g] = max(ps, key=lambda p: expr[p].mean())


def partial_spearman(x, y, covs):
    """秩变换后线性去协变量，再相关"""
    m = np.ones(len(x), bool)
    for v in [x, y] + list(covs):
        m &= np.isfinite(v)
    if m.sum() < 100:
        return np.nan, np.nan, m.sum()
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in covs])
    rx = stats.rankdata(x[m]); ry = stats.rankdata(y[m])
    bx = np.linalg.lstsq(X, rx, rcond=None)[0]
    by = np.linalg.lstsq(X, ry, rcond=None)[0]
    ex = rx - X @ bx; ey = ry - X @ by
    r, p = stats.pearsonr(ex, ey)
    return r, p, m.sum()


covs = [expr[comp_probe[g]].values for g in comp_probe if comp_probe[g]]
bmi = clin["bmi"].reindex(expr.index).values

print("\n" + "=" * 88)
print("【细胞组成校正】校正 CD68(巨噬细胞) + PLIN1(脂肪细胞) + ADIPOQ 前后，基因 vs BMI")
print("=" * 88)
print(f"{'gene':9s} {'raw_r':>8s} {'raw_p':>11s} {'adj_r':>8s} {'adj_p':>11s}  {'判定':<s}")
rows = []
for g in CORE:
    if g not in gene_probe:
        continue
    x = expr[gene_probe[g]].values
    r0, p0, n = partial_spearman(x, bmi, [np.zeros(len(x))])
    r1, p1, n = partial_spearman(x, bmi, covs)
    tag = "信号稳健" if abs(r1) > 0.1 and p1 < 0.01 else ("被稀释效应解释" if abs(r1) < abs(r0) * 0.5 else "无信号")
    print(f"{g:9s} {r0:+8.3f} {p0:11.2e} {r1:+8.3f} {p1:11.2e}  {tag}")
    rows.append([g, r0, p0, r1, p1, n, tag])

print("\n" + "=" * 88)
print("【非线性检验】BMI 五分位分组均值（log2）")
print("=" * 88)
q = pd.qcut(clin["bmi"].reindex(expr.index), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
print(f"{'gene':9s} " + " ".join(f"{l:>7s}" for l in ["Q1", "Q2", "Q3", "Q4", "Q5"]) + f" {'Q5-Q1':>8s} {'p_trend':>10s}")
for g in CORE:
    if g not in gene_probe:
        continue
    v = expr[gene_probe[g]]
    means = v.groupby(q, observed=True).mean()
    ord_idx = clin["bmi"].reindex(expr.index).rank()
    r, p = stats.spearmanr(ord_idx, v, nan_policy="omit")
    print(f"{g:9s} " + " ".join(f"{means[l]:7.3f}" for l in ["Q1", "Q2", "Q3", "Q4", "Q5"])
          + f" {means['Q5']-means['Q1']:+8.3f} {p:10.2e}")

pd.DataFrame(rows, columns=["gene", "raw_r", "raw_p", "adj_r", "adj_p", "n", "note"]).to_csv(
    os.path.join(OUT, "GSE70353_partial_corr.csv"), index=False)
print("\n已保存:", OUT)
