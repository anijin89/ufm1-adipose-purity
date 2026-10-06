#!/usr/bin/env python3
"""构建 UFMylation 轴复合评分并检验与代谢表型的关联
假说: "失稳态指数" = 修饰蛋白(UFM1)高 + 结合/去修饰机器(UFL1/DDRGK1/UFSP2/UBA5)低
"""
import numpy as np
import pandas as pd
from scipy import stats

X = pd.read_csv("results/GSE70353_UFMylation_core_expr.csv", index_col=0)
clin = pd.read_csv("results/GSE70353_clinical_aligned.csv", index_col=0)
df = X.join(clin, how="inner")
GENES = ["UFM1", "UFC1", "UBA5", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]
for g in GENES:
    df[g] = pd.to_numeric(df[g], errors="coerce")

# z 标准化
Z = df[GENES].apply(lambda s: (s - s.mean()) / s.std(ddof=0))

# 候选复合评分
df["UFM1_z"] = Z["UFM1"]
df["machine_z"] = Z[["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"]].mean(axis=1)   # 结合/去修饰机器
df["dysreg"] = Z["UFM1"] - df["machine_z"]                                   # 失稳态指数
df["axis_pc1"] = np.nan

# PCA: 取 UFMylation 轴 PC1
M = Z.dropna()
U, S, Vt = np.linalg.svd(M.values - M.values.mean(0), full_matrices=False)
pc1 = pd.Series(U[:, 0] * S[0], index=M.index)
# 定向: 使 PC1 与 UFM1 正相关
if stats.pearsonr(pc1, M["UFM1"])[0] < 0:
    pc1 = -pc1
df.loc[M.index, "axis_pc1"] = pc1
print("PC1 解释方差: %.3f" % (S[0] ** 2 / (S ** 2).sum()))
print("PC1 载荷:", dict(zip(GENES, np.round(Vt[0], 3))))

# 细胞组成校正协变量
COMP = ["PPARG", "FASN", "PLIN1", "ADIPOQ"]
# 从高变异矩阵取
hv = pd.read_csv("results/GSE70353_expr_hv8000.csv", index_col=0)
comp = hv[[c for c in COMP if c in hv.columns]]
df = df.join(comp, how="left")

PHENO = {"bmi": "BMI", "homair": "HOMA-IR", "matsuda": "Matsuda ISI", "p_adipon": "脂联素",
         "p_ins0": "空腹胰岛素", "p_gl0": "空腹血糖", "b_ghba1c": "HbA1c", "s_tottg": "甘油三酯",
         "s_hdlc": "HDL-C", "p_crp": "CRP", "whr": "腰臀比", "waist_circumference": "腰围",
         "p_ffa0": "游离脂肪酸"}
SCORES = ["UFM1_z", "machine_z", "dysreg", "axis_pc1"]

def assoc(x, y, covs):
    cols = list(dict.fromkeys([x, y] + covs))
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 50:
        return np.nan, np.nan, 0
    Xd = np.column_stack([np.ones(len(d))] + [d[c].values for c in covs]) if covs else np.ones((len(d), 1))
    def resid(v):
        b, *_ = np.linalg.lstsq(Xd, v, rcond=None)
        return v - Xd @ b
    r, p = stats.pearsonr(resid(d[x].values), resid(d[y].values))
    return r, p, len(d)

for sc in SCORES:
    print("\n" + "=" * 92)
    print(f"【{sc}】与代谢表型  (未校正 / 校正年龄+BMI / 校正年龄+BMI+细胞组成)")
    print("=" * 92)
    print(f"{'表型':<16}{'r0':>8}{'p0':>10}   {'r(age+bmi)':>11}{'p':>10}   {'r(+细胞组成)':>13}{'p':>10}")
    rows = []
    for ph, lab in PHENO.items():
        if ph not in df.columns:
            continue
        df[ph] = pd.to_numeric(df[ph], errors="coerce")
        r0, p0, n0 = assoc(sc, ph, [])
        r1, p1, n1 = assoc(sc, ph, ["age", "bmi"])
        cov2 = ["age", "bmi"] + [c for c in COMP if c in df.columns]
        r2, p2, n2 = assoc(sc, ph, cov2)
        rows.append(dict(score=sc, phenotype=ph, label=lab, n=n0, r0=r0, p0=p0,
                         r1=r1, p1=p1, r2=r2, p2=p2))
    RR = pd.DataFrame(rows).sort_values("p2")
    for _, r in RR.iterrows():
        def st(p):
            return "***" if p < 1e-3 else "**" if p < .01 else "*" if p < .05 else ""
        print(f"{r['label']:<16}{r['r0']:>+8.3f}{r['p0']:>10.2e}{st(r['p0']):>3}"
              f"{r['r1']:>+11.3f}{r['p1']:>10.2e}{st(r['p1']):>3}"
              f"{r['r2']:>+13.3f}{r['p2']:>10.2e}{st(r['p2']):>3}")
    RR.to_csv(f"results/GSE70353_assoc_{sc}.csv", index=False)

df[SCORES + GENES].to_csv("results/GSE70353_UFM_scores.csv")
print("\n评分已保存。")
