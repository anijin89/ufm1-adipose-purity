#!/usr/bin/env python3
"""UFM1 与 GSE70353 全部代谢表型的关联 (未校正 / 校正年龄 / 校正年龄+BMI)
目的: 确定论文主线结局变量
"""
import numpy as np
import pandas as pd
from scipy import stats

X = pd.read_csv("results/GSE70353_UFMylation_core_expr.csv", index_col=0)
clin = pd.read_csv("results/GSE70353_clinical_aligned.csv", index_col=0)
df = X.join(clin, how="inner")
print("样本:", df.shape)

PHENO = {
    "bmi": "BMI", "whr": "腰臀比", "waist_circumference": "腰围", "hip_circumference": "臀围",
    "ffmass": "去脂体重", "totfa": "总脂肪量", "p_adipon": "脂联素",
    "matsuda": "Matsuda 胰岛素敏感指数", "p_crp": "CRP", "b_ghba1c": "HbA1c",
    "homair": "HOMA-IR", "p_ffa0": "游离脂肪酸", "p_gl0": "空腹血糖",
    "p_ins0": "空腹胰岛素", "p_il1ra": "IL-1Ra", "p_proi0": " proinsulin",
    "s_hdlc": "HDL-C", "s_ldlc": "LDL-C", "s_totalc": "总胆固醇", "s_tottg": "甘油三酯",
    "systbp": "收缩压", "diastbp": "舒张压", "gfr": "GFR", "age": "年龄",
}
GENES = ["UFM1", "UFC1", "UBA5", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]

def partial_corr(x, y, covs):
    """y ~ x + covs 的偏相关"""
    ok = x.notna() & y.notna()
    for c in covs:
        ok &= df[c].notna()
    if ok.sum() < 50:
        return np.nan, np.nan, 0
    Xd = np.column_stack([np.ones(ok.sum())] + [df.loc[ok, c].values.astype(float) for c in covs])
    for v in (x, y):
        pass
    def resid(v):
        b, *_ = np.linalg.lstsq(Xd, v[ok].values.astype(float), rcond=None)
        return v[ok].values.astype(float) - Xd @ b
    rx, ry = resid(x), resid(y)
    r, p = stats.pearsonr(rx, ry)
    return r, p, int(ok.sum())

rows = []
for g in GENES:
    if g not in df.columns:
        continue
    x = pd.to_numeric(df[g], errors="coerce")
    for ph, lab in PHENO.items():
        if ph not in df.columns:
            continue
        y = pd.to_numeric(df[ph], errors="coerce")
        r0, p0, n0 = partial_corr(x, y, [])
        r1, p1, n1 = partial_corr(x, y, ["age"])
        r2, p2, n2 = partial_corr(x, y, ["age", "bmi"])
        rows.append(dict(gene=g, phenotype=ph, label=lab, n=n0,
                         r=r0, p=p0, r_adj_age=r1, p_adj_age=p1,
                         r_adj_age_bmi=r2, p_adj_age_bmi=p2))
R = pd.DataFrame(rows)
R.to_csv("results/GSE70353_UFM1_phenotype_assoc.csv", index=False)

def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

print("\n" + "=" * 100)
print("UFM1 与代谢表型的关联")
print("=" * 100)
print(f"{'表型':<26}{'n':>5}{'r':>9}{'p':>11}  {'r(校正年龄)':>12}{'p':>11}  {'r(校正年龄+BMI)':>15}{'p':>11}")
sub = R[R["gene"] == "UFM1"].copy()
sub["absr"] = sub["r_adj_age_bmi"].abs()
for _, r in sub.sort_values("absr", ascending=False).iterrows():
    print(f"{r['label']:<26}{int(r['n']):>5}{r['r']:>+9.3f}{r['p']:>11.2e}{star(r['p']):>3}"
          f"{r['r_adj_age']:>+12.3f}{r['p_adj_age']:>11.2e}{star(r['p_adj_age']):>3}"
          f"{r['r_adj_age_bmi']:>+15.3f}{r['p_adj_age_bmi']:>11.2e}{star(r['p_adj_age_bmi']):>3}")

print("\n" + "=" * 100)
print("其他 UFMylation 基因与 HOMA-IR / Matsuda 的关联 (校正年龄+BMI)")
print("=" * 100)
for g in GENES:
    if g == "UFM1":
        continue
    for ph in ["homair", "matsuda", "bmi", "p_adipon"]:
        r = R[(R["gene"] == g) & (R["phenotype"] == ph)]
        if len(r):
            rr = r.iloc[0]
            print(f"  {g:<10}{rr['label']:<26} r={rr['r_adj_age_bmi']:+.3f}  p={rr['p_adj_age_bmi']:.2e} {star(rr['p_adj_age_bmi'])}")
