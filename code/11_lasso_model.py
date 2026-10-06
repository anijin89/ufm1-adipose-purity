#!/usr/bin/env python3
"""LASSO-logistic 建模: UFMylation 轴特征预测肥胖与胰岛素抵抗
训练: GSE70353 (n=770 男性皮下脂肪)
外部验证: GSE59034 (n=16 肥胖术前 vs n=16 从不肥胖, 女性, 跨平台)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats

CORE = ["UFM1", "UFC1", "UBA5", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]

# ---------- 训练队列 ----------
X = pd.read_csv("results/GSE70353_UFMylation_core_expr.csv", index_col=0)
clin = pd.read_csv("results/GSE70353_clinical_aligned.csv", index_col=0)
df = X.join(clin, how="inner")
print("训练队列:", df.shape)

df["obese"] = (df["bmi"] >= 30).astype(int)
print("肥胖 (BMI>=30):", df["obese"].sum(), "/ 非肥胖:", (1 - df["obese"]).sum())

# HOMA-IR 上四分位 vs 下四分位
h = pd.to_numeric(df["homair"], errors="coerce")
q1, q3 = h.quantile(.25), h.quantile(.75)
ir = pd.Series(np.nan, index=df.index)
ir[h <= q1] = 0
ir[h >= q3] = 1
df["ir"] = ir
print("胰岛素抵抗 (HOMA-IR Q4 vs Q1):", int((ir == 1).sum()), "vs", int((ir == 0).sum()))

def run_lasso(feats, y, label, X_df, extra=None):
    d = X_df[feats].copy()
    if extra:
        for c in extra:
            if c in X_df.columns:
                d[c] = X_df[c]
    d = d.apply(pd.to_numeric, errors="coerce")
    keep = y.notna()
    d, yy = d[keep], y[keep]
    d = d.dropna()
    yy = yy.reindex(d.index)
    if d.shape[0] < 50 or yy.nunique() < 2:
        print(f"[{label}] 样本不足")
        return None
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegressionCV(Cs=np.logspace(-3, 2, 30), cv=5,
                                                 penalty="l1", solver="liblinear",
                                                 scoring="roc_auc", max_iter=5000,
                                                 random_state=0))])
    # 交叉验证 AUC
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    aucs = cross_val_score(pipe, d.values, yy.values, cv=skf, scoring="roc_auc")
    pipe.fit(d.values, yy.values)
    coefs = pd.Series(pipe.named_steps["lr"].coef_[0], index=d.columns)
    C = pipe.named_steps["lr"].C_[0]
    # hold-out
    Xtr, Xte, ytr, yte = train_test_split(d.values, yy.values, test_size=.3,
                                          stratify=yy.values, random_state=0)
    p2 = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=5000))])
    p2.fit(Xtr, ytr)
    auc_te = roc_auc_score(yte, p2.predict_proba(Xte)[:, 1])
    print(f"\n[{label}] n={d.shape[0]}  特征={list(d.columns)}")
    print(f"  5-fold CV AUC = {aucs.mean():.3f} ± {aucs.std():.3f}")
    print(f"  Hold-out AUC  = {auc_te:.3f}   (lambda: C={C:.4g})")
    print("  系数:")
    for g, c in coefs.items():
        mark = "  <-- 保留" if abs(c) > 1e-6 else "  (剔除)"
        print(f"    {g:<10} {c:+.4f}{mark}")
    return dict(label=label, n=int(d.shape[0]), cv_auc=aucs.mean(), cv_sd=aucs.std(),
                holdout_auc=auc_te, C=C, coefs=coefs, model=p2, feats=list(d.columns))

print("\n" + "=" * 70)
res = {}
res["obese_core"] = run_lasso(CORE, df["obese"], "肥胖 ~ UFMylation轴", df)
res["obese_core_age"] = run_lasso(CORE, df["obese"], "肥胖 ~ UFMylation轴+年龄", df, extra=["age"])
res["ir_core"] = run_lasso(CORE, df["ir"], "胰岛素抵抗 ~ UFMylation轴", df)
res["ir_core_age"] = run_lasso(CORE, df["ir"], "胰岛素抵抗 ~ UFMylation轴+年龄", df, extra=["age"])

# 单基因 AUC 对照
print("\n=== 单基因 AUC (肥胖) ===")
y = df["obese"]
for g in CORE:
    v = pd.to_numeric(df[g], errors="coerce")
    ok = v.notna() & y.notna()
    auc = roc_auc_score(y[ok], v[ok])
    print(f"  {g:<10} AUC={auc:.3f}")

# ---------- 外部验证 GSE59034 ----------
print("\n" + "=" * 70)
E = pd.read_csv("results/GSE59034_UFM_expr.csv", index_col=0)
meta = pd.read_csv("results/GSE59034_UFM_expr.csv", index_col=0)
# 重建分组
import gzip
titles, gsms = None, None
with gzip.open("data/GSE59034_series_matrix.txt.gz", "rt", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if line.startswith("!Sample_title"):
            titles = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_geo_accession"):
            gsms = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!series_matrix_table_begin"):
            break
grp = {}
for t, g in zip(titles, gsms):
    if "before bariatric" in t:
        grp[g] = 1
    elif "never-obese" in t:
        grp[g] = 0
E["y"] = [grp.get(i, np.nan) for i in E.index]
val = E.dropna(subset=["y"])
print("外部验证集: 肥胖术前 n=%d, 从不肥胖 n=%d" % ((val["y"] == 1).sum(), (val["y"] == 0).sum()))

common = [g for g in CORE if g in val.columns]
print("共同特征:", common)
if res.get("obese_core") and len(common) >= 2:
    m = res["obese_core"]["model"]
    feats = res["obese_core"]["feats"]
    # 用训练队列的均值/标准差做标准化，再套用系数
    tr = df[feats].apply(pd.to_numeric, errors="coerce").dropna()
    mu, sd = tr.mean(), tr.std(ddof=0)
    Xv = val[feats].apply(pd.to_numeric, errors="coerce")
    Xv = (Xv - mu) / sd.replace(0, 1)
    Xv = Xv.fillna(0)
    z = Xv.values @ m.named_steps["lr"].coef_[0] + m.named_steps["lr"].intercept_[0]
    auc_ext = roc_auc_score(val["y"].values, z)
    print(f"\n>>> 外部验证 AUC (GSE59034, 跨平台/跨性别) = {auc_ext:.3f}")
    # 单基因外部验证
    print("    单基因外部验证 AUC:")
    for g in common:
        a = roc_auc_score(val["y"].values, pd.to_numeric(val[g], errors="coerce").fillna(0))
        print(f"      {g:<10} AUC={a:.3f}")
    np.save("results/_dummy.npy", np.array([0]))
    pd.DataFrame({"sample": val.index, "score": z, "y": val["y"].values}).to_csv(
        "results/GSE59034_external_score.csv", index=False)

# 保存汇总
rows = []
for k, r in res.items():
    if r:
        rows.append(dict(model=k, n=r["n"], cv_auc=r["cv_auc"], cv_sd=r["cv_sd"],
                         holdout_auc=r["holdout_auc"], C=r["C"]))
pd.DataFrame(rows).to_csv("results/lasso_model_summary.csv", index=False)
print("\n汇总已保存。")
