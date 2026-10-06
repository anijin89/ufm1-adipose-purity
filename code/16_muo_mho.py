#!/usr/bin/env python3
"""关键测试: 在肥胖人群内部 (BMI>=30), UFMylation 轴能否区分代谢不健康/健康肥胖
(MUO vs MHO), 此时 BMI 不再是主导混杂
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats

sc = pd.read_csv("results/GSE70353_UFM_scores.csv", index_col=0)
clin = pd.read_csv("results/GSE70353_clinical_aligned.csv", index_col=0)
df = sc.join(clin, how="inner")
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

CORE = ["UFM1", "UFC1", "UBA5", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]

ob = df[df["bmi"] >= 30].copy()
print("肥胖子集 (BMI>=30): n =", len(ob), " BMI 范围 %.1f - %.1f" % (ob["bmi"].min(), ob["bmi"].max()))
print("HOMA-IR: 中位数 %.2f  范围 %.2f - %.2f" % (ob["homair"].median(), ob["homair"].min(), ob["homair"].max()))

# MUO/MHO 定义: HOMA-IR 上三分位 vs 下三分位 (在肥胖人群内)
t1, t2 = ob["homair"].quantile(1/3), ob["homair"].quantile(2/3)
ob["grp"] = np.nan
ob.loc[ob["homair"] >= t2, "grp"] = 1   # MUO
ob.loc[ob["homair"] <= t1, "grp"] = 0   # MHO
d = ob.dropna(subset=["grp"])
print(f"MUO n={int((d['grp']==1).sum())}  vs  MHO n={int((d['grp']==0).sum())}")

MODELS = {
    "Age + BMI (clinical)": ["age", "bmi"],
    "Age + BMI + UFMylation axis": ["age", "bmi"] + CORE,
    "Age + BMI + machinery score": ["age", "bmi", "machine_z"],
    "Age + BMI + dysreg index": ["age", "bmi", "dysreg"],
}
skf = StratifiedKFold(5, shuffle=True, random_state=0)
res = {}
for name, feats in MODELS.items():
    Xd = d[feats].apply(pd.to_numeric, errors="coerce")
    keep = Xd.notna().all(axis=1)
    Xd, y = Xd[keep], d["grp"][keep]
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=5000))])
    prob = cross_val_predict(pipe, Xd.values, y.values, cv=skf, method="predict_proba")[:, 1]
    auc = roc_auc_score(y.values, prob)
    res[name] = auc
    print(f"  {name:<32} AUC={auc:.3f}  n={len(y)}")

base = res["Age + BMI (clinical)"]
print(f"\n增量 ΔAUC (相对临床基准 {base:.3f}):")
for k, v in res.items():
    if "clinical" not in k:
        print(f"  {k:<32} Δ={v-base:+.3f}")

# 单变量: UFMylation 轴各成分在肥胖人群内 MUO vs MHO
print("\n=== 肥胖人群内: MUO vs MHO 的单变量比较 (校正年龄+BMI) ===")
def adj_compare(col):
    sub = ob.dropna(subset=[col, "grp", "age", "bmi"])
    y = sub["grp"].values
    Xd = np.column_stack([np.ones(len(sub)), sub["age"].values, sub["bmi"].values,
                          sub[col].values])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    Xr = np.column_stack([np.ones(len(sub)), sub["age"].values, sub["bmi"].values])
    br, *_ = np.linalg.lstsq(Xr, sub[col].values, rcond=None)
    rx = sub[col].values - Xr @ br
    r, p = stats.pearsonr(rx, resid)
    # t 检验
    a, b = sub.loc[sub["grp"] == 1, col], sub.loc[sub["grp"] == 0, col]
    t, pt = stats.ttest_ind(a, b, equal_var=False)
    return dict(var=col, mean_MUO=a.mean(), mean_MHO=b.mean(),
                smd=(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2),
                t=t, p_ttest=pt, r_adj=r, p_adj=p, n=len(sub))

rows = [adj_compare(c) for c in CORE + ["machine_z", "dysreg", "UFM1_z"]]
R = pd.DataFrame(rows)
print(R.round(4).to_string(index=False))
R.to_csv("results/MUO_vs_MHO.csv", index=False)

pd.DataFrame([dict(model=k, auc=v) for k, v in res.items()]).to_csv(
    "results/model_MUO_MHO_auc.csv", index=False)
