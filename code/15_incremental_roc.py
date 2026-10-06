#!/usr/bin/env python3
"""UFMylation 轴对胰岛素抵抗的增量预测价值 (在临床变量之上) + ROC 图"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
OUT = "figures"; os.makedirs(OUT, exist_ok=True)


def fmt_p(p):
    """AE&M wording rule: the complete word "p-value" and two decimals; a value
    that would round to 0.00 is reported as "p-value < 0.01"."""
    return "p-value < 0.01" if p < 0.005 else "p-value = {:.2f}".format(p)

sc = pd.read_csv("results/GSE70353_UFM_scores.csv", index_col=0)
clin = pd.read_csv("results/GSE70353_clinical_aligned.csv", index_col=0)
df = sc.join(clin, how="inner")
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

CORE = ["UFM1", "UFC1", "UBA5", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]
h = df["homair"]
q1, q3 = h.quantile(.25), h.quantile(.75)
ir = pd.Series(np.nan, index=df.index)
ir[h <= q1] = 0; ir[h >= q3] = 1
df["ir"] = ir

MODELS = {
    "Clinical (age + BMI)": ["age", "bmi"],
    "Clinical + UFMylation axis": ["age", "bmi"] + CORE,
    "Clinical + machinery score": ["age", "bmi", "machine_z"],
    "Clinical + dysregulation index": ["age", "bmi", "dysreg"],
}
d = df.dropna(subset=["ir"])
print("分析样本 (HOMA-IR Q4 vs Q1):", (d["ir"] == 1).sum(), "vs", (d["ir"] == 0).sum(),
      " 总 n =", d.shape[0])

skf = StratifiedKFold(5, shuffle=True, random_state=0)
results = {}
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
ax = axes[0]
pal = {"Clinical (age + BMI)": "#7f8c8d", "Clinical + UFMylation axis": "#2980b9",
       "Clinical + machinery score": "#c0392b", "Clinical + dysregulation index": "#8e44ad"}
for name, feats in MODELS.items():
    Xd = d[feats].apply(pd.to_numeric, errors="coerce")
    keep = Xd.notna().all(axis=1)
    Xd, y = Xd[keep], d["ir"][keep]
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l2", C=1.0, max_iter=5000))])
    prob = cross_val_predict(pipe, Xd.values, y.values, cv=skf, method="predict_proba")[:, 1]
    auc = roc_auc_score(y.values, prob)
    fpr, tpr, _ = roc_curve(y.values, prob)
    ax.plot(fpr, tpr, lw=1.7, color=pal[name], label=f"{name}  (AUC={auc:.3f})")
    results[name] = dict(auc=auc, n=int(len(y)), fpr=fpr, tpr=tpr, feats=feats)
    print(f"  {name:<34} AUC={auc:.3f}  n={len(y)}")

ax.plot([0, 1], [0, 1], "--", color="#999", lw=.9)
ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("a  Insulin resistance (HOMA-IR Q4 vs Q1)\n5-fold cross-validated ROC", fontsize=9.5, loc="left")
ax.legend(fontsize=7.2, loc="lower right", frameon=False)

base = results["Clinical (age + BMI)"]["auc"]
print(f"\n临床基准 AUC = {base:.3f}")
for name, r in results.items():
    if name.startswith("Clinical (age"):
        continue
    dauc = r["auc"] - base
    # DeLong 近似: 用 bootstrap 比较
    Xb = d[MODELS["Clinical (age + BMI)"]].apply(pd.to_numeric, errors="coerce")
    keep = Xb.notna().all(axis=1)
    Xa = d[r["feats"]].apply(pd.to_numeric, errors="coerce")
    keep = keep & Xa.notna().all(axis=1)
    yall = d["ir"][keep]
    pipeA = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=5000))])
    pipeB = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=5000))])
    rng = np.random.default_rng(0)
    diffs = []
    for b in range(200):
        idx = rng.choice(keep.sum(), keep.sum(), replace=True)
        Xtr = Xa[keep].values[idx]; ytr = yall.values[idx]
        if ytr.sum() < 5 or (1 - ytr).sum() < 5:
            continue
        try:
            pipeA.fit(Xtr, ytr)
            pa = pipeA.predict_proba(Xa[keep].values)[:, 1]
            Xtrb = Xb[keep].values[idx]
            pipeB.fit(Xtrb, ytr)
            pb = pipeB.predict_proba(Xb[keep].values)[:, 1]
            oob = np.setdiff1d(np.arange(keep.sum()), np.unique(idx))
            if len(oob) > 20:
                diffs.append(roc_auc_score(yall.values[oob], pa[oob]) - roc_auc_score(yall.values[oob], pb[oob]))
        except Exception:
            pass
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
    print(f"  ΔAUC {name:<34} = {dauc:+.3f}   95%CI [{lo:.3f}, {hi:.3f}]")

# 面板 b: machine score 的剂量效应 (按四分位的 IR 患病率)
ax = axes[1]
sub = df.dropna(subset=["ir", "machine_z"]).copy()
sub["q"] = pd.qcut(sub["machine_z"], 4, labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"])
prev = sub.groupby("q", observed=True)["ir"].agg(["mean", "sum", "count"])
xs = np.arange(4)
ax.bar(xs, prev["mean"] * 100, color=["#c0392b" if v else "#2980b9" for v in [1, 0, 0, 0]],
       width=.62, edgecolor="#333", lw=.6)
for i, (m, s, c) in enumerate(zip(prev["mean"], prev["sum"], prev["count"])):
    ax.text(i, m * 100 + 1.5, f"{int(s)}/{int(c)}\n{m*100:.0f}%", ha="center", fontsize=7.5)
# Cochran-Armitage 趋势检验近似: logistic 趋势
ordinal = sub["q"].cat.codes.values
Xtrend = np.column_stack([np.ones(len(sub)), ordinal])
beta_trend = np.linalg.lstsq(Xtrend, sub["ir"].values.astype(float), rcond=None)[0][1]
z = beta_trend / (np.std(sub["ir"].values) / np.sqrt(len(sub)) / np.std(ordinal))
from scipy.stats import chi2
r_trend, p_trend = stats.spearmanr(sub["machine_z"], sub["ir"])
ax.set_xticks(xs); ax.set_xticklabels(prev.index, fontsize=8)
ax.set_ylabel("Insulin resistance prevalence (%)")
# head-room so the bar labels clear the three-line panel title
ax.set_ylim(0, 80)
ax.set_title("b  Prevalence of insulin resistance\nby quartile of the machinery score\n"
             f"Spearman rho = {r_trend:+.2f}, {fmt_p(p_trend)}",
             fontsize=9.5, loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/Fig6_incremental_value.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}/Fig6_incremental_value.pdf", bbox_inches="tight")
plt.close()

pd.DataFrame([dict(model=k, auc=v["auc"], n=v["n"]) for k, v in results.items()]).to_csv(
    "results/model_comparison_auc.csv", index=False)
print("\nFig6 已生成。")
