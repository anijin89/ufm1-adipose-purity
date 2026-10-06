#!/usr/bin/env python3
"""GSE70353 (n=770 男性皮下脂肪): UFM1 全转录组共表达分析。
目的: (1) 找出与 UFM1 最相关的基因 -> 通路富集 (ORA/GSEA)
      (2) 为 LASSO 建模准备特征矩阵
同时输出 BMI 的全转录组相关以对照。
"""
import gzip
import numpy as np
import pandas as pd
from scipy import stats

DATA = "data/GSE70353_series_matrix.txt.gz"
ANN = "data/GPL13667_probe2gene.tsv"

# ---------- 读表达矩阵 ----------
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
print("表行数:", len(rows))
header = [x.strip('"') for x in rows[0].rstrip("\n").split("\t")]
samples = header[1:]
probes, mat = [], []
for line in rows[1:]:
    f = line.rstrip("\n").split("\t")
    probes.append(f[0].strip('"'))
    mat.append(np.array(f[1:], dtype=np.float32))
X = pd.DataFrame(np.vstack(mat), index=probes, columns=samples)
del rows, mat
print("表达矩阵:", X.shape)

# ---------- 注释 ----------
ann = pd.read_csv(ANN, sep="\t", dtype=str)
p2g = dict(zip(ann["probe"], ann["symbol"]))
X.index = [p2g.get(p, "") for p in X.index]
X = X[X.index != ""]
# 同基因多探针取均值
X = X.groupby(level=0).mean()
print("注释后基因数:", X.shape[0])

# ---------- 临床 ----------
clin = pd.read_csv("data/GSE70353_clinical.csv", index_col=0)
clin = clin.reindex(samples)
print("临床列:", list(clin.columns))

for c in clin.columns:
    clin[c] = pd.to_numeric(clin[c], errors="coerce")

# ---------- 目标基因表达向量 ----------
target = "UFM1"
if target not in X.index:
    raise SystemExit(f"{target} 不在矩阵中")
u = X.loc[target].astype(float)

bmi = clin["bmi"].astype(float) if "bmi" in clin else None

def cor_all(vec, name):
    v = vec.values.astype(float)
    ok = ~np.isnan(v)
    sub = X.loc[:, ok].astype(float)
    vv = v[ok]
    # Pearson
    vv_c = vv - vv.mean()
    M = sub.values - sub.values.mean(axis=1, keepdims=True)
    num = M @ vv_c
    den = np.sqrt((M ** 2).sum(axis=1) * (vv_c ** 2).sum())
    den[den == 0] = np.nan
    r = num / den
    n = ok.sum()
    # t 检验
    t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    p = 2 * stats.t.sf(np.abs(t), n - 2)
    out = pd.DataFrame({"gene": sub.index, f"r_{name}": r, f"p_{name}": p}).set_index("gene")
    # BH 校正
    pv = out[f"p_{name}"].values
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(pv) / (np.arange(len(pv)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    qq = np.empty_like(q)
    qq[order] = np.clip(q, 0, 1)
    out[f"q_{name}"] = qq
    return out.sort_values(f"r_{name}")

R_ufm1 = cor_all(u, "UFM1")
R_ufm1.to_csv("results/GSE70353_UFM1_transcriptome_corr.csv")
print("\n=== 与 UFM1 最正相关 top 25 ===")
print(R_ufm1.tail(25)[["r_UFM1", "q_UFM1"]].iloc[::-1].round(4).to_string())
print("\n=== 与 UFM1 最负相关 top 25 ===")
print(R_ufm1.head(25)[["r_UFM1", "q_UFM1"]].round(4).to_string())

# 保存 UFMylation 核心基因表达用于后续建模
core = ["UFM1", "UFC1", "UBA5", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]
core_present = [g for g in core if g in X.index]
X.loc[core_present].T.to_csv("results/GSE70353_UFMylation_core_expr.csv")
print("\n核心基因:", core_present)
print("缺失:", [g for g in core if g not in X.index])

if bmi is not None:
    R_bmi = cor_all(bmi, "BMI")
    R_bmi.to_csv("results/GSE70353_BMI_transcriptome_corr.csv")
    print("\n=== 与 BMI 最正相关 top 15 ===")
    print(R_bmi.tail(15)[["r_BMI", "q_BMI"]].iloc[::-1].round(4).to_string())

# 保存精简表达矩阵（用于 LASSO，只保留高变异基因以省内存）
hv = X.var(axis=1).sort_values(ascending=False).head(8000).index
X.loc[hv].T.to_csv("results/GSE70353_expr_hv8000.csv")
print("\n高变异基因矩阵已保存:", len(hv))

# 临床保存
clin.to_csv("results/GSE70353_clinical_clean.csv")
