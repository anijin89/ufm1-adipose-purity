#!/usr/bin/env python3
"""GSE59034: 女性腹部皮下脂肪，16 例肥胖患者 RYGB 术前/术后配对 + 16 例从不肥胖对照。
验证 UFMylation 轴 (主角 UFM1) 在肥胖中是否上调、减重后是否回落。
"""
import gzip
import numpy as np
import pandas as pd
from scipy import stats

DATA = "data/GSE59034_series_matrix.txt.gz"
ANN = "data/GPL11532_probe2gene.tsv"

# ---------- 1. 读样本元数据 ----------
meta_rows = []
with gzip.open(DATA, "rt", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if line.startswith("!Sample_title"):
            titles = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_geo_accession"):
            gsms = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            meta_rows.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
        elif line.startswith("!Sample_source_name_ch1") or line.startswith("!Sample_description"):
            meta_rows.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
        elif line.startswith("!series_matrix_table_begin"):
            break

meta = pd.DataFrame({"gsm": gsms, "title": titles})
for r in meta_rows:
    if len(r) != len(gsms):
        continue
    keys = {x.split(":")[0].strip() for x in r if ":" in x}
    if len(keys) == 1:
        k = keys.pop()
        meta[k] = [x.split(":", 1)[1].strip() if ":" in x else "" for x in r]

# 解析分组与配对编号
import re
meta["group"] = meta["obesity status"].map({
    "before bariatric surgery": "Obese_pre",
    "after bariatric surgery": "Obese_post",
    "never-obese": "Lean_control",
})
meta["pair"] = meta["title"].str.extract(r"rep(\d+)")[0]
print(meta[["gsm", "group", "pair", "gender"]].head(9).to_string(index=False))
print("\n分组计数:\n", meta["group"].value_counts().to_string())
print("配对数:", meta["pair"].nunique())

# ---------- 2. 探针注释 ----------
ann = pd.read_csv(ANN, sep="\t", dtype=str)
p2g = dict(zip(ann["probe"], ann["symbol"]))

GENES = {
    "UFM1": "UFM1", "UFC1": "UFC1", "UBA5": "UBA5", "UFL1": "KIAA0776",
    "UFSP1": "UFSP1", "UFSP2": "UFSP2", "DDRGK1": "DDRGK1", "CDK5RAP3": "CDK5RAP3",
}
POS = ["LEP", "CD68", "ADIPOQ", "PLIN1", "FASN", "PPARG"]

want_syms = set(GENES.values()) | set(POS)
want_probes = {p: s for p, s in p2g.items() if s in want_syms}

# ---------- 3. 提取表达 ----------
expr = {}
with gzip.open(DATA, "rt", encoding="utf-8", errors="ignore") as fh:
    in_table = False
    for line in fh:
        if line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        if line.startswith("!series_matrix_table_end"):
            break
        if not in_table:
            continue
        f = line.rstrip("\n").split("\t")
        pid = f[0].strip('"')
        if pid in want_probes:
            vals = pd.to_numeric(pd.Series(f[1:]).str.strip('"'), errors="coerce")
            expr[pid] = vals.values

E = pd.DataFrame(expr, index=gsms).dropna(axis=1, how="all")
E.columns = [want_probes[c] for c in E.columns]
print(f"\n提取表达矩阵: {E.shape[0]} 样本 x {E.shape[1]} 基因探针")
# 同基因多探针取均值
E = E.T.groupby(level=0).mean().T
E.to_csv("results/GSE59034_UFM_expr.csv")

# ---------- 4. 配对检验 ----------
piv = meta.set_index("gsm")


def paired(g, a, b):
    """同一 donor 内 a vs b 的配对检验"""
    x, y = [], []
    for p in sorted(piv["pair"].dropna().unique()):
        ga = piv[(piv["pair"] == p) & (piv["group"] == a)].index
        gb = piv[(piv["pair"] == p) & (piv["group"] == b)].index
        if len(ga) and len(gb):
            x.append(E.loc[ga, g].values[0])
            y.append(E.loc[gb, g].values[0])
    x, y = np.array(x), np.array(y)
    if len(x) < 3:
        return None
    t, p = stats.ttest_rel(x, y)
    d = x - y
    return dict(n=len(x), mean_a=x.mean(), mean_b=y.mean(),
                diff=d.mean(), log2fc=np.log2(x.mean() / y.mean()) if x.mean() * y.mean() > 0 else np.nan,
                t=t, p=p, d_cohen=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
                wilcoxon=stats.wilcoxon(d)[1])


out = []
allg = list(GENES.keys()) + POS
for g in allg:
    if g not in E.columns:
        print(f"!! {g} 不在矩阵中")
        continue
    r1 = paired(g, "Obese_pre", "Lean_control")   # 肥胖 vs 瘦
    r2 = paired(g, "Obese_post", "Obese_pre")     # 术后 vs 术前
    r3 = paired(g, "Obese_post", "Lean_control")  # 术后 vs 瘦
    for lab, r in [("Obese_pre vs Lean", r1), ("Post vs Pre (RYGB)", r2), ("Post vs Lean", r3)]:
        if r:
            r.update(gene=g, comparison=lab)
            out.append(r)

R = pd.DataFrame(out)[["gene", "comparison", "n", "mean_a", "mean_b", "diff", "log2fc",
                       "t", "p", "wilcoxon", "d_cohen"]]
R.to_csv("results/GSE59034_paired_tests.csv", index=False)

order = ["UFM1", "UFC1", "UBA5", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3",
         "LEP", "CD68", "ADIPOQ", "PLIN1", "FASN", "PPARG"]
for lab in ["Obese_pre vs Lean", "Post vs Pre (RYGB)", "Post vs Lean"]:
    print(f"\n=== {lab} ===")
    sub = R[R["comparison"] == lab].set_index("gene").reindex([g for g in order if g in set(R["gene"])])
    for g, row in sub.iterrows():
        star = "***" if row["p"] < .001 else "**" if row["p"] < .01 else "*" if row["p"] < .05 else "ns"
        print(f"  {g:<10} log2FC={row['log2fc']:+.3f}  p={row['p']:.2e} {star:<3} "
              f" Wilcoxon p={row['wilcoxon']:.2e}  Cohen d={row['d_cohen']:+.2f}")
