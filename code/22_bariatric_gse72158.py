"""
GSE72158 — 减重手术配对队列独立验证（Illumina HT-12，皮下脂肪，基线 vs 术后 1 年）
目标：独立重复 GSE59034 发现的"UFMylation 轴术后回落"（d = −1.16）
"""
import gzip
import numpy as np, pandas as pd
from scipy import stats

GSE = "GSE72158"
MAT = f"data/{GSE}_series_matrix.txt.gz"
ANN = "data/GPL10558_probe2gene.tsv"

AXIS_ALIAS = {
    "UFM1": ["UFM1"],
    "UBA5": ["UBA5"],
    "UFC1": ["UFC1"],
    "UFL1": ["UFL1", "KIAA0776"],
    "UFSP1": ["UFSP1"],
    "UFSP2": ["UFSP2"],
    "DDRGK1": ["DDRGK1"],
    "CDK5RAP3": ["CDK5RAP3"],
}
MARKERS = ["LEP", "ADIPOQ", "CD68", "FASN", "PPARG", "PLIN", "SLC2A4"]
CORE = ["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"]

# ---------- 注释 ----------
ann = pd.read_csv(ANN, sep="\t", header=None, names=["probe", "gene"], dtype=str)
p2g = dict(zip(ann.probe, ann.gene))
alias2std = {}
for std, als in AXIS_ALIAS.items():
    for a in als:
        alias2std[a] = std

# ---------- 读取矩阵 ----------
rows, hdr = [], None
meta = {}
with gzip.open(MAT, "rt") as f:
    for line in f:
        if line.startswith("!Sample_title"):
            meta["title"] = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            vals = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            key = vals[0].split(":")[0].strip().lower()
            meta[key] = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
        elif line.startswith("!series_matrix_table_begin"):
            hdr = next(f).rstrip("\n").split("\t")
            hdr = [x.strip().strip('"') for x in hdr]
            for line in f:
                if line.startswith("!series_matrix_table_end"):
                    break
                rows.append(line.rstrip("\n").split("\t"))

E = pd.DataFrame(rows, columns=None).set_index(0)
E.index = [x.strip().strip('"') for x in E.index]
E.columns = hdr[1:]
E = E.apply(pd.to_numeric, errors="coerce")
print(f"矩阵: {E.shape[0]} 探针 x {E.shape[1]} 样本")

samples = list(E.columns)
clin = pd.DataFrame({k: v for k, v in meta.items() if len(v) == len(samples)}, index=samples)
print("临床字段:", list(clin.columns))
for c in clin.columns:
    print(f"  {c}: {clin[c].nunique()} 个取值  {dict(list(clin[c].value_counts().head(4).items()))}")

sat = clin["tissue"].astype(str).str.contains("subcutaneous", case=False, na=False)
print(f"\n皮下脂肪样本: {sat.sum()} / {len(samples)}")
tp = clin["time point"].astype(str)
base = sat & tp.str.contains("0 days", case=False, na=False)
post = sat & tp.str.contains("1 year", case=False, na=False)
print(f"  基线: {base.sum()}   术后1年: {post.sum()}")

# ---------- 基因表达 ----------
def gene_expr(gene):
    probes = [p for p, g in p2g.items() if g == gene and p in E.index]
    if not probes:
        return None
    sub = E.loc[probes]
    return sub.mean(axis=0) if len(probes) > 1 else sub.iloc[0]

G = {}
for als in list(AXIS_ALIAS.values()) + [[m] for m in MARKERS]:
    for a in als:
        v = gene_expr(a)
        if v is not None:
            G[alias2std.get(a, a)] = v
print(f"\n成功映射基因 {len(G)} 个: {sorted(G.keys())}")

Gsat = pd.DataFrame({k: v[sat.values].astype(float) for k, v in G.items()})
sub_clin = clin[sat.values]
sid = sub_clin["subject id"].astype(str).values
tpv = sub_clin["time point"].astype(str).values
is_post = np.array(["1 year" in t.lower() for t in tpv])
is_base = np.array(["0 days" in t.lower() for t in tpv])

# 配对
from collections import defaultdict
d = defaultdict(dict)
for i, s in enumerate(sid):
    d[s]["post" if is_post[i] else ("base" if is_base[i] else "?")] = i
pairs = [(v["base"], v["post"]) for v in d.values() if "base" in v and "post" in v]
print(f"可配对受试者: {len(pairs)}")

bi = [p[0] for p in pairs]; pi = [p[1] for p in pairs]
print("\n" + "=" * 82)
print(f"GSE72158 减重手术配对（n={len(pairs)} 对，皮下脂肪，基线 vs 术后 1 年）")
print("=" * 82)
print(f"{'基因':<10}{'基线':>9}{'术后':>9}{'Δ(log2)':>10}{'p配对t':>11}{'p_Wilcoxon':>12}{'Cohen_d':>10}")
print("-" * 82)

out = []
Zall = Gsat.sub(Gsat.mean(axis=0), axis=1).div(Gsat.std(axis=0).replace(0, np.nan), axis=1)
scores = {
    "UFM1_z": Zall["UFM1"].values if "UFM1" in Zall else None,
    "machine_z": Zall[[c for c in CORE if c in Zall]].mean(axis=1).values,
}
if scores["UFM1_z"] is not None:
    scores["dysreg_z"] = scores["UFM1_z"] - scores["machine_z"]

targets = {g: Gsat[g].values for g in Gsat.columns}
targets.update({k: v for k, v in scores.items() if v is not None})

for name, v in targets.items():
    a, b = v[bi], v[pi]
    m = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[m], b[m]
    if len(a) < 5:
        continue
    t, pt = stats.ttest_rel(a, b)
    try:
        _, pw = stats.wilcoxon(a, b)
    except Exception:
        pw = np.nan
    diff = b - a
    d_cohen = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    out.append(dict(feature=name, base=a.mean(), post=b.mean(), delta=d_cohen and (b.mean()-a.mean()),
                    p_paired=pt, p_wilcoxon=pw, cohens_d=d_cohen, n=len(a)))
    print(f"{name:<10}{a.mean():>9.4f}{b.mean():>9.4f}{b.mean()-a.mean():>10.4f}"
          f"{pt:>11.2e}{pw:>12.2e}{d_cohen:>10.3f}")

res = pd.DataFrame(out)
res.to_csv("results/GSE72158_paired_weightloss.csv", index=False)

print("\n" + "=" * 82)
print("阳性对照（减重后预期方向：LEP↓, CD68↓, ADIPOQ↑, FASN↑, PPARG↑）")
print("=" * 82)
exp = {"LEP": -1, "CD68": -1, "ADIPOQ": +1, "FASN": +1, "PPARG": +1}
for g, e in exp.items():
    if g in res["feature"].values:
        row = res[res.feature == g].iloc[0]
        ok = (row["delta"] * e) > 0
        print(f"  {g:<8} Δ={row['delta']:+.4f}  p={row['p_paired']:.2e}  "
              f"方向{'正确' if ok else '错误'}")

print("\n" + "=" * 82)
print("与 GSE59034 的一致性判定")
print("=" * 82)
for k in ["UFM1_z", "dysreg_z", "machine_z"]:
    if k in res["feature"].values:
        row = res[res.feature == k].iloc[0]
        verdict = ("独立重复成功" if row["delta"] < 0 and row["p_paired"] < 0.05
                   else ("方向一致但不显著" if row["delta"] < 0
                         else "未重复（方向相反或无变化）"))
        print(f"  {k:<12} Δ={row['delta']:+.4f}  d={row['cohens_d']:+.3f}  "
              f"p={row['p_paired']:.2e}  -> {verdict}")

print("\n已保存 results/GSE72158_paired_weightloss.csv")
