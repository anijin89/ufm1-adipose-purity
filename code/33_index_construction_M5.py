"""
M5 敏感性：干预结论是否依赖"失衡指数"的机器定义方式？
复用 26_bariatric_gse53376.py 的 GSE53376 完全相同流水线（同一 Z、同一 pairs、
同一 dz 公式），只更换 conjugation-machinery 的基因集合，重算配对 Cohen's d。
产出 results/GSE53376_index_construction_M5.csv
"""
import gzip
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

GSE = "GSE53376"
MAT = f"data/{GSE}_series_matrix.txt.gz"
ANN = "data/GPL16686_probe2gene.tsv"
AXIS = ["UFM1", "UBA5", "UFC1", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]

ann = pd.read_csv(ANN, sep="\t", header=None, names=["probe", "gene"], dtype=str)
p2g = dict(zip(ann.probe, ann.gene))

rows, hdr, meta = [], None, {}
with gzip.open(MAT, "rt") as f:
    for line in f:
        if line.startswith("!Sample_title"):
            meta["title"] = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            vals = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            key = vals[0].split(":")[0].strip().lower()
            meta[key] = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
        elif line.startswith("!series_matrix_table_begin"):
            hdr = [x.strip().strip('"') for x in next(f).rstrip("\n").split("\t")]
            for line in f:
                if line.startswith("!series_matrix_table_end"):
                    break
                rows.append(line.rstrip("\n").split("\t"))

E = pd.DataFrame(rows).set_index(0)
E.index = [x.strip().strip('"') for x in E.index]
E.columns = hdr[1:]
E = E.apply(pd.to_numeric, errors="coerce")
samples = list(E.columns)
clin = pd.DataFrame({k: v for k, v in meta.items() if len(v) == len(samples)}, index=samples)

G = {}
for g in AXIS:
    probes = [p for p in E.index if p2g.get(p) == g]
    if probes:
        G[g] = E.loc[probes].mean(axis=0)
Gdf = pd.DataFrame(G)

cond = clin["condition"].astype(str).str.upper()
indiv = clin["individual"].astype(str)
is_post = cond.str.contains("POST")
is_pre = cond.str.contains("PRE") & ~is_post
d = defaultdict(dict)
for i, s in enumerate(indiv.values):
    d[s]["post" if is_post.iloc[i] else ("pre" if is_pre.iloc[i] else "?")] = i
pairs = [(v["pre"], v["post"]) for v in d.values() if "pre" in v and "post" in v]
bi = [p[0] for p in pairs]
pi = [p[1] for p in pairs]
print(f"配对 n={len(pairs)}")

# 与 26 脚本完全一致的全体 32 例内按样本 z 标准化（axis=1 因 Gdf 为 样本×基因）
Z = Gdf.sub(Gdf.mean(axis=0), axis=1).div(Gdf.std(axis=0, ddof=1).replace(0, np.nan), axis=1)

VARIANTS = {
    "Primary (UBA5, UFSP2, DDRGK1, CDK5RAP3)": ["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"],
    "E3 complex only (UFL1, DDRGK1, CDK5RAP3)": ["UFL1", "DDRGK1", "CDK5RAP3"],
    "Conjugation arm only (UBA5, UFC1, UFL1, DDRGK1, CDK5RAP3)": ["UBA5", "UFC1", "UFL1", "DDRGK1", "CDK5RAP3"],
    "UFSP2 excluded (UBA5, DDRGK1, CDK5RAP3)": ["UBA5", "DDRGK1", "CDK5RAP3"],
    "Deconjugation-only control (UFSP1, UFSP2)": ["UFSP1", "UFSP2"],
}

def paired_dz(score):
    a, b = np.asarray(score)[bi], np.asarray(score)[pi]
    m = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[m], b[m]
    diff = b - a
    sd = diff.std(ddof=1)
    dz = diff.mean() / sd if sd > 0 else np.nan
    t, pt = stats.ttest_rel(a, b)
    return dz, pt, len(a)

rec = []
for name, genes in VARIANTS.items():
    present = [g for g in genes if g in Z.columns]
    mach = Z[present].mean(axis=1)
    idx = Z["UFM1"] - mach
    dz, pt, n = paired_dz(idx.values)
    rec.append(dict(variant=name, genes=",".join(present), index_d=dz, p=pt, n=n))
    print(f"  {name:<58} index d={dz:+.3f}  p={pt:.3f}  n={n}")

# 单基因比值 UFM1/DDRGK1
idx2 = Z["UFM1"] - Z["DDRGK1"]
dz2, pt2, n2 = paired_dz(idx2.values)
rec.append(dict(variant="Simple UFM1/DDRGK1 ratio", genes="UFM1-DDRGK1", index_d=dz2, p=pt2, n=n2))
print(f"  {'Simple UFM1/DDRGK1 ratio':<58} index d={dz2:+.3f}  p={pt2:.3f}  n={n2}")

# 参照：UFM1 单独、machinery 单独（primary 集）
dz_u, p_u, _ = paired_dz(Z["UFM1"].values)
dz_m, p_m, _ = paired_dz(Z[["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"]].mean(axis=1).values)
print(f"\n参照: UFM1 alone d={dz_u:+.3f} (p={p_u:.3f}); primary machinery alone d={dz_m:+.3f} (p={p_m:.3f})")

pd.DataFrame(rec).to_csv("results/GSE53376_index_construction_M5.csv", index=False)
print("\n已保存 results/GSE53376_index_construction_M5.csv")
