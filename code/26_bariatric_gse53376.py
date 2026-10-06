"""
GSE53376 — 第三个减重手术配对队列（Affymetrix HuGene-2_0-st，皮下脂肪，16 名病态肥胖女性 PRE vs POST）
目的：把"UFMylation 轴术后回落"从 2 个队列扩展到 3 个队列，并对 3 个队列做合并效应量 meta 分析。
注释来源：scripts/26a_build_gpl16686.py 生成的 data/GPL16686_probe2gene.tsv
          （Bioconductor hugene20sttranscriptcluster.db -> Entrez -> NCBI Symbol/Synonyms）
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
MACHINE = ["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"]          # 与 GSE59034/GSE72158 定义一致
ER_ISR = ["EIF2AK3", "EIF2S1", "ATF4", "DDIT3", "HSPA5", "XBP1", "SEL1L", "ATF6"]
ER_CORE = ["EIF2AK3", "EIF2S1", "ATF4", "DDIT3", "HSPA5"]  # 先验定义的 eIF2α-ISR 复合评分
ADIPO = ["LEP", "ADIPOQ", "CD68", "FASN", "PPARG", "PLIN1", "SLC2A4", "PTPRC", "LYZ"]
# 原文 qPCR 验证的差异基因（PMID 26252355），用作本队列的方向性阳性对照
PAPER_UP = ["SLC27A2", "ELOVL6", "FASN", "GYS2", "LGALS12", "PKP2", "ACLY"]
PAPER_DOWN = ["FOS", "EGFL6", "PRG4", "AQP9", "DUSP1", "RGS1", "EGR1", "SPP1", "LYZ"]

# ---------- 注释 ----------
ann = pd.read_csv(ANN, sep="\t", header=None, names=["probe", "gene"], dtype=str)
p2g = dict(zip(ann.probe, ann.gene))
print(f"注释: {len(p2g)} 探针 -> {ann.gene.nunique()} 基因")

# ---------- 读取矩阵 ----------
rows, hdr = [], None
meta = {}
with gzip.open(MAT, "rt") as f:
    for line in f:
        if line.startswith("!Sample_title"):
            meta["title"] = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_geo_accession"):
            meta["gsm"] = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
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
print(f"矩阵: {E.shape[0]} 探针 x {E.shape[1]} 样本；取值域 [{np.nanmin(E.values):.2f}, {np.nanmax(E.values):.2f}]")

samples = list(E.columns)
clin = pd.DataFrame({k: v for k, v in meta.items() if len(v) == len(samples)}, index=samples)
print("临床字段:", list(clin.columns))
for c in clin.columns:
    print(f"  {c}: {clin[c].nunique()} 个取值 {dict(list(clin[c].value_counts().head(3).items()))}")

# ---------- 基因水平表达（多探针取均值）----------
want = sorted(set(AXIS + ER_ISR + ADIPO + PAPER_UP + PAPER_DOWN))
G = {}
for g in want:
    probes = [p for p in E.index if p2g.get(p) == g]
    if probes:
        G[g] = E.loc[probes].mean(axis=0)
print(f"\n映射成功 {len(G)}/{len(want)} 个基因；缺失: {[g for g in want if g not in G]}")

Gdf = pd.DataFrame(G)
cond = clin["condition"].astype(str).str.upper()
indiv = clin["individual"].astype(str)
is_post = cond.str.contains("POST")
is_pre = cond.str.contains("PRE") & ~is_post
print(f"PRE: {is_pre.sum()}   POST: {is_post.sum()}   个体数: {indiv.nunique()}")

d = defaultdict(dict)
for i, s in enumerate(indiv.values):
    d[s]["post" if is_post.iloc[i] else ("pre" if is_pre.iloc[i] else "?")] = i
pairs = [(v["pre"], v["post"]) for v in d.values() if "pre" in v and "post" in v]
print(f"可配对受试者: {len(pairs)}")

# ---------- 复合评分（全体 32 例内 z 标准化，定义与既往队列一致）----------
Z = Gdf.sub(Gdf.mean(axis=0), axis=1).div(Gdf.std(axis=0, ddof=1).replace(0, np.nan), axis=1)
Z["UFM1_z"] = Z["UFM1"]
Z["machine_z"] = Z[[c for c in MACHINE if c in Z]].mean(axis=1)
Z["dysreg_z"] = Z["UFM1_z"] - Z["machine_z"]
Z["ER_ISR_z"] = Z[[c for c in ER_CORE if c in Z]].mean(axis=1)

# ---------- 配对检验 ----------
bi = [p[0] for p in pairs]
pi = [p[1] for p in pairs]
print("\n" + "=" * 90)
print(f"GSE53376 减重手术配对（n={len(pairs)} 对，皮下脂肪，16 名病态肥胖女性，PRE -> POST）")
print("=" * 90)
print(f"{'特征':<12}{'PRE均值':>9}{'POST均值':>10}{'Δ':>10}{'p配对t':>11}{'p_Wilcoxon':>12}{'Cohen_d':>10}")
print("-" * 90)

out = []
targets = {g: Gdf[g].values for g in Gdf.columns}
targets.update({k: Z[k].values for k in ["UFM1_z", "machine_z", "dysreg_z", "ER_ISR_z"]})
for name, v in targets.items():
    a, b = np.asarray(v)[bi], np.asarray(v)[pi]
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
    sd = diff.std(ddof=1)
    d_cohen = diff.mean() / sd if sd > 0 else np.nan
    out.append(dict(feature=name, base=a.mean(), post=b.mean(), delta=diff.mean(),
                    p_paired=pt, p_wilcoxon=pw, cohens_d=d_cohen, n=len(a)))
    print(f"{name:<12}{a.mean():>9.4f}{b.mean():>10.4f}{diff.mean():>+10.4f}"
          f"{pt:>11.2e}{pw:>12.2e}{d_cohen:>+10.3f}")

res = pd.DataFrame(out)
res.to_csv(f"results/{GSE}_paired_weightloss.csv", index=False)

# ---------- 阳性对照：文献 qPCR 验证基因方向 ----------
print("\n" + "=" * 90)
print("阳性对照 1：原研究（PMID 26252355）qPCR 验证的差异基因方向")
print("=" * 90)
ok = 0
tot = 0
for g in PAPER_UP:
    r = res[res.feature == g]
    if len(r):
        r = r.iloc[0]
        good = r.delta > 0
        ok += good
        tot += 1
        print(f"  {g:<9} 期望↑  Δ={r.delta:+.4f}  p={r.p_paired:.2e}  {'一致' if good else '不一致'}")
for g in PAPER_DOWN:
    r = res[res.feature == g]
    if len(r):
        r = r.iloc[0]
        good = r.delta < 0
        ok += good
        tot += 1
        print(f"  {g:<9} 期望↓  Δ={r.delta:+.4f}  p={r.p_paired:.2e}  {'一致' if good else '不一致'}")
print(f"  -> 方向一致 {ok}/{tot}")

# ---------- 阳性对照 2：脂肪组织经典哨兵 ----------
print("\n" + "=" * 90)
print("阳性对照 2：减重后预期方向（LEP↓, CD68↓, ADIPOQ↑, FASN↑, PPARG↑）")
print("=" * 90)
exp = {"LEP": -1, "CD68": -1, "ADIPOQ": +1, "FASN": +1, "PPARG": +1, "PLIN1": +1, "SLC2A4": +1}
for g, e in exp.items():
    r = res[res.feature == g]
    if len(r):
        r = r.iloc[0]
        print(f"  {g:<8} Δ={r.delta:+.4f}  p={r.p_paired:.2e}  "
              f"方向{'正确' if r.delta * e > 0 else '错误'}")

# ---------- UFM1 与 ER/ISR 的横断面偶联（PRE 基线 n=16，及全部 32 例）----------
print("\n" + "=" * 90)
print("横断面偶联：UFM1 与 ER/ISR 基因的相关（PRE 基线 / 全部样本）")
print("=" * 90)
coup = []
for g in ER_ISR + ADIPO:
    if g not in Gdf:
        continue
    xb = Gdf["UFM1"].values[bi]
    yb = Gdf[g].values[bi]
    rb, pb = stats.pearsonr(xb, yb)
    ra, pa = stats.pearsonr(Gdf["UFM1"].values, Gdf[g].values)
    coup.append(dict(gene=g, r_pre=rb, p_pre=pb, r_all=ra, p_all=pa, n_pre=len(xb)))
    print(f"  {g:<9} PRE: r={rb:+.3f} p={pb:.2e} | 全部: r={ra:+.3f} p={pa:.2e}")
pd.DataFrame(coup).to_csv(f"results/{GSE}_UFM1_coupling.csv", index=False)

# ---------- 三队列 meta 分析 ----------
print("\n" + "=" * 90)
print("三队列合并：术后 vs 术前（配对标准化均数差，固定效应 inverse-variance）")
print("=" * 90)


def dz_se(d, n):
    return np.sqrt(1.0 / n + d ** 2 / (2.0 * n))


studies = []
for feat, key in [("UFM1 z-score", "UFM1_z"), ("Dysregulation index", "dysreg_z")]:
    # GSE59034（既有结果文件）
    s59034 = pd.read_csv("results/GSE59034_intervention_scores.csv")
    row = s59034[(s59034.score == ("UFM1_z" if feat.startswith("UFM1") else "dysreg")) &
                 (s59034.comparison == "Post_RYGB vs Obese_pre")].iloc[0]
    studies.append(dict(study="GSE59034 (n=16)", feature=feat, d=row.cohen_d, n=16))
    for f, lab in [("results/GSE72158_paired_weightloss.csv", "GSE72158 (n=42)"),
                   (f"results/{GSE}_paired_weightloss.csv", f"{GSE} (n={len(pairs)})")]:
        rr = pd.read_csv(f)
        r2 = rr[rr.feature == key].iloc[0]
        studies.append(dict(study=lab, feature=feat, d=r2.cohens_d, n=int(r2.n)))

S = pd.DataFrame(studies)
S["se"] = [dz_se(d, n) for d, n in zip(S.d, S.n)]
S["w"] = 1 / S.se ** 2
meta = []
for feat in S.feature.unique():
    s = S[S.feature == feat]
    w = s.w.values
    d = s.d.values
    k = len(d)
    D = float((w * d).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    z = D / se
    p = 2 * stats.norm.sf(abs(z))
    Q = float((w * (d - D) ** 2).sum())
    df = k - 1
    pQ = stats.chi2.sf(Q, df) if df > 0 else np.nan
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    # DerSimonian-Laird 随机效应（异质性明显时以随机效应为准）
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wR = 1 / (s.se.values ** 2 + tau2)
    DR = float((wR * d).sum() / wR.sum())
    seR = float(np.sqrt(1 / wR.sum()))
    zR = DR / seR
    pR = 2 * stats.norm.sf(abs(zR))
    meta.append(dict(feature=feat, n_pairs=int(s.n.sum()), d_fixed=D, se_fixed=se,
                     fixed_lo=D - 1.96 * se, fixed_hi=D + 1.96 * se, p_fixed=p,
                     d_random=DR, se_random=seR,
                     random_lo=DR - 1.96 * seR, random_hi=DR + 1.96 * seR, p_random=pR,
                     Q=Q, p_het=pQ, I2=I2, tau2=tau2))
    print(f"\n{feat}  合计 {int(s.n.sum())} 对")
    for _, r in s.iterrows():
        print(f"   {r.study:<18} d={r.d:+.3f} (95%CI {r.d-1.96*r.se:+.2f},{r.d+1.96*r.se:+.2f})")
    print(f"   固定效应 d={D:+.3f} (95%CI {D-1.96*se:+.2f},{D+1.96*se:+.2f})  p={p:.2e}")
    print(f"   随机效应 d={DR:+.3f} (95%CI {DR-1.96*seR:+.2f},{DR+1.96*seR:+.2f})  p={pR:.2e}")
    print(f"   异质性 Q={Q:.2f} (p={pQ:.3f}), I²={I2:.0f}%, τ²={tau2:.3f}")
pd.DataFrame(meta).to_csv("results/bariatric_three_cohort_meta.csv", index=False)

print("\n已保存:")
print(f"  results/{GSE}_paired_weightloss.csv")
print(f"  results/{GSE}_UFM1_coupling.csv")
print("  results/bariatric_three_cohort_meta.csv")
