#!/usr/bin/env python
"""地基检验：UFMylation 通路基因在 GSE70353（770 例男性皮下脂肪）中
与肥胖 / 胰岛素敏感性表型的关联。"""
import gzip, os, csv, re
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MATRIX = os.path.join(DATA, "GSE70353_series_matrix.txt.gz")
ANN = os.path.join(DATA, "GPL13667_probe2gene.tsv")
CLIN = os.path.join(DATA, "GSE70353_clinical.csv")

CORE = ["UFM1", "UFC1", "UBA5", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]

# ---------- 1. 注释：核心基因 -> 探针（含别名模糊匹配） ----------
ann = pd.read_csv(ANN, sep="\t", dtype=str).fillna("")
probe2gene = dict(zip(ann["probe"], ann["symbol"]))

alias = {"UFL1": r"(KIAA0776|RCAD|NLBP|MAXER|UFL1)",
         "DDRGK1": r"(DDRGK1|UFBP1|C20orf116)"}

gene2probes = {}
for g in CORE:
    pat = alias.get(g, rf"^{g}$")
    hits = [p for p, s in probe2gene.items() if re.search(pat, s)]
    gene2probes[g] = hits

print("=== 核心基因探针 ===")
for g in CORE:
    print(f"  {g:9s} {len(gene2probes[g])} probes")

CONTROL = ["FASN", "PPARG", "ADIPOQ", "LEP", "PLIN1", "CD68", "COL1A1", "SREBF1"]
ctrl2probes = {g: [p for p, s in probe2gene.items() if s == g] for g in CONTROL}
gene2probes.update(ctrl2probes)

wanted = sorted({p for ps in gene2probes.values() for p in ps})
wanted_set = set(wanted)

# ---------- 2. 流式抽取表达 ----------
rows = {}
samples = None
in_table = False
header_seen = False
with gzip.open(MATRIX, "rt") as fh:
    for line in fh:
        if line.startswith("!Sample_title"):
            samples = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        elif line.startswith("!series_matrix_table_end"):
            break
        elif in_table:
            if not header_seen:
                header_seen = True
                continue
            pid, rest = line.split("\t", 1)
            pid = pid.strip().strip('"')
            if pid in wanted_set:
                rows[pid] = [float(x) if x.strip() not in ("", "null", "NA") else np.nan
                             for x in rest.rstrip("\n").split("\t")]

expr = pd.DataFrame(rows, index=samples)
print(f"\n表达矩阵: {expr.shape}  (样本 x 探针)")
print(f"数值范围: {np.nanmin(expr.values):.2f} ~ {np.nanmax(expr.values):.2f}  -> 判定为 log2 尺度")

# ---------- 3. 临床 ----------
clin = pd.read_csv(CLIN, dtype=str).set_index("sample_id")
num_cols = ["age", "bmi", "whr", "waist_circumference", "hip_circumference",
            "ffmass", "totfa", "p_adipon", "matsuda", "p_crp",
            "homair", "b_ghba1c", "p_ins0", "p_gl0", "p_ffa0",
            "s_hdlc", "s_ldlc", "s_totalc", "s_tottg"]
num_cols = [c for c in num_cols if c in clin.columns]
for c in num_cols:
    clin[c] = pd.to_numeric(clin[c], errors="coerce")

common = expr.index.intersection(clin.index)
expr, clin = expr.loc[common], clin.loc[common]
print(f"匹配样本: {len(common)}")
print(clin[["bmi", "matsuda", "homair", "p_ins0", "totfa", "p_adipon", "p_crp"]]
      .describe().loc[["count", "mean", "std"]].round(2).to_string())

# ---------- 4. 每个基因选最佳探针（均值最高） ----------
def best_probe(gene):
    ps = [p for p in gene2probes[gene] if p in expr.columns]
    if not ps:
        return None
    return max(ps, key=lambda p: expr[p].mean())

print("\n" + "=" * 78)
print("【核心检验 1】与代谢表型的 Spearman 相关（n=%d）" % len(common))
print("=" * 78)
traits = ["bmi", "totfa", "matsuda", "homair", "p_ins0", "p_adipon", "p_crp"]
print(f"{'gene':9s} " + " ".join(f"{t:>16s}" for t in traits))
res_corr = {}
for g in CORE:
    bp = best_probe(g)
    if bp is None:
        print(f"{g:9s} 无探针")
        continue
    res_corr[g] = {}
    cells = []
    for t in traits:
        m = expr[bp].notna() & clin[t].notna()
        r, p = stats.spearmanr(expr[bp][m], clin[t][m])
        res_corr[g][t] = (r, p, m.sum())
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        cells.append(f"r={r:+.3f}{star:<3s}")
    print(f"{g:9s} " + " ".join(f"{c:>16s}" for c in cells))

# ---------- 5. 肥胖 vs  lean ----------
print("\n" + "=" * 78)
print("【核心检验 2】肥胖 (BMI≥30) vs  lean (BMI<25) 表达差异")
print("=" * 78)
ob = clin[clin["bmi"] >= 30].index
ln = clin[clin["bmi"] < 25].index
print(f"肥胖 n={len(ob)}    lean n={len(ln)}")
print(f"{'gene':9s} {'probe':16s} {'mean_obese':>11s} {'mean_lean':>10s} {'log2FC':>8s} {'p':>12s}")
res_grp = {}
for g in CORE:
    bp = best_probe(g)
    if bp is None:
        continue
    a, b = expr.loc[ob, bp].dropna(), expr.loc[ln, bp].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    lfc = a.mean() - b.mean()
    res_grp[g] = (lfc, p, len(a), len(b), bp)
    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    print(f"{g:9s} {bp:16s} {a.mean():11.3f} {b.mean():10.3f} {lfc:+8.3f} {p:12.3e} {star}")

# ---------- 6. 阳性对照 ----------
print("\n" + "=" * 78)
print("【阳性对照】已知肥胖相关基因应有明确信号")
print("=" * 78)
for g in CONTROL:
    ps = [p for p, s in probe2gene.items() if s == g and p in expr.columns]
    if not ps:
        print(f"{g:9s} 无探针")
        continue
    bp = max(ps, key=lambda p: expr[p].mean())
    m = expr[bp].notna() & clin["bmi"].notna()
    r, p = stats.spearmanr(expr[bp][m], clin["bmi"][m])
    a, b = expr.loc[ob, bp].dropna(), expr.loc[ln, bp].dropna()
    _, pg = stats.mannwhitneyu(a, b, alternative="two-sided")
    print(f"{g:9s} vs BMI: r={r:+.3f} (p={p:.2e}) | obese-lean log2FC={a.mean()-b.mean():+.3f} (p={pg:.2e})")

# ---------- 7. 保存 ----------
out = os.path.join(BASE, "results")
os.makedirs(out, exist_ok=True)
expr.to_csv(os.path.join(out, "GSE70353_UFMylation_expr.csv"))
with open(os.path.join(out, "GSE70353_corr_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gene", "probe", "trait", "spearman_r", "p", "n"])
    for g, d in res_corr.items():
        for t, (r, p, n) in d.items():
            w.writerow([g, best_probe(g), t, f"{r:.4f}", f"{p:.3e}", n])
print("\n已保存:", out)
