"""
GSE174475 — 纯化成熟脂肪细胞 RNA-seq（Rodbell 法分离）
决定性检验：在无细胞组成混杂的条件下，UFMylation 轴与胰岛素抵抗的关系，
以及"解耦"假说 vs "全轴代偿性上调"假说的判定。
"""
import gzip
import re
import numpy as np
import pandas as pd
from scipy import stats

DATA = "data/GSE174475_expr.txt.gz"
META_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE174nnn/GSE174475/matrix/GSE174475_series_matrix.txt.gz"
OUT = "results/"

# ---------- 读取表达矩阵 ----------
with gzip.open(DATA, "rt") as f:
    expr = pd.read_csv(f, sep="\t", index_col=0)
print(f"表达矩阵: {expr.shape[0]} 基因 x {expr.shape[1]} 样本")
expr = expr[~expr.index.duplicated(keep="first")]
print(f"去重后: {expr.shape[0]} 基因")

samples = list(expr.columns)
group = np.array(["IR" if "_IR_" in s else "IS" for s in samples])
print(f"分组: IR={int((group=='IR').sum())}, IS={int((group=='IS').sum())}")

# ---------- 读取临床元数据 ----------
import urllib.request
meta = {}
with urllib.request.urlopen(META_URL, timeout=120) as r:
    raw = gzip.decompress(r.read()).decode("utf-8", errors="ignore")

def parse_field(tag, cast=float):
    for line in raw.split("\n"):
        if line.startswith(tag):
            vals = [x.strip().strip('"') for x in line.split("\t")[1:]]
            out = []
            for v in vals:
                m = re.search(r":\s*(.+)$", v)
                v = m.group(1) if m else v
                try:
                    out.append(cast(v))
                except Exception:
                    out.append(np.nan)
            return np.array(out)
    return None

titles = None
for line in raw.split("\n"):
    if line.startswith("!Sample_title"):
        titles = [x.strip().strip('"') for x in line.split("\t")[1:]]
        break

clin = pd.DataFrame({"gsm_title": titles})
for tag, name in [("!Sample_characteristics_ch1", None)]:
    pass

# 逐字段解析（characteristics 有多行）
idx = 0
for line in raw.split("\n"):
    if line.startswith("!Sample_characteristics_ch1"):
        vals = [x.strip().strip('"') for x in line.split("\t")[1:]]
        key = vals[0].split(":")[0].strip().lower()
        vals = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
        clin[key] = vals

for c in ["age", "bmi", "homa-ir", "hdl", "ldl", "tg"]:
    if c in clin.columns:
        clin[c] = pd.to_numeric(clin[c], errors="coerce")
    else:
        clin[c] = np.nan

# 用 title 里的编号匹配表达矩阵列名
def sid(t):
    m = re.search(r"(\d+)\s*$", t)
    return m.group(1) if m else None

clin["sid"] = clin["gsm_title"].map(sid)
expr_sid = pd.Series([re.search(r"_(\d+)$", s).group(1) for s in samples], index=samples)
clin = clin.set_index("sid")
clin = clin.reindex(expr_sid.values)
clin.index = samples

print("\n临床变量可用性:")
for c in ["age", "bmi", "homa-ir", "hdl", "ldl", "tg"]:
    print(f"  {c}: 非缺失 {clin[c].notna().sum()}/{len(clin)}  "
          f"均值={clin[c].mean():.2f}" if clin[c].notna().sum() > 0 else f"  {c}: 缺失")

# ---------- 纯度验证 ----------
print("\n" + "=" * 78)
print("纯度验证：纯化脂肪细胞中应几乎无非脂肪细胞标记")
print("=" * 78)
ADIPO = ["ADIPOQ", "PLIN1", "FASN", "PPARG", "LEP", "CIDEC", "SLC2A4"]
IMMUNE = ["CD68", "CD14", "CSF1R", "ITGAM", "PTPRC", "CD3E", "LYZ", "MRC1"]
ENDOTH = ["PECAM1", "CDH5", "VWF"]
ASPC = ["PDGFRA", "PDGFRB", "THY1", "CD34"]
STROM = ["COL1A1", "COL3A1", "ACTA2"]

for label, genes in [("脂肪细胞", ADIPO), ("免疫", IMMUNE), ("内皮", ENDOTH),
                     ("前体", ASPC), ("基质/胶原", STROM)]:
    vals = []
    for g in genes:
        if g in expr.index:
            vals.append((g, expr.loc[g].mean()))
    if vals:
        s = "  ".join(f"{g}={v:.2f}" for g, v in vals)
        print(f"  {label:<8} {s}")

# ---------- UFMylation 轴基因定位 ----------
UFM_CORE = {
    "UFM1": ["UFM1"],
    "UBA5": ["UBA5"],
    "UFC1": ["UFC1"],
    "UFL1": ["UFL1", "KIAA0776", "RCAD", "NLBP", "MAXER", "RWFPL2"],
    "UFSP1": ["UFSP1"],
    "UFSP2": ["UFSP2"],
    "DDRGK1": ["DDRGK1", "UFBP1", "C20orf116"],
    "CDK5RAP3": ["CDK5RAP3"],
}
gene_row = {}
for std, aliases in UFM_CORE.items():
    hit = None
    for a in aliases:
        if a in expr.index:
            hit = a
            break
    if hit is None:
        pat = aliases[0]
        cand = [g for g in expr.index if g.upper().startswith(pat[:4])]
        if cand:
            hit = cand[0]
    if hit:
        gene_row[std] = hit
print("\nUFMylation 轴基因定位:")
for k, v in gene_row.items():
    print(f"  {k:<10} -> {v}")

missing = set(UFM_CORE) - set(gene_row)
if missing:
    print(f"  未找到: {missing}")

# ---------- 分组检验 IR vs IS ----------
print("\n" + "=" * 78)
print("纯化脂肪细胞：胰岛素抵抗(IR) vs 敏感(IS)")
print("=" * 78)
rows = []
for std, rowname in gene_row.items():
    x = expr.loc[rowname].astype(float)
    a = x[group == "IR"].values
    b = x[group == "IS"].values
    t, pt = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    d = (a.mean() - b.mean()) / sd if sd > 0 else np.nan
    rows.append(dict(gene=std, mean_IR=a.mean(), mean_IS=b.mean(),
                     log2FC=a.mean()-b.mean(), p_t=pt, p_mw=pu, cohens_d=d))
res = pd.DataFrame(rows).sort_values("p_t")

ER = ["PLIN1", "ATF6", "HSPA5", "EIF2AK3", "ERN1", "XBP1", "DDIT3", "SEL1L", "FASN", "PPARG", "ADIPOQ"]
for g in ER:
    if g in expr.index:
        x = expr.loc[g].astype(float)
        a, b = x[group == "IR"].values, x[group == "IS"].values
        t, pt = stats.ttest_ind(a, b, equal_var=False)
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
        sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
        res.loc[len(res)] = dict(gene=g + " *", mean_IR=a.mean(), mean_IS=b.mean(),
                                 log2FC=a.mean()-b.mean(), p_t=pt, p_mw=pu,
                                 cohens_d=(a.mean()-b.mean())/sd if sd > 0 else np.nan)

pd.set_option("display.width", 200)
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------- 连续型关联 ----------
def partial_corr(x, y, covs):
    df = pd.DataFrame({"x": x, "y": y})
    for i, c in enumerate(covs):
        df[f"c{i}"] = c
    df = df.dropna()
    if len(df) < 12:
        return np.nan, np.nan, len(df)
    X = np.column_stack([np.ones(len(df))] + [df[f"c{i}"].values for i in range(len(covs))])
    rx = df["x"].values - X @ np.linalg.lstsq(X, df["x"].values, rcond=None)[0]
    ry = df["y"].values - X @ np.linalg.lstsq(X, df["y"].values, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(df)

print("\n" + "=" * 78)
print("连续型关联（纯化脂肪细胞，细胞组成混杂已被设计消除）")
print("=" * 78)

homa = clin["homa-ir"].values.astype(float)
bmi = clin["bmi"].values.astype(float)
age = clin["age"].values.astype(float)

# 复合评分（z 分数，队列内）
core_for_score = [g for g in ["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"] if g in gene_row]
Z = expr.loc[[gene_row[g] for g in gene_row]].astype(float)
Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1).replace(0, np.nan), axis=0)
scores = {
    "UFM1_z": Z.loc[gene_row["UFM1"]].values,
    "machine_z": Z.loc[[gene_row[g] for g in core_for_score]].mean(axis=0).values,
}
scores["dysreg_z"] = scores["UFM1_z"] - scores["machine_z"]

targets = dict(scores)
for g in gene_row:
    targets[g] = expr.loc[gene_row[g]].astype(float).values

print(f"{'特征':<12}{'HOMA-IR r':>12}{'p':>10}{'| 校正BMI+年龄':>14}{'p':>10}{'| BMI r':>10}{'p':>10}")
print("-" * 88)
outcome_rows = []
for name, v in targets.items():
    r1, p1, n1 = partial_corr(v, homa, [])
    r2, p2, n2 = partial_corr(v, homa, [bmi, age])
    r3, p3, n3 = partial_corr(v, bmi, [])
    outcome_rows.append(dict(feature=name, r_homa=r1, p_homa=p1,
                             r_homa_adj=r2, p_homa_adj=p2, r_bmi=r3, p_bmi=p3, n=n1))
    print(f"{name:<12}{r1:>12.3f}{p1:>10.2e}{r2:>14.3f}{p2:>10.2e}{r3:>10.3f}{p3:>10.2e}")

pd.DataFrame(outcome_rows).to_csv(OUT + "GSE174475_purified_assoc.csv", index=False)

# ---------- 假说判定 ----------
print("\n" + "=" * 78)
print("假说判定")
print("=" * 78)
axis_up = {g: res.loc[res.gene == g, "log2FC"].iloc[0] for g in gene_row
           if (res.gene == g).any()}
mach_up = [axis_up[g] for g in core_for_score if g in axis_up]
print(f"UFM1   IR vs IS log2FC = {axis_up.get('UFM1', float('nan')):+.3f}")
print(f"机器基因 log2FC 均值 = {np.mean(mach_up):+.3f}  个体: "
      + ", ".join(f"{g}={v:+.3f}" for g, v in axis_up.items() if g in core_for_score))
u = axis_up.get("UFM1", np.nan)
m = np.mean(mach_up) if mach_up else np.nan
if abs(u) < 0.05 and abs(m) < 0.05:
    print("  -> 两者均无变化：轴与胰岛素抵抗无关，需重新定位")
elif u > 0 and m > 0:
    print("  -> 【全轴同步上调】解耦假说破产，应改写为'脂肪细胞内代偿性上调'")
elif u > 0 and abs(m) < 0.05:
    print("  -> 【解耦成立】UFM1 上调而机器不变，保留解耦叙事")
elif u > 0 and m < 0:
    print("  -> 【强解耦】UFM1 上调 + 机器下调，解耦假说得到最强支持")
else:
    print("  -> 方向与预期不符，需据实改写")

res.to_csv(OUT + "GSE174475_purified_IRvsIS.csv", index=False)
print("\n已保存: results/GSE174475_purified_IRvsIS.csv, results/GSE174475_purified_assoc.csv")
