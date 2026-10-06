"""
M7 敏感性：纯度校正是否依赖"标记面板 PC1/复合 z"这一特定纯度指征方式？
逐一把纯度替换为单个成熟脂肪细胞标记(ADIPOQ/PLIN1/CIDEC/SLC2A4)，
看 UFM1_z、machine_z 与 BMI 的粗关联校正后是否仍消失。
真实计算，产出 results/GSE174475_purity_sensitivity_M7.csv
"""
import gzip, re
import numpy as np, pandas as pd, urllib.request
from scipy import stats

with gzip.open("data/GSE174475_expr.txt.gz", "rt") as f:
    expr = pd.read_csv(f, sep="\t", index_col=0)
expr = expr[~expr.index.duplicated(keep="first")]
samples = list(expr.columns)

with urllib.request.urlopen(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE174nnn/GSE174475/matrix/"
        "GSE174475_series_matrix.txt.gz", timeout=120) as r:
    raw = gzip.decompress(r.read()).decode("utf-8", errors="ignore")

rows = {}
for line in raw.split("\n"):
    if line.startswith("!Sample_characteristics_ch1"):
        vals = [x.strip().strip('"') for x in line.split("\t")[1:]]
        key = vals[0].split(":")[0].strip().lower()
        rows[key] = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
    if line.startswith("!Sample_title"):
        titles = [x.strip().strip('"') for x in line.split("\t")[1:]]
clin = pd.DataFrame(rows)
clin["sid"] = [re.search(r"(\d+)\s*$", t).group(1) for t in titles]
clin = clin.set_index("sid").reindex([re.search(r"_(\d+)$", s).group(1) for s in samples])
clin.index = samples
clin["bmi"] = pd.to_numeric(clin["bmi"], errors="coerce")
bmi = clin["bmi"].values.astype(float)

CORE = ["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"]
AXIS = ["UFM1", "UBA5", "UFC1", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]
PURITY = ["ADIPOQ", "PLIN1", "CIDEC", "SLC2A4", "LEP"]
CONTAM = ["CD68", "PTPRC", "CD3E", "LYZ", "CD14", "CSF1R", "PECAM1", "CDH5", "PDGFRA", "COL1A1"]

def pear(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 10 or np.std(x[m]) == 0:
        return np.nan, np.nan
    return stats.pearsonr(x[m], y[m])

def partial_corr(x, y, covs):
    df = pd.DataFrame(np.column_stack([x, y] + list(covs)),
                      columns=["x", "y"] + [f"c{i}" for i in range(len(covs))])
    df = df.dropna()
    if len(df) < 12:
        return np.nan, np.nan, len(df)
    X = np.column_stack([np.ones(len(df))] + [df[f"c{i}"].values for i in range(len(covs))])
    rx = df["x"].values - X @ np.linalg.lstsq(X, df["x"].values, rcond=None)[0]
    ry = df["y"].values - X @ np.linalg.lstsq(X, df["y"].values, rcond=None)[0]
    rr, pp = stats.pearsonr(rx, ry)
    return rr, pp, len(df)

# 轴评分（与主分析一致）
Z = expr.loc[AXIS].astype(float)
Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1).replace(0, np.nan), axis=0)
ufm1_z = Z.loc["UFM1"].values
machine_z = Z.loc[CORE].mean(axis=0).values
dysreg_z = ufm1_z - machine_z

# 复合纯度 z（原主分析口径）
pur = expr.loc[[g for g in PURITY if g in expr.index]].astype(float)
pur_z = pur.sub(pur.mean(axis=1), axis=0).div(pur.std(axis=1).replace(0, np.nan), axis=0)
purity_composite = pur_z.mean(axis=0).values
con = expr.loc[[g for g in CONTAM if g in expr.index]].astype(float)
con_z = con.sub(con.mean(axis=1), axis=0).div(con.std(axis=1).replace(0, np.nan), axis=0)
contam_composite = con_z.mean(axis=0).values

print("粗关联（未校正）:")
for nm, v in [("UFM1_z", ufm1_z), ("machine_z", machine_z), ("dysreg_z", dysreg_z)]:
    rr, pp = pear(v, bmi)
    print(f"  {nm:<10} r={rr:+.3f} p={pp:.3f}")

# 单一标记口径：每行 = 用该单标记(±污染复合)作为纯度协变量后的偏相关
records = []
single_markers = [g for g in ["ADIPOQ", "PLIN1", "CIDEC", "SLC2A4"] if g in expr.index]
features = {"UFM1_z": ufm1_z, "machine_z": machine_z, "dysreg_z": dysreg_z}

for fname, fv in features.items():
    r0, p0 = pear(fv, bmi)
    # 原口径：复合纯度 + 复合污染
    ra, pa, na = partial_corr(fv, bmi, [purity_composite, contam_composite])
    rec = {"feature": fname, "r_crude": r0, "p_crude": p0,
           "r_adj_composite": ra, "p_adj_composite": pa, "n_composite": na}
    # 逐一单标记（配复合污染，保持与主分析同等污染校正，只换纯度指征）
    for g in single_markers:
        mv = expr.loc[g].astype(float).values
        rr, pp, nn = partial_corr(fv, bmi, [mv, contam_composite])
        rec[f"r_adj_{g}"] = rr
        rec[f"p_adj_{g}"] = pp
    records.append(rec)

out = pd.DataFrame(records)
out.to_csv("results/GSE174475_purity_sensitivity_M7.csv", index=False)
pd.set_option("display.width", 200, "display.max_columns", 50)
print("\nM7 敏感性汇总表:")
print(out.round(3).to_string(index=False))

# 报告单标记纯度的偏相关范围（UFM1_z 与 machine_z）
print("\n单标记口径下 partial r 范围:")
for fname in features:
    vals = [out.loc[out.feature == fname, f"r_adj_{g}"].values[0] for g in single_markers]
    ps = [out.loc[out.feature == fname, f"p_adj_{g}"].values[0] for g in single_markers]
    print(f"  {fname:<10} r ∈ [{min(vals):+.3f}, {max(vals):+.3f}], 全部 p = {[round(x,3) for x in ps]}")
