"""
GSE174475 敏感性分析：仅用"污染标记"校正（不含脂肪细胞功能基因，避免过度校正）
脂肪细胞不表达 CD68/PTPRC/CD3E/LYZ/CD14/CSF1R/PDGFRA/COL1A1，
因此这些基因的信号只可能来自污染，用它们校正是无偏的。
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
        rows[vals[0].split(":")[0].strip().lower()] = [
            v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
    if line.startswith("!Sample_title"):
        titles = [x.strip().strip('"') for x in line.split("\t")[1:]]
clin = pd.DataFrame(rows)
clin["sid"] = [re.search(r"(\d+)\s*$", t).group(1) for t in titles]
clin = clin.set_index("sid").reindex([re.search(r"_(\d+)$", s).group(1) for s in samples])
clin.index = samples
for c in ["age", "bmi", "homa-ir"]:
    clin[c] = pd.to_numeric(clin[c], errors="coerce")
bmi = clin["bmi"].values.astype(float)
homa = clin["homa-ir"].values.astype(float)
age = clin["age"].values.astype(float)

# 纯污染标记：脂肪细胞不表达
CONTAM = ["CD68", "PTPRC", "CD3E", "LYZ", "CD14", "CSF1R", "ITGAM", "MRC1",
          "PDGFRA", "COL1A1", "THY1", "CDH5"]
CONTAM = [g for g in CONTAM if g in expr.index]
print("污染标记（脂肪细胞不表达）:", ", ".join(CONTAM))

con = expr.loc[CONTAM].astype(float)
con_z = con.sub(con.mean(axis=1), axis=0).div(con.std(axis=1).replace(0, np.nan), axis=0)
contam_z = con_z.mean(axis=0).values
# 第一主成分也可作为污染轴
cz = (con_z.T - con_z.T.mean(axis=0)).values
u, s, vt = np.linalg.svd(cz, full_matrices=False)
contam_pc1 = u[:, 0] * s[0]
if np.corrcoef(contam_pc1, contam_z)[0, 1] < 0:
    contam_pc1 = -contam_pc1
print(f"污染 PC1 解释方差 {s[0]**2/np.sum(s**2)*100:.1f}%，与污染均值 z 相关 "
      f"{np.corrcoef(contam_pc1, contam_z)[0,1]:.3f}")

CORE = ["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"]
AXIS = ["UFM1", "UBA5", "UFC1", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]
Z = expr.loc[AXIS].astype(float)
Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1).replace(0, np.nan), axis=0)
sc = {"UFM1_z": Z.loc["UFM1"].values, "machine_z": Z.loc[CORE].mean(axis=0).values}
sc["dysreg_z"] = sc["UFM1_z"] - sc["machine_z"]

def pc(x, y, covs):
    df = pd.DataFrame(np.column_stack([x, y] + list(covs)),
                      columns=["x", "y"] + [f"c{i}" for i in range(len(covs))]).dropna()
    if len(df) < 12:
        return np.nan, np.nan, len(df)
    X = np.column_stack([np.ones(len(df))] + [df[f"c{i}"].values for i in range(len(covs))])
    rx = df["x"].values - X @ np.linalg.lstsq(X, df["x"].values, rcond=None)[0]
    ry = df["y"].values - X @ np.linalg.lstsq(X, df["y"].values, rcond=None)[0]
    rr, pp = stats.pearsonr(rx, ry)
    return rr, pp, len(df)

def r0(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    return stats.pearsonr(x[m], y[m])

print("\n" + "=" * 84)
print("仅校正污染（不过度校正脂肪细胞功能基因）")
print("=" * 84)
print(f"{'特征':<12}{'vs BMI 粗':>11}{'p':>10}{'| 校污染PC1':>12}{'p':>10}{'| 校污染+年龄':>13}{'p':>10}")
print("-" * 84)
res = {}
for name, v in sc.items():
    a, pa = r0(v, bmi)
    b, pb, _ = pc(v, bmi, [contam_pc1])
    c, pcc, _ = pc(v, bmi, [contam_pc1, age])
    res[name] = dict(r_bmi=a, p_bmi=pa, r_adj=b, p_adj=pb, r_adj2=c, p_adj2=pcc)
    print(f"{name:<12}{a:>11.3f}{pa:>10.2e}{b:>12.3f}{pb:>10.2e}{c:>13.3f}{pcc:>10.2e}")

print("\n" + "=" * 84)
print("ER 应激偶联是否存活？（UFM1 与 UPR 基因的关联，校正污染后）")
print("=" * 84)
UPR = ["EIF2AK3", "ATF6", "ERN1", "HSPA5", "SEL1L", "XBP1", "DDIT3", "ATF4", "EIF2S1"]
print(f"{'基因':<10}{'vs UFM1 粗':>12}{'p':>10}{'| 校污染':>10}{'p':>10}{'| vs dysreg 粗':>14}{'校污染':>10}{'p':>10}")
print("-" * 84)
up_rows = []
for g in UPR:
    if g not in expr.index:
        continue
    v = expr.loc[g].astype(float).values
    a, pa = r0(v, sc["UFM1_z"])
    b, pb, _ = pc(v, sc["UFM1_z"], [contam_pc1])
    c, pcc_ = r0(v, sc["dysreg_z"])
    d, pd_, _ = pc(v, sc["dysreg_z"], [contam_pc1])
    up_rows.append(dict(gene=g, r_ufm1=a, p_ufm1=pa, r_ufm1_adj=b, p_ufm1_adj=pb,
                        r_dys=c, p_dys=pcc_, r_dys_adj=d, p_dys_adj=pd_))
    print(f"{g:<10}{a:>12.3f}{pa:>10.3f}{b:>10.3f}{pb:>10.3f}{c:>14.3f}{d:>10.3f}{pd_:>10.3f}")

print("\n" + "=" * 84)
print("结论判定")
print("=" * 84)
uf = res["UFM1_z"]; ma = res["machine_z"]; dy = res["dysreg_z"]
def verdict(d, name):
    if d["p_bmi"] < 0.05 and d["p_adj"] < 0.05:
        return f"  {name}: 与 BMI 的关联在校正污染后仍然显著 -> 稳健"
    if d["p_bmi"] < 0.05 and d["p_adj"] >= 0.05:
        return f"  {name}: 粗关联显著但校正污染后消失 -> 由纯度差异驱动，不可信"
    return f"  {name}: 始终不显著 -> 无证据"
print(verdict(uf, "UFM1   "))
print(verdict(ma, "机器分  "))
print(verdict(dy, "失衡指数"))

surv = [x for x in up_rows if x["p_ufm1_adj"] < 0.05]
print(f"\n校正污染后仍与 UFM1 显著相关的 UPR 基因: "
      + (", ".join(f"{x['gene']}(r={x['r_ufm1_adj']:+.2f})" for x in surv) if surv else "无"))

pd.DataFrame(up_rows).to_csv("results/GSE174475_UPR_coupling.csv", index=False)
pd.DataFrame(res).T.reset_index().rename(columns={"index": "feature"}).to_csv(
    "results/GSE174475_contam_adjusted.csv", index=False)
print("已保存 results/GSE174475_UPR_coupling.csv, results/GSE174475_contam_adjusted.csv")
