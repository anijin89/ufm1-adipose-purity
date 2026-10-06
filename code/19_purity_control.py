"""
GSE174475 纯度混杂排除 + 关键关联的稳健性检验
核心问题：纯化脂肪细胞里"机器随 BMI 下降"是否可能由纯度差异驱动？
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
sid_map = {re.search(r"_(\d+)$", s).group(1): s for s in samples}
clin = clin.set_index("sid").reindex([re.search(r"_(\d+)$", s).group(1) for s in samples])
clin.index = samples
for c in ["age", "bmi", "homa-ir", "tg", "hdl", "ldl"]:
    clin[c] = pd.to_numeric(clin[c], errors="coerce")

bmi = clin["bmi"].values.astype(float)
homa = clin["homa-ir"].values.astype(float)
age = clin["age"].values.astype(float)
print(f"n={len(samples)}  BMI {np.nanmin(bmi):.1f}-{np.nanmax(bmi):.1f} (均值 {np.nanmean(bmi):.1f})")
print(f"HOMA-IR {np.nanmin(homa):.2f}-{np.nanmax(homa):.2f} (均值 {np.nanmean(homa):.2f})")
print(f"BMI vs HOMA-IR: r={stats.pearsonr(bmi, homa)[0]:.3f}, p={stats.pearsonr(bmi, homa)[1]:.2e}")

CORE = ["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"]
AXIS = ["UFM1", "UBA5", "UFC1", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3"]
PURITY = ["ADIPOQ", "PLIN1", "CIDEC", "SLC2A4", "LEP"]   # 脂肪细胞纯度
CONTAM = ["CD68", "PTPRC", "CD3E", "LYZ", "CD14", "CSF1R", "PECAM1", "CDH5", "PDGFRA", "COL1A1"]

def r(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 10 or np.std(x[m]) == 0:
        return np.nan, np.nan
    return stats.pearsonr(x[m], y[m])

print("\n" + "=" * 80)
print("第一步：纯度标记是否随 BMI 变化？（若变，则纯度是混杂）")
print("=" * 80)
print(f"{'基因':<10}{'vs BMI r':>10}{'p':>11}{'vs HOMA-IR r':>14}{'p':>11}")
for g in PURITY + CONTAM:
    if g not in expr.index:
        continue
    v = expr.loc[g].astype(float).values
    r1, p1 = r(v, bmi)
    r2, p2 = r(v, homa)
    print(f"{g:<10}{r1:>10.3f}{p1:>11.3f}{r2:>14.3f}{p2:>11.3f}")

# 纯度复合指标
pur = expr.loc[[g for g in PURITY if g in expr.index]].astype(float)
pur_z = pur.sub(pur.mean(axis=1), axis=0).div(pur.std(axis=1).replace(0, np.nan), axis=0)
purity_z = pur_z.mean(axis=0).values
con = expr.loc[[g for g in CONTAM if g in expr.index]].astype(float)
con_z = con.sub(con.mean(axis=1), axis=0).div(con.std(axis=1).replace(0, np.nan), axis=0)
contam_z = con_z.mean(axis=0).values
r1, p1 = r(purity_z, bmi); r2, p2 = r(contam_z, bmi)
print(f"\n脂肪细胞纯度复合 z vs BMI: r={r1:+.3f}, p={p1:.3f}")
print(f"污染标记复合 z   vs BMI: r={r2:+.3f}, p={p2:.3f}")
print("  -> 若二者 p>0.05，纯度不随 BMI 系统性改变，构成有效对照")

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

Z = expr.loc[AXIS].astype(float)
Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1).replace(0, np.nan), axis=0)
sc = {
    "UFM1_z": Z.loc["UFM1"].values,
    "machine_z": Z.loc[CORE].mean(axis=0).values,
}
sc["dysreg_z"] = sc["UFM1_z"] - sc["machine_z"]

print("\n" + "=" * 80)
print("第二步：校正纯度与污染后，核心关联是否存活？")
print("=" * 80)
print(f"{'特征':<12}{'vs BMI(粗)':>11}{'p':>10}{'| 校纯度+污染':>13}{'p':>10}{'| 校年龄':>10}{'p':>10}")
print("-" * 80)
for name, v in sc.items():
    r0, p0 = r(v, bmi)
    ra, pa, _ = partial_corr(v, bmi, [purity_z, contam_z])
    rb, pb, _ = partial_corr(v, bmi, [purity_z, contam_z, age])
    print(f"{name:<12}{r0:>11.3f}{p0:>10.2e}{ra:>13.3f}{pa:>10.2e}{rb:>10.3f}{pb:>10.2e}")

print("\n" + "=" * 80)
print("第三步：与 IR/IS 分组的关系（校正 BMI 后是否仍成立）")
print("=" * 80)
group = np.array(["IR" if "_IR_" in s else "IS" for s in samples])
is_ir = (group == "IR").astype(float)
print(f"{'特征':<12}{'IR vs IS':>10}{'p':>10}{'| 校正BMI':>10}{'p':>10}{'| 校BMI+年龄+纯度':>18}{'p':>10}")
for name, v in sc.items():
    r0, p0 = r(v, is_ir)
    ra, pa, _ = partial_corr(v, is_ir, [bmi])
    rb, pb, _ = partial_corr(v, is_ir, [bmi, age, purity_z, contam_z])
    print(f"{name:<12}{r0:>10.3f}{p0:>10.2e}{ra:>10.3f}{pa:>10.2e}{rb:>18.3f}{pb:>10.2e}")

print("\n" + "=" * 80)
print("第四步：UFMylation 轴 vs ER 应激 / 脂生成（纯化脂肪细胞内，无混杂）")
print("=" * 80)
PARTNERS = ["PLIN1", "FASN", "PPARG", "ADIPOQ", "HSPA5", "ATF6", "EIF2AK3",
            "ERN1", "XBP1", "DDIT3", "SEL1L", "SREBF1", "SCD", "LIPE", "PNPLA2"]
print(f"{'基因':<10}{'vs UFM1 r':>11}{'p':>10}{'vs machine r':>13}{'p':>10}{'vs dysreg r':>13}{'p':>10}")
print("-" * 80)
for g in PARTNERS:
    if g not in expr.index:
        continue
    v = expr.loc[g].astype(float).values
    a, pa = r(v, sc["UFM1_z"]); b, pb = r(v, sc["machine_z"]); c, pc = r(v, sc["dysreg_z"])
    print(f"{g:<10}{a:>11.3f}{pa:>10.3f}{b:>13.3f}{pb:>10.3f}{c:>13.3f}{pc:>10.3f}")

out = pd.DataFrame({"sample": samples, "group": group, "bmi": bmi, "homa_ir": homa,
                    "age": age, "purity_z": purity_z, "contam_z": contam_z, **sc})
out.to_csv("results/GSE174475_scores_purity.csv", index=False)
print("\n已保存 results/GSE174475_scores_purity.csv")
