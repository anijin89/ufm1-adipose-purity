"""
排除技术混杂：UFM1 与 UPR 基因的偶联是否只是"整体翻译机器丰度/文库质量"的代理？
做法：用非翻译类管家基因的 PC1 作为技术轴，校正后再看关联是否存活。
"""
import gzip, re
import numpy as np, pandas as pd, urllib.request
from scipy import stats

with gzip.open("data/GSE174475_expr.txt.gz", "rt") as f:
    expr = pd.read_csv(f, sep="\t", index_col=0)
expr = expr[~expr.index.duplicated(keep="first")]
samples = list(expr.columns)

HK = ["ACTB", "GAPDH", "B2M", "TUBB", "HPRT1", "TBP", "YWHAZ", "SDHA", "UBC",
      "PGK1", "PPIA", "GUSB", "TFRC", "VIM", "YBX1", "HNRNPA1", "NPM1", "CFL1"]
HK = [g for g in HK if g in expr.index]
print("非翻译类管家基因:", ", ".join(HK))

hk = expr.loc[HK].astype(float)
hkz = hk.sub(hk.mean(axis=1), axis=0).div(hk.std(axis=1).replace(0, np.nan), axis=0)
X = hkz.T.values - hkz.T.values.mean(axis=0)
u, s, vt = np.linalg.svd(X, full_matrices=False)
tech_pc1 = u[:, 0] * s[0]
print(f"技术轴 PC1 解释方差 {s[0]**2/np.sum(s**2)*100:.1f}%")

# 检测基因数（作为文库复杂度代理）
detect = (expr.astype(float) > 1).sum(axis=0).values.astype(float)
print(f"检出基因数范围: {detect.min():.0f}-{detect.max():.0f}")

CONTAM = [g for g in ["CD68", "PTPRC", "CD3E", "LYZ", "CD14", "CSF1R", "ITGAM",
                      "MRC1", "PDGFRA", "COL1A1", "THY1", "CDH5"] if g in expr.index]
con = expr.loc[CONTAM].astype(float)
conz = con.sub(con.mean(axis=1), axis=0).div(con.std(axis=1).replace(0, np.nan), axis=0)
cz = conz.T.values - conz.T.values.mean(axis=0)
u2, s2, _ = np.linalg.svd(cz, full_matrices=False)
contam_pc1 = u2[:, 0] * s2[0]

ufm1 = expr.loc["UFM1"].astype(float).values
ufm1_z = (ufm1 - np.nanmean(ufm1)) / np.nanstd(ufm1)

def pc(x, y, covs):
    df = pd.DataFrame(np.column_stack([x, y] + list(covs)),
                      columns=["x", "y"] + [f"c{i}" for i in range(len(covs))]).dropna()
    if len(df) < 12:
        return np.nan, np.nan
    Xm = np.column_stack([np.ones(len(df))] + [df[f"c{i}"].values for i in range(len(covs))])
    rx = df["x"].values - Xm @ np.linalg.lstsq(Xm, df["x"].values, rcond=None)[0]
    ry = df["y"].values - Xm @ np.linalg.lstsq(Xm, df["y"].values, rcond=None)[0]
    return stats.pearsonr(rx, ry)

def r0(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    return stats.pearsonr(x[m], y[m])

print("\n" + "=" * 88)
print("UFM1 与 UPR 基因的偶联：逐步施加严格校正")
print("=" * 88)
print(f"{'基因':<9}{'粗 r':>8}{'p':>9}{'| +技术轴':>10}{'p':>9}{'| +技术+污染':>13}{'p':>9}{'| +检出数':>10}{'p':>9}")
print("-" * 88)
rows = []
for g in ["EIF2S1", "EIF2AK3", "ATF6", "SEL1L", "ERN1", "HSPA5", "DDIT3", "ATF4", "XBP1"]:
    if g not in expr.index:
        continue
    v = expr.loc[g].astype(float).values
    a, pa = r0(v, ufm1_z)
    b, pb = pc(v, ufm1_z, [tech_pc1])
    c, pcc = pc(v, ufm1_z, [tech_pc1, contam_pc1])
    d, pd_ = pc(v, ufm1_z, [tech_pc1, contam_pc1, detect])
    rows.append(dict(gene=g, r=a, p=pa, r_tech=b, p_tech=pb,
                     r_tech_con=c, p_tech_con=pcc, r_full=d, p_full=pd_))
    print(f"{g:<9}{a:>8.3f}{pa:>9.3f}{b:>10.3f}{pb:>9.3f}{c:>13.3f}{pcc:>9.3f}{d:>10.3f}{pd_:>9.3f}")

print("\n" + "=" * 88)
print("判定")
print("=" * 88)
surv = [x for x in rows if x["p_full"] < 0.05]
if surv:
    print("校正技术轴 + 污染 + 检出基因数后仍显著:")
    for x in surv:
        print(f"  {x['gene']:<9} r={x['r_full']:+.3f}, p={x['p_full']:.4f}")
    if all(x["gene"] in ("EIF2S1", "EIF2AK3") for x in surv):
        print("  -> 仅剩 eIF2α 轴，需注意这可能是翻译机器丰度的共同调控")
    else:
        print("  -> 偶联稳健，非技术假象")
else:
    print("  -> 全部被技术轴解释，偶联不成立，需放弃此条主线")

pd.DataFrame(rows).to_csv("results/GSE174475_technical_check.csv", index=False)
print("\n已保存 results/GSE174475_technical_check.csv")
