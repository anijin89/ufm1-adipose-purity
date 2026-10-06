"""
GSE245948 偶联的技术混杂排除（与 21_technical_check.py 对 GSE174475 的做法对应）。
逐步校正：污染 PC1 / 管家基因技术轴 PC1 / 二者 / 再叠加检出基因数。
产出 results/GSE245948_full_adjust_check.csv（Panel A of Supplementary Table S10）。
"""
import gzip
import numpy as np, pandas as pd, urllib.request, urllib.parse
from scipy import stats

CPM = "data/GSE245948_cpm.csv.gz"
with gzip.open(CPM, "rt") as f:
    cpm = pd.read_csv(f, index_col=0)
cpm.index = [str(i).split(".")[0] for i in cpm.index]
cpm = cpm[~cpm.index.duplicated(keep="first")]

PARTNERS = ["EIF2S1", "EIF2AK3", "ATF6", "SEL1L", "ERN1", "HSPA5", "XBP1", "DDIT3", "ATF4", "PLIN1"]
CONTAM = ["CD68", "PTPRC", "CD3E", "LYZ", "CD14", "CSF1R", "ITGAM", "MRC1",
          "PDGFRA", "COL1A1", "THY1", "CDH5"]
HK = ["ACTB", "GAPDH", "B2M", "TUBB", "HPRT1", "TBP", "YWHAZ", "SDHA", "UBC",
      "PGK1", "PPIA", "GUSB", "TFRC", "VIM", "YBX1", "HNRNPA1", "NPM1", "CFL1"]
SYMBOLS = list(dict.fromkeys(["UFM1"] + PARTNERS + CONTAM + HK))

CACHE = "results/GSE245948_sym2ens.json"

# 硬编码 human Ensembl ID（境内 mygene 不稳定，改为静态表保证可复现）
# 每个 ID 均经 Ensembl xrefs/symbol 与 mygene(species=human) 双重核实；PLIN1/ITGAM/MRC1/THY1 曾因凭记忆出错已修正
SYMBOL2ENSG = {
    "UFM1": "ENSG00000120686", "PLIN1": "ENSG00000166819", "EIF2S1": "ENSG00000134001", "EIF2AK3": "ENSG00000172071",
    "ATF6": "ENSG00000118217", "SEL1L": "ENSG00000071537", "ERN1": "ENSG00000178607",
    "HSPA5": "ENSG00000044574", "XBP1": "ENSG00000100219", "DDIT3": "ENSG00000175197",
    "ATF4": "ENSG00000128272", "CD68": "ENSG00000129226", "PTPRC": "ENSG00000081237",
    "LYZ": "ENSG00000090382", "CD14": "ENSG00000170458", "CSF1R": "ENSG00000182578",
    "PDGFRA": "ENSG00000134853", "COL1A1": "ENSG00000108821", "PECAM1": "ENSG00000261371",
    "CDH5": "ENSG00000179776", "CD3E": "ENSG00000198851", "ITGAM": "ENSG00000169896",
    "MRC1": "ENSG00000260314", "THY1": "ENSG00000154096", "ACTB": "ENSG00000075624",
    "GAPDH": "ENSG00000111640", "B2M": "ENSG00000166710", "TUBB": "ENSG00000232575",
    "HPRT1": "ENSG00000165704", "TBP": "ENSG00000112592", "YWHAZ": "ENSG00000164924",
    "SDHA": "ENSG00000073578", "UBC": "ENSG00000150991", "PGK1": "ENSG00000102144",
    "PPIA": "ENSG00000196262", "GUSB": "ENSG00000169919", "TFRC": "ENSG00000072274",
    "VIM": "ENSG00000026025", "YBX1": "ENSG00000065978", "HNRNPA1": "ENSG00000135486",
    "NPM1": "ENSG00000181163", "CFL1": "ENSG00000172757",
}

sym2ens = SYMBOL2ENSG
print(f"静态映射 {len([s for s in SYMBOLS if s.upper() in sym2ens])}/{len(SYMBOLS)} symbol->ENSG")
missing = [s for s in SYMBOLS if s.upper() not in sym2ens]
if missing:
    print("未映射:", missing)

avail = {s: sym2ens[s.upper()] for s in SYMBOLS
         if s.upper() in sym2ens and sym2ens[s.upper()] in cpm.index}
E = pd.DataFrame({s: cpm.loc[e].astype(float) for s, e in avail.items()})
L = np.log2(E.clip(lower=0.01))                       # samples x genes
ufm1_z = (L["UFM1"] - L["UFM1"].mean()) / L["UFM1"].std(ddof=0)

def pc1(series_df):
    cols = [c for c in series_df.columns if c in L.columns]
    M = L[cols]
    Mz = (M - M.mean()) / M.std(ddof=0).replace(0, np.nan)
    X = Mz.values - Mz.values.mean(axis=0)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return pd.Series(u[:, 0] * s[0], index=Mz.index)

contam_pc1 = pc1(pd.DataFrame({c: L[c] for c in CONTAM if c in L.columns}))
tech_pc1 = pc1(pd.DataFrame({c: L[c] for c in HK if c in L.columns}))
detect = (E > 1).sum(axis=1).astype(float)            # 每样本检出基因数
print(f"技术轴可用管家基因: {[c for c in HK if c in L.columns]}")
print(f"污染标记可用: {[c for c in CONTAM if c in L.columns]}")

def pcorr(x, y, covs):
    cov_series = [c.rename(f"c{i}") if hasattr(c, "rename") else pd.Series(c, name=f"c{i}")
                  for i, c in enumerate(covs)]
    df = pd.concat([x.rename("x"), y.rename("y")] + cov_series, axis=1).dropna()
    if len(df) < 15:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(df))] + [df[f"c{i}"].values for i in range(len(covs))])
    rx = df["x"].values - X @ np.linalg.lstsq(X, df["x"].values, rcond=None)[0]
    ry = df["y"].values - X @ np.linalg.lstsq(X, df["y"].values, rcond=None)[0]
    return stats.pearsonr(rx, ry)

rows = []
for g in PARTNERS:
    if g not in L.columns:
        continue
    v = L[g]
    m = v.notna() & ufm1_z.notna()
    r_c, p_c = stats.pearsonr(v[m], ufm1_z[m])
    r_con, p_con = pcorr(v, ufm1_z, [contam_pc1])
    r_tech, p_tech = pcorr(v, ufm1_z, [tech_pc1])
    r_tc, p_tc = pcorr(v, ufm1_z, [tech_pc1, contam_pc1])
    r_full, p_full = pcorr(v, ufm1_z, [tech_pc1, contam_pc1, detect])
    rows.append(dict(gene=g, r_crude=r_c, p_crude=p_c, r_contam=r_con,
                     r_tech=r_tech, r_tech_contam=r_tc, r_full=r_full, p_full=p_full))
    print(f"{g:<9} crude={r_c:+.3f} contam={r_con:+.3f} tech={r_tech:+.3f} "
          f"tc={r_tc:+.3f} full={r_full:+.3f} p_full={p_full:.2e}")

pd.DataFrame(rows).to_csv("results/GSE245948_full_adjust_check.csv", index=False)
print("\n已保存 results/GSE245948_full_adjust_check.csv")
