"""
GSE245948 — 独立皮下脂肪 RNA-seq 队列（Vanderbilt）
GEO 元数据不含 BMI/HOMA-IR，故不做表型关联；改为验证 UFM1 的共表达偶联
（ER 应激 / eIF2α 轴 / 脂滴），该检验不需要临床表型。
"""
import gzip
import numpy as np, pandas as pd, urllib.request, json
from scipy import stats

CPM = "data/GSE245948_cpm.csv.gz"

with gzip.open(CPM, "rt") as f:
    cpm = pd.read_csv(f, index_col=0)
print(f"CPM 矩阵: {cpm.shape[0]} 基因 x {cpm.shape[1]} 样本")

# ---------- 目标基因 symbol -> ENSG ----------
TARGETS = ["UFM1", "UBA5", "UFC1", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3",
           "PLIN1", "FASN", "PPARG", "ADIPOQ", "LEP", "SLC2A4",
           "EIF2S1", "EIF2AK3", "ATF6", "ERN1", "HSPA5", "SEL1L", "XBP1", "DDIT3", "ATF4",
           "CD68", "PTPRC", "LYZ", "CD14", "CSF1R", "PDGFRA", "COL1A1", "PECAM1", "CDH5"]

import urllib.parse

def ensg_for(symbols):
    out = {}
    for i in range(0, len(symbols), 40):
        chunk = symbols[i:i + 40]
        payload = urllib.parse.urlencode(
            {"q": ",".join(chunk), "scopes": "symbol",
             "fields": "ensembl.gene", "species": "human"}).encode()
        try:
            req = urllib.request.Request("https://mygene.info/v3/query", data=payload)
            data = json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            print("  mygene 失败:", e)
            continue
        for d in data if isinstance(data, list) else []:
            q_ = d.get("query")
            e_ = d.get("ensembl", {}).get("gene") if isinstance(d.get("ensembl"), dict) else None
            if isinstance(e_, list):
                e_ = e_[0].get("gene") if isinstance(e_[0], dict) else e_[0]
            if q_ and e_:
                out[q_.upper()] = e_
    return out

print("查询 Ensembl ID ...")
sym2ens = ensg_for(TARGETS)
print(f"  成功映射 {len(sym2ens)} / {len(TARGETS)}")
missing = [t for t in TARGETS if t.upper() not in sym2ens]
if missing:
    print("  未映射:", missing)

# ---------- 提取表达 ----------
cpm.index = [str(i).split(".")[0] for i in cpm.index]
cpm = cpm[~cpm.index.duplicated(keep="first")]
expr = {}
for sym in TARGETS:
    e = sym2ens.get(sym.upper())
    if e and e in cpm.index:
        expr[sym] = cpm.loc[e].astype(float)
print(f"表达矩阵中定位到 {len(expr)} 个目标基因")
if "UFM1" not in expr:
    print("UFM1 未找到，终止")
    raise SystemExit(1)

E = pd.DataFrame(expr)
E = E.apply(pd.to_numeric, errors="coerce")
print(f"样本数: {E.shape[0]}")

# log2 转换（CPM）
L = np.log2(E.clip(lower=0.01))
n = L.shape[0]
print(f"\n注意：本队列无 BMI / HOMA-IR 元数据，仅做共表达偶联验证（n={n}）")

CORE = [c for c in ["UBA5", "UFSP2", "DDRGK1", "CDK5RAP3"] if c in L]
Z = (L - L.mean()) / L.std(ddof=0)
mach = Z[CORE].mean(axis=1)
ufm1_z = Z["UFM1"]
dysreg = ufm1_z - mach

CONTAM = [c for c in ["CD68", "PTPRC", "LYZ", "CD14", "CSF1R", "PDGFRA", "COL1A1",
                      "PECAM1", "CDH5"] if c in L]
contam_z = Z[CONTAM].mean(axis=1) if CONTAM else None

def r(a, b):
    m = a.notna() & b.notna()
    return stats.pearsonr(a[m], b[m])

def pcorr(x, y, cov):
    df = pd.concat([x, y, cov], axis=1).dropna()
    if len(df) < 15:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(df)), df.iloc[:, 2].values])
    rx = df.iloc[:, 0].values - X @ np.linalg.lstsq(X, df.iloc[:, 0].values, rcond=None)[0]
    ry = df.iloc[:, 1].values - X @ np.linalg.lstsq(X, df.iloc[:, 1].values, rcond=None)[0]
    return stats.pearsonr(rx, ry)

print("\n" + "=" * 84)
print("UFM1 共表达偶联（独立 RNA-seq 队列）")
print("=" * 84)
print(f"{'基因':<10}{'r(UFM1)':>10}{'p':>11}{'| 校污染':>10}{'p':>11}{'| r(失衡指数)':>13}{'p':>11}")
print("-" * 84)
rows = []
for g in ["EIF2S1", "EIF2AK3", "ATF6", "SEL1L", "HSPA5", "ERN1", "XBP1", "DDIT3", "ATF4",
          "PLIN1", "FASN", "PPARG", "ADIPOQ", "SLC2A4", "LEP"]:
    if g not in L:
        continue
    a, pa = r(L[g], ufm1_z)
    if contam_z is not None:
        b, pb = pcorr(L[g], ufm1_z, contam_z)
    else:
        b, pb = np.nan, np.nan
    c, pc = r(L[g], dysreg)
    rows.append(dict(gene=g, r_ufm1=a, p_ufm1=pa, r_ufm1_adj=b, p_ufm1_adj=pb,
                     r_dysreg=c, p_dysreg=pc))
    print(f"{g:<10}{a:>10.3f}{pa:>11.2e}{b:>10.3f}{pb:>11.2e}{c:>13.3f}{pc:>11.2e}")

print("\n" + "=" * 84)
print("轴内一致性（机器分与 UFM1 是否同向）")
print("=" * 84)
a, pa = r(ufm1_z, mach)
print(f"  UFM1 vs 机器分: r={a:+.3f}, p={pa:.2e}")

print("\n" + "=" * 84)
print("关键偶联的跨队列一致性")
print("=" * 84)
pri = ["EIF2S1", "EIF2AK3", "ATF6", "SEL1L", "PLIN1"]
for g in pri:
    row = next((x for x in rows if x["gene"] == g), None)
    if row:
        sig = "显著" if row["p_ufm1_adj"] < 0.05 else "不显著"
        print(f"  {g:<9} r={row['r_ufm1']:+.3f}  校污染后 r={row['r_ufm1_adj']:+.3f} ({sig})")

pd.DataFrame(rows).to_csv("results/GSE245948_coexpression.csv", index=False)
print("\n已保存 results/GSE245948_coexpression.csv")
