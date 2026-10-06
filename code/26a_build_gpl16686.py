"""
GPL16686 (Affymetrix HuGene-2_0-st, transcript/gene version) 探针 -> 基因符号注释
问题：GEO 平台表自带列无 Gene Symbol（只有 GB_ACC 与 hg19 区间），无法直接映射。
方案：使用 Bioconductor 官方注释包 hugene20sttranscriptcluster.db（probe_id -> Entrez Gene ID），
      再经 NCBI gene_info 的 Symbol / Synonyms 转成标准基因符号（含旧别名，如 KIAA0776 -> UFL1）。
输出：data/GPL16686_probe2gene.tsv （probe, gene），供 26_bariatric_gse53376.py 使用。
"""
import io
import os
import sqlite3
import tarfile

import pandas as pd
import requests

OUT = "data/GPL16686_probe2gene.tsv"
DB = "data/hugene20st.sqlite"
GI = "data/Homo_sapiens.gene_info.gz"

# ---------- 1. Bioconductor 注释包 (probe -> Entrez) ----------
if not os.path.exists(DB):
    u = ("https://bioconductor.org/packages/release/data/annotation/src/contrib/"
         "hugene20sttranscriptcluster.db_8.8.0.tar.gz")
    r = requests.get(u, timeout=300)
    tf = tarfile.open(fileobj=io.BytesIO(r.content))
    mem = [n for n in tf.getnames() if n.endswith("inst/extdata/hugene20sttranscriptcluster.sqlite")]
    open(DB, "wb").write(tf.extractfile(mem[0]).read())
    print("下载注释库:", mem[0])

con = sqlite3.connect(DB)
probes = pd.read_sql("select probe_id, gene_id from probes where gene_id is not null", con)
probes["probe_id"] = probes.probe_id.astype(str).str.strip()
probes["gene_id"] = probes.gene_id.astype(str).str.strip()
print(f"probe -> Entrez: {len(probes)} 条，{probes.probe_id.nunique()} 个探针")

# ---------- 2. NCBI gene_info (Entrez -> Symbol + 别名) ----------
if not os.path.exists(GI):
    u = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
    open(GI, "wb").write(requests.get(u, timeout=600).content)
gi = pd.read_csv(GI, sep="\t", dtype=str, usecols=["GeneID", "Symbol", "Synonyms"], low_memory=False)
id2sym = dict(zip(gi.GeneID.astype(str), gi.Symbol))
alias2sym = {}
for gid, sym, syn in zip(gi.GeneID.astype(str), gi.Symbol, gi.Synonyms.fillna("")):
    alias2sym[sym] = sym
    for a in str(syn).split("|"):
        a = a.strip()
        if a and a != "-" and a not in alias2sym:
            alias2sym[a] = sym

probes["gene"] = probes.gene_id.map(id2sym)
print("已映射符号:", probes.gene.notna().sum(), "/", len(probes))

# ---------- 3. 别名归一（UFL1/KIAA0776、UFSP1 等） ----------
AXIS_ALIAS = {
    "UFL1": ["UFL1", "KIAA0776", "NLBP", "RCAD", "UFM1L1"],
    "UFSP1": ["UFSP1"],
    "UFSP2": ["UFSP2", "C4orf20"],
    "DDRGK1": ["DDRGK1", "UFBP1", "C20orf116"],
    "CDK5RAP3": ["CDK5RAP3", "C53", "LZAP"],
    "UBA5": ["UBA5", "UBE1DC1"],
    "UFM1": ["UFM1", "C13orf20"],
    "UFC1": ["UFC1"],
}
alias2std = {a: k for k, v in AXIS_ALIAS.items() for a in v}
probes["gene_std"] = probes.gene.map(lambda g: alias2std.get(g, g))

probes[["probe_id", "gene_std"]].rename(columns={"probe_id": "probe", "gene_std": "gene"}) \
      .dropna().drop_duplicates().to_csv(OUT, sep="\t", index=False, header=False)
print("写出:", OUT, pd.read_csv(OUT, sep="\t", header=None).shape)

# ---------- 4. 自检 ----------
want = ["UFM1", "UBA5", "UFC1", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3",
        "EIF2AK3", "EIF2S1", "ATF4", "DDIT3", "HSPA5", "XBP1",
        "LEP", "ADIPOQ", "CD68", "FASN", "PPARG", "PLIN1", "SLC2A4"]
tab = pd.read_csv(OUT, sep="\t", header=None, names=["probe", "gene"], dtype=str)
for g in want:
    n = (tab.gene == g).sum()
    print(f"  {g:<10} {n} 个探针 {list(tab.probe[tab.gene == g])[:4]}")
