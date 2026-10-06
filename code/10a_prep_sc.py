#!/usr/bin/env python3
"""准备单细胞分析: 确定目标基因在 mtx 中的行号"""
import gzip

FEAT = "data/GSE176171_features.tsv.gz"
TARGETS = ["UFM1", "UFC1", "UBA5", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3",
           # 细胞类型标记
           "PPARG", "FASN", "ADIPOQ", "PLIN1", "LEP", "LPL", "CIDEC", "SREBF1", "CEBPA",
           "PDGFRA", "CD34", "THY1", "DCN", "LUM",
           "CD68", "PTPRC", "ITGAM", "CD14", "CSF1R", "MRC1",
           "PECAM1", "VWF", "CDH5", "CD3E", "CD2", "IL7R",
           # ER 应激 / UPR
           "HSPA5", "DDIT3", "ATF4", "ERN1", "EIF2AK3", "XBP1", "ATF6", "SEL1L",
           # 对照
           "ACTB", "GAPDH", "B2M"]

idx = {}
with gzip.open(FEAT, "rt") as fh:
    for i, line in enumerate(fh, start=1):
        f = line.rstrip("\n").split("\t")
        sym = f[1] if len(f) > 1 else ""
        if sym in TARGETS and sym not in idx:
            idx[sym] = i

print("找到基因:", len(idx))
miss = [g for g in TARGETS if g not in idx]
print("未找到:", miss)

with open("data/sc_target_rows.txt", "w") as out:
    for g, i in sorted(idx.items(), key=lambda x: x[1]):
        out.write(f"{i}\t{g}\n")
print("行数写入:", len(idx))
