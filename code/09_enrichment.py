#!/usr/bin/env python3
"""UFM1 共表达模块的通路富集分析 (Enrichr ORA + 相关性热图数据准备)"""
import re
import numpy as np
import pandas as pd

# ---------- 清理基因名 ----------
def clean(sym):
    s = str(sym).strip()
    if s in ("", "---", "nan"):
        return None
    s = s.split("///")[0].strip()
    if not re.match(r"^[A-Za-z0-9\-\._]+$", s):
        return None
    return s

R = pd.read_csv("results/GSE70353_UFM1_transcriptome_corr.csv", index_col=0)
R["sym"] = [clean(g) for g in R.index]
R = R.dropna(subset=["sym"])
print("可注释基因:", len(R))

SIG_POS = R[(R["r_UFM1"] > 0.30) & (R["q_UFM1"] < 0.05)]
SIG_NEG = R[(R["r_UFM1"] < -0.30) & (R["q_UFM1"] < 0.05)]
print(f"正相关模块 (r>0.30): {len(SIG_POS)}  负相关模块 (r<-0.30): {len(SIG_NEG)}")

pos_genes = sorted(set(SIG_POS["sym"]))
neg_genes = sorted(set(SIG_NEG["sym"]))
pd.DataFrame({"gene": pos_genes}).to_csv("results/UFM1_module_pos.txt", index=False, header=False)
pd.DataFrame({"gene": neg_genes}).to_csv("results/UFM1_module_neg.txt", index=False, header=False)

LIBS = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022",
        "MSigDB_Hallmark_2020", "WikiPathway_2023_Human"]

import gseapy as gp

for label, genes in [("POS_UFM1_correlated", pos_genes), ("NEG_UFM1_correlated", neg_genes)]:
    if len(genes) < 15:
        print(f"{label}: 基因太少 ({len(genes)})，跳过")
        continue
    print(f"\n{'='*70}\n{label}  (n={len(genes)})\n{'='*70}")
    try:
        enr = gp.enrichr(gene_list=genes, gene_sets=LIBS, organism="human",
                         outdir=None, cutoff=0.25, no_plot=True)
        df = enr.results
        df = df[df["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")
        df.to_csv(f"results/enrichr_{label}.csv", index=False)
        for lib in df["Gene_set"].unique():
            sub = df[df["Gene_set"] == lib].head(8)
            print(f"\n--- {lib} ---")
            for _, r in sub.iterrows():
                print(f"  {r['Term'][:70]:<70} adjP={r['Adjusted P-value']:.2e} "
                      f"OR={r['Odds Ratio']:.2f}")
    except Exception as e:
        print("富集失败:", e)
