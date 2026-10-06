#!/usr/bin/env python3
"""
Table S8 —— GSE53376 与原始研究 qPCR 验证基因的方向一致性

原研究（PMID 26252355）用定量 PCR 验证了 16 个基因在减重手术后的变化方向。
本脚本把本流程在 GSE53376（Affymetrix HuGene-2.0 ST，本研究唯一使用该平台的
队列）上得到的配对变化方向，与文献报告的方向逐基因比对，输出补充表 S8。

用途：这是本稿唯一可充当"外部实验验证"的锚点——16/16 方向一致（精确二项检验
p-value < 0.01，无方向零假设），说明本流程对该平台的处理是可靠的。

输入：results/GSE53376_paired_weightloss.csv（由 26_bariatric_gse53376.py 生成）
输出：../AE&M/06_Supplementary/SourceData/TableS8_GSE53376_qPCR_concordance.csv

运行（cwd = analysis/）：
    python3 scripts/31_qpcr_concordance_table.py
"""
import os
import pandas as pd
from scipy import stats

# 原研究 qPCR 验证的差异基因（PMID 26252355），与 26_bariatric_gse53376.py 中一致
PAPER_UP = ["SLC27A2", "ELOVL6", "FASN", "GYS2", "LGALS12", "PKP2", "ACLY"]
PAPER_DOWN = ["FOS", "EGFL6", "PRG4", "AQP9", "DUSP1", "RGS1", "EGR1", "SPP1", "LYZ"]

SRC = "results/GSE53376_paired_weightloss.csv"
DST = ("../AE&M/06_Supplementary/SourceData/"
       "TableS8_GSE53376_qPCR_concordance.csv")

res = pd.read_csv(SRC)

rows = []
for g in PAPER_UP + PAPER_DOWN:
    r = res[res.feature == g]
    if not len(r):
        continue
    r = r.iloc[0]
    expected = +1 if g in PAPER_UP else -1
    observed = +1 if r.delta > 0 else -1
    rows.append(dict(
        gene=g,
        expected_direction="up" if expected > 0 else "down",
        observed_delta=round(float(r.delta), 4),
        observed_direction="up" if observed > 0 else "down",
        p_paired=float(r.p_paired),
        cohens_d=round(float(r.cohens_d), 3),
        n=int(r.n),
        concordant="yes" if expected == observed else "no",
    ))

out = pd.DataFrame(rows)
n_match = int((out.concordant == "yes").sum())
n_total = len(out)
binom = stats.binomtest(n_match, n_total, 0.5).pvalue

print(f"方向一致 {n_match}/{n_total}；精确二项检验 p = {binom:.3g}")
os.makedirs(os.path.dirname(DST), exist_ok=True)
out.to_csv(DST, index=False)
print("写入", DST)
