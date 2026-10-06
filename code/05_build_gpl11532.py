#!/usr/bin/env python3
"""从 GPL11532 平台表解析探针 -> Gene Symbol 映射 (Affymetrix HTA 2.0)。
gene_assignment 字段格式: "NM_xxx // SYMBOL // ... /// ..."
"""
import re
import csv

SRC = "data/GPL11532_part.txt"
OUT = "data/GPL11532_probe2gene.tsv"

rows = 0
mapped = 0
with open(SRC, encoding="utf-8", errors="ignore") as fh, open(OUT, "w", newline="") as out:
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    w.writerow(["probe", "symbol"])
    started = False
    for line in fh:
        if not started:
            if line.startswith("!platform_table_begin"):
                started = True
            continue
        if line.startswith("!platform_table_end"):
            break
        f = line.rstrip("\n").split("\t")
        if len(f) < 10:
            continue
        probe = f[0].strip()
        ga = f[9].strip()
        if not probe or not ga or ga == "---":
            continue
        rows += 1
        # 取第一个 assignment 的 symbol
        sym = None
        for part in ga.split("///"):
            seg = [x.strip() for x in part.split("//")]
            if len(seg) >= 2 and seg[1] and seg[1] != "---":
                sym = seg[1]
                break
        if sym:
            mapped += 1
            w.writerow([probe, sym])

print(f"rows parsed: {rows}, mapped: {mapped}")

TARGETS = ["UFM1", "UFC1", "UBA5", "UFL1", "UFSP1", "UFSP2", "DDRGK1", "CDK5RAP3",
           "FASN", "PPARG", "ADIPOQ", "LEP", "CD68", "PLIN1", "KIAA0776", "RCAD",
           "NLBP", "MAXER", "UFBP1", "C20orf116", "PD1", "SREBF1", "CEBPA", "LPL"]
d = {}
with open(OUT) as fh:
    next(fh)
    for line in fh:
        p, s = line.rstrip("\n").split("\t")
        d.setdefault(s, []).append(p)

print("--- target genes on platform ---")
for g in TARGETS:
    ps = d.get(g, [])
    print(f"{g}: {len(ps)} probes" + (f"  e.g. {ps[:3]}" if ps else "  ** MISSING **"))
