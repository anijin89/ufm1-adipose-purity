#!/usr/bin/env python
"""从 GPL13667 SOFT 前缀中抽取 probe -> gene symbol 映射"""
import os, csv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "GPL13667_part.txt")
OUT = os.path.join(BASE, "data", "GPL13667_probe2gene.tsv")

TARGET = {"UFM1", "UFC1", "UBA5", "UFL1", "UFSP1", "UFSP2", "DDRGK1",
          "CDK5RAP3", "RPL26", "GAPDH", "PPARG", "FASN", "ADIPOQ", "LEP"}

mapping = {}
found = {g: [] for g in TARGET}
in_table = False
header = None
n = 0

with open(SRC, "rt", errors="replace") as fh:
    for line in fh:
        if line.startswith("!platform_table_begin"):
            in_table = True
            continue
        if line.startswith("!platform_table_end"):
            break
        if not in_table:
            continue
        parts = line.rstrip("\n").split("\t")
        if header is None:
            header = parts
            print("cols:", len(header))
            continue
        if len(parts) < 15:
            continue
        pid = parts[0].strip()
        sym = parts[header.index("Gene Symbol")].strip()
        n += 1
        mapping[pid] = sym
        if sym in found:
            found[sym].append(pid)

print("probe rows parsed:", n)
print("unique symbols:", len(set(v for v in mapping.values() if v)))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["probe", "symbol"])
    for k, v in mapping.items():
        w.writerow([k, v])

print("\n=== target genes -> probes ===")
for g in sorted(found):
    print(f"{g:10s} n={len(found[g]):2d}  {found[g][:6]}")
