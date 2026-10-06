#!/usr/bin/env python
"""流式解析 GSE70353 series matrix：抽出样本临床信息 + 探针列表（不加载 452MB 全矩阵）"""
import gzip, csv, json, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(BASE, "data", "GSE70353_series_matrix.txt.gz")
OUT = os.path.join(BASE, "data")

samples = None
chars = []
probe_ids = []
in_table = False
header_seen = False

want_fields = ["tissue", "gender", "age", "bmi", "whr", "waist_circumference",
               "hip_circumference", "ffmass", "totfa", "p_adipon", "matsuda",
               "p_crp", "b_ghba1c", "diastbp", "systbp", "gfr", "homair",
               "p_ffa0", "p_gl0", "p_il1ra", "p_ins0", "p_proi0", "s_hdlc",
               "s_ldlc", "s_totalc", "s_tottg"]

with gzip.open(MATRIX, "rt") as fh:
    for line in fh:
        if line.startswith("!Sample_title"):
            samples = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            chars.append([x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]])
        elif line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        elif line.startswith("!series_matrix_table_end"):
            break
        elif in_table:
            if not header_seen:
                header_seen = True
                continue
            probe_ids.append(line.split("\t", 1)[0].strip().strip('"'))

print("samples:", len(samples) if samples else None)
print("chars rows:", len(chars))
print("probes:", len(probe_ids))

# 组装临床表
rows = {}
for row in chars:
    for sid, val in zip(samples, row):
        if ":" not in val:
            continue
        k, v = val.split(":", 1)
        k = k.strip().lower()
        if k in want_fields:
            rows.setdefault(sid, {})[k] = v.strip()

with open(os.path.join(OUT, "GSE70353_clinical.csv"), "w", newline="") as f:
    w = csv.writer(f)
    cols = ["sample_id"] + want_fields
    w.writerow(cols)
    for sid in samples:
        d = rows.get(sid, {})
        w.writerow([sid] + [d.get(c, "") for c in want_fields])

with open(os.path.join(OUT, "GSE70353_probes.txt"), "w") as f:
    f.write("\n".join(probe_ids))

print("clinical rows written:", len(samples))
