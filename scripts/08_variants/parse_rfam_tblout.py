#!/usr/bin/env python3
import pandas as pd

def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

rows = []

with open("rfam.tblout") as f:
    for line in f:
        if line.startswith("#"):
            continue

        fields = line.rstrip("\n").split()

        # Core identifiers
        rfam_name = fields[0]   # target_name
        rfam_acc  = fields[1]   # RFxxxxx
        query     = fields[2]   # query_name

        # Stop parsing at ! or ? (annotation marker)
        if "!" in fields:
            cut = fields.index("!")
        elif "?" in fields:
            cut = fields.index("?")
        else:
            cut = len(fields)

        numeric = [float(x) for x in fields[:cut] if is_float(x)]
        if len(numeric) < 2:
            continue

        score, evalue = numeric[-2], numeric[-1]

        # Optional description
        desc = ""
        if cut < len(fields):
            desc = " ".join(fields[cut+1:])

        rows.append({
            "query": query,
            "rfam_acc": rfam_acc,
            "rfam_name": rfam_name,
            "rfam_desc": desc,
            "evalue": evalue,
            "score": score
        })

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No Rfam hits parsed from rfam.tblout")

# ---------- classification ----------
def classify_type(name):
    n = name.lower()
    if n == "trna":
        return "tRNA"
    if n in {"u1","u2","u4","u5","u6"}:
        return "snRNA"
    if n.startswith("5s_rrna"):
        return "rRNA_5S"
    if n.startswith("5_8s_rrna"):
        return "rRNA_5_8S"
    if "ssu_rrna" in n:
        return "rRNA_SSU"
    if "lsu_rrna" in n:
        return "rRNA_LSU"
    return "other"

def organism_rank(name):
    n = name.lower()
    if "eukarya" in n:
        return 0
    if "microsporidia" in n:
        return 1
    if "archaea" in n:
        return 2
    if "bacteria" in n:
        return 3
    return 4

df["rna_type"] = df["rfam_name"].apply(classify_type)
df["org_rank"] = df["rfam_name"].apply(organism_rank)

type_rank = {
    "snRNA": 0,
    "rRNA_5_8S": 0,
    "rRNA_5S": 0,
    "rRNA_SSU": 0,
    "rRNA_LSU": 0,
    "tRNA": 1,
    "other": 2
}
df["type_rank"] = df["rna_type"].map(type_rank).fillna(2)

# ---------- deterministic resolution ----------
best = []
for q, sub in df.groupby("query", sort=False):
    sub = sub.sort_values(
        ["org_rank", "type_rank", "evalue", "score"],
        ascending=[True, True, True, False]
    )
    best.append(sub.iloc[0])

df_best = pd.DataFrame(best)

# Parse PDB / chain
df_best["pdb_id"] = df_best["query"].str[:4]
df_best["chain_id"] = df_best["query"].str[4:].str.lstrip("_")

df_best.to_csv("rna_rfam_annotations.csv", index=False)

print("✔ Written rna_rfam_annotations.csv")
print("\nRNA type counts:")
print(df_best["rna_type"].value_counts())
print("\nRfam family counts:")
print(df_best["rfam_name"].value_counts())