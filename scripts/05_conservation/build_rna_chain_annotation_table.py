#!/usr/bin/env python3
import pandas as pd
import sys

IN_FILE = "rna_rfam_annotations.csv"
OUT_FILE = "rna_chain_annotations_minimal.csv"

# Load annotated Rfam results
df = pd.read_csv(IN_FILE)

required = {"pdb_id", "chain_id", "rna_type", "rfam_name"}
missing = required - set(df.columns)
if missing:
    sys.exit(f"ERROR: Missing required columns in {IN_FILE}: {missing}")

# One row per RNA chain (defensive, in case of duplicates)
out = (
    df[["pdb_id", "chain_id", "rna_type", "rfam_name"]]
    .drop_duplicates()
    .sort_values(["pdb_id", "chain_id"])
)

out.to_csv(OUT_FILE, index=False)

print(f"✔ Written minimal RNA annotation table to {OUT_FILE}")
print(f"Rows: {len(out)}")
print("\nRNA type counts:")
print(out["rna_type"].value_counts())