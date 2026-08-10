#!/usr/bin/env python3

import pandas as pd
import glob
import os

# -------------------------
# Inputs
# -------------------------
INTERFACE_FILE = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/25_ene_26/interface_areas_per_residue_normalized.csv"
PDB_UNIPROT_MAP = "pdb_uniprot_position_map.csv"
VARIANTS_DIR = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/20_ago_25"

OUT_FULL = "interface_residues_with_uniprot_variants.csv"
OUT_INTERFACE_ONLY = "interface_residues_with_variants_interface_only.csv"

# -------------------------
# Load structural dataset
# -------------------------
print("Loading interface residues...")
df = pd.read_csv(INTERFACE_FILE, low_memory=False)
print("Rows (structural residues):", len(df))

# -------------------------
# Load PDB → UniProt map
# -------------------------
print("Loading PDB–UniProt position map...")
map_df = pd.read_csv(PDB_UNIPROT_MAP)

df = df.merge(
    map_df,
    on=["pdb_id", "auth_asym_id", "auth_seq_id"],
    how="left"
)

print("Mapped to UniProt:", df["uniprot_id"].notna().sum())

# -------------------------
# Load ALL UniProt variants
# -------------------------
print("Loading UniProt variant files...")
variant_tables = []

for fp in glob.glob(os.path.join(VARIANTS_DIR, "*_variants.csv")):
    uniprot_id = os.path.basename(fp).split("_")[0]
    vdf = pd.read_csv(fp)
    vdf["uniprot_id"] = uniprot_id
    variant_tables.append(vdf)

if not variant_tables:
    raise RuntimeError("No variant CSVs found!")

variants_df = pd.concat(variant_tables, ignore_index=True)

# Normalize column names
variants_df = variants_df.rename(columns={
    "Position": "uniprot_pos",
    "Wild_Type": "wild_type",
    "Mutated_Type": "mutated_type",
    "Consequence": "consequence",
    "Disease_Association": "disease",
    "Prediction_PolyPhen": "polyphen",
    "Prediction_SIFT": "sift"
})

# -------------------------
# Merge variants
# -------------------------
print("Merging variants...")
df = df.merge(
    variants_df,
    on=["uniprot_id", "uniprot_pos"],
    how="left"
)

print("With variants:", df["mutated_type"].notna().sum())

# -------------------------
# Save FULL dataset (heavy)
# -------------------------
print("Saving full annotated dataset...")
df.to_csv(OUT_FULL, index=False)

# -------------------------
# Save FILTERED dataset (recommended)
# -------------------------
df_interface = df[
    (df["interface_class"] != "non_interface") &
    (df["mutated_type"].notna())
]

print("Interface residues with variants:", len(df_interface))
df_interface.to_csv(OUT_INTERFACE_ONLY, index=False)

print("✔ Done")
print("Saved:")
print(" -", OUT_FULL)
print(" -", OUT_INTERFACE_ONLY)
