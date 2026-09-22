#!/usr/bin/env python3

import pandas as pd

# --------------------------------------------------
# INPUT FILES
# --------------------------------------------------

CONSURF = "/media/usuario/Seagate Basic1/Tesis_Maestria/04_feb_26/consurf_mapping/consurf_ALL.csv"
MAP     = "/media/usuario/Seagate Basic1/Tesis_Maestria/01_feb_26/pdb_uniprot_position_map.csv"

OUT     = "consurf_ALL_mapped_to_uniprot.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading tables...")

df_con = pd.read_csv(CONSURF, dtype=str)
df_map = pd.read_csv(MAP, dtype=str)

# --------------------------------------------------
# NORMALIZE KEYS  (VERY IMPORTANT)
# --------------------------------------------------

# ConSurf naming
df_con["pdb_id"] = df_con["PDBID"].str.upper().str.strip()
df_con["auth_asym_id"] = df_con["Chain"].str.strip()

# force numbering consistency
df_con["auth_seq_id"] = (
    pd.to_numeric(df_con["auth_seq_id"], errors="coerce")
    .astype("Int64")
    .astype(str)
)

df_map["pdb_id"] = df_map["pdb_id"].str.upper().str.strip()
df_map["auth_asym_id"] = df_map["auth_asym_id"].str.strip()
df_map["auth_seq_id"] = df_map["auth_seq_id"].astype(str)

print("ConSurf rows:", len(df_con))
print("Mapping rows:", len(df_map))

# --------------------------------------------------
# MERGE
# --------------------------------------------------

print("Merging ConSurf with UniProt mapping...")

df = df_con.merge(
    df_map,
    on=["pdb_id","auth_asym_id","auth_seq_id"],
    how="left",
    validate="many_to_one"   # safety check
)

# --------------------------------------------------
# SANITY CHECKS
# --------------------------------------------------

mapped = df["uniprot_id"].notna().sum()
total  = len(df)

print("\n=== SANITY CHECK ===")
print(f"Mapped residues: {mapped}/{total} ({mapped/total*100:.2f}%)")

missing = df[df["uniprot_id"].isna()]
if len(missing) > 0:
    print("⚠ Unmapped residues:", len(missing))

# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(OUT, index=False)

print("\n✔ Output written to:", OUT)