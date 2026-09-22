#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

BASE = "/media/usuario/Seagate Basic1/Tesis_Maestria/23_feb_26"

# ----------------------------------------------------
# 1️⃣ Load backbone (SASA)
# ----------------------------------------------------
print("Loading SASA backbone...")
df_master = pd.read_csv(f"{BASE}/sasa_mapped_to_uniprot.csv", low_memory=False)

# Normalize keys
df_master["pdb_id"] = df_master["pdb_id"].str.upper().str.strip()
df_master["uniprot_id"] = df_master["uniprot_id"].str.strip()
df_master["uniprot_pos"] = df_master["uniprot_pos"].astype(int)

print("Backbone rows:", len(df_master))


# ----------------------------------------------------
# 2️⃣ Load ConSurf
# ----------------------------------------------------
print("Merging ConSurf...")
df_cs = pd.read_csv(f"{BASE}/consurf_ALL_mapped_to_uniprot.csv", low_memory=False)

# normalize ids
df_cs["pdb_id"] = df_cs["pdb_id"].astype(str).str.upper().str.strip()

# convert safely to numeric (avoids IntCastingNaNError)
df_cs["uniprot_pos"] = pd.to_numeric(df_cs["uniprot_pos"], errors="coerce")

# 🔴 IMPORTANT: remove residues without UniProt mapping
df_cs = df_cs.dropna(subset=["uniprot_pos","uniprot_id"])

# now safe to convert to int
df_cs["uniprot_pos"] = df_cs["uniprot_pos"].astype(int)

cols_cs = [
    "pdb_id","uniprot_id","uniprot_pos",
    "consurf_score","consurf_color","msa_coverage","is_low_conf"
]

df_master = df_master.merge(
    df_cs[cols_cs],
    on=["pdb_id","uniprot_id","uniprot_pos"],
    how="left"
)

# ----------------------------------------------------
# 3️⃣ Load IUPred
# ----------------------------------------------------
print("Merging IUPred...")
df_iup = pd.read_csv(f"{BASE}/iupred_results_mapped_to_uniprot.csv", low_memory=False)

# normalize id safely
df_iup["pdb_id"] = df_iup["pdb_id"].astype(str).str.upper().str.strip()

# safe numeric conversion (handles NaN correctly)
df_iup["uniprot_pos"] = pd.to_numeric(df_iup["uniprot_pos"], errors="coerce")

# remove residues that cannot be mapped to UniProt coordinates
df_iup = df_iup.dropna(subset=["uniprot_pos","uniprot_id"])

# now integer conversion is safe
df_iup["uniprot_pos"] = df_iup["uniprot_pos"].astype(int)

cols_iup = [
    "pdb_id","uniprot_id","uniprot_pos",
    "iupred_score","is_disordered"
]

df_master = df_master.merge(
    df_iup[cols_iup],
    on=["pdb_id","uniprot_id","uniprot_pos"],
    how="left"
)


# ----------------------------------------------------
# 4️⃣ Load WATER DATA
# ----------------------------------------------------
print("Merging water-mediated residues...")
df_water = pd.read_csv(f"{BASE}/water_mediated_residues_chain_uniprot_COLLAPSED.csv")

df_water["pdb_id"] = df_water["pdb_id"].str.upper().str.strip()
df_water["uniprot_pos"] = df_water["uniprot_pos"].astype(int)

df_master = df_master.merge(
    df_water,
    on=["pdb_id","uniprot_id","uniprot_pos"],
    how="left"
)


# ----------------------------------------------------
# 5️⃣ Load RING outputs (many files → one df)
# ----------------------------------------------------
print("Loading RING files...")

ring_files = list(Path(f"{BASE}/ring_outputs").glob("*_interface_residues_with_uniprot.csv"))

ring_list = []
for f in ring_files:
    tmp = pd.read_csv(f)
    if not tmp.empty:
        ring_list.append(tmp)

df_ring = pd.concat(ring_list, ignore_index=True)

# normalize id safely
df_ring["pdb_id"] = df_ring["pdb_id"].astype(str).str.upper().str.strip()

# safe numeric conversion
df_ring["uniprot_pos"] = pd.to_numeric(df_ring["uniprot_pos"], errors="coerce")

# keep only valid UniProt-mapped residues
df_ring = df_ring.dropna(subset=["uniprot_pos","uniprot_id"])

# now safe
df_ring["uniprot_pos"] = df_ring["uniprot_pos"].astype(int)

cols_ring = [
    "pdb_id","uniprot_id","uniprot_pos",
    "ring_interaction_count"
]

df_master = df_master.merge(
    df_ring[cols_ring],
    on=["pdb_id","uniprot_id","uniprot_pos"],
    how="left"
)


# ----------------------------------------------------
# 6️⃣ Final cleanup
# ----------------------------------------------------
print("Finalizing...")

df_master["has_ring_interface"] = df_master["ring_interaction_count"].notna()

out = f"{BASE}/MASTER_interface_dataset.csv"
df_master.to_csv(out, index=False)

print("MASTER file written:", out)
print("Total rows:", len(df_master))