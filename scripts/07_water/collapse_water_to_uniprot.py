#!/usr/bin/env python3
import pandas as pd

IN = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/07_feb_26/water_mediated_residues_uniprot.csv"
OUT = "water_mediated_residues_uniprot_COLLAPSED.csv"

# --------------------------------------------------
# Load
# --------------------------------------------------
df = pd.read_csv(IN, dtype=str)

# numeric cols
df["n_bridging_waters"] = df["n_bridging_waters"].astype(int)
df["n_bridging_waters_angle_filtered"] = df["n_bridging_waters_angle_filtered"].astype(int)

# --------------------------------------------------
# Collapse to UniProt residue level
# --------------------------------------------------
grp = (
    df.groupby(
        ["pdb_id", "uniprot_id", "uniprot_pos"],
        as_index=False
    )
    .agg({
        "resname": "first",   # identical across chains
        "n_bridging_waters": "max",
        "n_bridging_waters_angle_filtered": "max"
    })
)

# --------------------------------------------------
# Derive MASTER-friendly annotations
# --------------------------------------------------
grp["has_bridging_water"] = grp["n_bridging_waters"] > 0
grp["has_angle_filtered_water"] = grp["n_bridging_waters_angle_filtered"] > 0

# same logic you used before in analysis tables
grp["is_recurrent_water_site"] = grp["n_bridging_waters"] >= 2

# --------------------------------------------------
# Save
# --------------------------------------------------
grp.to_csv(OUT, index=False)

print("✔ Collapsed UniProt-level water dataset created")
print("Rows:", len(grp))