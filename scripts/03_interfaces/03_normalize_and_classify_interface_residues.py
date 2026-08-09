#!/usr/bin/env python3
import pandas as pd
import numpy as np

# ============================================================
# INPUT / OUTPUT
# ============================================================
INFILE = "interface_areas_per_residue.csv"
OUT_NORM = "interface_areas_per_residue_normalized.csv"
OUT_LABELED = "interface_hotspots_labeled.csv"

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(INFILE)

# ============================================================
# Sanity checks
# ============================================================
required_cols = {
    "pdb_id",
    "polymer_class",
    "label_asym_id",
    "auth_asym_id",
    "resnum_used",
    "resname",
    "label_seq_id",
    "auth_seq_id",
    "sasa_unbound",
    "sasa_bound",
    "delta_sasa",
}

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {INFILE}: {missing}")

# Force numeric types (robust against weird CSVs)
for col in ["sasa_unbound", "sasa_bound", "delta_sasa"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace NaN / inf early
df[["sasa_unbound", "sasa_bound", "delta_sasa"]] = (
    df[["sasa_unbound", "sasa_bound", "delta_sasa"]]
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0.0)
)

# ============================================================
# 1) Buried fraction per residue (ΔSASA / SASA_unbound)
# ============================================================
df["buried_fraction_residue"] = 0.0

mask = (df["sasa_unbound"] > 0.0) & (df["delta_sasa"] > 0.0)
df.loc[mask, "buried_fraction_residue"] = (
    df.loc[mask, "delta_sasa"] / df.loc[mask, "sasa_unbound"]
)

# Clamp numerical noise
df["buried_fraction_residue"] = df["buried_fraction_residue"].clip(0.0, 1.0)

# Explicit alias (useful later)
df["delta_sasa_norm"] = df["buried_fraction_residue"]

# ============================================================
# 2) Percentiles
# ============================================================

# Global percentile (across ALL residues)
df["delta_sasa_percentile_global"] = (
    df["delta_sasa"]
      .rank(method="average", pct=True)
      .fillna(0.0) * 100.0
)

# Percentile within each chain (mmCIF-stable)
df["delta_sasa_percentile_within_chain"] = (
    df.groupby(["pdb_id", "polymer_class", "label_asym_id"])["delta_sasa"]
      .rank(method="average", pct=True)
      .fillna(0.0) * 100.0
)

# ============================================================
# 3) Interface classification (biologically motivated)
# ============================================================

# Absolute interface: minimal physical contact
df["is_interface_abs"] = df["delta_sasa"] > 1.0

# Strong interface: top 10 % globally
df["is_strong_interface"] = df["delta_sasa_percentile_global"] >= 90.0

# Core hotspot: extreme burial + high relative contribution
df["is_core_hotspot"] = (
    (df["delta_sasa_percentile_global"] >= 97.0) &
    (df["buried_fraction_residue"] >= 0.25)
)

def classify_residue(row):
    if row["is_core_hotspot"]:
        return "core_hotspot"
    elif row["is_strong_interface"]:
        return "strong_interface"
    elif row["is_interface_abs"]:
        return "interface"
    else:
        return "non_interface"

df["interface_class"] = df.apply(classify_residue, axis=1)

# ============================================================
# 4) Save outputs
# ============================================================
df.to_csv(OUT_NORM, index=False)
df.to_csv(OUT_LABELED, index=False)

# ============================================================
# 5) Quick summaries
# ============================================================
print("\nInterface class counts:")
print(df["interface_class"].value_counts())

print("\nTop 15 core hotspots:")
cols = [
    "pdb_id",
    "polymer_class",
    "label_asym_id",
    "auth_asym_id",
    "resnum_used",
    "resname",
    "delta_sasa",
    "buried_fraction_residue",
    "delta_sasa_percentile_global",
]
print(
    df[df["interface_class"] == "core_hotspot"]
      .sort_values("delta_sasa", ascending=False)
      .head(15)[cols]
)

print(f"\nSaved normalized table: {OUT_NORM}")
print(f"Saved labeled hotspots: {OUT_LABELED}")
