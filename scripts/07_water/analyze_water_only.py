#!/usr/bin/env python3

import pandas as pd
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--water", required=True)
parser.add_argument("--out_prefix", default="water")
args = parser.parse_args()


water = pd.read_csv(args.water)

# =========================
# Limpieza
# =========================

num_cols = [
    "n_bridging_waters",
    "n_bridging_waters_angle_filtered",
    "n_chains_with_water",
]

for c in num_cols:
    if c in water.columns:
        water[c] = pd.to_numeric(water[c], errors="coerce").fillna(0)

bool_cols = [
    "has_bridging_water",
    "has_angle_filtered_water",
    "is_recurrent_water_site",
]

for c in bool_cols:
    if c in water.columns:
        water[c] = water[c].astype(str).str.lower().isin(["true", "1", "yes"])


# =========================
# 1. Frecuencia global
# =========================

summary = {
    "n_rows": len(water),
    "n_unique_pdb": water["pdb_id"].nunique(),
    "n_unique_uniprot": water["uniprot_id"].nunique(),
    "n_unique_positions": water[["uniprot_id", "uniprot_pos"]].drop_duplicates().shape[0],
    "fraction_angle_filtered": water["has_angle_filtered_water"].mean()
        if "has_angle_filtered_water" in water.columns else None,
    "fraction_recurrent": water["is_recurrent_water_site"].mean()
        if "is_recurrent_water_site" in water.columns else None,
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f"{args.out_prefix}_01_global_summary.csv", index=False)
print(summary_df)


# =========================
# 2. Distribución de aguas
# =========================

dist = water.agg({
    "n_bridging_waters": ["mean", "median", "max"],
    "n_bridging_waters_angle_filtered": ["mean", "median", "max"],
    "n_chains_with_water": ["mean", "median", "max"],
})

dist.to_csv(f"{args.out_prefix}_02_distribution.csv")
print("\nDistribución:")
print(dist)


# =========================
# 3. Residuos involucrados
# =========================

res = (
    water.groupby("resname")
    .agg(
        n_sites=("resname", "size"),
        mean_waters=("n_bridging_waters", "mean"),
        mean_chains=("n_chains_with_water", "mean"),
    )
    .reset_index()
    .sort_values("n_sites", ascending=False)
)

res.to_csv(f"{args.out_prefix}_03_residue_summary.csv", index=False)
print("\nTop residuos:")
print(res.head(15))


# =========================
# 4. Sitios recurrentes
# =========================

if "is_recurrent_water_site" in water.columns:
    recurrent = water[water["is_recurrent_water_site"]]

    rec_summary = (
        recurrent.groupby("resname")
        .size()
        .reset_index(name="n_recurrent")
        .sort_values("n_recurrent", ascending=False)
    )

    rec_summary.to_csv(f"{args.out_prefix}_04_recurrent_sites.csv", index=False)
    print("\nSitios recurrentes:")
    print(rec_summary.head(10))


# =========================
# 5. Distribución por complejo
# =========================

per_pdb = (
    water.groupby("pdb_id")
    .agg(
        n_sites=("pdb_id", "size"),
        total_waters=("n_bridging_waters", "sum"),
        mean_waters=("n_bridging_waters", "mean"),
    )
    .reset_index()
    .sort_values("n_sites", ascending=False)
)

per_pdb.to_csv(f"{args.out_prefix}_05_per_pdb.csv", index=False)

print("\nComplejos con más agua:")
print(per_pdb.head(10))


# =========================
# 6. Figuras
# =========================

plt.figure()
plt.hist(water["n_bridging_waters"], bins=50)
plt.xlabel("Aguas puente por residuo")
plt.ylabel("Frecuencia")
plt.title("Distribución de aguas puente")
plt.savefig(f"{args.out_prefix}_fig1_hist_waters.png", dpi=300)
plt.close()


plt.figure()
plt.hist(water["n_chains_with_water"], bins=30)
plt.xlabel("Chains con agua")
plt.ylabel("Frecuencia")
plt.title("Distribución de chains involucradas")
plt.savefig(f"{args.out_prefix}_fig2_hist_chains.png", dpi=300)
plt.close()


top = res.head(10)

plt.figure()
plt.bar(top["resname"], top["n_sites"])
plt.xlabel("Residuo")
plt.ylabel("Cantidad")
plt.title("Residuos más frecuentes en sitios con agua")
plt.savefig(f"{args.out_prefix}_fig3_residues.png", dpi=300)
plt.close()


plt.figure()
plt.hist(per_pdb["n_sites"], bins=50)
plt.xlabel("Sitios por PDB")
plt.ylabel("Frecuencia")
plt.title("Distribución por complejo")
plt.savefig(f"{args.out_prefix}_fig4_pdb_distribution.png", dpi=300)
plt.close()


print("\nListo.")