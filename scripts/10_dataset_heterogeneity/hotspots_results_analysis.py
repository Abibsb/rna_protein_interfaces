import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact

# =========================
# CONFIG
# =========================

INPUT = "dataset_sasa_ring.csv"
OUTDIR = Path("hotspots_results")
OUTDIR.mkdir(exist_ok=True)

# =========================
# LOAD
# =========================

df = pd.read_csv(INPUT)

print("Dataset:", df.shape)
print(df.columns.tolist())

# =========================
# NORMALIZACIÓN
# =========================

interface_col = "is_interface_abs" if "is_interface_abs" in df.columns else "is_interface"

for col in [interface_col, "is_strong_interface", "is_core_hotspot"]:
    df[col] = df[col].fillna(0).astype(int)

df["ring_interaction_count"] = df["ring_interaction_count"].fillna(0).astype(int)
df["has_ring_contact"] = df["ring_interaction_count"] > 0

df["interface_class"] = "non_interface"
df.loc[df[interface_col] == 1, "interface_class"] = "interface"
df.loc[df["is_strong_interface"] == 1, "interface_class"] = "strong_interface"
df.loc[df["is_core_hotspot"] == 1, "interface_class"] = "core_hotspot"

class_order = ["non_interface", "interface", "strong_interface", "core_hotspot"]

# =========================
# 1. PROPORCIÓN DE RESIDUOS
# =========================

prop = (
    df["interface_class"]
    .value_counts()
    .reindex(class_order)
    .fillna(0)
    .reset_index()
)

prop.columns = ["class", "n"]
prop["fraction_total"] = prop["n"] / len(df)
prop.to_csv(OUTDIR / "01_residue_class_proportions.csv", index=False)

plt.figure(figsize=(7, 5))
plt.bar(prop["class"], prop["fraction_total"])
plt.ylabel("Fracción del total de residuos")
plt.xlabel("Clase")
plt.xticks(rotation=30, ha="right")
plt.title("Proporción de residuos por clase estructural")
plt.tight_layout()
plt.savefig(OUTDIR / "01_residue_class_proportions.png", dpi=300)
plt.close()

# =========================
# 2. DISTRIBUCIÓN DENTRO DE INTERFACES
# =========================

interface_df = df[df[interface_col] == 1].copy()

inside_prop = (
    interface_df["interface_class"]
    .value_counts()
    .reindex(["interface", "strong_interface", "core_hotspot"])
    .fillna(0)
    .reset_index()
)

inside_prop.columns = ["class", "n"]
inside_prop["fraction_within_interface"] = inside_prop["n"] / len(interface_df)
inside_prop.to_csv(OUTDIR / "02_distribution_within_interfaces.csv", index=False)

plt.figure(figsize=(7, 5))
plt.bar(inside_prop["class"], inside_prop["fraction_within_interface"])
plt.ylabel("Fracción dentro de interfaz")
plt.xlabel("Clase")
plt.xticks(rotation=30, ha="right")
plt.title("Distribución de hotspots dentro de interfaces")
plt.tight_layout()
plt.savefig(OUTDIR / "02_distribution_within_interfaces.png", dpi=300)
plt.close()

# =========================
# 3. RELACIÓN HOTSPOTS - CONTACTOS RING
# =========================

ring_by_class = (
    df.groupby("interface_class")["has_ring_contact"]
    .agg(n="count", n_ring="sum", fraction_ring="mean")
    .reindex(class_order)
    .reset_index()
)

ring_by_class.to_csv(OUTDIR / "03_ring_contact_by_class.csv", index=False)

plt.figure(figsize=(7, 5))
plt.bar(ring_by_class["interface_class"], ring_by_class["fraction_ring"])
plt.ylabel("Fracción con contacto RING")
plt.xlabel("Clase")
plt.xticks(rotation=30, ha="right")
plt.title("Contactos directos RING por clase estructural")
plt.tight_layout()
plt.savefig(OUTDIR / "03_ring_contact_by_class.png", dpi=300)
plt.close()

# Fisher hotspot vs no hotspot dentro de interfaces
tmp = df[df[interface_col] == 1].copy()
tmp["is_hotspot"] = tmp["is_core_hotspot"] == 1

a = ((tmp["is_hotspot"]) & (tmp["has_ring_contact"])).sum()
b = ((tmp["is_hotspot"]) & (~tmp["has_ring_contact"])).sum()
c = ((~tmp["is_hotspot"]) & (tmp["has_ring_contact"])).sum()
d = ((~tmp["is_hotspot"]) & (~tmp["has_ring_contact"])).sum()

or_value, fisher_p = fisher_exact([[a, b], [c, d]])

with open(OUTDIR / "03_fisher_hotspot_ring.txt", "w") as f:
    f.write("Fisher exact test: hotspot vs contacto RING dentro de interfaz\n")
    f.write(f"hotspot + RING: {a}\n")
    f.write(f"hotspot + no RING: {b}\n")
    f.write(f"no hotspot + RING: {c}\n")
    f.write(f"no hotspot + no RING: {d}\n")
    f.write(f"Odds ratio: {or_value}\n")
    f.write(f"p-value: {fisher_p}\n")

# =========================
# 4. DISTRIBUCIÓN DE ENTERRAMIENTO
# =========================

def boxplot_by_class(value_col, filename, ylabel, title):
    plot_df = df[df[value_col].notna()].copy()

    data = [
        plot_df.loc[plot_df["interface_class"] == cls, value_col].values
        for cls in class_order
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=class_order, showfliers=False)
    plt.ylabel(ylabel)
    plt.xlabel("Clase")
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=300)
    plt.close()

boxplot_by_class(
    "delta_sasa",
    "04_delta_sasa_by_class.png",
    "ΔSASA",
    "Distribución de ΔSASA por clase estructural"
)

if "buried_fraction_residue" in df.columns:
    boxplot_by_class(
        "buried_fraction_residue",
        "05_buried_fraction_by_class.png",
        "Fracción enterrada",
        "Distribución de fracción enterrada por clase estructural"
    )

# =========================
# 5. NÚMERO DE CONTACTOS RING
# =========================

boxplot_by_class(
    "ring_interaction_count",
    "06_ring_count_by_class.png",
    "Número de contactos RING",
    "Distribución de contactos RING por clase estructural"
)

ring_count_summary = (
    df.groupby("interface_class")["ring_interaction_count"]
    .describe()
    .reindex(class_order)
)

ring_count_summary.to_csv(OUTDIR / "06_ring_count_summary_by_class.csv")

# =========================
# 6. ΔSASA VS CONTACTOS
# =========================

tmp = df[["delta_sasa", "ring_interaction_count"]].dropna()

rho, p_spearman = spearmanr(tmp["delta_sasa"], tmp["ring_interaction_count"])

with open(OUTDIR / "07_spearman_delta_sasa_ring.txt", "w") as f:
    f.write("Spearman: ΔSASA vs ring_interaction_count\n")
    f.write(f"rho: {rho}\n")
    f.write(f"p-value: {p_spearman}\n")

plt.figure(figsize=(7, 5))
plt.scatter(
    tmp["ring_interaction_count"],
    tmp["delta_sasa"],
    s=4,
    alpha=0.1
)
plt.xlabel("Número de contactos RING")
plt.ylabel("ΔSASA")
plt.title("Relación entre ΔSASA y número de contactos")
plt.tight_layout()
plt.savefig(OUTDIR / "07_delta_sasa_vs_ring_count.png", dpi=300)
plt.close()

# =========================
# 7. ΔSASA POR GRUPOS DE CONTACTOS
# =========================

df["ring_contact_group"] = pd.cut(
    df["ring_interaction_count"],
    bins=[-0.1, 0, 1, 2, 3, np.inf],
    labels=["0", "1", "2", "3", ">=4"]
)

delta_by_contact = (
    df.groupby("ring_contact_group", observed=False)["delta_sasa"]
    .agg(["count", "mean", "median"])
    .reset_index()
)

delta_by_contact.to_csv(OUTDIR / "08_delta_sasa_by_ring_contact_group.csv", index=False)

plt.figure(figsize=(7, 5))
plt.bar(delta_by_contact["ring_contact_group"].astype(str), delta_by_contact["median"])
plt.xlabel("Número de contactos RING")
plt.ylabel("Mediana de ΔSASA")
plt.title("Enterramiento según número de contactos RING")
plt.tight_layout()
plt.savefig(OUTDIR / "08_delta_sasa_by_ring_contact_group.png", dpi=300)
plt.close()

# =========================
# 8. CENTRALIDAD ESTRUCTURAL OPERATIVA
# =========================
# No es centralidad de grafo estricta.
# Es un score compuesto: ΔSASA + fracción enterrada + contactos.

def zscore(s):
    s = s.fillna(0)
    if s.std() == 0:
        return s * 0
    return (s - s.mean()) / s.std()

df["z_delta_sasa"] = zscore(df["delta_sasa"])

if "buried_fraction_residue" in df.columns:
    df["z_buried_fraction"] = zscore(df["buried_fraction_residue"])
else:
    df["z_buried_fraction"] = 0

df["z_ring_count"] = zscore(df["ring_interaction_count"])

df["structural_centrality_score"] = (
    df["z_delta_sasa"] +
    df["z_buried_fraction"] +
    df["z_ring_count"]
)

centrality_summary = (
    df.groupby("interface_class")["structural_centrality_score"]
    .describe()
    .reindex(class_order)
)

centrality_summary.to_csv(OUTDIR / "09_structural_centrality_summary.csv")

boxplot_by_class(
    "structural_centrality_score",
    "09_structural_centrality_by_class.png",
    "Score de centralidad estructural",
    "Centralidad estructural operativa por clase"
)

# =========================
# 9. HOTSPOTS POR COMPLEJO
# =========================

complex_summary = (
    df.groupby("pdb_id")
    .agg(
        n_residues=("pdb_id", "size"),
        n_interface=(interface_col, "sum"),
        n_hotspot=("is_core_hotspot", "sum"),
        n_strong=("is_strong_interface", "sum"),
        n_ring=("has_ring_contact", "sum"),
        mean_delta_sasa=("delta_sasa", "mean"),
        median_delta_sasa=("delta_sasa", "median"),
    )
    .reset_index()
)

complex_summary["fraction_interface"] = complex_summary["n_interface"] / complex_summary["n_residues"]
complex_summary["fraction_hotspot"] = complex_summary["n_hotspot"] / complex_summary["n_residues"]
complex_summary["fraction_ring"] = complex_summary["n_ring"] / complex_summary["n_residues"]

complex_summary.to_csv(OUTDIR / "10_hotspots_by_complex.csv", index=False)

plt.figure(figsize=(7, 5))
plt.hist(complex_summary["fraction_hotspot"], bins=50)
plt.xlabel("Fracción de hotspots por complejo")
plt.ylabel("Número de complejos")
plt.title("Distribución de hotspots entre complejos")
plt.tight_layout()
plt.savefig(OUTDIR / "10_fraction_hotspots_by_complex.png", dpi=300)
plt.close()

# =========================
# 10. TABLA FINAL RESUMEN
# =========================

summary = {
    "n_total_residues": len(df),
    "n_interface": int((df[interface_col] == 1).sum()),
    "n_strong_interface": int((df["is_strong_interface"] == 1).sum()),
    "n_core_hotspot": int((df["is_core_hotspot"] == 1).sum()),
    "n_ring_contact": int((df["has_ring_contact"]).sum()),
    "fraction_interface": float((df[interface_col] == 1).mean()),
    "fraction_strong_interface": float((df["is_strong_interface"] == 1).mean()),
    "fraction_core_hotspot": float((df["is_core_hotspot"] == 1).mean()),
    "fraction_ring_contact": float(df["has_ring_contact"].mean()),
    "spearman_delta_sasa_ring_rho": float(rho),
    "spearman_delta_sasa_ring_p": float(p_spearman),
    "fisher_hotspot_ring_or": float(or_value),
    "fisher_hotspot_ring_p": float(fisher_p),
}

pd.DataFrame([summary]).to_csv(OUTDIR / "00_hotspots_global_summary.csv", index=False)

print("Análisis terminado.")
print("Output:", OUTDIR.resolve())