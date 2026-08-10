#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

infile = sys.argv[1]
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(infile, low_memory=False)

# =========================
# columnas reales de tu master
# =========================

df["consurf"] = pd.to_numeric(df["consurf_score"], errors="coerce")
df["delta_sasa"] = pd.to_numeric(df["delta_sasa"], errors="coerce")
df["buried_fraction_residue"] = pd.to_numeric(df["buried_fraction_residue"], errors="coerce")
df["ring"] = pd.to_numeric(df["ring_interaction_count"], errors="coerce").fillna(0)
df["has_ring"] = pd.to_numeric(df["has_ring_interface"], errors="coerce").fillna(0)

df["is_interface"] = df["interface_class"].astype(str).ne("non_interface").astype(int)
df["is_hotspot"] = df["interface_class"].astype(str).eq("core_hotspot").astype(int)
df["is_strong"] = df["interface_class"].astype(str).eq("strong_interface").astype(int)

df["resname"] = df["resname_x"].fillna(df.get("resname_y", np.nan))

protein = df[
    df["polymer_class"].astype(str).str.contains("protein", case=False, na=False)
].copy()

protein = protein.dropna(subset=["consurf", "delta_sasa"])


# =========================
# helper
# =========================

def savefig(name):
    plt.tight_layout()
    plt.savefig(outdir / name, dpi=300)
    plt.close()


def sample_df(data, n=100000):
    if len(data) > n:
        return data.sample(n, random_state=42)
    return data


# =========================
# 1. ConSurf por clase estructural
# =========================

order = ["non_interface", "interface", "strong_interface", "core_hotspot"]

plot_df = protein[protein["interface_class"].isin(order)].copy()

groups = [
    plot_df.loc[plot_df["interface_class"] == cls, "consurf"].dropna()
    for cls in order
]

plt.figure(figsize=(7, 4))
plt.boxplot(groups, labels=order, showfliers=False)
plt.ylabel("ConSurf score")
plt.xlabel("Clase estructural")
plt.title("Conservación proteica según clase de interfaz")
plt.xticks(rotation=25, ha="right")
savefig("01_consurf_por_clase_interface.png")


# =========================
# 2. ConSurf: interfaz vs no interfaz
# =========================

groups = [
    protein.loc[protein["is_interface"] == 0, "consurf"].dropna(),
    protein.loc[protein["is_interface"] == 1, "consurf"].dropna()
]

plt.figure(figsize=(5, 4))
plt.boxplot(groups, labels=["No interfaz", "Interfaz"], showfliers=False)
plt.ylabel("ConSurf score")
plt.title("Conservación en residuos de interfaz vs no interfaz")
savefig("02_consurf_interface_vs_nointerface.png")


# =========================
# 3. ConSurf: con contacto RING vs sin contacto RING
# =========================

groups = [
    protein.loc[protein["has_ring"] == 0, "consurf"].dropna(),
    protein.loc[protein["has_ring"] == 1, "consurf"].dropna()
]

plt.figure(figsize=(5, 4))
plt.boxplot(groups, labels=["Sin RING", "Con RING"], showfliers=False)
plt.ylabel("ConSurf score")
plt.title("Conservación según contacto directo RING")
savefig("03_consurf_ring_vs_noring.png")


# =========================
# 4. ConSurf según número de contactos RING
# =========================

protein["ring_bin"] = pd.cut(
    protein["ring"],
    bins=[-0.1, 0, 1, 2, 3, np.inf],
    labels=["0", "1", "2", "3", "≥4"]
)

groups = [
    protein.loc[protein["ring_bin"] == b, "consurf"].dropna()
    for b in ["0", "1", "2", "3", "≥4"]
]

plt.figure(figsize=(6, 4))
plt.boxplot(groups, labels=["0", "1", "2", "3", "≥4"], showfliers=False)
plt.xlabel("Número de contactos RING")
plt.ylabel("ConSurf score")
plt.title("Conservación según número de contactos directos")
savefig("04_consurf_por_numero_contactos_ring.png")


# =========================
# 5. Scatter ΔSASA vs ConSurf
# =========================

sub = sample_df(protein.dropna(subset=["delta_sasa", "consurf"]))

rho, p = spearmanr(sub["delta_sasa"], sub["consurf"])

plt.figure(figsize=(5, 4))
plt.scatter(sub["delta_sasa"], sub["consurf"], s=3, alpha=0.2)
plt.xlabel("ΔSASA")
plt.ylabel("ConSurf score")
plt.title(f"ΔSASA vs ConSurf\nSpearman rho={rho:.3f}, p={p:.2e}")
savefig("05_scatter_delta_sasa_vs_consurf.png")


# =========================
# 6. Scatter buried fraction vs ConSurf
# =========================

sub = sample_df(protein.dropna(subset=["buried_fraction_residue", "consurf"]))

rho, p = spearmanr(sub["buried_fraction_residue"], sub["consurf"])

plt.figure(figsize=(5, 4))
plt.scatter(sub["buried_fraction_residue"], sub["consurf"], s=3, alpha=0.2)
plt.xlabel("Fracción enterrada")
plt.ylabel("ConSurf score")
plt.title(f"Fracción enterrada vs ConSurf\nSpearman rho={rho:.3f}, p={p:.2e}")
savefig("06_scatter_buried_fraction_vs_consurf.png")


# =========================
# 7. Heatmap simple: aminoácido vs clase estructural
# =========================

aa_order = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
]

heat = (
    protein[protein["resname"].isin(aa_order)]
    .groupby(["resname", "interface_class"])
    .size()
    .unstack(fill_value=0)
)

heat = heat.reindex(index=aa_order, columns=order, fill_value=0)

# proporción por clase
heat_prop = heat.div(heat.sum(axis=0), axis=1)

plt.figure(figsize=(6, 7))
plt.imshow(heat_prop.values, aspect="auto")
plt.xticks(range(len(order)), order, rotation=30, ha="right")
plt.yticks(range(len(aa_order)), aa_order)
plt.colorbar(label="Proporción dentro de clase")
plt.title("Composición aminoacídica por clase estructural")
savefig("07_heatmap_aminoacidos_por_clase.png")

heat_prop.to_csv(outdir / "07_heatmap_aminoacidos_por_clase.csv")


# =========================
# 8. Conservación media por aminoácido
# =========================

aa_summary = (
    protein[protein["resname"].isin(aa_order)]
    .groupby("resname")
    .agg(
        n=("consurf", "count"),
        consurf_median=("consurf", "median"),
        consurf_mean=("consurf", "mean"),
        interface_fraction=("is_interface", "mean"),
        hotspot_fraction=("is_hotspot", "mean"),
        ring_fraction=("has_ring", "mean")
    )
    .reset_index()
)

aa_summary = aa_summary.sort_values("consurf_median")

plt.figure(figsize=(8, 4))
plt.bar(aa_summary["resname"], aa_summary["consurf_median"])
plt.ylabel("Mediana ConSurf")
plt.xlabel("Aminoácido")
plt.title("Conservación mediana por aminoácido")
plt.xticks(rotation=45)
savefig("08_consurf_mediana_por_aminoacido.png")

aa_summary.to_csv(outdir / "08_resumen_conservacion_por_aminoacido.csv", index=False)


# =========================
# 9. Fracción de interfaz / hotspot por aminoácido
# =========================

aa_summary2 = aa_summary.sort_values("interface_fraction", ascending=False)

plt.figure(figsize=(8, 4))
plt.bar(aa_summary2["resname"], aa_summary2["interface_fraction"])
plt.ylabel("Fracción en interfaz")
plt.xlabel("Aminoácido")
plt.title("Fracción de residuos en interfaz por aminoácido")
plt.xticks(rotation=45)
savefig("09_fraccion_interface_por_aminoacido.png")

aa_summary3 = aa_summary.sort_values("hotspot_fraction", ascending=False)

plt.figure(figsize=(8, 4))
plt.bar(aa_summary3["resname"], aa_summary3["hotspot_fraction"])
plt.ylabel("Fracción core hotspot")
plt.xlabel("Aminoácido")
plt.title("Fracción de hotspots por aminoácido")
plt.xticks(rotation=45)
savefig("10_fraccion_hotspot_por_aminoacido.png")


# =========================
# 10. Relación conservación-contactos-estructura
# =========================

summary_class = (
    protein
    .groupby("interface_class")
    .agg(
        n=("consurf", "count"),
        consurf_median=("consurf", "median"),
        delta_sasa_median=("delta_sasa", "median"),
        buried_fraction_median=("buried_fraction_residue", "median"),
        ring_mean=("ring", "mean"),
        ring_fraction=("has_ring", "mean")
    )
    .reset_index()
)

summary_class.to_csv(outdir / "10_summary_clase_estructura_conservacion.csv", index=False)

print("Listo. Figuras guardadas en:", outdir.resolve())