import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

infile = sys.argv[1]
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(infile, low_memory=False)

df["shannon_mean"] = pd.to_numeric(df["shannon_mean"], errors="coerce")
df["delta_sasa"] = pd.to_numeric(df["delta_sasa"], errors="coerce")
df["buried_fraction_residue"] = pd.to_numeric(df["buried_fraction_residue"], errors="coerce")
df["ring"] = pd.to_numeric(df["has_ring_interface"], errors="coerce").fillna(0)

df["is_interface"] = pd.to_numeric(df["is_interface_abs"], errors="coerce").fillna(0).astype(int)
df["is_hotspot"] = pd.to_numeric(df["is_core_hotspot"], errors="coerce").fillna(0).astype(int)
df["resname"] = df["resname"].astype(str)

sub = df.dropna(subset=["shannon_mean"]).copy()

print("Filas con RNA Shannon:", len(sub))
print("PDBs:", sub["pdb_id"].nunique())
print("Residuos únicos PDB-chain-pos:", sub[["pdb_id","auth_asym_id","auth_seq_id"]].drop_duplicates().shape[0])

def savefig(name):
    plt.tight_layout()
    plt.savefig(outdir / name, dpi=300)
    plt.close()

# 1. Shannon del RNA contactado por clase estructural proteica
order = ["non_interface", "interface", "strong_interface", "core_hotspot"]
plot_df = sub[sub["interface_class"].isin(order)].copy()

groups = [plot_df.loc[plot_df["interface_class"] == x, "shannon_mean"].dropna() for x in order]

plt.figure(figsize=(7,4))
plt.boxplot(groups, labels=order, showfliers=False)
plt.ylabel("Shannon medio del RNA contactado")
plt.xlabel("Clase estructural del residuo proteico")
plt.title("Conservación del RNA contactado según clase proteica")
plt.xticks(rotation=25, ha="right")
savefig("01_rna_shannon_por_clase_proteica.png")

# 2. Interfaz vs no interfaz
groups = [
    sub.loc[sub["is_interface"] == 0, "shannon_mean"],
    sub.loc[sub["is_interface"] == 1, "shannon_mean"]
]

plt.figure(figsize=(5,4))
plt.boxplot(groups, labels=["No interfaz", "Interfaz"], showfliers=False)
plt.ylabel("Shannon medio del RNA contactado")
plt.title("RNA contactado: interfaz vs no interfaz proteica")
savefig("02_rna_shannon_interface_vs_nointerface.png")

# 3. Hotspot vs no hotspot
groups = [
    sub.loc[sub["is_hotspot"] == 0, "shannon_mean"],
    sub.loc[sub["is_hotspot"] == 1, "shannon_mean"]
]

plt.figure(figsize=(5,4))
plt.boxplot(groups, labels=["No hotspot", "Hotspot"], showfliers=False)
plt.ylabel("Shannon medio del RNA contactado")
plt.title("RNA contactado por hotspots proteicos")
savefig("03_rna_shannon_hotspot_vs_nohotspot.png")

# 4. ΔSASA vs Shannon del RNA contactado
tmp = sub.dropna(subset=["delta_sasa", "shannon_mean"])
rho, p = spearmanr(tmp["delta_sasa"], tmp["shannon_mean"])

plt.figure(figsize=(5,4))
plt.scatter(tmp["delta_sasa"], tmp["shannon_mean"], s=3, alpha=0.2)
plt.xlabel("ΔSASA del residuo proteico")
plt.ylabel("Shannon medio del RNA contactado")
plt.title(f"ΔSASA proteína vs Shannon RNA\nrho={rho:.3f}, p={p:.2e}")
savefig("04_delta_sasa_vs_rna_shannon.png")

# 5. Fracción enterrada vs Shannon
tmp = sub.dropna(subset=["buried_fraction_residue", "shannon_mean"])
rho, p = spearmanr(tmp["buried_fraction_residue"], tmp["shannon_mean"])

plt.figure(figsize=(5,4))
plt.scatter(tmp["buried_fraction_residue"], tmp["shannon_mean"], s=3, alpha=0.2)
plt.xlabel("Fracción enterrada del residuo proteico")
plt.ylabel("Shannon medio del RNA contactado")
plt.title(f"Fracción enterrada vs Shannon RNA\nrho={rho:.3f}, p={p:.2e}")
savefig("05_buried_fraction_vs_rna_shannon.png")

# 6. Shannon RNA por aminoácido contactante
aa_order = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
]

aa = (
    sub[sub["resname"].isin(aa_order)]
    .groupby("resname")
    .agg(
        n=("shannon_mean", "count"),
        shannon_median=("shannon_mean", "median"),
        shannon_mean=("shannon_mean", "mean"),
        interface_fraction=("is_interface", "mean"),
        hotspot_fraction=("is_hotspot", "mean")
    )
    .reset_index()
)

aa = aa.sort_values("shannon_median")

plt.figure(figsize=(8,4))
plt.bar(aa["resname"], aa["shannon_median"])
plt.ylabel("Mediana Shannon del RNA contactado")
plt.xlabel("Aminoácido proteico")
plt.title("Conservación del RNA contactado por aminoácido")
plt.xticks(rotation=45)
savefig("06_rna_shannon_por_aminoacido_proteico.png")

aa.to_csv(outdir / "06_rna_shannon_por_aminoacido_proteico.csv", index=False)

# 7. Nucleótidos contactados
if "rna_nt_contacted" in sub.columns:
    nt_rows = []
    for _, row in sub.dropna(subset=["rna_nt_contacted"]).iterrows():
        nts = str(row["rna_nt_contacted"]).split(",")
        for nt in nts:
            nt = nt.strip()
            if nt in ["A", "U", "G", "C"]:
                nt_rows.append({
                    "nt": nt,
                    "shannon_mean": row["shannon_mean"],
                    "is_interface": row["is_interface"],
                    "is_hotspot": row["is_hotspot"]
                })

    nt_df = pd.DataFrame(nt_rows)

    if len(nt_df) > 0:
        nt_summary = (
            nt_df.groupby("nt")
            .agg(
                n=("shannon_mean", "count"),
                shannon_median=("shannon_mean", "median"),
                interface_fraction=("is_interface", "mean"),
                hotspot_fraction=("is_hotspot", "mean")
            )
            .reset_index()
        )

        nt_summary.to_csv(outdir / "07_rna_nt_summary.csv", index=False)

        nt_summary = nt_summary.sort_values("shannon_median")

        plt.figure(figsize=(4,4))
        plt.bar(nt_summary["nt"], nt_summary["shannon_median"])
        plt.ylabel("Mediana Shannon")
        plt.xlabel("Nucleótido contactado")
        plt.title("Conservación por nucleótido contactado")
        savefig("07_rna_shannon_por_nt_contactado.png")

# 8. resumen por clase
summary = (
    sub.groupby("interface_class")
    .agg(
        n=("shannon_mean", "count"),
        shannon_mean=("shannon_mean", "mean"),
        shannon_median=("shannon_mean", "median"),
        delta_sasa_median=("delta_sasa", "median"),
        buried_fraction_median=("buried_fraction_residue", "median"),
        hotspot_fraction=("is_hotspot", "mean")
    )
    .reset_index()
)

summary.to_csv(outdir / "08_summary_rna_shannon_por_clase.csv", index=False)

print("Listo. Figuras en:", outdir)