#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

INPUT_CSV = "complex_level_summary.csv"
OUTDIR = Path("dataset_imbalance_summary")
OUTDIR.mkdir(exist_ok=True)

# =========================
# Cargar datos
# =========================
df = pd.read_csv(INPUT_CSV)

print("Columnas disponibles:")
print(df.columns.tolist())

# =========================
# 1. Desbalance en número de cadenas
# =========================
if "n_chains" in df.columns:
    n_chains = pd.to_numeric(df["n_chains"], errors="coerce")

    desc = n_chains.describe()
    quant = n_chains.quantile([0.5, 0.75, 0.9, 0.95, 0.99])

    frac_gt_5 = (n_chains > 5).mean()
    frac_gt_20 = (n_chains > 20).mean()
    frac_gt_50 = (n_chains > 50).mean()

    desc.to_csv(OUTDIR / "n_chains_describe.csv", header=["value"])

    quant_df = quant.reset_index()
    quant_df.columns = ["quantile", "value"]
    quant_df.to_csv(OUTDIR / "n_chains_quantiles.csv", index=False)

    pd.DataFrame([
        {"metric": "fraction_n_chains_gt_5", "value": frac_gt_5},
        {"metric": "fraction_n_chains_gt_20", "value": frac_gt_20},
        {"metric": "fraction_n_chains_gt_50", "value": frac_gt_50},
    ]).to_csv(OUTDIR / "n_chains_tail_fractions.csv", index=False)

# =========================
# 2. Desbalance en oligomerización
# =========================
if "oligomer_type" in df.columns:
    olig_counts = df["oligomer_type"].value_counts(dropna=False)
    olig_props = df["oligomer_type"].value_counts(dropna=False, normalize=True)

    olig_df = pd.DataFrame({
        "oligomer_type": olig_counts.index,
        "count": olig_counts.values,
        "proportion": olig_props.reindex(olig_counts.index).values
    })
    olig_df.to_csv(OUTDIR / "oligomer_type_distribution.csv", index=False)

# =========================
# 3. Desbalance por tamaño por secuencia
# =========================
if "size_category_seq" in df.columns:
    size_seq_counts = df["size_category_seq"].value_counts(dropna=False)
    size_seq_props = df["size_category_seq"].value_counts(dropna=False, normalize=True)

    size_seq_df = pd.DataFrame({
        "size_category_seq": size_seq_counts.index,
        "count": size_seq_counts.values,
        "proportion": size_seq_props.reindex(size_seq_counts.index).values
    })
    size_seq_df.to_csv(OUTDIR / "size_category_seq_distribution.csv", index=False)

# =========================
# 4. Desbalance por tamaño por MW
# =========================
if "size_category_mw" in df.columns:
    size_mw_counts = df["size_category_mw"].value_counts(dropna=False)
    size_mw_props = df["size_category_mw"].value_counts(dropna=False, normalize=True)

    size_mw_df = pd.DataFrame({
        "size_category_mw": size_mw_counts.index,
        "count": size_mw_counts.values,
        "proportion": size_mw_props.reindex(size_mw_counts.index).values
    })
    size_mw_df.to_csv(OUTDIR / "size_category_mw_distribution.csv", index=False)

# =========================
# 5. Desbalance temporal
# =========================
year_col = None
for candidate in ["Publication Year", "publication_year", "PublicationYear"]:
    if candidate in df.columns:
        year_col = candidate
        break

if year_col is not None:
    years = pd.to_numeric(df[year_col], errors="coerce")

    year_counts = years.value_counts(dropna=False).sort_index()
    year_props = years.value_counts(dropna=False, normalize=True).sort_index()

    year_df = pd.DataFrame({
        "publication_year": year_counts.index,
        "count": year_counts.values,
        "proportion": year_props.reindex(year_counts.index).values
    })
    year_df.to_csv(OUTDIR / "publication_year_distribution.csv", index=False)

# =========================
# Resumen de texto automático
# =========================
with open(OUTDIR / "imbalance_summary.txt", "w", encoding="utf-8") as fh:
    fh.write("Dataset imbalance summary\n")
    fh.write("========================\n\n")

    if "n_chains" in df.columns:
        fh.write("1. Chain count imbalance\n")
        fh.write("------------------------\n")
        fh.write(desc.to_string())
        fh.write("\n\nQuantiles:\n")
        fh.write(quant.to_string())
        fh.write("\n\n")
        fh.write(f"Fraction with n_chains > 5:  {frac_gt_5:.4f}\n")
        fh.write(f"Fraction with n_chains > 20: {frac_gt_20:.4f}\n")
        fh.write(f"Fraction with n_chains > 50: {frac_gt_50:.4f}\n\n")

    if "oligomer_type" in df.columns:
        fh.write("2. Oligomer type imbalance\n")
        fh.write("--------------------------\n")
        fh.write(olig_df.to_string(index=False))
        fh.write("\n\n")

    if "size_category_seq" in df.columns:
        fh.write("3. Size category by sequence length\n")
        fh.write("-----------------------------------\n")
        fh.write(size_seq_df.to_string(index=False))
        fh.write("\n\n")

    if "size_category_mw" in df.columns:
        fh.write("4. Size category by molecular weight\n")
        fh.write("------------------------------------\n")
        fh.write(size_mw_df.to_string(index=False))
        fh.write("\n\n")

    if year_col is not None:
        fh.write("5. Publication year imbalance\n")
        fh.write("-----------------------------\n")
        fh.write(year_df.to_string(index=False))
        fh.write("\n\n")

print(f"Listo. Resúmenes guardados en: {OUTDIR.resolve()}")