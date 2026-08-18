#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# =========================
# Configuración
# =========================
INPUT_CSV = "/media/usuario/Seagate Basic1/Tesis_Maestria/24_jul_25/rcsb_pdb_custom_report_20250724182030.csv"
OUTDIR = Path("dataset_metadata_summary_2")
OUTDIR.mkdir(exist_ok=True)

# =========================
# Funciones auxiliares
# =========================
def first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_sequence(seq):
    if pd.isna(seq):
        return np.nan
    seq = str(seq).strip().upper().replace(" ", "")
    return seq if seq != "" else np.nan

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def save_value_counts(series, outpath, index_name, value_name="count"):
    vc = series.value_counts(dropna=False).rename_axis(index_name).reset_index()
    vc.columns = [index_name, value_name]
    vc.to_csv(outpath, index=False)
    return vc

def classify_size_by_length(length):
    if pd.isna(length):
        return "unknown"
    elif length < 1000:
        return "small"
    elif length < 10000:
        return "medium"
    else:
        return "large"

def classify_size_by_mw(mw):
    if pd.isna(mw):
        return "unknown"
    elif mw < 1000:
        return "small"
    elif mw < 10000:
        return "medium"
    else:
        return "large"

def classify_oligomer(n_chains):
    try:
        if pd.isna(n_chains):
            return "unknown"
        n_chains = float(n_chains)
    except Exception:
        return "unknown"

    if n_chains == 1:
        return "monomer"
    elif n_chains == 2:
        return "dimer"
    elif 3 <= n_chains <= 5:
        return "oligomer_3_5"
    else:
        return "large_complex"

# =========================
# Lectura CSV
# =========================
df = pd.read_csv(INPUT_CSV, header=None, low_memory=False)

header1 = df.iloc[0].fillna("")
header2 = df.iloc[1].fillna("")

columns = []
for h1, h2 in zip(header1, header2):
    columns.append(h2.strip() if h2 != "" else h1.strip())

df = df[2:].copy()
df.columns = [c.strip() for c in columns]

# 🔧 FIX CRÍTICO
df = df.loc[:, df.columns != ""]
df = df.loc[:, ~df.columns.duplicated()]

print("Columnas detectadas:")
print(df.columns.tolist())

pd.DataFrame({"column_name": df.columns}).to_csv(OUTDIR / "detected_columns.csv", index=False)

# =========================
# Columnas clave
# =========================
entry_col = first_existing_column(df, ["Entry ID"])
pdb_col = first_existing_column(df, ["PDB ID"])
method_col = first_existing_column(df, ["Experimental Method"])
release_col = first_existing_column(df, ["Release Date"])
pubyear_col = first_existing_column(df, ["Publication Year"])
seq_col = first_existing_column(df, ["Sequence"])
asym_col = first_existing_column(df, ["Asym ID"])
auth_asym_col = first_existing_column(df, ["Auth Asym ID"])
mw_model_col = first_existing_column(df, ["Molecular Weight per Deposited Model"])
entity_mw_col = first_existing_column(df, ["Molecular Weight (Entity)"])
gene_col = first_existing_column(df, ["Gene Name"])
host_col = first_existing_column(df, ["Expression Host"])

if entry_col is None and pdb_col is not None:
    entry_col = pdb_col

if entry_col is None:
    raise ValueError("No se encontró columna de Entry ID / PDB ID")

# =========================
# Limpieza
# =========================
df[entry_col] = df[entry_col].ffill()

for col in [pdb_col, method_col, release_col, pubyear_col, mw_model_col]:
    if col is not None:
        df[col] = df[col].ffill()

if seq_col is not None:
    df[seq_col] = df[seq_col].apply(normalize_sequence)
    df["seq_len"] = df[seq_col].str.len()
else:
    df["seq_len"] = np.nan

if mw_model_col is not None:
    df[mw_model_col] = safe_numeric(df[mw_model_col])

if entity_mw_col is not None:
    df[entity_mw_col] = safe_numeric(df[entity_mw_col])

if pubyear_col is not None:
    df[pubyear_col] = safe_numeric(df[pubyear_col])

# =========================
# Resumen entidad
# =========================
entity_summary = df.copy()
entity_summary.to_csv(OUTDIR / "entity_level_summary.csv", index=False)

# =========================
# Número de cadenas
# =========================
if asym_col is not None:
    chain_counts = (
        df[[entry_col, asym_col]]
        .dropna()
        .drop_duplicates()
        .groupby(entry_col)[asym_col]
        .nunique()
        .reset_index()
    )
    chain_counts.columns = [entry_col, "n_chains"]
else:
    chain_counts = pd.DataFrame({
        entry_col: df[entry_col].drop_duplicates(),
        "n_chains": np.nan
    })

# =========================
# Resumen complejo
# =========================
complex_summary = df.groupby(entry_col, dropna=False).agg({
    "seq_len": "sum"
}).reset_index()

complex_summary = complex_summary.rename(columns={
    "seq_len": "total_sequence_length"
})

complex_summary = complex_summary.merge(chain_counts, on=entry_col, how="left")

# 🔧 FIX CLAVE
complex_summary["n_chains"] = pd.to_numeric(complex_summary["n_chains"], errors="coerce")

# Clasificaciones
complex_summary["size_category_seq"] = complex_summary["total_sequence_length"].apply(classify_size_by_length)

if mw_model_col is not None and mw_model_col in df.columns:
    mw_map = df.groupby(entry_col)[mw_model_col].first().reset_index()
    complex_summary = complex_summary.merge(mw_map, on=entry_col, how="left")
    complex_summary["size_category_mw"] = complex_summary[mw_model_col].apply(classify_size_by_mw)
else:
    complex_summary["size_category_mw"] = "unknown"

complex_summary["oligomer_type"] = complex_summary["n_chains"].apply(classify_oligomer)

complex_summary.to_csv(OUTDIR / "complex_level_summary.csv", index=False)

# =========================
# Métricas
# =========================
global_metrics = {
    "n_complexes": complex_summary[entry_col].nunique(),
    "mean_n_chains": float(complex_summary["n_chains"].mean()),
    "median_n_chains": float(complex_summary["n_chains"].median()),
    "mean_length": float(complex_summary["total_sequence_length"].mean()),
    "median_length": float(complex_summary["total_sequence_length"].median()),
}

pd.DataFrame(list(global_metrics.items()), columns=["metric", "value"]).to_csv(
    OUTDIR / "global_metrics.csv", index=False
)

# =========================
# Tablas
# =========================
save_value_counts(complex_summary["oligomer_type"], OUTDIR / "oligomer_type_counts.csv", "oligomer_type")
save_value_counts(complex_summary["size_category_seq"], OUTDIR / "size_category_seq_counts.csv", "size_category_seq")

# =========================
# Gráficos
# =========================
plt.figure()
complex_summary["oligomer_type"].value_counts().plot(kind="bar")
plt.tight_layout()
plt.savefig(OUTDIR / "oligomer_type_distribution.png")
plt.close()

plt.figure()
plt.hist(complex_summary["n_chains"].dropna(), bins=30)
plt.tight_layout()
plt.savefig(OUTDIR / "chain_count_distribution.png")
plt.close()

print("OK - script terminado sin errores")