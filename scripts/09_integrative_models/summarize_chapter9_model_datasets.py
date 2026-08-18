#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/"
    "Tesis_Maestria"
)

BESTROW = BASE / "01_mar_26/variants_bestrow.csv"

BESTROW_RING = (
    BASE / "01_mar_26/variants_bestrow_ring.csv"
)

BESTROW_RING_RNA = (
    BASE / "02_mar_26/3_rna/"
    "variants_bestrow_ring_rna.csv"
)

OUTDIR = (
    BASE / "02_ago_26_ultimos_retoques"
)

OUTPUT_CSV = (
    OUTDIR / "chapter9_model_dataset_summary.csv"
)

OUTPUT_TXT = (
    OUTDIR / "chapter9_model_dataset_summary.txt"
)


# ============================================================
# TARGETS
# ============================================================
def add_targets(df):
    df = df.copy()

    # yA: Disease_Association presente y no vacía
    da = df["Disease_Association"]

    df["yA"] = (
        da.notna()
        & da.astype("string").str.strip().ne("")
    ).astype(int)

    # yB:
    # PolyPhen damaging AND SIFT deleterious
    pp = (
        df["Prediction_PolyPhen"]
        .astype("string")
        .str.lower()
    )

    sf = (
        df["Prediction_SIFT"]
        .astype("string")
        .str.lower()
    )

    damaging_pp = pp.str.contains(
        "damaging",
        na=False
    )

    deleterious_sf = sf.str.contains(
        "deleterious",
        na=False
    )

    df["yB"] = (
        damaging_pp &
        deleterious_sf
    ).astype(int)

    return df


# ============================================================
# CLEANING HELPERS
# ============================================================

def numeric(df, columns):
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


def remove_low_conf(df):
    """
    Reproduce el tratamiento usado en los scripts
    de modelado para is_low_conf.
    """
    df = df.copy()

    if "is_low_conf" not in df.columns:
        return df

    low = (
        df["is_low_conf"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df[
        ~low.isin(["true", "1", "yes"])
    ].copy()


def summarize(
    name,
    df,
    source,
    filters,
    expected_n=None,
):
    n = len(df)

    if "uniprot_id" in df.columns:
        proteins = df["uniprot_id"].nunique()
    else:
        proteins = np.nan

    row = {
        "analysis": name,
        "source_file": source,
        "filters": filters,
        "n_variants": n,
        "n_proteins": proteins,
        "yA_positive_n": (
            int(df["yA"].sum())
            if "yA" in df.columns else np.nan
        ),
        "yA_positive_fraction": (
            float(df["yA"].mean())
            if "yA" in df.columns else np.nan
        ),
        "yB_positive_n": (
            int(df["yB"].sum())
            if "yB" in df.columns else np.nan
        ),
        "yB_positive_fraction": (
            float(df["yB"].mean())
            if "yB" in df.columns else np.nan
        ),
        "expected_n": expected_n,
        "n_matches_expected": (
            n == expected_n
            if expected_n is not None
            else np.nan
        ),
    }

    return row


# ============================================================
# LOAD
# ============================================================

print("Cargando datasets...")

best = pd.read_csv(
    BESTROW,
    low_memory=False
)

ring = pd.read_csv(
    BESTROW_RING,
    low_memory=False
)

rna = pd.read_csv(
    BESTROW_RING_RNA,
    low_memory=False
)

best = add_targets(best)
ring = add_targets(ring)
rna = add_targets(rna)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

continuous = [
    "consurf_score",
    "msa_coverage",
    "iupred_score",
    "buried_fraction_residue",
    "delta_sasa_norm",
    "rna_contact_count",
]

best = numeric(best, continuous)
ring = numeric(ring, continuous)
rna = numeric(rna, continuous)

for df in [best, ring, rna]:
    if "has_ring_interface" in df.columns:
        df["has_ring_interface"] = pd.to_numeric(
            df["has_ring_interface"],
            errors="coerce"
        )


# ============================================================
# SUMMARIES
# ============================================================

rows = []


# ------------------------------------------------------------
# 1. Dataset representativo completo
# ------------------------------------------------------------

rows.append(
    summarize(
        name="Dataset representativo de variantes",
        df=best,
        source="variants_bestrow.csv",
        filters="Sin filtros adicionales",
        expected_n=278221,
    )
)


# ------------------------------------------------------------
# 2. Modelo general inicial yB
#
# modelo.py:
# consurf, iupred, msa_coverage disponibles
# is_low_conf excluido
# buried_fraction disponible
# ------------------------------------------------------------

m_general = best[
    best["consurf_score"].notna() &
    best["iupred_score"].notna()
].copy()

for c in [
    "consurf_score",
    "buried_fraction_residue",
    "iupred_score",
]:
    m_general[c] = pd.to_numeric(
        m_general[c],
        errors="coerce"
    )

m_general = m_general.dropna(
    subset=[
        "consurf_score",
        "buried_fraction_residue",
        "iupred_score",
    ]
)

m_general["is_low_conf"] = (
    m_general["is_low_conf"]
    .astype(str)
    .str.lower()
)

m_general = m_general[
    ~m_general["is_low_conf"].isin(
        ["true", "1", "yes"]
    )
].copy()

rows.append(
    summarize(
        name="Modelo general inicial yB",
        df=m_general,
        source="variants_bestrow.csv",
        filters=(
            "ConSurf + IUPred disponibles; "
            "buried_fraction completa; "
            "exclusión de is_low_conf"
        ),
        expected_n=64489,
    )
)


# ------------------------------------------------------------
# 3. Modelo con RING como predictor
#
# modelo_ring.py
# ------------------------------------------------------------

m_ring_predictor = ring[
    ring["consurf_score"].notna() &
    ring["iupred_score"].notna()
].copy()

m_ring_predictor = remove_low_conf(
    m_ring_predictor
)

m_ring_predictor = m_ring_predictor.dropna(
    subset=[
        "consurf_score",
        "buried_fraction_residue",
        "iupred_score",
    ]
)

rows.append(
    summarize(
        name="Modelo con RING como predictor",
        df=m_ring_predictor,
        source="variants_bestrow_ring.csv",
        filters=(
            "ConSurf + IUPred disponibles; "
            "exclusión low_conf; "
            "buried_fraction completa"
        ),
        expected_n=93700,
    )
)


# ------------------------------------------------------------
# 4. Modelo inferencial restringido a RING
#
# modelo_ring_igual1.py / modelo_final_semana4_yA.py
#
# Esperado histórico: 2507
# ------------------------------------------------------------

m_ring1 = m_ring_predictor[
    m_ring_predictor[
        "has_ring_interface"
    ] == 1
].copy()

rows.append(
    summarize(
        name="Subconjunto inferencial RING=1",
        df=m_ring1,
        source="variants_bestrow_ring.csv",
        filters=(
            "Filtros del modelo RING + "
            "has_ring_interface == 1"
        ),
        expected_n=2507,
    )
)


# ------------------------------------------------------------
# 5. Subconjunto común para CV estructural
#
# semana_2.ipynb:
# RING=1, ConSurf, IUPred, buried,
# uniprot_id y target disponibles
#
# Esperado: 2048
# ------------------------------------------------------------

m_cv = ring[
    ring["consurf_score"].notna() &
    ring["iupred_score"].notna()
].copy()

m_cv = remove_low_conf(m_cv)

m_cv = m_cv[
    m_cv["has_ring_interface"] == 1
].copy()

m_cv = m_cv.dropna(
    subset=[
        "consurf_score",
        "buried_fraction_residue",
        "iupred_score",
        "uniprot_id",
    ]
)

rows.append(
    summarize(
        name="Subconjunto CV estructural RING=1",
        df=m_cv,
        source="variants_bestrow_ring.csv",
        filters=(
            "RING=1; ConSurf, IUPred, buried_fraction "
            "y UniProt disponibles; exclusión low_conf"
        ),
        expected_n=2507,
    )
)


# ------------------------------------------------------------
# 6. Dataset ARN completo
# ------------------------------------------------------------

rna["rna_contact_count"] = (
    pd.to_numeric(
        rna["rna_contact_count"],
        errors="coerce"
    )
    .fillna(0)
)

rna["rna_contact"] = (
    rna["rna_contact_count"] > 0
).astype(int)

rows.append(
    summarize(
        name="Dataset de variantes con variables ARN",
        df=rna,
        source="variants_bestrow_ring_rna.csv",
        filters="Sin filtros adicionales",
        expected_n=278221,
    )
)


# ------------------------------------------------------------
# 7. Modelo ampliado ARN yA
#
# modelo_control_rna_contanct.py
#
# Esperado: 99084
# ------------------------------------------------------------

m_rna_control = rna[
    rna["consurf_score"].notna() &
    rna["iupred_score"].notna()
].copy()

m_rna_control = m_rna_control.dropna(
    subset=[
        "consurf_score",
        "delta_sasa_norm",
        "iupred_score",
        "uniprot_id",
    ]
)

rows.append(
    summarize(
        name="Modelo ampliado ARN yA",
        df=m_rna_control,
        source="variants_bestrow_ring_rna.csv",
        filters=(
            "ConSurf + IUPred + delta_sasa_norm "
            "+ UniProt disponibles"
        ),
        expected_n=99084,
    )
)


# ------------------------------------------------------------
# 8. CV incremental ARN
#
# cv_feature_increment_rna_v2.py
#
# Esperado: 2507
# ------------------------------------------------------------

m_rna_cv = rna[
    rna["consurf_score"].notna() &
    rna["iupred_score"].notna()
].copy()

for c in [
    "consurf_score",
    "iupred_score",
    "has_ring_interface",
    "is_core_hotspot",
    "rna_contact_count",
]:
    if c in m_rna_cv.columns:
        m_rna_cv[c] = pd.to_numeric(
            m_rna_cv[c],
            errors="coerce"
        )

m_rna_cv = remove_low_conf(m_rna_cv)

m_rna_cv["is_core_hotspot"] = (
    m_rna_cv["is_core_hotspot"]
    .fillna(0)
    .astype(int)
)

m_rna_cv = m_rna_cv[
    m_rna_cv["has_ring_interface"] == 1
].copy()

m_rna_cv = m_rna_cv.dropna(
    subset=[
        "consurf_score",
        "iupred_score",
        "rna_contact_count",
        "uniprot_id",
    ]
)

rows.append(
    summarize(
        name="CV incremental de contacto ARN",
        df=m_rna_cv,
        source="variants_bestrow_ring_rna.csv",
        filters=(
            "ConSurf + IUPred + rna_contact_count "
            "+ UniProt disponibles; "
            "subconjunto reproducido por script CV"
        ),
        expected_n=2507,
    )
)


# ============================================================
# OUTPUT
# ============================================================

summary = pd.DataFrame(rows)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    OUTPUT_CSV,
    index=False
)

with OUTPUT_TXT.open(
    "w",
    encoding="utf-8"
) as fh:

    fh.write(
        "CHAPTER 9 MODEL DATASET SUMMARY\n"
    )

    fh.write("=" * 100 + "\n\n")

    fh.write(
        summary.to_string(index=False)
    )

    fh.write("\n\n")

    fh.write(
        "AUDITORÍA DE N ESPERADOS\n"
    )

    fh.write("=" * 100 + "\n\n")

    for _, row in summary.iterrows():
        if pd.isna(row["expected_n"]):
            continue

        status = (
            "OK"
            if row["n_matches_expected"]
            else "REVISAR"
        )

        fh.write(
            f"{status}: "
            f"{row['analysis']} -> "
            f"N={row['n_variants']} "
            f"(esperado={row['expected_n']})\n"
        )


print("\nResumen final:\n")

cols = [
    "analysis",
    "n_variants",
    "n_proteins",
    "yA_positive_fraction",
    "yB_positive_fraction",
    "expected_n",
    "n_matches_expected",
]

print(
    summary[cols].to_string(
        index=False
    )
)

print("\nArchivos escritos:")
print(OUTPUT_CSV)
print(OUTPUT_TXT)