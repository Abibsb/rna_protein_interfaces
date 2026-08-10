#!/usr/bin/env python3
"""
Versión de bajo consumo de memoria:
asociación entre aguas puente, contactos RING y core hotspots.

Unidad de análisis:
    un residuo proteico único por (pdb_id, uniprot_id, uniprot_pos)

Análisis:
    1) has_bridging_water frente a has_ring;
    2) has_bridging_water frente a is_core_hotspot;
    3) ring_interaction_count entre residuos con y sin agua;
    4) Spearman entre n_bridging_waters y ring_interaction_count
       solo entre residuos con agua.

Uso:
    python water_vs_ring_hotspot_lowmem.py \
      --master MASTER_interface_dataset.csv \
      --out_prefix resultados_water_ring_hotspot

Opcional:
    --chunksize 250000
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def to_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(TRUE_VALUES)
    )


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """
    Tabla 2x2:
                  outcome=1  outcome=0
    exposure=1        a          b
    exposure=0        c          d

    Se aplica corrección de Haldane–Anscombe si alguna celda es cero.
    """
    cells = np.array([a, b, c, d], dtype=float)
    if np.any(cells == 0):
        cells += 0.5

    aa, bb, cc, dd = cells
    or_value = (aa * dd) / (bb * cc)
    se_log_or = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lower = math.exp(math.log(or_value) - 1.96 * se_log_or)
    upper = math.exp(math.log(or_value) + 1.96 * se_log_or)
    return or_value, lower, upper


def binary_association(
    df: pd.DataFrame,
    exposure_col: str,
    outcome_col: str,
    label: str
) -> dict:
    tab = pd.crosstab(df[exposure_col], df[outcome_col]).reindex(
        index=[True, False],
        columns=[True, False],
        fill_value=0
    )

    a = int(tab.loc[True, True])
    b = int(tab.loc[True, False])
    c = int(tab.loc[False, True])
    d = int(tab.loc[False, False])

    fisher_or, p_value = fisher_exact([[a, b], [c, d]])
    corrected_or, ci_low, ci_high = odds_ratio_ci(a, b, c, d)

    p_outcome_exposed = a / (a + b) if (a + b) else np.nan
    p_outcome_unexposed = c / (c + d) if (c + d) else np.nan

    return {
        "comparison": label,
        "a_exposure1_outcome1": a,
        "b_exposure1_outcome0": b,
        "c_exposure0_outcome1": c,
        "d_exposure0_outcome0": d,
        "fisher_odds_ratio": fisher_or,
        "odds_ratio_for_ci": corrected_or,
        "or_ci95_low": ci_low,
        "or_ci95_high": ci_high,
        "fisher_p_value_two_sided": p_value,
        "outcome_fraction_exposed": p_outcome_exposed,
        "outcome_fraction_unexposed": p_outcome_unexposed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--out_prefix", default="water_vs_ring_hotspot")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    header = pd.read_csv(args.master, nrows=0)
    available = set(header.columns)

    required = {
        "pdb_id",
        "uniprot_id",
        "uniprot_pos",
        "polymer_class",
        "has_bridging_water",
        "ring_interaction_count",
    }

    hotspot_source = None
    if "is_core_hotspot" in available:
        hotspot_source = "is_core_hotspot"
    elif "interface_class" in available:
        hotspot_source = "interface_class"
    else:
        raise SystemExit(
            "No se encontró ni 'is_core_hotspot' ni 'interface_class' "
            "para definir los core hotspots."
        )

    missing = sorted(required - available)
    if missing:
        raise SystemExit(f"Faltan columnas obligatorias: {missing}")

    usecols = list(required)
    usecols.append(hotspot_source)
    has_n_waters = "n_bridging_waters" in available
    if has_n_waters:
        usecols.append("n_bridging_waters")

    reduced_chunks = []
    processed = 0

    for chunk in pd.read_csv(
        args.master,
        usecols=usecols,
        chunksize=args.chunksize,
        low_memory=False
    ):
        processed += len(chunk)

        chunk = chunk[
            chunk["polymer_class"].astype(str).str.lower().eq("protein")
        ].copy()

        if chunk.empty:
            print(f"Procesadas {processed:,} filas", end="\r")
            continue

        chunk["has_water"] = to_bool(chunk["has_bridging_water"])

        if hotspot_source == "is_core_hotspot":
            chunk["is_core_hotspot"] = to_bool(chunk["is_core_hotspot"])
        else:
            chunk["is_core_hotspot"] = (
                chunk["interface_class"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("core_hotspot")
            )

        chunk["ring_interaction_count"] = pd.to_numeric(
            chunk["ring_interaction_count"],
            errors="coerce",
            downcast="integer"
        ).fillna(0)

        chunk["uniprot_pos"] = pd.to_numeric(
            chunk["uniprot_pos"],
            errors="coerce",
            downcast="integer"
        )

        if has_n_waters:
            chunk["n_bridging_waters"] = pd.to_numeric(
                chunk["n_bridging_waters"],
                errors="coerce",
                downcast="integer"
            ).fillna(0)
        else:
            chunk["n_bridging_waters"] = chunk["has_water"].astype("int8")

        chunk = chunk.dropna(
            subset=["pdb_id", "uniprot_id", "uniprot_pos"]
        )

        chunk = chunk.drop_duplicates(
            subset=["pdb_id", "uniprot_id", "uniprot_pos"]
        )

        reduced_chunks.append(
            chunk[[
                "pdb_id",
                "uniprot_id",
                "uniprot_pos",
                "has_water",
                "is_core_hotspot",
                "ring_interaction_count",
                "n_bridging_waters",
            ]]
        )

        print(f"Procesadas {processed:,} filas", end="\r")

    print()

    if not reduced_chunks:
        raise SystemExit("No quedaron residuos proteicos analizables.")

    df = pd.concat(reduced_chunks, ignore_index=True)
    del reduced_chunks

    df = df.drop_duplicates(
        subset=["pdb_id", "uniprot_id", "uniprot_pos"]
    )

    df["has_ring"] = df["ring_interaction_count"] >= 1

    fisher_results = [
        binary_association(
            df,
            exposure_col="has_ring",
            outcome_col="has_water",
            label="bridging_water_in_ring_vs_non_ring_residues",
        ),
        binary_association(
            df,
            exposure_col="is_core_hotspot",
            outcome_col="has_water",
            label="bridging_water_in_core_hotspot_vs_non_core_residues",
        ),
    ]

    fisher_df = pd.DataFrame(fisher_results)
    fisher_df.to_csv(
        f"{args.out_prefix}_01_fisher.csv",
        index=False
    )

    ring_with_water = df.loc[
        df["has_water"], "ring_interaction_count"
    ].to_numpy(dtype=np.float64)

    ring_without_water = df.loc[
        ~df["has_water"], "ring_interaction_count"
    ].to_numpy(dtype=np.float64)

    if len(ring_with_water) and len(ring_without_water):
        u_stat, mw_p = mannwhitneyu(
            ring_with_water,
            ring_without_water,
            alternative="two-sided"
        )
    else:
        u_stat, mw_p = np.nan, np.nan

    ring_summary = pd.DataFrame([{
        "comparison": "ring_interaction_count_with_vs_without_bridging_water",
        "n_with_water": len(ring_with_water),
        "n_without_water": len(ring_without_water),
        "mean_ring_with_water": np.mean(ring_with_water),
        "mean_ring_without_water": np.mean(ring_without_water),
        "median_ring_with_water": np.median(ring_with_water),
        "median_ring_without_water": np.median(ring_without_water),
        "mannwhitney_u": u_stat,
        "p_value_two_sided": mw_p,
    }])

    ring_summary.to_csv(
        f"{args.out_prefix}_02_ring_count_mannwhitney.csv",
        index=False
    )

    water_only = df[df["has_water"]]

    if (
        len(water_only) >= 3
        and water_only["n_bridging_waters"].nunique() > 1
        and water_only["ring_interaction_count"].nunique() > 1
    ):
        rho, sp_p = spearmanr(
            water_only["n_bridging_waters"].to_numpy(),
            water_only["ring_interaction_count"].to_numpy(),
            nan_policy="omit"
        )
    else:
        rho, sp_p = np.nan, np.nan

    spearman_df = pd.DataFrame([{
        "analysis": (
            "n_bridging_waters_vs_ring_interaction_count_"
            "among_water_sites"
        ),
        "n": len(water_only),
        "spearman_rho": rho,
        "p_value": sp_p,
    }])

    spearman_df.to_csv(
        f"{args.out_prefix}_03_spearman.csv",
        index=False
    )

    descriptive = pd.DataFrame([
        {
            "group": "ring",
            "n_residues": int(df["has_ring"].sum()),
            "fraction_with_water": df.loc[
                df["has_ring"], "has_water"
            ].mean(),
        },
        {
            "group": "non_ring",
            "n_residues": int((~df["has_ring"]).sum()),
            "fraction_with_water": df.loc[
                ~df["has_ring"], "has_water"
            ].mean(),
        },
        {
            "group": "core_hotspot",
            "n_residues": int(df["is_core_hotspot"].sum()),
            "fraction_with_water": df.loc[
                df["is_core_hotspot"], "has_water"
            ].mean(),
        },
        {
            "group": "non_core",
            "n_residues": int((~df["is_core_hotspot"]).sum()),
            "fraction_with_water": df.loc[
                ~df["is_core_hotspot"], "has_water"
            ].mean(),
        },
    ])

    descriptive.to_csv(
        f"{args.out_prefix}_04_descriptive.csv",
        index=False
    )

    print("\nFisher:")
    print(fisher_df.to_string(index=False))

    print("\nConteo RING:")
    print(ring_summary.to_string(index=False))

    print("\nSpearman:")
    print(spearman_df.to_string(index=False))

    print("\nDescriptiva:")
    print(descriptive.to_string(index=False))


if __name__ == "__main__":
    main()
