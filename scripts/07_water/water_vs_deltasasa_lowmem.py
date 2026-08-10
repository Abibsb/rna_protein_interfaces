#!/usr/bin/env python3
"""
Versión de bajo consumo de memoria:
asociación entre presencia/cantidad de aguas puente y ΔSASA.

Lee únicamente las columnas necesarias y procesa el MASTER por bloques.
La unidad de análisis se mantiene en:
    (pdb_id, uniprot_id, uniprot_pos)

Uso:
    python water_vs_deltasasa_lowmem.py \
      --master MASTER_interface_dataset.csv \
      --out_prefix resultados_water_deltasasa

Opcional:
    --chunksize 250000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr


TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def to_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(TRUE_VALUES)
    )


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    return (2.0 * u_stat) / (n1 * n2) - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--out_prefix", default="water_vs_deltasasa")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    header = pd.read_csv(args.master, nrows=0)
    available = set(header.columns)

    required = {
        "pdb_id", "uniprot_id", "uniprot_pos",
        "polymer_class", "has_bridging_water", "delta_sasa"
    }
    missing = sorted(required - available)
    if missing:
        raise SystemExit(f"Faltan columnas obligatorias: {missing}")

    usecols = list(required)
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
        chunk["delta_sasa"] = pd.to_numeric(
            chunk["delta_sasa"], errors="coerce", downcast="float"
        )
        chunk["uniprot_pos"] = pd.to_numeric(
            chunk["uniprot_pos"], errors="coerce", downcast="integer"
        )

        if has_n_waters:
            chunk["n_bridging_waters"] = pd.to_numeric(
                chunk["n_bridging_waters"], errors="coerce", downcast="integer"
            ).fillna(0)
        else:
            chunk["n_bridging_waters"] = chunk["has_water"].astype("int8")

        chunk = chunk.dropna(
            subset=["pdb_id", "uniprot_id", "uniprot_pos", "delta_sasa"]
        )

        # Elimina duplicados dentro de cada bloque.
        chunk = chunk.drop_duplicates(
            subset=["pdb_id", "uniprot_id", "uniprot_pos"]
        )

        reduced_chunks.append(
            chunk[[
                "pdb_id", "uniprot_id", "uniprot_pos",
                "has_water", "delta_sasa", "n_bridging_waters"
            ]]
        )

        print(f"Procesadas {processed:,} filas", end="\r")

    print()

    if not reduced_chunks:
        raise SystemExit("No quedaron residuos proteicos analizables.")

    # Solo se concatenan seis columnas, no el MASTER completo.
    df = pd.concat(reduced_chunks, ignore_index=True)
    del reduced_chunks

    # Elimina duplicados que hayan quedado repartidos entre bloques.
    df = df.drop_duplicates(
        subset=["pdb_id", "uniprot_id", "uniprot_pos"]
    )

    with_water = df.loc[df["has_water"], "delta_sasa"].to_numpy(dtype=np.float64)
    without_water = df.loc[~df["has_water"], "delta_sasa"].to_numpy(dtype=np.float64)

    if len(with_water) == 0 or len(without_water) == 0:
        raise SystemExit("Uno de los grupos quedó vacío.")

    u_stat, p_value = mannwhitneyu(
        with_water,
        without_water,
        alternative="two-sided"
    )
    rbc = rank_biserial_from_u(
        u_stat, len(with_water), len(without_water)
    )

    descriptive = pd.DataFrame([
        {
            "group": "without_bridging_water",
            "count": len(without_water),
            "mean": np.mean(without_water),
            "median": np.median(without_water),
            "std": np.std(without_water, ddof=1),
            "min": np.min(without_water),
            "max": np.max(without_water),
        },
        {
            "group": "with_bridging_water",
            "count": len(with_water),
            "mean": np.mean(with_water),
            "median": np.median(with_water),
            "std": np.std(with_water, ddof=1),
            "min": np.min(with_water),
            "max": np.max(with_water),
        },
    ])
    descriptive.to_csv(
        f"{args.out_prefix}_01_descriptive.csv", index=False
    )

    test = pd.DataFrame([{
        "comparison": "delta_sasa_with_vs_without_bridging_water",
        "n_with_water": len(with_water),
        "n_without_water": len(without_water),
        "mannwhitney_u": u_stat,
        "p_value_two_sided": p_value,
        "rank_biserial_correlation": rbc,
        "median_with_water": np.median(with_water),
        "median_without_water": np.median(without_water),
        "median_difference": np.median(with_water) - np.median(without_water),
    }])
    test.to_csv(
        f"{args.out_prefix}_02_mannwhitney.csv", index=False
    )

    water_only = df[df["has_water"]]
    if (
        len(water_only) >= 3
        and water_only["n_bridging_waters"].nunique() > 1
        and water_only["delta_sasa"].nunique() > 1
    ):
        rho, spearman_p = spearmanr(
            water_only["n_bridging_waters"].to_numpy(),
            water_only["delta_sasa"].to_numpy(),
            nan_policy="omit"
        )
    else:
        rho, spearman_p = np.nan, np.nan

    correlation = pd.DataFrame([{
        "analysis": "n_bridging_waters_vs_delta_sasa_among_water_sites",
        "n": len(water_only),
        "spearman_rho": rho,
        "p_value": spearman_p,
    }])
    correlation.to_csv(
        f"{args.out_prefix}_03_spearman.csv", index=False
    )

    upper = np.quantile(
        np.concatenate([with_water, without_water]), 0.99
    )
    plot_without = without_water[without_water <= upper]
    plot_with = with_water[with_water <= upper]

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [plot_without, plot_with],
        tick_labels=["Sin agua", "Con agua"],
        showfliers=False
    )
    plt.ylabel("ΔSASA (Å²)")
    plt.title("ΔSASA según presencia de agua puente")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_boxplot.png", dpi=300)
    plt.close()

    print("\nDescriptiva:")
    print(descriptive.to_string(index=False))
    print("\nMann–Whitney:")
    print(test.to_string(index=False))
    print("\nSpearman:")
    print(correlation.to_string(index=False))


if __name__ == "__main__":
    main()
