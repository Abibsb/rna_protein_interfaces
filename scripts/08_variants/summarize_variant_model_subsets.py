#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "uniprot_id",
    "Disease_Association",
    "Prediction_PolyPhen",
    "Prediction_SIFT",
    "consurf_score",
    "iupred_score",
    "buried_fraction_residue",
    "is_low_conf",
    "has_ring_interface",
    "is_core_hotspot",
]


def parse_bool(series: pd.Series) -> pd.Series:
    """Convierte representaciones habituales de booleanos a True/False."""
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {"true", "1", "yes", "y", "si", "sí", "t"}
    )


def nonempty(series: pd.Series) -> pd.Series:
    """Identifica cadenas con contenido real."""
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return (
        normalized.notna()
        & ~normalized.isin(
            {
                "",
                "nan",
                "none",
                "null",
                "na",
                "n/a",
                ".",
            }
        )
    )


def make_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Construye los outcomes usados en los modelos."""
    out = df.copy()

    # yA: presencia de alguna anotación en Disease_Association.
    out["yA"] = nonempty(out["Disease_Association"])

    polyphen = (
        out["Prediction_PolyPhen"]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("")
    )

    sift = (
        out["Prediction_SIFT"]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("")
    )

    # yB: PolyPhen damaging y SIFT deleterious simultáneamente.
    out["yB"] = (
        polyphen.str.contains("damaging", regex=False)
        & sift.str.contains("deleterious", regex=False)
    )

    out["has_ring_interface_bool"] = parse_bool(
        out["has_ring_interface"]
    )
    out["is_core_hotspot_bool"] = parse_bool(
        out["is_core_hotspot"]
    )
    out["is_low_conf_bool"] = parse_bool(
        out["is_low_conf"]
    )

    for column in [
        "consurf_score",
        "iupred_score",
        "buried_fraction_residue",
    ]:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    return out


def summarize_subset(
    df: pd.DataFrame,
    subset_name: str,
    subset_mask: pd.Series,
) -> list[dict]:
    subset = df.loc[subset_mask].copy()

    # Reproduce el universo de modelado usado en los modelos previos:
    # ConSurf, IUPred y buried_fraction disponibles y sin baja confianza.
    eligible = subset.loc[
        subset["consurf_score"].notna()
        & subset["iupred_score"].notna()
        & subset["buried_fraction_residue"].notna()
        & ~subset["is_low_conf_bool"]
    ].copy()

    rows = []

    for universe_name, current in [
        ("before_model_filters", subset),
        ("after_model_filters", eligible),
    ]:
        for outcome in ["yA", "yB"]:
            n = len(current)
            positives = int(current[outcome].sum())
            negatives = int(n - positives)

            rows.append(
                {
                    "subset": subset_name,
                    "universe": universe_name,
                    "outcome": outcome,
                    "n_rows": n,
                    "n_uniprot": current[
                        "uniprot_id"
                    ].nunique(dropna=True),
                    "n_positive": positives,
                    "n_negative": negatives,
                    "positive_fraction": (
                        positives / n if n else np.nan
                    ),
                    "positive_percent": (
                        100 * positives / n if n else np.nan
                    ),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume los subconjuntos usados en los modelos de variantes "
            "y calcula las prevalencias reales de yA e yB."
        )
    )
    parser.add_argument(
        "input_csv",
        help="variants_bestrow_ring.csv",
    )
    parser.add_argument(
        "--out-prefix",
        default="variant_model_subset_summary",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)

    if not input_csv.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo: {input_csv}"
        )

    header = pd.read_csv(input_csv, nrows=0)
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in header.columns
    ]

    if missing:
        raise KeyError(
            "Faltan columnas requeridas: "
            + ", ".join(missing)
        )

    df = pd.read_csv(
        input_csv,
        usecols=REQUIRED_COLUMNS,
        low_memory=False,
    )

    df = make_outcomes(df)

    subset_definitions = {
        "all_variants": pd.Series(
            True,
            index=df.index,
        ),
        "ring_only": df["has_ring_interface_bool"],
        "hotspot_only": df["is_core_hotspot_bool"],
        "ring_and_hotspot": (
            df["has_ring_interface_bool"]
            & df["is_core_hotspot_bool"]
        ),
    }

    summary_rows = []

    for subset_name, subset_mask in subset_definitions.items():
        summary_rows.extend(
            summarize_subset(
                df=df,
                subset_name=subset_name,
                subset_mask=subset_mask,
            )
        )

    summary = pd.DataFrame(summary_rows)

    out_csv = f"{args.out_prefix}.csv"
    out_txt = f"{args.out_prefix}.txt"

    summary.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as handle:
        handle.write("RESUMEN DE SUBCONJUNTOS DE MODELADO\n")
        handle.write("=" * 76 + "\n\n")

        for subset_name in subset_definitions:
            handle.write(f"{subset_name}\n")
            handle.write("-" * 76 + "\n")

            current = summary.loc[
                summary["subset"].eq(subset_name)
            ]

            for _, row in current.iterrows():
                handle.write(
                    f"{row['universe']} | {row['outcome']}: "
                    f"N={int(row['n_rows']):,}; "
                    f"UniProt={int(row['n_uniprot']):,}; "
                    f"positivos={int(row['n_positive']):,}; "
                    f"negativos={int(row['n_negative']):,}; "
                    f"prevalencia={row['positive_percent']:.4f}%\n"
                )

            handle.write("\n")

    print("=" * 76)
    print("RESUMEN DE SUBCONJUNTOS DE MODELADO")
    print("=" * 76)
    print(
        summary.to_string(
            index=False,
            formatters={
                "positive_fraction": "{:.6f}".format,
                "positive_percent": "{:.4f}".format,
            },
        )
    )

    print()
    print(f"Guardado: {out_csv}")
    print(f"Guardado: {out_txt}")

    hotspot_yA = summary.loc[
        summary["subset"].eq("hotspot_only")
        & summary["universe"].eq("after_model_filters")
        & summary["outcome"].eq("yA")
    ]

    if not hotspot_yA.empty:
        row = hotspot_yA.iloc[0]

        print()
        print("=" * 76)
        print("DATO CLAVE PARA CORREGIR EL MODELO HOTSPOT yA")
        print("=" * 76)
        print(
            f"N = {int(row['n_rows']):,}\n"
            f"Positivos yA = {int(row['n_positive']):,}\n"
            f"Fracción yA = {row['positive_fraction']:.6f}\n"
            f"Porcentaje yA = {row['positive_percent']:.4f}%"
        )


if __name__ == "__main__":
    main()