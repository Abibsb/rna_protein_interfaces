#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis estricto SASA vs RING usando Opción A:
contacto fuerte = ring_interaction_count >= threshold

Este script:
1. carga interface_areas_per_residue_normalized.csv
2. carga todos los *_interface_residues_with_uniprot.csv
3. mergea por (pdb_id, auth_asym_id, auth_seq_id)
4. define:
   - interface_bool
   - strong_contact
   - strict_category
5. genera tablas, estadísticas y figuras

Uso:
python analyze_sasa_ring_intersection_strict_contacts.py \
    --interface_csv "/ruta/interface_areas_per_residue_normalized.csv" \
    --ring_dir "/ruta/ring_outputs" \
    --outdir "/ruta/salida" \
    --interface_col is_interface_abs \
    --contact_threshold 2

Opcional:
--delta_sasa_threshold 20
Si se da este parámetro, además genera una clasificación "strict_both"
donde interfaz fuerte = delta_sasa >= threshold y contacto fuerte = ring_interaction_count >= threshold.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface_csv", required=True,
                        help="Ruta a interface_areas_per_residue_normalized.csv")
    parser.add_argument("--ring_dir", required=True,
                        help="Directorio con archivos *_interface_residues_with_uniprot.csv")
    parser.add_argument("--outdir", required=True,
                        help="Directorio de salida")
    parser.add_argument("--interface_col", default="is_interface_abs",
                        help="Columna booleana de interfaz a usar (default: is_interface_abs)")
    parser.add_argument("--contact_threshold", type=int, default=2,
                        help="Threshold para definir contacto fuerte (default: 2)")
    parser.add_argument("--delta_sasa_threshold", type=float, default=None,
                        help="Threshold opcional para definir interfaz fuerte por delta_sasa")
    return parser.parse_args()


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    converted = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    mask_na = converted.isna()
    if mask_na.any():
        numeric = pd.to_numeric(series, errors="coerce")
        converted.loc[mask_na] = numeric.loc[mask_na].fillna(0).astype(int).astype(bool)

    return converted.fillna(False).astype(bool)


def load_interface_data(interface_csv: str, interface_col: str) -> pd.DataFrame:
    df = pd.read_csv(interface_csv)

    required_cols = [
        "pdb_id", "auth_asym_id", "auth_seq_id",
        "delta_sasa", interface_col
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en interface CSV: {missing}")

    df["pdb_id"] = df["pdb_id"].astype(str).str.upper()
    df["auth_asym_id"] = df["auth_asym_id"].astype(str).str.strip()
    df["auth_seq_id"] = pd.to_numeric(df["auth_seq_id"], errors="coerce")
    df["delta_sasa"] = pd.to_numeric(df["delta_sasa"], errors="coerce")

    df = df.dropna(subset=["auth_seq_id"]).copy()
    df["auth_seq_id"] = df["auth_seq_id"].astype(int)

    df["interface_bool"] = normalize_bool_series(df[interface_col])

    return df


def load_ring_data(ring_dir: str) -> pd.DataFrame:
    pattern = os.path.join(ring_dir, "*_interface_residues_with_uniprot.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No encontré archivos *_interface_residues_with_uniprot.csv en {ring_dir}"
        )

    dfs = []
    for f in files:
        tmp = pd.read_csv(f)

        required_cols = ["pdb_id", "auth_asym_id", "auth_seq_id", "ring_interaction_count"]
        missing = [c for c in required_cols if c not in tmp.columns]
        if missing:
            raise ValueError(f"En {f} faltan columnas: {missing}")

        tmp["pdb_id"] = tmp["pdb_id"].astype(str).str.upper()
        tmp["auth_asym_id"] = tmp["auth_asym_id"].astype(str).str.strip()
        tmp["auth_seq_id"] = pd.to_numeric(tmp["auth_seq_id"], errors="coerce")
        tmp["ring_interaction_count"] = pd.to_numeric(tmp["ring_interaction_count"], errors="coerce")

        tmp = tmp.dropna(subset=["auth_seq_id"]).copy()
        tmp["auth_seq_id"] = tmp["auth_seq_id"].astype(int)
        tmp["ring_interaction_count"] = tmp["ring_interaction_count"].fillna(0)

        tmp = (
            tmp.groupby(["pdb_id", "auth_asym_id", "auth_seq_id"], as_index=False)
               .agg(ring_interaction_count=("ring_interaction_count", "sum"))
        )

        dfs.append(tmp)

    ring = pd.concat(dfs, ignore_index=True)

    ring = (
        ring.groupby(["pdb_id", "auth_asym_id", "auth_seq_id"], as_index=False)
            .agg(ring_interaction_count=("ring_interaction_count", "sum"))
    )

    ring["has_ring_contact"] = ring["ring_interaction_count"] > 0
    return ring


def merge_data(df_interface: pd.DataFrame, df_ring: pd.DataFrame,
               contact_threshold: int,
               delta_sasa_threshold: float | None = None) -> pd.DataFrame:
    df = df_interface.merge(
        df_ring,
        how="left",
        on=["pdb_id", "auth_asym_id", "auth_seq_id"]
    )

    df["ring_interaction_count"] = df["ring_interaction_count"].fillna(0)
    df["has_ring_contact"] = df["has_ring_contact"].fillna(False)

    # Clasificación original
    conditions = [
        (df["interface_bool"]) & (df["has_ring_contact"]),
        (df["interface_bool"]) & (~df["has_ring_contact"]),
        (~df["interface_bool"]) & (df["has_ring_contact"]),
        (~df["interface_bool"]) & (~df["has_ring_contact"]),
    ]
    labels = [
        "interface_and_contact",
        "interface_only",
        "contact_only",
        "neither",
    ]
    df["category"] = np.select(conditions, labels, default="unknown")

    # Opción A
    df["strong_contact"] = df["ring_interaction_count"] >= contact_threshold

    strict_conditions = [
        (df["interface_bool"]) & (df["strong_contact"]),
        (df["interface_bool"]) & (~df["strong_contact"]),
        (~df["interface_bool"]) & (df["strong_contact"]),
        (~df["interface_bool"]) & (~df["strong_contact"]),
    ]
    strict_labels = [
        "interface_and_strong_contact",
        "interface_only",
        "strong_contact_only",
        "neither",
    ]
    df["strict_category"] = np.select(strict_conditions, strict_labels, default="unknown")

    # Opcional: interfaz fuerte por delta_sasa
    if delta_sasa_threshold is not None:
        df["strong_interface"] = df["delta_sasa"] >= delta_sasa_threshold

        both_conditions = [
            (df["strong_interface"]) & (df["strong_contact"]),
            (df["strong_interface"]) & (~df["strong_contact"]),
            (~df["strong_interface"]) & (df["strong_contact"]),
            (~df["strong_interface"]) & (~df["strong_contact"]),
        ]
        both_labels = [
            "strong_interface_and_strong_contact",
            "strong_interface_only",
            "strong_contact_only",
            "neither",
        ]
        df["strict_both_category"] = np.select(both_conditions, both_labels, default="unknown")

    return df


def save_basic_outputs(df: pd.DataFrame, outdir: Path):
    df.to_csv(outdir / "merged_interface_ring_strict.csv", index=False)

    counts = df["strict_category"].value_counts(dropna=False).rename_axis("strict_category").reset_index(name="n")
    counts["fraction"] = counts["n"] / len(df)
    counts.to_csv(outdir / "strict_category_counts.csv", index=False)

    summary = (
        df.groupby("strict_category", dropna=False)
          .agg(
              n=("strict_category", "size"),
              delta_sasa_mean=("delta_sasa", "mean"),
              delta_sasa_median=("delta_sasa", "median"),
              ring_interaction_count_mean=("ring_interaction_count", "mean"),
              ring_interaction_count_median=("ring_interaction_count", "median"),
          )
          .reset_index()
    )
    summary["fraction"] = summary["n"] / len(df)
    summary.to_csv(outdir / "strict_category_summary.csv", index=False)

    if "strict_both_category" in df.columns:
        counts2 = df["strict_both_category"].value_counts(dropna=False).rename_axis("strict_both_category").reset_index(name="n")
        counts2["fraction"] = counts2["n"] / len(df)
        counts2.to_csv(outdir / "strict_both_category_counts.csv", index=False)

        summary2 = (
            df.groupby("strict_both_category", dropna=False)
              .agg(
                  n=("strict_both_category", "size"),
                  delta_sasa_mean=("delta_sasa", "mean"),
                  delta_sasa_median=("delta_sasa", "median"),
                  ring_interaction_count_mean=("ring_interaction_count", "mean"),
                  ring_interaction_count_median=("ring_interaction_count", "median"),
              )
              .reset_index()
        )
        summary2["fraction"] = summary2["n"] / len(df)
        summary2.to_csv(outdir / "strict_both_category_summary.csv", index=False)


def save_strict_intersection_table(df: pd.DataFrame, outdir: Path):
    strict_inter = df[
        (df["interface_bool"]) &
        (df["strong_contact"])
    ].copy()

    strict_inter = strict_inter.sort_values(
        by=["ring_interaction_count", "delta_sasa"],
        ascending=[False, False]
    )

    strict_inter.to_csv(outdir / "strict_intersection_residues.csv", index=False)


def run_statistics(df: pd.DataFrame, outdir: Path):
    results = []

    # Comparación dentro de interfaz: strong_contact vs no strong_contact
    g1 = df[(df["interface_bool"]) & (df["strong_contact"])]["delta_sasa"].dropna()
    g2 = df[(df["interface_bool"]) & (~df["strong_contact"])]["delta_sasa"].dropna()

    if len(g1) > 0 and len(g2) > 0:
        stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
        results.append({
            "comparison": "delta_sasa: interface_and_strong_contact vs interface_only",
            "test": "mannwhitneyu",
            "n_group1": len(g1),
            "n_group2": len(g2),
            "statistic": stat,
            "pvalue": p,
            "group1_median": g1.median(),
            "group2_median": g2.median(),
            "group1_mean": g1.mean(),
            "group2_mean": g2.mean(),
        })

    # Correlación en intersección estricta
    strict_inter = df[(df["interface_bool"]) & (df["strong_contact"])].copy()
    if len(strict_inter) > 1:
        rho, p = spearmanr(strict_inter["ring_interaction_count"], strict_inter["delta_sasa"], nan_policy="omit")
        results.append({
            "comparison": "spearman: ring_interaction_count vs delta_sasa in strict_intersection",
            "test": "spearmanr",
            "n_group1": len(strict_inter),
            "n_group2": np.nan,
            "statistic": rho,
            "pvalue": p,
            "group1_median": strict_inter["ring_interaction_count"].median(),
            "group2_median": strict_inter["delta_sasa"].median(),
            "group1_mean": strict_inter["ring_interaction_count"].mean(),
            "group2_mean": strict_inter["delta_sasa"].mean(),
        })

    # Opcional: strong_interface + strong_contact si existe
    if "strong_interface" in df.columns:
        h1 = df[(df["strong_interface"]) & (df["strong_contact"])]["delta_sasa"].dropna()
        h2 = df[(df["strong_interface"]) & (~df["strong_contact"])]["delta_sasa"].dropna()

        if len(h1) > 0 and len(h2) > 0:
            stat, p = mannwhitneyu(h1, h2, alternative="two-sided")
            results.append({
                "comparison": "delta_sasa: strong_interface_and_strong_contact vs strong_interface_only",
                "test": "mannwhitneyu",
                "n_group1": len(h1),
                "n_group2": len(h2),
                "statistic": stat,
                "pvalue": p,
                "group1_median": h1.median(),
                "group2_median": h2.median(),
                "group1_mean": h1.mean(),
                "group2_mean": h2.mean(),
            })

    pd.DataFrame(results).to_csv(outdir / "strict_statistics.csv", index=False)


def plot_bar_strict_categories(df: pd.DataFrame, outdir: Path):
    order = ["interface_and_strong_contact", "interface_only", "strong_contact_only", "neither"]
    counts = df["strict_category"].value_counts().reindex(order).fillna(0)

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.ylabel("Number of residues")
    plt.xlabel("Category")
    plt.title("Residue categories: interface vs strong direct contact")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "bar_strict_category_counts.png", dpi=300)
    plt.close()


def plot_box_strict_within_interface(df: pd.DataFrame, outdir: Path):
    sub = df[df["interface_bool"]].copy()
    if sub.empty:
        return

    labels = ["interface_only", "interface_and_strong_contact"]
    data = [
        sub.loc[~sub["strong_contact"], "delta_sasa"].dropna().values,
        sub.loc[sub["strong_contact"], "delta_sasa"].dropna().values,
    ]

    plt.figure(figsize=(6, 5))
    plt.boxplot(data, labels=labels)
    plt.ylabel("delta_sasa")
    plt.xlabel("Group")
    plt.title("delta_sasa within interface residues (strict contact)")
    plt.tight_layout()
    plt.savefig(outdir / "boxplot_delta_sasa_interface_strict_contact_vs_no_strict_contact.png", dpi=300)
    plt.close()


def plot_box_strict_categories(df: pd.DataFrame, outdir: Path):
    order = ["interface_and_strong_contact", "interface_only", "strong_contact_only", "neither"]
    data = [df.loc[df["strict_category"] == cat, "delta_sasa"].dropna().values for cat in order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=order)
    plt.ylabel("delta_sasa")
    plt.xlabel("Category")
    plt.title("delta_sasa by strict residue category")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "boxplot_delta_sasa_by_strict_category.png", dpi=300)
    plt.close()


def plot_hist_strict_intersection(df: pd.DataFrame, outdir: Path):
    strict_inter = df[(df["interface_bool"]) & (df["strong_contact"])].copy()
    if strict_inter.empty:
        return

    values = strict_inter["ring_interaction_count"].dropna().astype(int)
    bins = np.arange(values.min(), values.max() + 2) - 0.5

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins)
    plt.xlabel("ring_interaction_count")
    plt.ylabel("Number of residues")
    plt.title("Strong direct-contact counts in the strict intersection")
    plt.xticks(sorted(values.unique()))
    plt.tight_layout()
    plt.savefig(outdir / "hist_strict_intersection_ring_interaction_count.png", dpi=300)
    plt.close()


def plot_scatter_strict_intersection(df: pd.DataFrame, outdir: Path):
    strict_inter = df[(df["interface_bool"]) & (df["strong_contact"])].copy()
    if strict_inter.empty:
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(strict_inter["ring_interaction_count"], strict_inter["delta_sasa"], alpha=0.7)
    plt.xlabel("ring_interaction_count")
    plt.ylabel("delta_sasa")
    plt.title("Strict intersection: strong direct contacts vs delta_sasa")
    plt.tight_layout()
    plt.savefig(outdir / "scatter_strict_intersection_contacts_vs_delta_sasa.png", dpi=300)
    plt.close()


def write_report(df: pd.DataFrame, outdir: Path, contact_threshold: int, delta_sasa_threshold: float | None):
    n_total = len(df)
    n_interface = int(df["interface_bool"].sum())
    n_strong_contact = int(df["strong_contact"].sum())
    n_strict_inter = int(((df["interface_bool"]) & (df["strong_contact"])).sum())

    strict_inter = df[(df["interface_bool"]) & (df["strong_contact"])]
    interface_only = df[(df["interface_bool"]) & (~df["strong_contact"])]

    lines = [
        "Strict SASA–RING analysis (Option A)",
        "===================================",
        "",
        f"contact_threshold (ring_interaction_count >=): {contact_threshold}",
        f"delta_sasa_threshold: {delta_sasa_threshold if delta_sasa_threshold is not None else 'not used'}",
        "",
        f"Total residues analyzed: {n_total}",
        f"Interface residues: {n_interface} ({n_interface / n_total:.4f})" if n_total else "Interface residues: NA",
        f"Strong-contact residues: {n_strong_contact} ({n_strong_contact / n_total:.4f})" if n_total else "Strong-contact residues: NA",
        f"Strict intersection (interface + strong contact): {n_strict_inter} ({n_strict_inter / n_total:.4f})" if n_total else "Strict intersection: NA",
        f"Strict intersection within interface: {n_strict_inter / n_interface:.4f}" if n_interface else "Strict intersection within interface: NA",
        f"Strict intersection within strong-contact residues: {n_strict_inter / n_strong_contact:.4f}" if n_strong_contact else "Strict intersection within strong-contact residues: NA",
        "",
    ]

    if len(strict_inter) > 0:
        lines.extend([
            "Strict intersection summary",
            "---------------------------",
            f"delta_sasa mean: {strict_inter['delta_sasa'].mean():.4f}",
            f"delta_sasa median: {strict_inter['delta_sasa'].median():.4f}",
            f"ring_interaction_count mean: {strict_inter['ring_interaction_count'].mean():.4f}",
            f"ring_interaction_count median: {strict_inter['ring_interaction_count'].median():.4f}",
            "",
        ])

    if len(interface_only) > 0:
        lines.extend([
            "Interface-only summary (against strict contact definition)",
            "---------------------------------------------------------",
            f"delta_sasa mean: {interface_only['delta_sasa'].mean():.4f}",
            f"delta_sasa median: {interface_only['delta_sasa'].median():.4f}",
            "",
        ])

    with open(outdir / "strict_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading interface data...")
    df_interface = load_interface_data(args.interface_csv, args.interface_col)

    print("[2/7] Loading RING residue-level data...")
    df_ring = load_ring_data(args.ring_dir)

    print("[3/7] Merging datasets and defining strict contact categories...")
    df = merge_data(
        df_interface=df_interface,
        df_ring=df_ring,
        contact_threshold=args.contact_threshold,
        delta_sasa_threshold=args.delta_sasa_threshold
    )

    print("[4/7] Saving tables...")
    save_basic_outputs(df, outdir)
    save_strict_intersection_table(df, outdir)

    print("[5/7] Running statistics...")
    run_statistics(df, outdir)

    print("[6/7] Generating plots...")
    plot_bar_strict_categories(df, outdir)
    plot_box_strict_within_interface(df, outdir)
    plot_box_strict_categories(df, outdir)
    plot_hist_strict_intersection(df, outdir)
    plot_scatter_strict_intersection(df, outdir)

    print("[7/7] Writing report...")
    write_report(
        df=df,
        outdir=outdir,
        contact_threshold=args.contact_threshold,
        delta_sasa_threshold=args.delta_sasa_threshold
    )

    print("Done.")
    print(f"Results written to: {outdir}")


if __name__ == "__main__":
    main()