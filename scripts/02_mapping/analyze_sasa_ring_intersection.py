#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analiza la intersección entre:
1) residuos de interfaz definidos por delta SASA
2) residuos con contacto directo ARN-proteína definidos por RING

Inputs esperados:
- interface_areas_per_residue_normalized.csv
- uno o más archivos *_interface_residues_with_uniprot.csv dentro de ring_dir

Uso:
python analyze_sasa_ring_intersection.py \
    --interface_csv /ruta/interface_areas_per_residue_normalized.csv \
    --ring_dir /ruta/ring_outputs \
    --outdir /ruta/salida

Opcional:
--interface_col is_interface_abs
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
    return parser.parse_args()


def normalize_bool_series(series: pd.Series) -> pd.Series:
    """
    Convierte una serie heterogénea a booleana.
    Acepta True/False, 'True'/'False', 1/0, etc.
    """
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

    # Si hay valores que no matchearon, intentar numérico
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

    # Estandarizar claves
    df["pdb_id"] = df["pdb_id"].astype(str).str.upper()
    df["auth_asym_id"] = df["auth_asym_id"].astype(str).str.strip()
    df["auth_seq_id"] = pd.to_numeric(df["auth_seq_id"], errors="coerce")

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

        # En caso de duplicados por alguna razón, sumar conteos por residuo
        tmp = (
            tmp.groupby(["pdb_id", "auth_asym_id", "auth_seq_id"], as_index=False)
               .agg(ring_interaction_count=("ring_interaction_count", "sum"))
        )

        dfs.append(tmp)

    ring = pd.concat(dfs, ignore_index=True)

    # Y de nuevo consolidar por si un PDB quedó repetido entre archivos
    ring = (
        ring.groupby(["pdb_id", "auth_asym_id", "auth_seq_id"], as_index=False)
            .agg(ring_interaction_count=("ring_interaction_count", "sum"))
    )

    ring["has_ring_contact"] = ring["ring_interaction_count"] > 0
    return ring


def merge_data(df_interface: pd.DataFrame, df_ring: pd.DataFrame) -> pd.DataFrame:
    df = df_interface.merge(
        df_ring,
        how="left",
        on=["pdb_id", "auth_asym_id", "auth_seq_id"]
    )

    df["ring_interaction_count"] = df["ring_interaction_count"].fillna(0)
    df["has_ring_contact"] = df["has_ring_contact"].fillna(False)

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

    return df


def save_summary_tables(df: pd.DataFrame, outdir: Path):
    # Tabla global de categorías
    counts = df["category"].value_counts(dropna=False).rename_axis("category").reset_index(name="n")
    counts["fraction"] = counts["n"] / len(df)
    counts.to_csv(outdir / "category_counts.csv", index=False)

    # Fracciones de interfaz/contacto/intersección
    n_total = len(df)
    n_interface = int(df["interface_bool"].sum())
    n_contact = int(df["has_ring_contact"].sum())
    n_intersection = int(((df["interface_bool"]) & (df["has_ring_contact"])).sum())

    metrics = pd.DataFrame([
        {"metric": "n_total", "value": n_total},
        {"metric": "n_interface", "value": n_interface},
        {"metric": "n_contact", "value": n_contact},
        {"metric": "n_intersection", "value": n_intersection},
        {"metric": "fraction_interface_total", "value": n_interface / n_total if n_total else np.nan},
        {"metric": "fraction_contact_total", "value": n_contact / n_total if n_total else np.nan},
        {"metric": "fraction_intersection_total", "value": n_intersection / n_total if n_total else np.nan},
        {"metric": "fraction_intersection_within_interface", "value": n_intersection / n_interface if n_interface else np.nan},
        {"metric": "fraction_intersection_within_contact", "value": n_intersection / n_contact if n_contact else np.nan},
    ])
    metrics.to_csv(outdir / "global_metrics.csv", index=False)

    # Resumen por categoría
    summary = (
        df.groupby("category", dropna=False)
          .agg(
              n=("category", "size"),
              delta_sasa_mean=("delta_sasa", "mean"),
              delta_sasa_median=("delta_sasa", "median"),
              ring_interaction_count_mean=("ring_interaction_count", "mean"),
              ring_interaction_count_median=("ring_interaction_count", "median"),
          )
          .reset_index()
    )
    summary["fraction"] = summary["n"] / len(df)
    summary.to_csv(outdir / "category_summary_stats.csv", index=False)


def run_stats(df: pd.DataFrame, outdir: Path):
    results = []

    # Interfaz con contacto vs interfaz sin contacto
    g1 = df[(df["interface_bool"]) & (df["has_ring_contact"])]["delta_sasa"].dropna()
    g2 = df[(df["interface_bool"]) & (~df["has_ring_contact"])]["delta_sasa"].dropna()

    if len(g1) > 0 and len(g2) > 0:
        stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
        results.append({
            "comparison": "delta_sasa: interface_and_contact vs interface_only",
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

    # Correlación en la intersección: contactos vs delta_sasa
    inter = df[(df["interface_bool"]) & (df["has_ring_contact"])].copy()
    if len(inter) > 1:
        rho, p = spearmanr(inter["ring_interaction_count"], inter["delta_sasa"], nan_policy="omit")
        results.append({
            "comparison": "spearman: ring_interaction_count vs delta_sasa in intersection",
            "test": "spearmanr",
            "n_group1": len(inter),
            "n_group2": np.nan,
            "statistic": rho,
            "pvalue": p,
            "group1_median": inter["ring_interaction_count"].median(),
            "group2_median": inter["delta_sasa"].median(),
            "group1_mean": inter["ring_interaction_count"].mean(),
            "group2_mean": inter["delta_sasa"].mean(),
        })

    pd.DataFrame(results).to_csv(outdir / "statistics.csv", index=False)


def plot_bar_categories(df: pd.DataFrame, outdir: Path):
    order = ["interface_and_contact", "interface_only", "contact_only", "neither"]
    counts = df["category"].value_counts().reindex(order).fillna(0)

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.ylabel("Number of residues")
    plt.xlabel("Category")
    plt.title("Residue categories: interface vs direct contact")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "bar_category_counts.png", dpi=300)
    plt.close()


def plot_box_delta_sasa_interface(df: pd.DataFrame, outdir: Path):
    sub = df[df["interface_bool"]].copy()
    if sub.empty:
        return

    labels = ["interface_only", "interface_and_contact"]
    data = [
        sub.loc[~sub["has_ring_contact"], "delta_sasa"].dropna().values,
        sub.loc[sub["has_ring_contact"], "delta_sasa"].dropna().values,
    ]

    plt.figure(figsize=(6, 5))
    plt.boxplot(data, labels=labels)
    plt.ylabel("delta_sasa")
    plt.xlabel("Group")
    plt.title("delta_sasa within interface residues")
    plt.tight_layout()
    plt.savefig(outdir / "boxplot_delta_sasa_interface_contact_vs_no_contact.png", dpi=300)
    plt.close()


def plot_box_delta_sasa_all_categories(df: pd.DataFrame, outdir: Path):
    order = ["interface_and_contact", "interface_only", "contact_only", "neither"]
    data = [df.loc[df["category"] == cat, "delta_sasa"].dropna().values for cat in order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=order)
    plt.ylabel("delta_sasa")
    plt.xlabel("Category")
    plt.title("delta_sasa by residue category")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "boxplot_delta_sasa_by_category.png", dpi=300)
    plt.close()


def plot_hist_intersection_contacts(df: pd.DataFrame, outdir: Path):
    inter = df[(df["interface_bool"]) & (df["has_ring_contact"])].copy()
    if inter.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(inter["ring_interaction_count"], bins=20)
    plt.xlabel("ring_interaction_count")
    plt.ylabel("Number of residues")
    plt.title("Direct-contact counts in the intersection")
    plt.tight_layout()
    plt.savefig(outdir / "hist_intersection_ring_interaction_count.png", dpi=300)
    plt.close()


def plot_scatter_intersection(df: pd.DataFrame, outdir: Path):
    inter = df[(df["interface_bool"]) & (df["has_ring_contact"])].copy()
    if inter.empty:
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(inter["ring_interaction_count"], inter["delta_sasa"], alpha=0.7)
    plt.xlabel("ring_interaction_count")
    plt.ylabel("delta_sasa")
    plt.title("Intersection: direct contacts vs delta_sasa")
    plt.tight_layout()
    plt.savefig(outdir / "scatter_intersection_contacts_vs_delta_sasa.png", dpi=300)
    plt.close()


def save_intersection_table(df: pd.DataFrame, outdir: Path):
    inter = df[(df["interface_bool"]) & (df["has_ring_contact"])].copy()
    inter = inter.sort_values(
        by=["ring_interaction_count", "delta_sasa"],
        ascending=[False, False]
    )
    inter.to_csv(outdir / "intersection_residues.csv", index=False)


def write_report(df: pd.DataFrame, outdir: Path):
    n_total = len(df)
    n_interface = int(df["interface_bool"].sum())
    n_contact = int(df["has_ring_contact"].sum())
    n_inter = int(((df["interface_bool"]) & (df["has_ring_contact"])).sum())

    inter = df[(df["interface_bool"]) & (df["has_ring_contact"])]
    interface_only = df[(df["interface_bool"]) & (~df["has_ring_contact"])]

    report_lines = [
        "SASA–RING intersection analysis",
        "==============================",
        "",
        f"Total residues analyzed: {n_total}",
        f"Interface residues: {n_interface} ({n_interface / n_total:.4f})" if n_total else "Interface residues: NA",
        f"Residues with direct contact: {n_contact} ({n_contact / n_total:.4f})" if n_total else "Residues with direct contact: NA",
        f"Intersection (interface + direct contact): {n_inter} ({n_inter / n_total:.4f})" if n_total else "Intersection: NA",
        f"Intersection within interface: {n_inter / n_interface:.4f}" if n_interface else "Intersection within interface: NA",
        f"Intersection within contact residues: {n_inter / n_contact:.4f}" if n_contact else "Intersection within contact residues: NA",
        "",
    ]

    if len(inter) > 0:
        report_lines.extend([
            "Intersection summary",
            "--------------------",
            f"delta_sasa mean: {inter['delta_sasa'].mean():.4f}",
            f"delta_sasa median: {inter['delta_sasa'].median():.4f}",
            f"ring_interaction_count mean: {inter['ring_interaction_count'].mean():.4f}",
            f"ring_interaction_count median: {inter['ring_interaction_count'].median():.4f}",
            "",
        ])

    if len(interface_only) > 0:
        report_lines.extend([
            "Interface-only summary",
            "----------------------",
            f"delta_sasa mean: {interface_only['delta_sasa'].mean():.4f}",
            f"delta_sasa median: {interface_only['delta_sasa'].median():.4f}",
            "",
        ])

    with open(outdir / "report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading interface data...")
    df_interface = load_interface_data(args.interface_csv, args.interface_col)

    print("[2/7] Loading RING residue-level data...")
    df_ring = load_ring_data(args.ring_dir)

    print("[3/7] Merging datasets...")
    df = merge_data(df_interface, df_ring)
    df.to_csv(outdir / "merged_interface_ring.csv", index=False)

    print("[4/7] Saving summary tables...")
    save_summary_tables(df, outdir)
    save_intersection_table(df, outdir)

    print("[5/7] Running statistics...")
    run_stats(df, outdir)

    print("[6/7] Generating plots...")
    plot_bar_categories(df, outdir)
    plot_box_delta_sasa_interface(df, outdir)
    plot_box_delta_sasa_all_categories(df, outdir)
    plot_hist_intersection_contacts(df, outdir)
    plot_scatter_intersection(df, outdir)

    print("[7/7] Writing report...")
    write_report(df, outdir)

    print("Done.")
    print(f"Results written to: {outdir}")


if __name__ == "__main__":
    main()