#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    required_min = ["delta_sasa", "buried_fraction_residue"]
    missing = [c for c in required_min if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas mínimas requeridas en el CSV: {missing}")

    return df


def reconstruct_interface_class(df: pd.DataFrame) -> pd.DataFrame:
    if "interface_class" in df.columns:
        df["interface_class"] = df["interface_class"].astype(str)
        return df

    for col in ["is_core_hotspot", "is_strong_interface", "is_interface"]:
        if col not in df.columns:
            df[col] = 0

    def classify(row: pd.Series) -> str:
        if int(row.get("is_core_hotspot", 0)) == 1:
            return "core_hotspot"
        elif int(row.get("is_strong_interface", 0)) == 1:
            return "strong_interface"
        elif int(row.get("is_interface", 0)) == 1:
            return "interface"
        else:
            return "non_interface"

    df["interface_class"] = df.apply(classify, axis=1)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = ["delta_sasa", "buried_fraction_residue"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["delta_sasa", "buried_fraction_residue", "interface_class"])
    df = df[(df["buried_fraction_residue"] >= 0) & (df["buried_fraction_residue"] <= 1)]

    return df


def save_class_summary(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    counts = (
        df["interface_class"]
        .value_counts(dropna=False)
        .rename_axis("interface_class")
        .reset_index(name="count")
    )
    counts["proportion"] = counts["count"] / counts["count"].sum()
    counts["percentage"] = counts["proportion"] * 100

    counts.to_csv(outdir / "tabla_conteos_proporciones_clase.csv", index=False)

    with open(outdir / "tabla_conteos_proporciones_clase.txt", "w", encoding="utf-8") as fh:
        fh.write("Conteos y proporciones por clase estructural\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(counts.to_string(index=False))
        fh.write("\n")

    return counts


def save_class_summary_by_polymer(df: pd.DataFrame, outdir: Path) -> None:
    if "polymer_class" not in df.columns:
        return

    summary = (
        df.groupby(["polymer_class", "interface_class"])
        .size()
        .reset_index(name="count")
    )

    totals = (
        df.groupby("polymer_class")
        .size()
        .reset_index(name="total_polymer")
    )

    summary = summary.merge(totals, on="polymer_class", how="left")
    summary["proportion_within_polymer"] = summary["count"] / summary["total_polymer"]
    summary["percentage_within_polymer"] = summary["proportion_within_polymer"] * 100

    summary.to_csv(outdir / "tabla_clases_por_polymer_class.csv", index=False)


def freedman_diaconis_bins(x: np.ndarray, fallback: int = 50) -> int:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return fallback

    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return fallback

    bin_width = 2 * iqr / np.cbrt(len(x))
    if bin_width <= 0:
        return fallback

    n_bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return max(10, min(n_bins, 100))


def simple_kde(
    x: np.ndarray,
    grid: np.ndarray,
    max_points: int = 50000,
    random_seed: int = 123,
) -> np.ndarray:
    """
    KDE gaussiana simple con regla de Scott, sin scipy.
    Si hay demasiados puntos, toma una muestra aleatoria para evitar explotar RAM.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.zeros_like(grid)

    if len(x) > max_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(x), size=max_points, replace=False)
        x = x[idx]

    std = np.std(x, ddof=1)
    if std == 0:
        return np.zeros_like(grid)

    bw = std * (len(x) ** (-1 / 5))
    if bw <= 0:
        return np.zeros_like(grid)

    diff = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diff**2).sum(axis=1)
    dens /= (len(x) * bw * np.sqrt(2 * np.pi))
    return dens


def plot_hist_with_density(
    series: pd.Series,
    xlabel: str,
    title: str,
    outpath: Path,
    xlim: tuple[float, float] | None = None,
    use_kde: bool = True,
    kde_max_points: int = 50000,
) -> None:
    values = series.dropna().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = freedman_diaconis_bins(values)
    ax.hist(values, bins=bins, density=True, alpha=0.7, edgecolor="black")

    if use_kde and len(values) > 1:
        grid = np.linspace(values.min(), values.max(), 400)
        dens = simple_kde(values, grid, max_points=kde_max_points)
        if np.any(dens > 0):
            ax.plot(grid, dens, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densidad")
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def ordered_classes(df: pd.DataFrame) -> list[str]:
    preferred = ["non_interface", "interface", "strong_interface", "core_hotspot"]
    present = [c for c in preferred if c in set(df["interface_class"])]
    other = [c for c in sorted(df["interface_class"].unique()) if c not in present]
    return present + other


def collect_group_values(df: pd.DataFrame, ycol: str, class_order: list[str]) -> list[np.ndarray]:
    values = []
    for cls in class_order:
        arr = df.loc[df["interface_class"] == cls, ycol].dropna().to_numpy(dtype=float)
        values.append(arr)
    return values


def plot_boxplot_by_class(
    df: pd.DataFrame,
    ycol: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    class_order = ordered_classes(df)
    data = collect_group_values(df, ycol, class_order)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, tick_labels=class_order, showfliers=False)
    ax.set_xlabel("Clase estructural")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_violin_by_class(
    df: pd.DataFrame,
    ycol: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    class_order = ordered_classes(df)
    data = collect_group_values(df, ycol, class_order)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    ax.set_xticks(range(1, len(class_order) + 1))
    ax.set_xticklabels(class_order, rotation=25, ha="right")
    ax.set_xlabel("Clase estructural")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_barplot(counts_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(counts_df["interface_class"], counts_df["count"], edgecolor="black")
    ax.set_xlabel("Clase estructural")
    ax.set_ylabel("Cantidad de residuos")
    ax.set_title("Conteos por clase estructural")

    for i, (_, row) in enumerate(counts_df.iterrows()):
        ax.text(i, row["count"], f"{row['percentage']:.1f}%", ha="center", va="bottom")

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def write_numeric_summary(df: pd.DataFrame, outdir: Path) -> None:
    summary = (
        df.groupby("interface_class")[["delta_sasa", "buried_fraction_residue"]]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )
    summary.to_csv(outdir / "resumen_numerico_por_clase.csv")

    with open(outdir / "resumen_numerico_por_clase.txt", "w", encoding="utf-8") as fh:
        fh.write("Resumen numérico por clase estructural\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(summary.to_string())
        fh.write("\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python plot_definicion_interfaces.py interface_areas_per_residue_normalized.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    outdir = Path("resultados_definicion_interfaces")
    ensure_output_dir(outdir)

    df = read_data(csv_file)
    df = reconstruct_interface_class(df)
    df = clean_data(df)

    if df.empty:
        raise ValueError("No quedaron filas válidas luego del filtrado.")

    counts_df = save_class_summary(df, outdir)
    save_class_summary_by_polymer(df, outdir)
    write_numeric_summary(df, outdir)

    plot_hist_with_density(
        df["delta_sasa"],
        xlabel="ΔSASA",
        title="Distribución de ΔSASA a nivel de residuo",
        outpath=outdir / "01_hist_densidad_delta_sasa.png",
        use_kde=False,
    )

    plot_hist_with_density(
        df["buried_fraction_residue"],
        xlabel="Fracción enterrada",
        title="Distribución de fracción enterrada a nivel de residuo",
        outpath=outdir / "02_hist_densidad_fraccion_enterrada.png",
        xlim=(0, 1),
        use_kde=False,
    )

    plot_boxplot_by_class(
        df,
        ycol="delta_sasa",
        ylabel="ΔSASA",
        title="ΔSASA por clase estructural",
        outpath=outdir / "03_boxplot_delta_sasa_por_clase.png",
    )

    plot_boxplot_by_class(
        df,
        ycol="buried_fraction_residue",
        ylabel="Fracción enterrada",
        title="Fracción enterrada por clase estructural",
        outpath=outdir / "04_boxplot_fraccion_enterrada_por_clase.png",
    )

    plot_violin_by_class(
        df,
        ycol="delta_sasa",
        ylabel="ΔSASA",
        title="Distribución de ΔSASA por clase estructural",
        outpath=outdir / "05_violin_delta_sasa_por_clase.png",
    )

    plot_violin_by_class(
        df,
        ycol="buried_fraction_residue",
        ylabel="Fracción enterrada",
        title="Distribución de fracción enterrada por clase estructural",
        outpath=outdir / "06_violin_fraccion_enterrada_por_clase.png",
    )

    plot_class_barplot(
        counts_df,
        outpath=outdir / "07_barplot_conteos_por_clase.png",
    )

    print(f"Listo. Resultados guardados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()