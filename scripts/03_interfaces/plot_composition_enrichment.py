#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESIDUE_FILE = "interface_residue_enrichment.csv"
GROUP_FILE = "interface_group_enrichment.csv"
OUTDIR = "composition_plots"

TARGET_ORDER = ["is_interface", "is_strong_interface", "is_core_hotspot"]
TARGET_LABELS = {
    "is_interface": "Interface",
    "is_strong_interface": "Strong interface",
    "is_core_hotspot": "Core hotspot",
}


def safe_p(p):
    p = pd.to_numeric(p, errors="coerce")
    return p.replace(0, np.nextafter(0, 1))


def savefig(name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def load_data():
    residues = pd.read_csv(RESIDUE_FILE)
    groups = pd.read_csv(GROUP_FILE)

    residues["target"] = pd.Categorical(residues["target"], TARGET_ORDER, ordered=True)
    groups["target"] = pd.Categorical(groups["target"], TARGET_ORDER, ordered=True)

    return residues, groups


def plot_1_protein_residue_log2or(residues):
    df = residues[
        (residues["analysis"] == "protein_aminoacid") &
        (residues["target"].isin(TARGET_ORDER))
    ].copy()

    pivot = df.pivot(index="feature", columns="target", values="log2_odds_ratio")
    pivot = pivot[TARGET_ORDER]
    pivot = pivot.sort_values("is_core_hotspot")

    ax = pivot.plot(kind="barh", figsize=(8, 9), width=0.8)

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title("Protein residue enrichment across interface classes")
    ax.set_xlabel("log2(odds ratio)")
    ax.set_ylabel("Amino acid")
    handles, labels = ax.get_legend_handles_labels()
    labels = [TARGET_LABELS[l] for l in labels]
    ax.legend(handles, labels, title="Class")

    savefig("01_protein_residue_log2OR.png")


def plot_2_protein_residue_fractions(residues):
    df = residues[
        (residues["analysis"] == "protein_aminoacid") &
        (residues["target"].isin(TARGET_ORDER))
    ].copy()

    amino_order = (
        df[df["target"] == "is_core_hotspot"]
        .sort_values("fraction_in_target", ascending=False)["feature"]
        .tolist()
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    for ax, target in zip(axes, TARGET_ORDER):
        sub = df[df["target"] == target].set_index("feature").loc[amino_order]

        x = np.arange(len(sub))
        width = 0.4

        ax.bar(x - width/2, sub["fraction_in_target"], width, label="Target")
        ax.bar(x + width/2, sub["fraction_in_background"], width, label="Background")

        ax.set_title(TARGET_LABELS[target])
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, max(sub["fraction_in_target"].max(), sub["fraction_in_background"].max()) * 1.25)

        ax.legend()

    axes[-1].set_xticks(np.arange(len(amino_order)))
    axes[-1].set_xticklabels(amino_order, rotation=45, ha="right")

    fig.suptitle("Protein residue composition: target vs background", y=1.01)

    savefig("02_protein_residue_fractions_interface_strong_hotspot.png")


def plot_3_group_log2or(groups):
    df = groups[
        (groups["analysis"] == "protein_group") &
        (groups["target"].isin(TARGET_ORDER))
    ].copy()

    pivot = df.pivot(index="feature", columns="target", values="log2_odds_ratio")
    pivot = pivot[TARGET_ORDER]
    pivot = pivot.sort_values("is_core_hotspot")

    ax = pivot.plot(kind="barh", figsize=(8, 6), width=0.8)

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title("Protein physicochemical group enrichment")
    ax.set_xlabel("log2(odds ratio)")
    ax.set_ylabel("Group")
    handles, labels = ax.get_legend_handles_labels()
    labels = [TARGET_LABELS[l] for l in labels]
    ax.legend(handles, labels, title="Class")

    savefig("03_protein_group_log2OR_interface_strong_hotspot.png")


def plot_4_heatmap_residues(residues):
    df = residues[
        (residues["analysis"] == "protein_aminoacid") &
        (residues["target"].isin(TARGET_ORDER))
    ].copy()

    pivot = df.pivot(index="feature", columns="target", values="log2_odds_ratio")
    pivot = pivot[TARGET_ORDER]
    pivot = pivot.sort_values("is_core_hotspot", ascending=False)

    data = pivot.values
    vmax = np.nanmax(np.abs(data))

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(TARGET_ORDER)))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title("Protein residue enrichment heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2(odds ratio)")

    savefig("04_protein_residue_heatmap.png")


def plot_5_volcano(residues):
    df = residues[
        (residues["analysis"] == "protein_aminoacid") &
        (residues["target"] == "is_core_hotspot")
    ].copy()

    df["minus_log10_padj"] = -np.log10(safe_p(df["p_adj_fdr_bh"]))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(df["log2_odds_ratio"], df["minus_log10_padj"], s=45)

    for _, row in df.iterrows():
        ax.text(row["log2_odds_ratio"], row["minus_log10_padj"], row["feature"], fontsize=8)

    ax.axvline(0, linestyle="--", linewidth=1, label="log2(OR)=0")
    ax.set_title("Volcano plot: protein residues in core hotspots")
    ax.set_xlabel("log2(odds ratio)")
    ax.set_ylabel("-log10(FDR-adjusted p-value)")

    savefig("05_volcano_protein_core_hotspot.png")


def plot_6_rna_nucleotide_log2or(residues):
    df = residues[
        (residues["analysis"] == "rna_nucleotide") &
        (residues["target"].isin(TARGET_ORDER))
    ].copy()

    pivot = df.pivot(index="feature", columns="target", values="log2_odds_ratio")
    pivot = pivot[TARGET_ORDER]
    pivot = pivot.loc[[x for x in ["A", "C", "G", "U"] if x in pivot.index]]

    ax = pivot.plot(kind="bar", figsize=(7, 5), width=0.8)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("RNA nucleotide enrichment across interface classes")
    ax.set_ylabel("log2(odds ratio)")
    ax.set_xlabel("Nucleotide")
    handles, labels = ax.get_legend_handles_labels()
    labels = [TARGET_LABELS[l] for l in labels]
    ax.legend(handles, labels, title="Class")

    savefig("06_rna_nucleotide_log2OR.png")


def plot_7_combined_protein_rna(groups, residues):
    protein = groups[
        (groups["analysis"] == "protein_group") &
        (groups["target"] == "is_core_hotspot")
    ].copy()

    rna = residues[
        (residues["analysis"] == "rna_nucleotide") &
        (residues["target"] == "is_core_hotspot")
    ].copy()

    protein = protein.sort_values("log2_odds_ratio")
    rna = rna.set_index("feature").loc[[x for x in ["A", "C", "G", "U"] if x in rna["feature"].values]].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(protein["feature"], protein["log2_odds_ratio"])
    axes[0].axvline(0, linestyle="--", linewidth=1, label="log2(OR)=0")
    axes[0].set_title("Protein groups")
    axes[0].set_xlabel("log2(odds ratio)")
    axes[0].set_ylabel("Group")

    axes[1].bar(rna["feature"], rna["log2_odds_ratio"])
    axes[1].axhline(0, linestyle="--", linewidth=1, label="log2(OR)=0")
    axes[1].set_title("RNA nucleotides")
    axes[1].set_ylabel("log2(odds ratio)")
    axes[1].set_xlabel("Nucleotide")

    fig.suptitle("Composition of core hotspots: protein and RNA sides", y=1.02)

    savefig("07_combined_protein_groups_rna_nucleotides_core_hotspot.png")


def main():
    residues, groups = load_data()

    plot_1_protein_residue_log2or(residues)
    plot_2_protein_residue_fractions(residues)
    plot_3_group_log2or(groups)
    plot_4_heatmap_residues(residues)
    plot_5_volcano(residues)
    plot_6_rna_nucleotide_log2or(residues)
    plot_7_combined_protein_rna(groups, residues)


if __name__ == "__main__":
    main()