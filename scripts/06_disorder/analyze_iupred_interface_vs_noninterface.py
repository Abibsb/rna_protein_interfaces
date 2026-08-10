#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, fisher_exact, spearmanr


def boolify(s):
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "yes", "y", "t"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("merged_csv")
    parser.add_argument("--outdir", default="iupred_interface_vs_noninterface")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged_csv, low_memory=False)

    interface_col = "is_interface_abs"
    strong_col = "is_strong_interface"
    hotspot_col = "is_core_hotspot"
    delta_col = "delta_sasa"
    ring_col = "ring_interaction_count"
    has_ring_col = "has_ring_contact"

    df = df.dropna(subset=["iupred_score"]).copy()
    df["iupred_score"] = pd.to_numeric(df["iupred_score"], errors="coerce")
    df = df.dropna(subset=["iupred_score"]).copy()

    df[interface_col] = boolify(df[interface_col])
    df[strong_col] = boolify(df[strong_col])
    df[hotspot_col] = boolify(df[hotspot_col])

    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
    df[ring_col] = pd.to_numeric(df[ring_col], errors="coerce").fillna(0)

    if has_ring_col in df.columns:
        df[has_ring_col] = boolify(df[has_ring_col])
    else:
        df[has_ring_col] = df[ring_col] > 0

    df["is_disordered_iupred"] = df["iupred_score"] > args.threshold

    print("Rows with IUPred:", len(df))

    # -------------------------
    # Summaries
    # -------------------------
    global_summary = pd.DataFrame([{
        "n_residues": len(df),
        "mean_iupred": df["iupred_score"].mean(),
        "median_iupred": df["iupred_score"].median(),
        "q25_iupred": df["iupred_score"].quantile(0.25),
        "q75_iupred": df["iupred_score"].quantile(0.75),
        "fraction_disordered": df["is_disordered_iupred"].mean(),
    }])
    global_summary.to_csv(outdir / "summary_global_merged_iupred.csv", index=False)

    def grouped_summary(col, filename):
        summary = (
            df.groupby(col)
            .agg(
                n=("iupred_score", "size"),
                mean_iupred=("iupred_score", "mean"),
                median_iupred=("iupred_score", "median"),
                q25_iupred=("iupred_score", lambda x: x.quantile(0.25)),
                q75_iupred=("iupred_score", lambda x: x.quantile(0.75)),
                n_disordered=("is_disordered_iupred", "sum"),
                fraction_disordered=("is_disordered_iupred", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(outdir / filename, index=False)
        return summary

    interface_summary = grouped_summary(
        interface_col,
        "summary_interface_vs_noninterface.csv"
    )

    strong_summary = grouped_summary(
        strong_col,
        "summary_strong_interface.csv"
    )

    hotspot_summary = grouped_summary(
        hotspot_col,
        "summary_hotspot.csv"
    )

    ring_binary_summary = grouped_summary(
        has_ring_col,
        "summary_has_ring_contact.csv"
    )

    # RING bins
    df["ring_bin"] = pd.cut(
        df[ring_col],
        bins=[-0.1, 0, 1, 2, np.inf],
        labels=["0", "1", "2", ">=3"]
    )

    ring_bin_summary = grouped_summary(
        "ring_bin",
        "summary_ring_contact_bins.csv"
    )

    # -------------------------
    # Tests
    # -------------------------
    tests = []

    def mannwhitney_for_bool(col, label):
        x = df.loc[df[col], "iupred_score"].dropna()
        y = df.loc[~df[col], "iupred_score"].dropna()

        if len(x) == 0 or len(y) == 0:
            return

        u, p = mannwhitneyu(x, y, alternative="two-sided")

        tests.append({
            "test": f"Mann-Whitney IUPred score: {label} true vs false",
            "n_true": len(x),
            "n_false": len(y),
            "median_true": x.median(),
            "median_false": y.median(),
            "mean_true": x.mean(),
            "mean_false": y.mean(),
            "delta_median_true_minus_false": x.median() - y.median(),
            "statistic": u,
            "p_value": p,
        })

    def fisher_for_bool(col, label):
        group = df[col]
        dis = df["is_disordered_iupred"]

        a = int((group & dis).sum())
        b = int((group & ~dis).sum())
        c = int((~group & dis).sum())
        d = int((~group & ~dis).sum())

        OR, p = fisher_exact([[a, b], [c, d]])

        tests.append({
            "test": f"Fisher disorder enrichment: {label} true vs false",
            "a_true_disordered": a,
            "b_true_ordered": b,
            "c_false_disordered": c,
            "d_false_ordered": d,
            "odds_ratio": OR,
            "p_value": p,
            "fraction_disordered_true": a / (a + b) if (a + b) else np.nan,
            "fraction_disordered_false": c / (c + d) if (c + d) else np.nan,
        })

    for col, label in [
        (interface_col, "interface"),
        (strong_col, "strong_interface"),
        (hotspot_col, "core_hotspot"),
        (has_ring_col, "has_ring_contact"),
    ]:
        mannwhitney_for_bool(col, label)
        fisher_for_bool(col, label)

    # Correlations
    sub = df[["iupred_score", delta_col]].dropna()
    rho, p = spearmanr(sub["iupred_score"], sub[delta_col])
    tests.append({
        "test": "Spearman IUPred score vs delta_sasa",
        "n": len(sub),
        "spearman_rho": rho,
        "p_value": p,
    })

    sub = df[["iupred_score", ring_col]].dropna()
    rho, p = spearmanr(sub["iupred_score"], sub[ring_col])
    tests.append({
        "test": "Spearman IUPred score vs ring_interaction_count",
        "n": len(sub),
        "spearman_rho": rho,
        "p_value": p,
    })

    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(outdir / "statistical_tests.csv", index=False)

    # -------------------------
    # Plots
    # -------------------------
    data_non = df.loc[~df[interface_col], "iupred_score"].dropna()
    data_int = df.loc[df[interface_col], "iupred_score"].dropna()

    plt.figure(figsize=(7, 5))
    plt.hist(data_non, bins=60, density=True, alpha=0.5, label="Non-interface")
    plt.hist(data_int, bins=60, density=True, alpha=0.5, label="Interface")
    plt.xlabel("IUPred score")
    plt.ylabel("Density")
    plt.title("IUPred distribution: interface vs non-interface")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_01_iupred_interface_vs_noninterface_hist.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [data_non, data_int],
        labels=["Non-interface", "Interface"],
        showfliers=False
    )
    plt.ylabel("IUPred score")
    plt.title("IUPred score: interface vs non-interface")
    plt.tight_layout()
    plt.savefig(outdir / "fig_02_iupred_interface_vs_noninterface_boxplot.png", dpi=300)
    plt.close()

    frac = (
        df.groupby(interface_col)["is_disordered_iupred"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(5, 5))
    plt.bar(frac[interface_col].astype(str), frac["is_disordered_iupred"])
    plt.xlabel("Interface")
    plt.ylabel("Fraction disordered")
    plt.title("Fraction of disordered residues")
    plt.tight_layout()
    plt.savefig(outdir / "fig_03_fraction_disordered_interface_vs_noninterface.png", dpi=300)
    plt.close()

    # IUPred by structural class
    classes = [
        ("Non-interface", ~df[interface_col]),
        ("Interface", df[interface_col]),
        ("Strong interface", df[strong_col]),
        ("Core hotspot", df[hotspot_col]),
        ("RING contact", df[has_ring_col]),
    ]

    data = [df.loc[mask, "iupred_score"].dropna() for _, mask in classes]
    labels = [label for label, _ in classes]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("IUPred score")
    plt.title("IUPred score across structural classes")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "fig_04_iupred_structural_classes_boxplot.png", dpi=300)
    plt.close()

    # IUPred vs delta_sasa
    sample = df[[delta_col, "iupred_score"]].dropna()
    if len(sample) > 300000:
        sample = sample.sample(300000, random_state=1)

    plt.figure(figsize=(6, 5))
    plt.scatter(sample["iupred_score"], sample[delta_col], s=3, alpha=0.2)
    plt.xlabel("IUPred score")
    plt.ylabel("ΔSASA")
    plt.title("IUPred score vs ΔSASA")
    plt.tight_layout()
    plt.savefig(outdir / "fig_05_iupred_vs_delta_sasa.png", dpi=300)
    plt.close()

    print("\nSummary interface vs non-interface:")
    print(interface_summary.to_string(index=False))

    print("\nSummary hotspot:")
    print(hotspot_summary.to_string(index=False))

    print("\nSummary RING bins:")
    print(ring_bin_summary.to_string(index=False))

    print("\nStatistical tests:")
    print(tests_df.to_string(index=False))

    print(f"\nOutput directory: {outdir}")


if __name__ == "__main__":
    main()