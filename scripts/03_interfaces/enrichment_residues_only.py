#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
}

RNA = {"A", "U", "G", "C", "ADE", "URA", "GUA", "CYT"}


def norm(x):
    if pd.isna(x):
        return np.nan
    return str(x).upper().strip()


def fisher_enrichment(df, feature_col, target_col):
    rows = []

    target = df[target_col].astype(int) == 1

    for feature in sorted(df[feature_col].dropna().unique()):
        feat = df[feature_col] == feature

        a = int((feat & target).sum())
        b = int((feat & ~target).sum())
        c = int((~feat & target).sum())
        d = int((~feat & ~target).sum())

        odds, p = fisher_exact([[a, b], [c, d]])

        rows.append({
            "target": target_col,
            "feature": feature,
            "a_feature_target": a,
            "b_feature_background": b,
            "c_other_target": c,
            "d_other_background": d,
            "odds_ratio": odds,
            "log2_odds_ratio": np.log2(odds) if odds > 0 and np.isfinite(odds) else np.nan,
            "p_value": p,
            "fraction_in_target": a / (a + c) if (a + c) else np.nan,
            "fraction_in_background": b / (b + d) if (b + d) else np.nan,
        })

    out = pd.DataFrame(rows)

    if len(out):
        out["p_adj_fdr_bh"] = multipletests(out["p_value"], method="fdr_bh")[1]

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-prefix", default="composition")
    args = parser.parse_args()

    usecols = [
        "polymer_class",
        "resname",
        "is_interface",
        "is_strong_interface",
        "is_core_hotspot",
    ]

    df = pd.read_csv(args.input, usecols=lambda c: c in usecols)

    df["polymer_class"] = df["polymer_class"].astype(str).str.lower()
    df["resname"] = df["resname"].map(norm)

    targets = [
        c for c in ["is_interface", "is_strong_interface", "is_core_hotspot"]
        if c in df.columns
    ]

    results = []

    protein = df[
        (df["polymer_class"] == "protein") &
        (df["resname"].isin(AA))
    ].copy()

    rna = df[
        (df["polymer_class"] == "rna") &
        (df["resname"].isin(RNA))
    ].copy()

    for target in targets:
        if protein[target].nunique() > 1:
            out = fisher_enrichment(protein, "resname", target)
            out.insert(0, "analysis", "protein_aminoacid")
            results.append(out)

        if rna[target].nunique() > 1:
            out = fisher_enrichment(rna, "resname", target)
            out.insert(0, "analysis", "rna_nucleotide")
            results.append(out)

    final = pd.concat(results, ignore_index=True)
    final = final.sort_values(["analysis", "target", "p_adj_fdr_bh", "odds_ratio"],
                              ascending=[True, True, True, False])

    final.to_csv(f"{args.output_prefix}_residue_enrichment.csv", index=False)

    print(f"Saved: {args.output_prefix}_residue_enrichment.csv")


if __name__ == "__main__":
    main()