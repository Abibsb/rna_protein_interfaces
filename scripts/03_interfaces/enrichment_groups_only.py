#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


AA_GROUPS = {
    "positive_basic": {"LYS", "ARG", "HIS"},
    "negative_acidic": {"ASP", "GLU"},
    "polar_uncharged": {"SER", "THR", "ASN", "GLN", "TYR", "CYS"},
    "hydrophobic": {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"},
    "aromatic": {"PHE", "TYR", "TRP"},
    "small": {"GLY", "ALA", "SER", "CYS"},
    "flexible": {"GLY", "SER"},
    "rigid_special": {"PRO"},
}

RNA_GROUPS = {
    "purine": {"A", "G", "ADE", "GUA"},
    "pyrimidine": {"C", "U", "CYT", "URA"},
}


def norm(x):
    if pd.isna(x):
        return np.nan
    return str(x).upper().strip()


def fisher_group(df, residues, target_col):
    target = df[target_col].astype(int) == 1
    feat = df["resname"].isin(residues)

    a = int((feat & target).sum())
    b = int((feat & ~target).sum())
    c = int((~feat & target).sum())
    d = int((~feat & ~target).sum())

    odds, p = fisher_exact([[a, b], [c, d]])

    return {
        "target": target_col,
        "a_group_target": a,
        "b_group_background": b,
        "c_other_target": c,
        "d_other_background": d,
        "odds_ratio": odds,
        "log2_odds_ratio": np.log2(odds) if odds > 0 and np.isfinite(odds) else np.nan,
        "p_value": p,
        "fraction_in_target": a / (a + c) if (a + c) else np.nan,
        "fraction_in_background": b / (b + d) if (b + d) else np.nan,
    }


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

    rows = []

    protein = df[df["polymer_class"] == "protein"].copy()
    rna = df[df["polymer_class"] == "rna"].copy()

    for target in targets:
        for group, residues in AA_GROUPS.items():
            row = fisher_group(protein, residues, target)
            row["analysis"] = "protein_group"
            row["feature"] = group
            rows.append(row)

        for group, residues in RNA_GROUPS.items():
            row = fisher_group(rna, residues, target)
            row["analysis"] = "rna_group"
            row["feature"] = group
            rows.append(row)

    out = pd.DataFrame(rows)

    out["p_adj_fdr_bh"] = multipletests(out["p_value"], method="fdr_bh")[1]

    out = out[
        [
            "analysis",
            "target",
            "feature",
            "odds_ratio",
            "log2_odds_ratio",
            "p_value",
            "p_adj_fdr_bh",
            "fraction_in_target",
            "fraction_in_background",
            "a_group_target",
            "b_group_background",
            "c_other_target",
            "d_other_background",
        ]
    ]

    out = out.sort_values(["analysis", "target", "p_adj_fdr_bh", "odds_ratio"],
                          ascending=[True, True, True, False])

    out.to_csv(f"{args.output_prefix}_group_enrichment.csv", index=False)

    print(f"Saved: {args.output_prefix}_group_enrichment.csv")


if __name__ == "__main__":
    main()