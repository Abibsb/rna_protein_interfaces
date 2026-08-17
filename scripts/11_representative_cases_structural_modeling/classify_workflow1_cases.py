#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact-summary", required=True)
    ap.add_argument("--prepared-summary", required=True)
    ap.add_argument("--best-models", required=False)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    contacts = pd.read_csv(args.contact_summary, sep="\t")
    prepared = pd.read_csv(args.prepared_summary, sep="\t")

    wide = contacts.pivot_table(
        index="case_id",
        columns="model",
        values=[
            "min_distance_A",
            "n_atom_pairs_lt_5A",
            "n_unique_rna_nt_lt_5A",
        ],
        aggfunc="first",
    )

    wide.columns = [f"{metric}__{model}" for metric, model in wide.columns]
    wide = wide.reset_index()

    df = prepared.merge(wide, on="case_id", how="left")

    if args.best_models:
        bm = pd.read_csv(args.best_models, sep="\t")
        bm_wide = bm.pivot_table(
            index="case_id",
            columns="condition",
            values=[c for c in ["ranking_score", "iptm", "ptm", "fraction_disordered", "has_clash"] if c in bm.columns],
            aggfunc="first",
        )
        bm_wide.columns = [f"{metric}__{cond}" for metric, cond in bm_wide.columns]
        bm_wide = bm_wide.reset_index()
        df = df.merge(bm_wide, on="case_id", how="left")

    wt_nt = pd.to_numeric(df.get("n_unique_rna_nt_lt_5A__AF_CIF_WT"), errors="coerce")
    mut_nt = pd.to_numeric(df.get("n_unique_rna_nt_lt_5A__AF_CIF_MUT"), errors="coerce")
    wt_min = pd.to_numeric(df.get("min_distance_A__AF_CIF_WT"), errors="coerce")
    mut_min = pd.to_numeric(df.get("min_distance_A__AF_CIF_MUT"), errors="coerce")

    classes = []
    for i in range(len(df)):
        if pd.isna(wt_nt.iloc[i]) or pd.isna(mut_nt.iloc[i]):
            cls = "missing_measurement"
        elif wt_nt.iloc[i] >= 1 and mut_nt.iloc[i] == 0:
            cls = "geometry_contact_loss_candidate"
        elif wt_nt.iloc[i] >= 1 and mut_nt.iloc[i] < wt_nt.iloc[i]:
            cls = "geometry_partial_contact_reduction"
        elif wt_nt.iloc[i] == 0:
            cls = "AF_WT_no_general_RNA_contact"
        elif mut_nt.iloc[i] >= wt_nt.iloc[i]:
            cls = "no_geometry_loss_detected"
        else:
            cls = "ambiguous"

        classes.append(cls)

    df["workflow1_interpretation_class"] = classes
    df["delta_min_distance_MUT_minus_WT"] = mut_min - wt_min
    df["delta_n_unique_nt_lt5_MUT_minus_WT"] = mut_nt - wt_nt

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)

    print(df[[
        "case_id",
        "variant_id",
        "pdb_id",
        "uniprot_id",
        "workflow1_interpretation_class",
        "min_distance_A__AF_CIF_WT",
        "min_distance_A__AF_CIF_MUT",
        "n_unique_rna_nt_lt_5A__AF_CIF_WT",
        "n_unique_rna_nt_lt_5A__AF_CIF_MUT",
        "delta_min_distance_MUT_minus_WT",
        "delta_n_unique_nt_lt5_MUT_minus_WT",
    ]].to_string(index=False))

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()