#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def str_bool_series(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--max-iupred", type=float, default=0.5)
    ap.add_argument("--require-a-best", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.candidates, low_memory=False)

    required = [
        "pdb_id", "uniprot_id", "uniprot_pos", "variant_id",
        "wild_type", "mutated_type",
        "alphafold_candidate_score",
        "is_interface", "is_strong_interface", "is_core_hotspot",
        "has_ring_interface",
        "ring_interaction_count", "rna_unique_nucleotides",
        "delta_sasa", "buried_fraction_residue",
        "iupred_score",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    sub = df.copy()

    if args.require_a_best and "candidate_tier" in sub.columns:
        sub = sub[sub["candidate_tier"].astype(str) == "A_best"]

    sub = sub[
        str_bool_series(sub["is_interface"]) &
        str_bool_series(sub["is_strong_interface"]) &
        str_bool_series(sub["is_core_hotspot"]) &
        str_bool_series(sub["has_ring_interface"]) &
        (pd.to_numeric(sub["ring_interaction_count"], errors="coerce") > 0) &
        (pd.to_numeric(sub["rna_unique_nucleotides"], errors="coerce") > 0) &
        (pd.to_numeric(sub["iupred_score"], errors="coerce") <= args.max_iupred)
    ].copy()

    sub["has_consurf"] = pd.to_numeric(sub.get("consurf_score"), errors="coerce").notna()

    sort_cols = [
        "has_consurf",
        "alphafold_candidate_score",
        "ring_interaction_count",
        "rna_unique_nucleotides",
        "delta_sasa",
        "buried_fraction_residue",
    ]

    sub = sub.sort_values(sort_cols, ascending=[False, False, False, False, False, False])

    keep = [
        "candidate_tier", "alphafold_candidate_score",
        "pdb_id", "uniprot_id", "gene", "protein_chain",
        "uniprot_pos", "variant_id", "wild_type", "mutated_type",
        "is_interface", "is_strong_interface", "is_core_hotspot",
        "has_ring_interface", "ring_interaction_count",
        "rna_unique_nucleotides", "delta_sasa",
        "buried_fraction_residue", "consurf_score", "iupred_score",
        "is_pathogenic_or_damaging", "polyphen_damaging",
        "sift_deleterious", "bad_mutation", "disease_association",
    ]
    keep = [c for c in keep if c in sub.columns]

    out = sub[keep].head(args.top_n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)

    print(out.to_string(index=False))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()