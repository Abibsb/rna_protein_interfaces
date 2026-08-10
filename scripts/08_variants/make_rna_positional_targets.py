#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def target_info(target_name):
    """
    Returns positional-mapping candidate info for Rfam families.
    reference_id is intentionally left blank unless it is unambiguous from your own metadata.
    You will later fill exact RefSeq/RNAcentral/Ensembl reference IDs.
    """
    mapping = {
        "Telomerase-vert": {
            "candidate_gene_symbol": "TERC",
            "candidate_rna_name": "telomerase RNA",
            "positional_tier": "tier1_high",
            "mapping_comment": "Usually feasible: specific functional RNA gene.",
        },
        "U6atac": {
            "candidate_gene_symbol": "RNU6ATAC",
            "candidate_rna_name": "U6atac snRNA",
            "positional_tier": "tier1_high",
            "mapping_comment": "Usually feasible if reference sequence is confirmed.",
        },
        "U11": {
            "candidate_gene_symbol": "RNU11",
            "candidate_rna_name": "U11 snRNA",
            "positional_tier": "tier1_high",
            "mapping_comment": "Usually feasible if exact reference is recovered.",
        },
        "U12": {
            "candidate_gene_symbol": "RNU12",
            "candidate_rna_name": "U12 snRNA",
            "positional_tier": "tier1_high",
            "mapping_comment": "Usually feasible if exact reference is recovered.",
        },
        "7SK": {
            "candidate_gene_symbol": "RN7SK",
            "candidate_rna_name": "7SK RNA",
            "positional_tier": "tier1_high",
            "mapping_comment": "Potentially feasible; confirm exact reference.",
        },
        "U4": {
            "candidate_gene_symbol": "RNU4A/RNU4-1",
            "candidate_rna_name": "U4 snRNA",
            "positional_tier": "tier2_medium",
            "mapping_comment": "Family-level Rfam hit; exact gene copy may be ambiguous.",
        },
        "U5": {
            "candidate_gene_symbol": "RNU5A/RNU5A-1",
            "candidate_rna_name": "U5 snRNA",
            "positional_tier": "tier2_medium",
            "mapping_comment": "Family-level Rfam hit; exact gene copy may be ambiguous.",
        },
        "U6": {
            "candidate_gene_symbol": "RNU6-1/RNU6",
            "candidate_rna_name": "U6 snRNA",
            "positional_tier": "tier2_medium",
            "mapping_comment": "Family-level Rfam hit; multiple copies possible.",
        },
        "U2": {
            "candidate_gene_symbol": "RNU2/RNU2-1",
            "candidate_rna_name": "U2 snRNA",
            "positional_tier": "tier2_medium",
            "mapping_comment": "Family-level Rfam hit; multiple copies possible.",
        },
        "U1": {
            "candidate_gene_symbol": "RNU1",
            "candidate_rna_name": "U1 snRNA",
            "positional_tier": "tier2_medium",
            "mapping_comment": "Family-level Rfam hit; multiple copies possible.",
        },
        "Metazoa_SRP": {
            "candidate_gene_symbol": "RN7SL1/RN7SL",
            "candidate_rna_name": "7SL/SRP RNA",
            "positional_tier": "tier2_medium",
            "mapping_comment": "Potentially feasible, but RN7SL repeats/pseudogenes can complicate mapping.",
        },
        "let-7": {
            "candidate_gene_symbol": "MIRLET7 family",
            "candidate_rna_name": "let-7 microRNA precursor family",
            "positional_tier": "tier2_medium",
            "mapping_comment": "Family-level miRNA; exact MIRLET7 copy required for positional variants.",
        },
        "mir-16": {
            "candidate_gene_symbol": "MIR16 family",
            "candidate_rna_name": "mir-16 microRNA precursor family",
            "positional_tier": "tier2_medium",
            "mapping_comment": "Family-level miRNA; exact MIR16-1/MIR16-2 distinction required.",
        },
        "U3": {
            "candidate_gene_symbol": "SNORD3/U3",
            "candidate_rna_name": "U3 snoRNA",
            "positional_tier": "tier2_medium",
            "mapping_comment": "Potentially feasible, but exact snoRNA copy should be confirmed.",
        },
    }

    return mapping.get(str(target_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classified",
        required=True,
        help="rfam_classified/rna_unique_sequences_rfam_classified.csv"
    )
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.classified)

    rows = []

    for _, row in df.iterrows():
        info = target_info(row.get("target_name", ""))

        if info is None:
            continue

        out = row.to_dict()
        out.update(info)

        # To be filled later
        out["reference_id"] = ""
        out["reference_source"] = ""
        out["reference_sequence_available"] = False
        out["ready_for_positional_mapping"] = False

        rows.append(out)

    targets = pd.DataFrame(rows)

    targets.to_csv(outdir / "rna_positional_mapping_targets.csv", index=False)

    summary = (
        targets
        .groupby(
            ["target_name", "target_accession", "candidate_gene_symbol",
             "candidate_rna_name", "positional_tier"],
            dropna=False
        )
        .agg(
            n_unique_sequences=("unique_rna_id", "nunique"),
            n_chains=("n_members", "sum"),
            example_pdb_chains=("pdb_chains", lambda x: list(x.dropna().astype(str))[0] if len(x.dropna()) else ""),
            mapping_comment=("mapping_comment", "first"),
        )
        .reset_index()
        .sort_values(["positional_tier", "n_chains"], ascending=[True, False])
    )

    summary.to_csv(outdir / "summary_rna_positional_mapping_targets.csv", index=False)

    print("Done.")
    print(outdir / "rna_positional_mapping_targets.csv")
    print(outdir / "summary_rna_positional_mapping_targets.csv")


if __name__ == "__main__":
    main()