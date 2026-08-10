#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def classify_rna_type(target_name, description):
    text = f"{clean(target_name)} {clean(description)}".lower()

    if "trna" in text or "transfer rna" in text:
        return "tRNA"
    if "rrna" in text or "ribosomal rna" in text:
        return "rRNA"
    if "spliceosomal" in text or target_name in {"U1", "U2", "U3", "U4", "U5", "U6", "U11", "U12"}:
        if target_name == "U3" or "small nucleolar" in text:
            return "snoRNA"
        return "snRNA"
    if "snorna" in text or "small nucleolar" in text:
        return "snoRNA"
    if "microrna" in text or "mir-" in text or "let-7" in text:
        return "miRNA_precursor"
    if "telomerase" in text:
        return "telomerase_RNA"
    if "signal recognition particle" in text or "srp" in text or "7sl" in text:
        return "SRP_RNA"
    if "7sk" in text:
        return "7SK_RNA"
    if "secis" in text:
        return "SECIS_element"
    if "riboswitch" in text:
        return "riboswitch"
    if "ires" in text:
        return "viral_IRES"
    if "ribozyme" in text:
        return "ribozyme"
    return "other_RNA"


def classify_biological_scope(target_name, description):
    text = f"{clean(target_name)} {clean(description)}".lower()

    if "bacteria" in text or "bacterial" in text or "riboswitch" in text or "glms" in text:
        return "non_human_bacterial_or_prokaryotic"

    if "virus" in text or "viral" in text or "hepatitis" in text or "enterovirus" in text or "cripavirus" in text or "hdv" in text or "hcv" in text:
        return "viral"

    if "metazoa" in text or "vertebrate" in text or "mammalian" in text:
        return "human_relevant_family"

    if "eukarya" in text or "eukaryotic" in text:
        return "ambiguous_eukaryotic"

    # Spliceosomal RNAs, 7SK, telomerase, miRNA families are relevant but not exact human IDs
    if target_name in {"U1", "U2", "U3", "U4", "U5", "U6", "U11", "U12"}:
        return "human_relevant_family"

    if target_name in {"7SK", "let-7"} or "mir-" in clean(target_name).lower():
        return "human_relevant_family"

    if "trna" in text:
        return "ambiguous_tRNA_family"

    return "family_only_ambiguous"


def disease_annotation_priority(rna_type, biological_scope):
    if biological_scope in {"viral", "non_human_bacterial_or_prokaryotic"}:
        return "exclude_from_human_rna_disease_step"

    if rna_type in {
        "snRNA",
        "snoRNA",
        "miRNA_precursor",
        "telomerase_RNA",
        "SRP_RNA",
        "7SK_RNA",
        "SECIS_element",
    }:
        return "prioritize_for_rna_disease_databases"

    if rna_type in {"rRNA", "tRNA"}:
        return "annotate_with_caution_repetitive_or_family_level"

    return "low_priority_family_level"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-rfam", required=True, help="rfam_best_hit_per_unique_sequence.csv")
    parser.add_argument("--unique-csv", required=True, help="rna_unique_sequences.csv")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(args.best_rfam)
    unique = pd.read_csv(args.unique_csv)

    df = unique.merge(
        best[["unique_rna_id", "target_name", "target_accession", "evalue", "score", "description"]],
        on="unique_rna_id",
        how="left"
    )

    df["has_rfam_hit"] = df["target_name"].notna()

    df["rfam_rna_type"] = df.apply(
        lambda r: classify_rna_type(r.get("target_name", ""), r.get("description", "")),
        axis=1
    )

    df["rfam_biological_scope"] = df.apply(
        lambda r: classify_biological_scope(r.get("target_name", ""), r.get("description", "")),
        axis=1
    )

    df["disease_annotation_priority"] = df.apply(
        lambda r: disease_annotation_priority(r["rfam_rna_type"], r["rfam_biological_scope"]),
        axis=1
    )

    df.to_csv(outdir / "rna_unique_sequences_rfam_classified.csv", index=False)

    by_type = (
        df[df["has_rfam_hit"]]
        .groupby(["rfam_rna_type", "rfam_biological_scope", "disease_annotation_priority"], dropna=False)
        .agg(
            n_unique_sequences=("unique_rna_id", "nunique"),
            n_chains=("n_members", "sum"),
            n_families=("target_name", "nunique"),
        )
        .reset_index()
        .sort_values(["n_chains", "n_unique_sequences"], ascending=False)
    )

    by_type.to_csv(outdir / "summary_rfam_by_type_scope.csv", index=False)

    by_family = (
        df[df["has_rfam_hit"]]
        .groupby(
            ["target_name", "target_accession", "description", "rfam_rna_type",
             "rfam_biological_scope", "disease_annotation_priority"],
            dropna=False
        )
        .agg(
            n_unique_sequences=("unique_rna_id", "nunique"),
            n_chains=("n_members", "sum"),
        )
        .reset_index()
        .sort_values(["n_chains", "n_unique_sequences"], ascending=False)
    )

    by_family.to_csv(outdir / "summary_rfam_families_classified.csv", index=False)

    priority = df[
        df["disease_annotation_priority"].isin([
            "prioritize_for_rna_disease_databases",
            "annotate_with_caution_repetitive_or_family_level",
        ])
    ].copy()

    priority.to_csv(outdir / "rna_sequences_priority_for_disease_annotation.csv", index=False)

    print("Done.")
    print(outdir / "rna_unique_sequences_rfam_classified.csv")
    print(outdir / "summary_rfam_by_type_scope.csv")
    print(outdir / "summary_rfam_families_classified.csv")
    print(outdir / "rna_sequences_priority_for_disease_annotation.csv")


if __name__ == "__main__":
    main()