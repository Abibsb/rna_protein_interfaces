#!/usr/bin/env python3

import argparse
import gzip
import re
from pathlib import Path
import pandas as pd


def normalize_gene_set(x):
    if pd.isna(x):
        return set()
    s = str(x).upper()
    parts = re.split(r"[;|,/\s]+", s)
    return {p.strip() for p in parts if p.strip()}


def pathogenic_bucket(clinsig):
    s = str(clinsig).lower()

    if "pathogenic" in s and "benign" not in s:
        return "pathogenic_or_likely_pathogenic"
    if "uncertain" in s or "vus" in s:
        return "vus_or_uncertain"
    if "benign" in s:
        return "benign_or_likely_benign"
    if "conflicting" in s:
        return "conflicting"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-summary", required=True)
    parser.add_argument("--gene-set", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    genes = pd.read_csv(args.gene_set)
    target_genes = set(genes["candidate_gene"].astype(str).str.upper())

    with gzip.open(args.variant_summary, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")

    wanted = [
        "AlleleID",
        "Type",
        "Name",
        "GeneID",
        "GeneSymbol",
        "HGNC_ID",
        "ClinicalSignificance",
        "ClinSigSimple",
        "LastEvaluated",
        "RS# (dbSNP)",
        "RCVaccession",
        "PhenotypeIDS",
        "PhenotypeList",
        "Origin",
        "OriginSimple",
        "Assembly",
        "ChromosomeAccession",
        "Chromosome",
        "Start",
        "Stop",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "ReviewStatus",
        "NumberSubmitters",
        "VariationID",
        "PositionVCF",
    ]

    usecols = [c for c in wanted if c in header]

    chunks = []

    for chunk in pd.read_csv(
        args.variant_summary,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        dtype=str,
        chunksize=500000,
        low_memory=False,
    ):
        mask = chunk["GeneSymbol"].apply(lambda x: bool(normalize_gene_set(x) & target_genes))
        sub = chunk[mask].copy()
        if not sub.empty:
            chunks.append(sub)

    if chunks:
        cv = pd.concat(chunks, ignore_index=True)
    else:
        cv = pd.DataFrame(columns=usecols)

    expanded = []

    for _, row in cv.iterrows():
        clinvar_genes = normalize_gene_set(row.get("GeneSymbol", ""))
        hits = sorted(clinvar_genes & target_genes)

        for g in hits:
            r = row.to_dict()
            r["candidate_gene"] = g
            expanded.append(r)

    cv_expanded = pd.DataFrame(expanded)

    if cv_expanded.empty:
        print("No variants found.")
        cv_expanded.to_csv(outdir / "clinvar_expanded_rna_gene_level.csv", index=False)
        return

    cv_expanded = cv_expanded.merge(
        genes,
        on="candidate_gene",
        how="left"
    )

    cv_expanded["clinical_bucket"] = cv_expanded["ClinicalSignificance"].apply(pathogenic_bucket)

    cv_expanded.to_csv(outdir / "clinvar_expanded_rna_gene_level.csv", index=False)

    summary_gene = (
        cv_expanded
        .groupby(["tier", "rfam_target", "candidate_gene", "clinical_bucket"], dropna=False)
        .agg(
            n_variants=("VariationID", "nunique"),
            n_rows=("VariationID", "size"),
            diseases=("PhenotypeList", lambda x: "; ".join(sorted(set(map(str, x.dropna())))[:20])),
        )
        .reset_index()
        .sort_values(["tier", "rfam_target", "candidate_gene", "n_variants"], ascending=[True, True, True, False])
    )

    summary_gene.to_csv(outdir / "summary_clinvar_expanded_by_gene_bucket.csv", index=False)

    summary_tier = (
        cv_expanded
        .groupby(["tier", "clinical_bucket"], dropna=False)
        .agg(
            n_variants=("VariationID", "nunique"),
            n_genes=("candidate_gene", "nunique"),
            n_rows=("VariationID", "size"),
        )
        .reset_index()
        .sort_values(["tier", "n_variants"], ascending=[True, False])
    )

    summary_tier.to_csv(outdir / "summary_clinvar_expanded_by_tier.csv", index=False)

    print("Done.")
    print("Gene-level rows:", len(cv_expanded))
    print("Unique variants:", cv_expanded["VariationID"].nunique())
    print("Unique candidate genes:", cv_expanded["candidate_gene"].nunique())
    print(outdir / "clinvar_expanded_rna_gene_level.csv")
    print(outdir / "summary_clinvar_expanded_by_gene_bucket.csv")
    print(outdir / "summary_clinvar_expanded_by_tier.csv")


if __name__ == "__main__":
    main()