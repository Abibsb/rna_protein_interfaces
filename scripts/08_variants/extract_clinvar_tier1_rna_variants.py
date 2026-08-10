#!/usr/bin/env python3

import argparse
import gzip
import re
from pathlib import Path
import pandas as pd


TIER1_GENES = {"TERC", "RNU11", "RN7SK", "RNU6ATAC", "RNU12"}


def normalize_gene_set(x):
    if pd.isna(x):
        return set()
    s = str(x).upper()
    # ClinVar GeneSymbol can contain multiple symbols separated by ;, |, comma
    parts = re.split(r"[;|,/\s]+", s)
    return {p.strip() for p in parts if p.strip()}


def extract_reference_pos_from_name(name, gene):
    """
    Attempts to extract transcript/RNA position from ClinVar preferred Name.
    This is conservative and should be QC'd.

    Examples that may occur:
    NM_...:c.123A>G
    NR_...:n.123A>G
    TERC:n.107G>A
    RNU4ATAC:n.51G>A
    g. coordinates are NOT converted here.
    """
    if pd.isna(name):
        return None, None, None, None

    s = str(name)

    # Prefer n. notation for noncoding RNA
    # Captures n.123A>G, n.123+1G>A will not be treated as simple position
    patterns = [
        r"[:\(]n\.([0-9]+)([ACGTU])>([ACGTU])",
        r"[:\(]r\.([0-9]+)([ACGTU])>([ACGTU])",
        r"[:\(]c\.([0-9]+)([ACGTU])>([ACGTU])",
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            pos = int(m.group(1))
            ref = m.group(2).upper().replace("T", "U")
            alt = m.group(3).upper().replace("T", "U")
            notation = pat.split("\\.")[0].replace("[:\\(]", "") + "."
            return pos, ref, alt, notation

    return None, None, None, None


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
    parser.add_argument(
        "--variant-summary",
        required=True,
        help="ClinVar variant_summary.txt.gz"
    )
    parser.add_argument(
        "--outdir",
        required=True
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read only needed columns if present
    # variant_summary columns can vary slightly over time, so read header first.
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
        "nsv/esv (dbVar)",
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
        "Guidelines",
        "TestedInGTR",
        "OtherIDs",
        "SubmitterCategories",
        "VariationID",
        "PositionVCF",
    ]

    usecols = [c for c in wanted if c in header]

    df_iter = pd.read_csv(
        args.variant_summary,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        dtype=str,
        chunksize=500000,
        low_memory=False
    )

    chunks = []

    for chunk in df_iter:
        if "GeneSymbol" not in chunk.columns:
            raise ValueError("GeneSymbol column not found in ClinVar variant_summary.")

        mask = chunk["GeneSymbol"].apply(lambda x: bool(normalize_gene_set(x) & TIER1_GENES))
        sub = chunk[mask].copy()

        if not sub.empty:
            chunks.append(sub)

    if chunks:
        cv = pd.concat(chunks, ignore_index=True)
    else:
        cv = pd.DataFrame(columns=usecols)

    # Normalize gene match
    def matched_gene(x):
        gs = normalize_gene_set(x)
        hits = sorted(gs & TIER1_GENES)
        return ";".join(hits)

    cv["reference_key"] = cv["GeneSymbol"].apply(matched_gene)

    # Keep one row per matched gene if GeneSymbol has multiple matches
    expanded_rows = []
    for _, row in cv.iterrows():
        genes = str(row["reference_key"]).split(";")
        for g in genes:
            if not g:
                continue
            r = row.to_dict()
            r["reference_key"] = g
            expanded_rows.append(r)

    cv = pd.DataFrame(expanded_rows)

    if cv.empty:
        cv.to_csv(outdir / "clinvar_tier1_rna_variants_gene_level.csv", index=False)
        pd.DataFrame().to_csv(outdir / "clinvar_tier1_rna_variants_position_resolved.csv", index=False)
        print("No ClinVar variants found for Tier 1 genes.")
        return

    # Try to derive RNA/transcript position from Name
    parsed = cv.apply(
        lambda r: extract_reference_pos_from_name(r.get("Name", ""), r.get("reference_key", "")),
        axis=1
    )

    cv["reference_pos"] = [x[0] for x in parsed]
    cv["ref_nt_from_hgvs"] = [x[1] for x in parsed]
    cv["alt_nt_from_hgvs"] = [x[2] for x in parsed]
    cv["hgvs_notation_type"] = [x[3] for x in parsed]

    if "ClinicalSignificance" in cv.columns:
        cv["clinical_bucket"] = cv["ClinicalSignificance"].apply(pathogenic_bucket)
    else:
        cv["clinical_bucket"] = "unknown"

    # Gene-level output
    cv.to_csv(outdir / "clinvar_tier1_rna_variants_gene_level.csv", index=False)

    # Position-resolved output
    pos = cv[cv["reference_pos"].notna()].copy()
    pos["reference_pos"] = pd.to_numeric(pos["reference_pos"], errors="coerce").astype("Int64")

    # Rename to expected merge-friendly fields
    rename = {
        "VariationID": "variant_id",
        "Name": "variant_name",
        "ClinicalSignificance": "clinical_significance",
        "ReviewStatus": "review_status",
        "PhenotypeList": "disease",
        "ReferenceAlleleVCF": "ref_allele_vcf",
        "AlternateAlleleVCF": "alt_allele_vcf",
    }
    pos = pos.rename(columns={k: v for k, v in rename.items() if k in pos.columns})

    pos["source"] = "ClinVar_variant_summary"
    pos["evidence_level"] = "clinvar_position_parsed_from_hgvs_name"

    pos.to_csv(outdir / "clinvar_tier1_rna_variants_position_resolved.csv", index=False)

    # Summaries
    summary_gene = (
        cv.groupby(["reference_key", "clinical_bucket"], dropna=False)
        .size()
        .reset_index(name="n_variants")
        .sort_values(["reference_key", "n_variants"], ascending=[True, False])
    )
    summary_gene.to_csv(outdir / "summary_clinvar_tier1_by_gene_bucket.csv", index=False)

    summary_pos = (
        pos.groupby(["reference_key", "clinical_bucket"], dropna=False)
        .size()
        .reset_index(name="n_position_resolved_variants")
        .sort_values(["reference_key", "n_position_resolved_variants"], ascending=[True, False])
    )
    summary_pos.to_csv(outdir / "summary_clinvar_tier1_position_resolved.csv", index=False)

    print("Done.")
    print(f"Gene-level ClinVar variants: {len(cv)}")
    print(f"Position-resolved variants: {len(pos)}")
    print(outdir / "clinvar_tier1_rna_variants_gene_level.csv")
    print(outdir / "clinvar_tier1_rna_variants_position_resolved.csv")
    print(outdir / "summary_clinvar_tier1_by_gene_bucket.csv")
    print(outdir / "summary_clinvar_tier1_position_resolved.csv")


if __name__ == "__main__":
    main()