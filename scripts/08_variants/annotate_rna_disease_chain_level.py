#!/usr/bin/env python3

import argparse
import pandas as pd
from pathlib import Path


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def collapse_unique(series, sep="; "):
    vals = sorted({str(x).strip() for x in series.dropna() if str(x).strip()})
    return sep.join(vals)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate RNA chains with RNA-disease associations at chain/gene level."
    )
    parser.add_argument("--rna-metadata", required=True, help="CSV with RNA chain metadata from PDB/mmCIF.")
    parser.add_argument("--disease-db", required=True, help="Normalized CSV with RNA-disease associations.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--species-filter", default="Homo sapiens", help="Species name to prioritize.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rna = pd.read_csv(args.rna_metadata)
    disease = pd.read_csv(args.disease_db)

    # Normalize expected columns if absent
    for col in ["gene_symbol", "gene_id", "rnacentral_urs", "refseq_id", "rna_type", "species"]:
        if col not in rna.columns:
            rna[col] = pd.NA

    for col in [
        "source", "rna_name", "gene_symbol", "gene_id", "rnacentral_urs",
        "refseq_id", "rna_type", "disease_name", "evidence_type", "pubmed_id", "species"
    ]:
        if col not in disease.columns:
            disease[col] = pd.NA

    # Normalized keys
    rna["gene_symbol_norm"] = rna["gene_symbol"].map(norm_text)
    rna["gene_id_norm"] = rna["gene_id"].map(norm_text)
    rna["rnacentral_urs_norm"] = rna["rnacentral_urs"].map(norm_text)
    rna["refseq_id_norm"] = rna["refseq_id"].map(norm_text)
    rna["species_norm"] = rna["species"].map(norm_text)

    disease["gene_symbol_norm"] = disease["gene_symbol"].map(norm_text)
    disease["gene_id_norm"] = disease["gene_id"].map(norm_text)
    disease["rnacentral_urs_norm"] = disease["rnacentral_urs"].map(norm_text)
    disease["refseq_id_norm"] = disease["refseq_id"].map(norm_text)
    disease["species_norm"] = disease["species"].map(norm_text)

    target_species_norm = norm_text(args.species_filter)

    # Filter disease DB to human when species is available
    disease_human = disease[
        (disease["species_norm"] == "") |
        (disease["species_norm"] == target_species_norm) |
        (disease["species_norm"].str.contains("HUMAN", na=False))
    ].copy()

    # Build long matches by multiple identifiers
    match_frames = []

    match_specs = [
        ("gene_symbol_norm", "gene_symbol"),
        ("gene_id_norm", "gene_id"),
        ("rnacentral_urs_norm", "rnacentral_urs"),
        ("refseq_id_norm", "refseq_id"),
    ]

    rna = rna.reset_index().rename(columns={"index": "rna_row_id"})

    for key_norm, key_name in match_specs:
        left = rna[rna[key_norm] != ""].copy()
        right = disease_human[disease_human[key_norm] != ""].copy()

        if left.empty or right.empty:
            continue

        merged = left.merge(
            right,
            on=key_norm,
            how="inner",
            suffixes=("_rna", "_disease")
        )
        merged["match_key"] = key_name
        match_frames.append(merged)

    if match_frames:
        matches = pd.concat(match_frames, ignore_index=True)
        matches = matches.drop_duplicates(
            subset=[
                "rna_row_id",
                "source",
                "disease_name",
                "evidence_type",
                "pubmed_id",
                "match_key"
            ]
        )
    else:
        matches = pd.DataFrame(columns=["rna_row_id"])

    matches.to_csv(outdir / "rna_disease_matches_long.csv", index=False)

    # Aggregate disease evidence per RNA chain
    if not matches.empty:
        agg = matches.groupby("rna_row_id").agg(
            has_disease_association=("disease_name", lambda x: True),
            n_disease_associations=("disease_name", lambda x: len(set([v for v in x.dropna() if str(v).strip()]))),
            disease_names=("disease_name", collapse_unique),
            disease_sources=("source", collapse_unique),
            evidence_types=("evidence_type", collapse_unique),
            pubmed_ids=("pubmed_id", collapse_unique),
            match_keys=("match_key", collapse_unique),
        ).reset_index()
    else:
        agg = pd.DataFrame({
            "rna_row_id": rna["rna_row_id"],
            "has_disease_association": False,
            "n_disease_associations": 0,
            "disease_names": "",
            "disease_sources": "",
            "evidence_types": "",
            "pubmed_ids": "",
            "match_keys": "",
        })

    annotated = rna.merge(agg, on="rna_row_id", how="left")

    annotated["has_disease_association"] = annotated["has_disease_association"].fillna(False)
    annotated["n_disease_associations"] = annotated["n_disease_associations"].fillna(0).astype(int)

    for col in ["disease_names", "disease_sources", "evidence_types", "pubmed_ids", "match_keys"]:
        annotated[col] = annotated[col].fillna("")

    # Basic mapping confidence, conservative default
    def infer_confidence(row):
        if norm_text(row.get("species", "")) not in ["", target_species_norm] and "HUMAN" not in norm_text(row.get("species", "")):
            return "low"
        if norm_text(row.get("gene_id", "")) or norm_text(row.get("refseq_id", "")):
            return "high"
        if norm_text(row.get("gene_symbol", "")) or norm_text(row.get("rnacentral_urs", "")):
            return "medium"
        return "unmapped"

    annotated["mapping_confidence"] = annotated.apply(infer_confidence, axis=1)

    annotated.to_csv(outdir / "rna_disease_chain_level.csv", index=False)

    # Summary tables
    summary = {
        "n_rna_chains": len(annotated),
        "n_with_gene_symbol": int((annotated["gene_symbol"].map(norm_text) != "").sum()),
        "n_with_gene_id": int((annotated["gene_id"].map(norm_text) != "").sum()),
        "n_with_rnacentral_urs": int((annotated["rnacentral_urs"].map(norm_text) != "").sum()),
        "n_with_refseq": int((annotated["refseq_id"].map(norm_text) != "").sum()),
        "n_disease_associated": int(annotated["has_disease_association"].sum()),
        "fraction_disease_associated": float(annotated["has_disease_association"].mean()) if len(annotated) else 0.0,
    }

    pd.DataFrame([summary]).to_csv(outdir / "summary_rna_disease_annotation.csv", index=False)

    by_type = (
        annotated
        .assign(rna_type_clean=annotated["rna_type"].fillna("unknown").astype(str))
        .groupby("rna_type_clean")
        .agg(
            n_chains=("rna_row_id", "count"),
            n_disease_associated=("has_disease_association", "sum"),
        )
        .reset_index()
    )
    by_type["fraction_disease_associated"] = (
        by_type["n_disease_associated"] / by_type["n_chains"]
    )
    by_type.to_csv(outdir / "summary_by_rna_type.csv", index=False)

    if not matches.empty:
        disease_counts = (
            matches
            .groupby("disease_name")
            .agg(
                n_rna_chains=("rna_row_id", "nunique"),
                sources=("source", collapse_unique),
                evidence_types=("evidence_type", collapse_unique),
            )
            .reset_index()
            .sort_values("n_rna_chains", ascending=False)
        )
        disease_counts.to_csv(outdir / "summary_top_diseases.csv", index=False)

    print("Done.")
    print(f"Annotated RNA chains: {outdir / 'rna_disease_chain_level.csv'}")
    print(f"Long matches:          {outdir / 'rna_disease_matches_long.csv'}")
    print(f"Summary:               {outdir / 'summary_rna_disease_annotation.csv'}")
    print(f"By RNA type:           {outdir / 'summary_by_rna_type.csv'}")


if __name__ == "__main__":
    main()