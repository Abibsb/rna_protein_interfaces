#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def s(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def low(x):
    return s(x).lower()


def classify_mapping(row):
    desc = low(row.get("entity_description", ""))
    gene = low(row.get("gene_symbol_normalized", ""))
    ncbi_name = low(row.get("ncbi_gene_name", ""))
    ncbi_desc = low(row.get("ncbi_gene_description", ""))
    combined_gene = " ".join([gene, ncbi_name, ncbi_desc])

    # Good specific RNA matches
    if "u4atac" in desc and "rnu4atac" in combined_gene:
        return "high_confidence"

    if "telomerase" in desc and "terc" in combined_gene:
        return "high_confidence"

    if ("srp" in desc or "7sl" in desc) and ("rn7sl" in combined_gene or "signal recognition particle" in combined_gene):
        return "high_confidence"

    if "u4" in desc and ("rnu4" in combined_gene or "small nuclear" in combined_gene):
        return "high_confidence"

    if "secis" in desc:
        # SECIS belongs to an mRNA/gene context; useful but not the same as ncRNA identity
        return "gene_context_rna_element"

    # Mitochondrial RNAs
    if ("12s" in desc or "16s" in desc or "rrna" in desc or "ribosomal rna" in desc):
        if "mt-rnr" in combined_gene or "ribosomal rna" in combined_gene:
            return "high_confidence"
        if gene in ["nd1", "nd2", "nd3", "nd4", "nd4l", "nd5", "nd6", "cox1", "cox2", "cox3", "cytb", "atp6", "atp8"]:
            return "suspicious_protein_gene_for_rna"

    if "trna" in desc or "t-rna" in desc:
        if "trn" in combined_gene or "trna" in combined_gene or "transfer rna" in combined_gene:
            return "high_confidence"
        return "suspicious_protein_gene_for_trna"

    if "mir" in desc or "microrna" in desc or "pri-mir" in desc:
        if "mir" in combined_gene or "microrna" in combined_gene:
            return "high_confidence"
        return "suspicious_host_or_wrong_gene_for_mirna"

    if "snrna" in desc or "small nuclear" in desc:
        if "rnu" in combined_gene or "small nuclear" in combined_gene:
            return "high_confidence"
        return "suspicious_protein_gene_for_snrna"

    # Suspicious disease abbreviations or protein-coding genes detected from your output
    suspicious_genes = {
        "cd4", "fecd", "trk1", "huwe1", "nd1"
    }

    if gene in suspicious_genes:
        return "suspicious_manual_review"

    if gene == "" or gene == "nan":
        return "unmapped"

    return "needs_manual_review"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    df["mapping_qc"] = df.apply(classify_mapping, axis=1)

    df["has_pathology_bool"] = (
        df["has_pathology"]
        .astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "yes"])
    )

    df["is_confident_pathology"] = (
        df["has_pathology_bool"]
        & df["mapping_qc"].isin(["high_confidence", "gene_context_rna_element"])
    )

    df.to_csv(outdir / "rna_ncbi_pathology_with_mapping_qc.csv", index=False)

    summary_qc = (
        df.groupby("mapping_qc")
        .agg(
            n_entities=("pdb_id", "size"),
            n_pdbs=("pdb_id", "nunique"),
            n_pathology=("has_pathology_bool", "sum"),
            n_confident_pathology=("is_confident_pathology", "sum"),
            genes=("gene_symbol_normalized", lambda x: "; ".join(sorted(set(map(str, x.dropna()))))),
        )
        .reset_index()
        .sort_values("n_entities", ascending=False)
    )

    summary_qc.to_csv(outdir / "summary_mapping_qc.csv", index=False)

    confident = df[df["is_confident_pathology"]].copy()
    confident.to_csv(outdir / "rna_entities_confident_pathology.csv", index=False)

    suspicious = df[
        df["has_pathology_bool"]
        & ~df["is_confident_pathology"]
    ].copy()

    suspicious.to_csv(outdir / "rna_entities_suspicious_pathology.csv", index=False)

    by_gene_confident = (
        confident
        .groupby(["gene_symbol_normalized", "ncbi_gene_id", "ncbi_gene_name"], dropna=False)
        .agg(
            n_entities=("pdb_id", "size"),
            n_pdbs=("pdb_id", "nunique"),
            entity_descriptions=("entity_description", lambda x: "; ".join(sorted(set(map(str, x.dropna()))))),
            pathology_texts=("pathology_text", lambda x: "; ".join(sorted(set(map(str, x.dropna()))))),
            mapping_qc=("mapping_qc", lambda x: "; ".join(sorted(set(map(str, x.dropna()))))),
        )
        .reset_index()
        .sort_values(["n_pdbs", "n_entities"], ascending=False)
    )

    by_gene_confident.to_csv(outdir / "summary_confident_pathology_by_gene.csv", index=False)

    print("Done.")
    print("Outputs:")
    print(outdir / "rna_ncbi_pathology_with_mapping_qc.csv")
    print(outdir / "summary_mapping_qc.csv")
    print(outdir / "rna_entities_confident_pathology.csv")
    print(outdir / "rna_entities_suspicious_pathology.csv")
    print(outdir / "summary_confident_pathology_by_gene.csv")


if __name__ == "__main__":
    main()