#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def collapse_unique(values, sep="; "):
    clean = sorted({
        clean_text(v)
        for v in values
        if clean_text(v) not in ["", "nan", "None"]
    })
    return sep.join(clean)


def infer_rna_type(row):
    """
    Conservative RNA type inference from available text fields.
    This does not overwrite raw annotations; it creates a broad category.
    """
    text = " ".join([
        clean_text(row.get("entity_description", "")),
        clean_text(row.get("ncbi_gene_name", "")),
        clean_text(row.get("ncbi_gene_description", "")),
        clean_text(row.get("gene_symbol_normalized", "")),
    ]).lower()

    if "mir" in text or "microrna" in text or "micro-rna" in text:
        return "miRNA"
    if "lncrna" in text or "long non-coding" in text or "long noncoding" in text:
        return "lncRNA"
    if "trna" in text or "tRNA".lower() in text:
        return "tRNA"
    if "rrna" in text or "ribosomal rna" in text:
        return "rRNA"
    if "snrna" in text or "small nuclear" in text:
        return "snRNA"
    if "snorna" in text or "small nucleolar" in text:
        return "snoRNA"
    if "srp" in text or "signal recognition particle" in text or "7sl" in text:
        return "SRP_RNA"
    if "riboswitch" in text:
        return "riboswitch"
    if "rna" in text:
        return "other_RNA"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize RNA disease/pathology annotations from NCBI-mapped PDB RNA metadata."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV, e.g. rna_metadata_ncbi_pathologies_flagged.csv"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory"
    )
    args = parser.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inpath)

    required = [
        "pdb_id",
        "entity_id",
        "chain_ids",
        "entity_description",
        "gene_symbol_normalized",
        "ncbi_gene_id",
        "ncbi_gene_name",
        "ncbi_gene_description",
        "ncbi_gene_summary",
        "ncbi_gene_organism",
        "has_pathology",
        "pathology_text",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing expected columns:\n"
            + "\n".join(missing)
            + "\nAvailable columns are:\n"
            + "\n".join(df.columns)
        )

    # Normalize booleans and key fields
    df["has_pathology_bool"] = (
        df["has_pathology"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    df["is_human_bool"] = (
        df["ncbi_gene_organism"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("homo sapiens")
    )

    df["has_gene_id"] = df["ncbi_gene_id"].notna() & (df["ncbi_gene_id"].astype(str).str.strip() != "")
    df["has_gene_symbol"] = (
        df["gene_symbol_normalized"].notna()
        & (df["gene_symbol_normalized"].astype(str).str.strip() != "")
    )

    df["rna_type_inferred"] = df.apply(infer_rna_type, axis=1)

    # Keep a cleaned annotated full table
    df.to_csv(outdir / "rna_ncbi_pathology_annotated_clean.csv", index=False)

    # 1. General coverage summary
    n_total = len(df)
    n_pdbs = df["pdb_id"].nunique()
    n_human = int(df["is_human_bool"].sum())
    n_gene_id = int(df["has_gene_id"].sum())
    n_gene_symbol = int(df["has_gene_symbol"].sum())
    n_pathology = int(df["has_pathology_bool"].sum())

    summary = pd.DataFrame([{
        "n_rows_rna_entities": n_total,
        "n_unique_pdbs": n_pdbs,
        "n_human_entities": n_human,
        "fraction_human_entities": n_human / n_total if n_total else 0,
        "n_with_ncbi_gene_id": n_gene_id,
        "fraction_with_ncbi_gene_id": n_gene_id / n_total if n_total else 0,
        "n_with_gene_symbol": n_gene_symbol,
        "fraction_with_gene_symbol": n_gene_symbol / n_total if n_total else 0,
        "n_with_pathology": n_pathology,
        "fraction_with_pathology": n_pathology / n_total if n_total else 0,
        "n_unique_pathology_genes": df.loc[df["has_pathology_bool"], "gene_symbol_normalized"].nunique(),
        "n_unique_pathology_pdbs": df.loc[df["has_pathology_bool"], "pdb_id"].nunique(),
    }])

    summary.to_csv(outdir / "summary_annotation_coverage.csv", index=False)

    # 2. RNA entities with pathology
    patho_cols = [
        "pdb_id",
        "entity_id",
        "chain_ids",
        "entity_description",
        "rna_type_inferred",
        "gene_symbol_normalized",
        "ncbi_gene_id",
        "ncbi_gene_name",
        "ncbi_gene_description",
        "ncbi_gene_organism",
        "pathology_text",
    ]

    patho = (
        df.loc[df["has_pathology_bool"], patho_cols]
        .sort_values(["gene_symbol_normalized", "pdb_id", "entity_id"])
    )

    patho.to_csv(outdir / "rna_entities_with_pathology.csv", index=False)

    # 3. Summary by gene
    by_gene = (
        df.loc[df["has_pathology_bool"]]
        .groupby(["gene_symbol_normalized", "ncbi_gene_id", "ncbi_gene_name"], dropna=False)
        .agg(
            n_rows=("pdb_id", "size"),
            n_unique_pdbs=("pdb_id", "nunique"),
            n_unique_entities=("entity_id", "nunique"),
            chain_ids=("chain_ids", collapse_unique),
            rna_types=("rna_type_inferred", collapse_unique),
            entity_descriptions=("entity_description", collapse_unique),
            pathology_texts=("pathology_text", collapse_unique),
        )
        .reset_index()
        .sort_values(["n_unique_pdbs", "n_rows"], ascending=False)
    )

    by_gene.to_csv(outdir / "summary_pathology_by_gene.csv", index=False)

    # 4. Summary by PDB
    by_pdb = (
        df.loc[df["has_pathology_bool"]]
        .groupby("pdb_id", dropna=False)
        .agg(
            n_pathology_entities=("entity_id", "size"),
            n_pathology_genes=("gene_symbol_normalized", "nunique"),
            genes=("gene_symbol_normalized", collapse_unique),
            rna_types=("rna_type_inferred", collapse_unique),
            pathology_texts=("pathology_text", collapse_unique),
        )
        .reset_index()
        .sort_values(["n_pathology_entities", "n_pathology_genes"], ascending=False)
    )

    by_pdb.to_csv(outdir / "summary_pathology_by_pdb.csv", index=False)

    # 5. Summary by RNA type
    by_type = (
        df.groupby("rna_type_inferred", dropna=False)
        .agg(
            n_entities=("pdb_id", "size"),
            n_human=("is_human_bool", "sum"),
            n_with_gene_id=("has_gene_id", "sum"),
            n_with_pathology=("has_pathology_bool", "sum"),
            n_unique_pdbs=("pdb_id", "nunique"),
            n_unique_genes=("gene_symbol_normalized", "nunique"),
        )
        .reset_index()
    )

    by_type["fraction_with_pathology"] = by_type["n_with_pathology"] / by_type["n_entities"]
    by_type = by_type.sort_values(["n_with_pathology", "n_entities"], ascending=False)

    by_type.to_csv(outdir / "summary_pathology_by_rna_type.csv", index=False)

    # 6. Non-human / problematic rows, useful for limitations
    nonhuman_or_unmapped = df.loc[
        (~df["is_human_bool"]) | (~df["has_gene_id"]),
        [
            "pdb_id",
            "entity_id",
            "chain_ids",
            "entity_description",
            "external_db",
            "external_accession",
            "gene_symbol_scraped",
            "gene_symbol_normalized",
            "ncbi_gene_id",
            "ncbi_gene_organism",
            "rna_type_inferred",
            "has_pathology",
        ],
    ]

    nonhuman_or_unmapped.to_csv(outdir / "nonhuman_or_unmapped_rna_entities.csv", index=False)

    print("\nDone.")
    print(f"Input rows: {n_total}")
    print(f"Unique PDBs: {n_pdbs}")
    print(f"Human RNA entities: {n_human}")
    print(f"With NCBI Gene ID: {n_gene_id}")
    print(f"With pathology flag: {n_pathology}")
    print("\nSaved outputs:")
    for f in [
        "rna_ncbi_pathology_annotated_clean.csv",
        "summary_annotation_coverage.csv",
        "rna_entities_with_pathology.csv",
        "summary_pathology_by_gene.csv",
        "summary_pathology_by_pdb.csv",
        "summary_pathology_by_rna_type.csv",
        "nonhuman_or_unmapped_rna_entities.csv",
    ]:
        print(f"  - {outdir / f}")


if __name__ == "__main__":
    main()