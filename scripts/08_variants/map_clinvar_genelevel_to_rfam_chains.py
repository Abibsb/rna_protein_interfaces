#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def expand_pdb_chains(pdb_chains):
    out = []
    for pc in str(pdb_chains).split(";"):
        pc = pc.strip()
        if "_" not in pc:
            continue
        pdb, ch = pc.split("_", 1)
        out.append((pdb, ch))
    return out


def n_unique_pdb_chains(df):
    if df.empty:
        return 0
    return df[["pdb_id", "chain_id"]].drop_duplicates().shape[0]


def count_pathogenic_variants(group):
    if "clinical_bucket" not in group.columns or "VariationID" not in group.columns:
        return 0
    mask = group["clinical_bucket"].astype(str).eq("pathogenic_or_likely_pathogenic")
    return group.loc[mask, "VariationID"].dropna().nunique()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfam-targets", required=True, help="rna_positional_mapping_targets.csv")
    parser.add_argument("--gene-set", required=True, help="expanded_rna_clinvar_gene_set_tier1_tier2.csv")
    parser.add_argument("--clinvar-gene-level", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(args.rfam_targets)
    gene_set = pd.read_csv(args.gene_set)
    clinvar = pd.read_csv(args.clinvar_gene_level)

    # Normalize keys
    targets["target_name"] = targets["target_name"].astype(str)
    gene_set["rfam_target"] = gene_set["rfam_target"].astype(str)
    gene_set["candidate_gene"] = gene_set["candidate_gene"].astype(str).str.upper()
    clinvar["candidate_gene"] = clinvar["candidate_gene"].astype(str).str.upper()

    # Make sure these exist even if ClinVar had no variants for some genes
    if "VariationID" not in clinvar.columns:
        clinvar["VariationID"] = pd.NA
    if "clinical_bucket" not in clinvar.columns:
        clinvar["clinical_bucket"] = pd.NA

    # Map each Rfam target to candidate genes
    target_gene = targets.merge(
        gene_set,
        left_on="target_name",
        right_on="rfam_target",
        how="inner",
        suffixes=("", "_geneset")
    )

    # Add ClinVar gene-level variants
    mapped = target_gene.merge(
        clinvar,
        on=["candidate_gene", "tier", "rfam_target"],
        how="left",
        suffixes=("_chain", "_clinvar")
    )

    mapped.to_csv(outdir / "clinvar_genelevel_mapped_to_rfam_unique_sequences.csv", index=False)

    # Expand to PDB chains
    rows = []
    for _, row in mapped.iterrows():
        for pdb, ch in expand_pdb_chains(row.get("pdb_chains", "")):
            r = row.to_dict()
            r["pdb_id"] = str(pdb).upper()
            r["chain_id"] = str(ch)
            rows.append(r)

    expanded = pd.DataFrame(rows)

    if expanded.empty:
        expanded.to_csv(outdir / "clinvar_genelevel_mapped_to_rfam_pdb_chains.csv", index=False)
        pd.DataFrame().to_csv(outdir / "summary_genelevel_clinvar_by_rfam_gene.csv", index=False)
        pd.DataFrame().to_csv(outdir / "summary_genelevel_clinvar_by_tier.csv", index=False)
        print("No expanded PDB-chain rows produced.")
        return

    expanded["has_clinvar_variant"] = expanded["VariationID"].notna()
    expanded["is_pathogenic_or_likely_pathogenic"] = (
        expanded["clinical_bucket"].astype(str).eq("pathogenic_or_likely_pathogenic")
    )

    expanded.to_csv(outdir / "clinvar_genelevel_mapped_to_rfam_pdb_chains.csv", index=False)

    # Summary by Rfam target and candidate gene
    summary_rows = []

    group_cols = ["tier", "rfam_target", "candidate_gene"]

    for keys, g in expanded.groupby(group_cols, dropna=False):
        tier, rfam_target, candidate_gene = keys

        summary_rows.append({
            "tier": tier,
            "rfam_target": rfam_target,
            "candidate_gene": candidate_gene,
            "n_pdb_chains": n_unique_pdb_chains(g),
            "n_unique_sequences": g["unique_rna_id"].nunique() if "unique_rna_id" in g.columns else 0,
            "n_clinvar_variants": g["VariationID"].dropna().nunique(),
            "n_pathogenic_variants": g.loc[
                g["is_pathogenic_or_likely_pathogenic"], "VariationID"
            ].dropna().nunique(),
            "n_vus_or_uncertain": g.loc[
                g["clinical_bucket"].astype(str).eq("vus_or_uncertain"), "VariationID"
            ].dropna().nunique(),
            "n_benign_or_likely_benign": g.loc[
                g["clinical_bucket"].astype(str).eq("benign_or_likely_benign"), "VariationID"
            ].dropna().nunique(),
        })

    summary_chain = pd.DataFrame(summary_rows).sort_values(
        ["tier", "rfam_target", "n_clinvar_variants"],
        ascending=[True, True, False]
    )

    summary_chain.to_csv(outdir / "summary_genelevel_clinvar_by_rfam_gene.csv", index=False)

    # Summary by tier
    tier_rows = []

    for tier, g in expanded.groupby("tier", dropna=False):
        tier_rows.append({
            "tier": tier,
            "n_pdb_chains": n_unique_pdb_chains(g),
            "n_unique_sequences": g["unique_rna_id"].nunique() if "unique_rna_id" in g.columns else 0,
            "n_candidate_genes": g["candidate_gene"].nunique(),
            "n_clinvar_variants": g["VariationID"].dropna().nunique(),
            "n_pathogenic_variants": g.loc[
                g["is_pathogenic_or_likely_pathogenic"], "VariationID"
            ].dropna().nunique(),
            "n_vus_or_uncertain": g.loc[
                g["clinical_bucket"].astype(str).eq("vus_or_uncertain"), "VariationID"
            ].dropna().nunique(),
            "n_benign_or_likely_benign": g.loc[
                g["clinical_bucket"].astype(str).eq("benign_or_likely_benign"), "VariationID"
            ].dropna().nunique(),
        })

    summary_tier = pd.DataFrame(tier_rows).sort_values("tier")
    summary_tier.to_csv(outdir / "summary_genelevel_clinvar_by_tier.csv", index=False)

    print("Done.")
    print("Expanded PDB-chain rows:", len(expanded))
    print("Unique PDB chains:", n_unique_pdb_chains(expanded))
    print("Unique ClinVar variants:", expanded["VariationID"].dropna().nunique())
    print("Pathogenic/Likely pathogenic variants:",
          expanded.loc[expanded["is_pathogenic_or_likely_pathogenic"], "VariationID"].dropna().nunique())
    print(outdir / "clinvar_genelevel_mapped_to_rfam_unique_sequences.csv")
    print(outdir / "clinvar_genelevel_mapped_to_rfam_pdb_chains.csv")
    print(outdir / "summary_genelevel_clinvar_by_rfam_gene.csv")
    print(outdir / "summary_genelevel_clinvar_by_tier.csv")


if __name__ == "__main__":
    main()