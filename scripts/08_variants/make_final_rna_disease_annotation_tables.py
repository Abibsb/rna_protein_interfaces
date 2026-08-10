#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def safe_read(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(
        description="Create final consolidated tables for RNA disease annotation analysis."
    )

    parser.add_argument(
        "--rfam-coverage",
        default="summary_rfam_annotation_coverage.csv"
    )
    parser.add_argument(
        "--rfam-families",
        default="summary_rfam_families_weighted.csv"
    )
    parser.add_argument(
        "--rfam-classified-summary",
        default="rfam_classified/summary_rfam_by_type_scope.csv"
    )

    parser.add_argument(
        "--genelevel-summary",
        default="clinvar_tier1_tier2_gene_level_mapped_to_rfam/summary_genelevel_clinvar_by_rfam_gene.csv"
    )
    parser.add_argument(
        "--genelevel-pathogenic-summary",
        default="clinvar_tier1_tier2_gene_level_mapped_to_rfam/summary_genelevel_pathogenic_by_rfam_gene.csv"
    )
    parser.add_argument(
        "--genelevel-tier-summary",
        default="clinvar_tier1_tier2_gene_level_mapped_to_rfam/summary_genelevel_clinvar_by_tier.csv"
    )

    parser.add_argument(
        "--position-qc",
        default="clinvar_tier1/tier1_rna_variants_mapped_to_structure/clinvar_tier1_pathogenic_position_resolved_QC.csv"
    )
    parser.add_argument(
        "--position-qc-summary",
        default="clinvar_tier1/tier1_rna_variants_mapped_to_structure/summary_pathogenic_position_resolved_QC_by_reference.csv"
    )
    parser.add_argument(
        "--position-all",
        default="clinvar_tier1/tier1_rna_variants_mapped_to_structure/clinvar_tier1_rna_variants_mapped_to_structure.csv"
    )

    parser.add_argument(
        "--tier2-rescue-summary",
        default="tier2_positional_rescue/reference_mapping_RNA_only/summary_tier2_best_reference_mapping.csv"
    )

    parser.add_argument(
        "--outdir",
        default="final_rna_disease_annotation_tables"
    )

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # =========================
    # FINAL 1: Rfam annotation
    # =========================
    rfam_cov = safe_read(args.rfam_coverage)
    rfam_fam = safe_read(args.rfam_families)
    rfam_class = safe_read(args.rfam_classified_summary, required=False)

    rfam_final = rfam_fam.copy()

    if not rfam_class.empty:
        # Attach type/scope if possible
        # summary_rfam_families_weighted.csv: target_name, target_accession, description, n_unique_sequences, n_chains
        # rfam_classified summary is grouped by type/scope, not family, so keep it separate too.
        rfam_class.to_csv(outdir / "FINAL_1B_Rfam_annotation_by_type_scope.csv", index=False)

    rfam_final["annotation_level"] = "rfam_family"
    rfam_final["interpretation"] = (
        "RNA family annotation by sequence similarity; does not imply exact human gene identity."
    )

    rfam_final.to_csv(outdir / "FINAL_1_Rfam_annotation_summary.csv", index=False)

    # =========================
    # FINAL 2: ClinVar gene-level
    # =========================
    gene_sum = safe_read(args.genelevel_summary)
    gene_path = safe_read(args.genelevel_pathogenic_summary, required=False)
    gene_tier = safe_read(args.genelevel_tier_summary, required=False)

    gene_final = gene_sum.copy()

    gene_final["annotation_level"] = "gene_level"
    gene_final["position_resolved"] = False
    gene_final["interpretation"] = (
        "RNA family or candidate gene has ClinVar variants; variants are not necessarily mapped "
        "to observed structural nucleotides."
    )

    # Add stricter caveat by tier
    def caveat(row):
        tier = str(row.get("tier", ""))
        rfam = str(row.get("rfam_target", ""))
        if tier == "tier1_high":
            return "Tier 1 gene identity has stronger support; still gene-level unless position-resolved table is used."
        if rfam in {"U1", "U2", "U3", "U4", "U5", "U6", "let-7", "mir-16", "Metazoa_SRP"}:
            return "Tier 2 family-level candidate; gene copy ambiguity may exist."
        return "Gene-level candidate."

    gene_final["caveat"] = gene_final.apply(caveat, axis=1)

    gene_final.to_csv(outdir / "FINAL_2_ClinVar_genelevel_summary.csv", index=False)

    if not gene_path.empty:
        gene_path.to_csv(outdir / "FINAL_2B_ClinVar_genelevel_pathogenic_summary.csv", index=False)

    if not gene_tier.empty:
        gene_tier.to_csv(outdir / "FINAL_2C_ClinVar_genelevel_by_tier_summary.csv", index=False)

    # =========================
    # FINAL 3: ClinVar position-resolved strict QC
    # =========================
    pos_qc = safe_read(args.position_qc, required=False)
    pos_qc_summary = safe_read(args.position_qc_summary, required=False)
    pos_all = safe_read(args.position_all, required=False)

    if not pos_qc.empty:
        pos_final = pos_qc.copy()

        pos_final["annotation_level"] = "position_resolved"
        pos_final["position_resolved"] = True
        pos_final["clinical_filter"] = "pathogenic_or_likely_pathogenic"
        pos_final["interpretation"] = (
            "ClinVar variant position maps directly to an RNA reference position and then to an observed "
            "PDB RNA nucleotide with structural metrics."
        )

        # Select and reorder useful columns if present
        preferred_cols = [
            "reference_key",
            "reference_pos",
            "reference_nt",
            "ref_nt_from_hgvs",
            "alt_nt_from_hgvs",
            "variant_id",
            "variant_name",
            "clinical_significance",
            "clinical_bucket",
            "review_status",
            "disease",
            "source",
            "evidence_level",
            "pdb_id",
            "chain_id",
            "pdb_seq_pos",
            "pdb_nt",
            "nt",
            "shannon_conservation",
            "gap_fraction",
            "sasa_unbound",
            "sasa_bound",
            "delta_sasa",
            "buried_fraction_residue",
            "is_interface_abs",
            "is_strong_interface",
            "is_core_hotspot",
            "ring_interaction_count",
            "has_ring_contact",
            "variant_ref_matches_reference",
            "mapping_quality",
            "identity",
            "query_coverage",
            "reference_span_density",
            "annotation_level",
            "position_resolved",
            "clinical_filter",
            "interpretation",
        ]

        final_cols = [c for c in preferred_cols if c in pos_final.columns]
        remaining_cols = [c for c in pos_final.columns if c not in final_cols]
        pos_final = pos_final[final_cols + remaining_cols]

        pos_final.to_csv(outdir / "FINAL_3_ClinVar_position_resolved_QC.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "FINAL_3_ClinVar_position_resolved_QC.csv", index=False)

    if not pos_qc_summary.empty:
        pos_qc_summary.to_csv(outdir / "FINAL_3B_ClinVar_position_resolved_QC_summary.csv", index=False)

    if not pos_all.empty:
        # broader non-pathogenic/non-QC strict position-resolved table
        pos_all_out = pos_all.copy()
        pos_all_out["annotation_level"] = "position_resolved_all_clinvar"
        pos_all_out["interpretation"] = (
            "All ClinVar position-resolved mappings before pathogenic-only filtering."
        )
        pos_all_out.to_csv(outdir / "FINAL_3C_ClinVar_position_resolved_all.csv", index=False)

    # =========================
    # Tier 2 positional rescue summary
    # =========================
    tier2_rescue = safe_read(args.tier2_rescue_summary, required=False)

    if not tier2_rescue.empty:
        tier2_rescue["position_rescue_result"] = tier2_rescue["mapping_quality"].apply(
            lambda x: "accepted_position_resolved" if str(x) == "high_unambiguous" else "not_position_resolved"
        )
        tier2_rescue["interpretation"] = tier2_rescue["mapping_quality"].apply(
            lambda x: (
                "Accepted as Tier 2 position-resolved rescue."
                if str(x) == "high_unambiguous"
                else "Retained as gene-level/family-level because mapping is ambiguous, medium, or low."
            )
        )
        tier2_rescue.to_csv(outdir / "FINAL_4_Tier2_positional_rescue_summary.csv", index=False)

    # =========================
    # Overall counts summary
    # =========================
    counts = []

    # Rfam coverage
    if not rfam_cov.empty:
        r = rfam_cov.iloc[0].to_dict()
        counts.append({
            "analysis_level": "Rfam sequence annotation",
            "main_count": r.get("n_unique_sequences_with_rfam_hit", None),
            "secondary_count": r.get("n_chains_with_rfam_hit", None),
            "description": "Unique sequences and original chains with Rfam hit."
        })

    # Gene-level ClinVar
    if not gene_tier.empty:
        total_variants = gene_tier["n_clinvar_variants"].sum() if "n_clinvar_variants" in gene_tier.columns else None
        total_path = gene_tier["n_pathogenic_variants"].sum() if "n_pathogenic_variants" in gene_tier.columns else None
        total_chains = gene_tier["n_pdb_chains"].sum() if "n_pdb_chains" in gene_tier.columns else None
        counts.append({
            "analysis_level": "ClinVar gene-level Tier1+Tier2",
            "main_count": total_variants,
            "secondary_count": total_path,
            "description": f"ClinVar gene-level variants; secondary count is pathogenic/likely pathogenic. PDB-chain sum by tier={total_chains}."
        })

    # Position-resolved all
    if not pos_all.empty:
        variant_col = "variant_id" if "variant_id" in pos_all.columns else "VariationID"
        counts.append({
            "analysis_level": "ClinVar position-resolved all",
            "main_count": pos_all[variant_col].nunique() if variant_col in pos_all.columns else len(pos_all),
            "secondary_count": pos_all[["reference_key", "reference_pos"]].drop_duplicates().shape[0] if {"reference_key", "reference_pos"}.issubset(pos_all.columns) else None,
            "description": "All ClinVar variants mapped to RNA reference positions and PDB RNA positions before pathogenic-only filtering."
        })

    # Position-resolved pathogenic QC
    if not pos_qc.empty:
        variant_col = "variant_id" if "variant_id" in pos_qc.columns else "VariationID"
        counts.append({
            "analysis_level": "ClinVar position-resolved pathogenic QC",
            "main_count": pos_qc[variant_col].nunique() if variant_col in pos_qc.columns else len(pos_qc),
            "secondary_count": pos_qc[["reference_key", "reference_pos"]].drop_duplicates().shape[0] if {"reference_key", "reference_pos"}.issubset(pos_qc.columns) else None,
            "description": "Pathogenic/likely pathogenic position-resolved variants after reference-allele QC."
        })

    counts_df = pd.DataFrame(counts)
    counts_df.to_csv(outdir / "FINAL_0_overall_counts_summary.csv", index=False)

    print("Done.")
    print(f"Final tables written to: {outdir}")
    for p in sorted(outdir.glob("FINAL_*.csv")):
        print(p)


if __name__ == "__main__":
    main()