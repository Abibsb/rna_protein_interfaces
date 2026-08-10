#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


OUTDIR = Path("final_rna_disease_annotation_tables")
POS_ALL = OUTDIR / "FINAL_3C_ClinVar_position_resolved_all.csv"
GENELEVEL = OUTDIR / "FINAL_2_ClinVar_genelevel_summary.csv"


def clinical_bucket_fixed(x):
    s = str(x).lower()

    if "conflicting" in s:
        return "conflicting"
    if "uncertain" in s or "vus" in s:
        return "vus_or_uncertain"
    if "benign" in s:
        return "benign_or_likely_benign"
    if "pathogenic" in s:
        return "pathogenic_or_likely_pathogenic"
    return "other"


def main():
    if not POS_ALL.exists():
        raise FileNotFoundError(POS_ALL)

    df = pd.read_csv(POS_ALL)

    sig_col = "clinical_significance"
    if sig_col not in df.columns and "ClinicalSignificance" in df.columns:
        sig_col = "ClinicalSignificance"

    df["clinical_bucket_fixed"] = df[sig_col].apply(clinical_bucket_fixed)

    if "variant_ref_matches_reference" in df.columns:
        df = df[df["variant_ref_matches_reference"] == True].copy()

    # Deduplicate duplicated GRCh37/GRCh38 rows for the same variant-structure mapping.
    dedup_keys = [
        "reference_key",
        "reference_pos",
        "variant_id",
        "pdb_id",
        "chain_id",
        "pdb_seq_pos",
    ]
    dedup_keys = [c for c in dedup_keys if c in df.columns]

    df_dedup = df.drop_duplicates(subset=dedup_keys).copy()

    patho = df_dedup[
        df_dedup["clinical_bucket_fixed"].eq("pathogenic_or_likely_pathogenic")
    ].copy()

    patho.to_csv(
        OUTDIR / "FINAL_3D_ClinVar_position_resolved_pathogenic_QC_DEDUP.csv",
        index=False
    )

    summary = (
        patho.groupby("reference_key")
        .agg(
            n_rows=("reference_pos", "count"),
            n_variants=("variant_id", "nunique"),
            n_positions=("reference_pos", "nunique"),
            n_pdb_chains=("pdb_id", lambda x: patho.loc[x.index, ["pdb_id", "chain_id"]].drop_duplicates().shape[0]),
            n_interface=("is_interface_abs", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(bool).sum()),
            n_strong_interface=("is_strong_interface", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(bool).sum()),
            n_core_hotspot=("is_core_hotspot", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(bool).sum()),
            mean_delta_sasa=("delta_sasa", "mean"),
            mean_shannon=("shannon_conservation", "mean"),
            mean_ring_contacts=("ring_interaction_count", "mean"),
        )
        .reset_index()
        if not patho.empty else pd.DataFrame()
    )

    summary.to_csv(
        OUTDIR / "FINAL_3E_summary_position_resolved_pathogenic_QC_DEDUP.csv",
        index=False
    )

    bucket_summary = (
        df_dedup.groupby(["reference_key", "clinical_bucket_fixed"])
        .agg(
            n_rows=("reference_pos", "count"),
            n_variants=("variant_id", "nunique"),
            n_positions=("reference_pos", "nunique"),
        )
        .reset_index()
    )

    bucket_summary.to_csv(
        OUTDIR / "FINAL_3F_summary_position_resolved_all_buckets_DEDUP.csv",
        index=False
    )

    overall = pd.DataFrame([
        {
            "analysis_level": "ClinVar position-resolved all, deduplicated",
            "main_count": df_dedup["variant_id"].nunique(),
            "secondary_count": df_dedup[["reference_key", "reference_pos"]].drop_duplicates().shape[0],
            "description": "All position-resolved ClinVar variants after reference-allele QC and GRCh37/GRCh38 deduplication."
        },
        {
            "analysis_level": "ClinVar position-resolved pathogenic QC, deduplicated",
            "main_count": patho["variant_id"].nunique(),
            "secondary_count": patho[["reference_key", "reference_pos"]].drop_duplicates().shape[0] if not patho.empty else 0,
            "description": "Pathogenic/likely pathogenic position-resolved variants after corrected clinical bucket, reference-allele QC, and GRCh37/GRCh38 deduplication."
        }
    ])

    overall.to_csv(
        OUTDIR / "FINAL_3G_overall_position_resolved_DEDUP_counts.csv",
        index=False
    )

    print("Done.")
    print("Wrote:")
    print(OUTDIR / "FINAL_3D_ClinVar_position_resolved_pathogenic_QC_DEDUP.csv")
    print(OUTDIR / "FINAL_3E_summary_position_resolved_pathogenic_QC_DEDUP.csv")
    print(OUTDIR / "FINAL_3F_summary_position_resolved_all_buckets_DEDUP.csv")
    print(OUTDIR / "FINAL_3G_overall_position_resolved_DEDUP_counts.csv")
    print("\nCorrected pathogenic summary:")
    print(summary.to_string(index=False))
    print("\nBucket summary:")
    print(bucket_summary.to_string(index=False))


if __name__ == "__main__":
    main()