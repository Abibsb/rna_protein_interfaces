#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def as_bool_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_unique_case_id(df):
    if "case_id" in df.columns:
        return df

    required = ["pdb_id", "uniprot_id", "uniprot_pos", "wild_type", "mutated_type"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        df["case_id"] = df.index.map(lambda x: f"case_{x}")
        return df

    df["case_id"] = (
        df["pdb_id"].astype(str).str.upper()
        + "_"
        + df["uniprot_id"].astype(str)
        + "_"
        + df["wild_type"].astype(str)
        + df["uniprot_pos"].astype(str)
        + df["mutated_type"].astype(str)
    )

    df["case_id"] = (
        df["pdb_id"].astype(str).str.upper()
        + "_"
        + df["uniprot_id"].astype(str)
        + "_"
        + df["wild_type"].astype(str)
        + df["uniprot_pos"].astype(str)
        + df["mutated_type"].astype(str)
    )

    return df


def pick_top(df, n, score_col, already_selected, bucket, reason):
    if n <= 0 or df.empty:
        return pd.DataFrame()

    pool = df[~df["case_id"].isin(already_selected)].copy()
    if pool.empty:
        return pd.DataFrame()

    pool = pool.sort_values(score_col, ascending=False, na_position="last")
    picked = pool.head(n).copy()
    picked["selection_bucket"] = bucket
    picked["selection_reason"] = reason
    return picked


def pick_top_per_group(df, group_col, n_total, score_col, already_selected, bucket, reason, per_group=1):
    if n_total <= 0 or df.empty or group_col not in df.columns:
        return pd.DataFrame()

    pool = df[~df["case_id"].isin(already_selected)].copy()
    if pool.empty:
        return pd.DataFrame()

    rows = []
    for _, g in pool.sort_values(score_col, ascending=False, na_position="last").groupby(group_col, sort=False):
        rows.append(g.head(per_group))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out

    out = out.sort_values(score_col, ascending=False, na_position="last").head(n_total).copy()
    out["selection_bucket"] = bucket
    out["selection_reason"] = reason
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Select a stratified AlphaFold panel from the full candidate dataset."
    )
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--panel-size", type=int, default=60)
    ap.add_argument("--score-col", default="alphafold_candidate_score")
    ap.add_argument("--only-modelable", action="store_true")
    ap.add_argument("--random-seed", type=int, default=123)

    ap.add_argument("--n-top-score", type=int, default=20)
    ap.add_argument("--n-top-per-interface", type=int, default=20)
    ap.add_argument("--n-multimut-residue", type=int, default=10)
    ap.add_argument("--n-high-ring", type=int, default=5)
    ap.add_argument("--n-controls", type=int, default=5)

    args = ap.parse_args()

    np.random.seed(args.random_seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.candidates, low_memory=False)
    df = add_unique_case_id(df)

    # -----------------------------
    # Detect columns
    # -----------------------------
    pdb_col = find_col(df, ["pdb_id", "PDBID", "pdb"])
    protein_chain_col = find_col(df, ["protein_chain", "auth_asym_id", "label_asym_id"])
    uniprot_col = find_col(df, ["uniprot_id", "UniProt", "uniprot"])
    pos_col = find_col(df, ["uniprot_pos", "position", "pos"])
    wt_col = find_col(df, ["wild_type", "Wild_Type"])
    mut_col = find_col(df, ["mutated_type", "Mutated_Type"])
    wt_match_col = find_col(df, ["observed_matches_variant_wt"])
    cif_mut_col = find_col(df, ["cif_mut_written"])
    status_col = find_col(df, ["status"])
    ring_col = find_col(df, ["has_ring_interface", "has_ring_contact"])
    ring_count_col = find_col(df, ["ring_interaction_count", "rna_contact_count"])
    rna_nt_col = find_col(df, ["rna_unique_nucleotides", "n_rna_unique_nucleotides"])
    core_col = find_col(df, ["is_core_hotspot"])
    strong_col = find_col(df, ["is_strong_interface"])
    interface_col = find_col(df, ["is_interface", "is_interface_abs"])
    delta_sasa_col = find_col(df, ["delta_sasa"])
    buried_col = find_col(df, ["buried_fraction_residue"])
    pathogenic_col = find_col(df, ["is_pathogenic_or_damaging"])

    score_col = args.score_col if args.score_col in df.columns else None
    if score_col is None:
        raise ValueError(f"Score column not found: {args.score_col}")

    df["_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(-np.inf)

    if pdb_col:
        df["_pdb_id"] = df[pdb_col].astype(str).str.upper()
    else:
        df["_pdb_id"] = "unknown"

    if protein_chain_col:
        df["_protein_chain"] = df[protein_chain_col].astype(str)
    else:
        df["_protein_chain"] = "unknown"

    df["_interface_id"] = df["_pdb_id"] + "|protein_chain=" + df["_protein_chain"]

    if uniprot_col:
        df["_uniprot_id"] = df[uniprot_col].astype(str)
    else:
        df["_uniprot_id"] = "unknown"

    if pos_col:
        df["_uniprot_pos"] = pd.to_numeric(df[pos_col], errors="coerce")
    else:
        df["_uniprot_pos"] = np.nan

    df["_residue_key"] = (
        df["_pdb_id"].astype(str)
        + "|"
        + df["_protein_chain"].astype(str)
        + "|"
        + df["_uniprot_id"].astype(str)
        + "|"
        + df["_uniprot_pos"].astype("Int64").astype(str)
    )

    if ring_count_col:
        df["_ring_interaction_count"] = pd.to_numeric(df[ring_count_col], errors="coerce").fillna(0)
    else:
        df["_ring_interaction_count"] = 0

    if rna_nt_col:
        df["_rna_unique_nucleotides"] = pd.to_numeric(df[rna_nt_col], errors="coerce").fillna(0)
    else:
        df["_rna_unique_nucleotides"] = 0

    if ring_col:
        df["_has_ring_interface"] = as_bool_series(df[ring_col])
    else:
        df["_has_ring_interface"] = False

    df["_has_ring_or_rna_contact"] = (
        df["_has_ring_interface"]
        | (df["_ring_interaction_count"] > 0)
        | (df["_rna_unique_nucleotides"] > 0)
    )

    df["_is_core_hotspot"] = as_bool_series(df[core_col]) if core_col else False
    df["_is_strong_interface"] = as_bool_series(df[strong_col]) if strong_col else False
    df["_is_interface"] = as_bool_series(df[interface_col]) if interface_col else False
    df["_is_pathogenic_or_damaging"] = as_bool_series(df[pathogenic_col]) if pathogenic_col else False

    if delta_sasa_col:
        df["_delta_sasa"] = pd.to_numeric(df[delta_sasa_col], errors="coerce").fillna(0)
    else:
        df["_delta_sasa"] = 0

    if buried_col:
        df["_buried_fraction_residue"] = pd.to_numeric(df[buried_col], errors="coerce").fillna(0)
    else:
        df["_buried_fraction_residue"] = 0

    # -----------------------------
    # Modelability filter
    # -----------------------------
    df["_modelable"] = True

    if wt_match_col:
        df["_wt_match"] = as_bool_series(df[wt_match_col])
        df["_modelable"] &= df["_wt_match"]

    if cif_mut_col:
        df["_cif_mut_written"] = as_bool_series(df[cif_mut_col])
        df["_modelable"] &= df["_cif_mut_written"]

    if status_col:
        df["_status_ok"] = df[status_col].astype(str).str.upper().eq("OK")
        df["_modelable"] &= df["_status_ok"]

    if args.only_modelable:
        universe = df[df["_modelable"]].copy()
    else:
        universe = df.copy()

    # Require at least some RNA/interface evidence for the positive candidate pool
    positive_pool = universe[
        universe["_has_ring_or_rna_contact"]
        & (
            universe["_is_interface"]
            | universe["_is_strong_interface"]
            | universe["_is_core_hotspot"]
            | (universe["_ring_interaction_count"] > 0)
        )
    ].copy()

    if positive_pool.empty:
        positive_pool = universe.copy()

    selected_parts = []
    selected_ids = set()

    def add_selection(part):
        nonlocal selected_ids, selected_parts
        if part is None or part.empty:
            return
        part = part[~part["case_id"].isin(selected_ids)].copy()
        if part.empty:
            return
        selected_ids.update(part["case_id"].tolist())
        selected_parts.append(part)

    # -----------------------------
    # Bucket 1: top score
    # -----------------------------
    add_selection(
        pick_top(
            positive_pool,
            args.n_top_score,
            "_score",
            selected_ids,
            "top_score",
            "Highest global AlphaFold candidate score among eligible candidates.",
        )
    )

    # -----------------------------
    # Bucket 2: top per interface
    # -----------------------------
    add_selection(
        pick_top_per_group(
            positive_pool,
            "_interface_id",
            args.n_top_per_interface,
            "_score",
            selected_ids,
            "top_per_interface",
            "Top ranked candidate per PDB/protein-chain interface to improve structural diversity.",
            per_group=1,
        )
    )

    # -----------------------------
    # Bucket 3: multiple mutations same residue
    # -----------------------------
    residue_counts = (
        positive_pool.groupby("_residue_key")["case_id"]
        .nunique()
        .reset_index(name="n_variants_at_residue")
    )
    multi_keys = residue_counts[residue_counts["n_variants_at_residue"] >= 2]["_residue_key"]

    multi_pool = positive_pool[positive_pool["_residue_key"].isin(multi_keys)].copy()
    multi_pool = multi_pool.sort_values(
        ["_residue_key", "_score"],
        ascending=[True, False],
        na_position="last",
    )

    add_selection(
        pick_top(
            multi_pool,
            args.n_multimut_residue,
            "_score",
            selected_ids,
            "multi_mutation_residue",
            "Residues represented by multiple alternative substitutions.",
        )
    )

    # -----------------------------
    # Bucket 4: high RING/RNA contacts
    # -----------------------------
    high_ring_pool = positive_pool.sort_values(
        ["_ring_interaction_count", "_rna_unique_nucleotides", "_score"],
        ascending=[False, False, False],
        na_position="last",
    )

    add_selection(
        pick_top(
            high_ring_pool,
            args.n_high_ring,
            "_score",
            selected_ids,
            "high_ring_or_rna_contacts",
            "Candidates with high RING/RNA contact support.",
        )
    )

    # -----------------------------
    # Bucket 5: controls
    # -----------------------------
    # Controls are modelable cases with lower score or weaker class, but still technically runnable.
    control_pool = universe[
        (~universe["case_id"].isin(selected_ids))
        & universe["_modelable"]
    ].copy()

    # Prefer lower score, non-core, non-strong if available.
    control_pool["_control_rank"] = (
        control_pool["_is_core_hotspot"].astype(int) * 3
        + control_pool["_is_strong_interface"].astype(int) * 2
        + control_pool["_is_interface"].astype(int)
    )

    control_pool = control_pool.sort_values(
        ["_control_rank", "_score"],
        ascending=[True, True],
        na_position="last",
    )

    add_selection(
        pick_top(
            control_pool,
            args.n_controls,
            "_score",
            selected_ids,
            "controls_lower_priority",
            "Runnable lower-priority controls to reduce positive-selection bias.",
        )
    )

    # -----------------------------
    # Fill remaining slots by diversity
    # -----------------------------
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()

    remaining = args.panel_size - len(selected)
    if remaining > 0:
        add_selection(
            pick_top_per_group(
                positive_pool,
                "_interface_id",
                remaining,
                "_score",
                selected_ids,
                "fill_diverse_top_per_interface",
                "Additional high-ranking candidates selected with interface diversity.",
                per_group=1,
            )
        )

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()

    if len(selected) > args.panel_size:
        selected = selected.head(args.panel_size).copy()

    # -----------------------------
    # Write outputs
    # -----------------------------
    selected.to_csv(outdir / "stratified_alphafold_panel.tsv", sep="\t", index=False)

    # Also write a CSV compatible with prepare_af_fragment_workflow.py
    selected.to_csv(outdir / "stratified_alphafold_panel.csv", index=False)

    counts = (
        selected["selection_bucket"]
        .value_counts(dropna=False)
        .reset_index()
    )
    counts.columns = ["selection_bucket", "n"]
    counts.to_csv(outdir / "stratified_panel_bucket_counts.tsv", sep="\t", index=False)

    by_interface = (
        selected["_interface_id"]
        .value_counts(dropna=False)
        .reset_index()
    )
    by_interface.columns = ["interface_id", "n_selected"]
    by_interface.to_csv(outdir / "stratified_panel_by_interface.tsv", sep="\t", index=False)

    by_pdb = (
        selected["_pdb_id"]
        .value_counts(dropna=False)
        .reset_index()
    )
    by_pdb.columns = ["pdb_id", "n_selected"]
    by_pdb.to_csv(outdir / "stratified_panel_by_pdb.tsv", sep="\t", index=False)

    # Universe accounting
    accounting = pd.DataFrame([
        {"metric": "n_input_rows", "value": len(df)},
        {"metric": "n_modelable_rows", "value": int(df["_modelable"].sum())},
        {"metric": "n_universe_after_filter", "value": len(universe)},
        {"metric": "n_positive_pool", "value": len(positive_pool)},
        {"metric": "n_selected_panel", "value": len(selected)},
        {"metric": "n_selected_unique_pdb", "value": selected["_pdb_id"].nunique() if not selected.empty else 0},
        {"metric": "n_selected_unique_interface", "value": selected["_interface_id"].nunique() if not selected.empty else 0},
        {"metric": "n_selected_unique_residue_key", "value": selected["_residue_key"].nunique() if not selected.empty else 0},
    ])
    accounting.to_csv(outdir / "stratified_panel_accounting.tsv", sep="\t", index=False)

    print("Wrote:")
    for f in sorted(outdir.glob("*")):
        print(f)

    print("\nPanel accounting:")
    print(accounting.to_string(index=False))

    print("\nBucket counts:")
    print(counts.to_string(index=False))

    cols = [
        "case_id",
        "selection_bucket",
        "selection_reason",
        "pdb_id",
        "uniprot_id",
        "uniprot_pos",
        "variant_id",
        "wild_type",
        "mutated_type",
        "_interface_id",
        "_score",
        "_ring_interaction_count",
        "_rna_unique_nucleotides",
        "_is_core_hotspot",
        "_is_strong_interface",
        "_is_interface",
    ]
    cols = [c for c in cols if c in selected.columns]
    print("\nSelected panel preview:")
    print(selected[cols].head(100).to_string(index=False))


if __name__ == "__main__":
    main()