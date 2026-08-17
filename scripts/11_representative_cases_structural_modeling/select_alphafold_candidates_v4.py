#!/usr/bin/env python3
"""
Select AlphaFold candidate mutations from thesis interface/variant tables.

v3 changes:
- Robust to duplicate column names created by table merges.
- Row-wise numeric maximum now coerces each candidate column safely.

v2 changes vs previous script:
- Does NOT trust generic has_contact flags for RNA contacts.
- Builds direct RNA-contact evidence from explicit RNA/RING columns.
- Combines numeric contact columns by row-wise maximum instead of taking the first matching column.
- Penalizes or excludes rows without direct RNA-contact evidence.

Recommended input:
  variants_bestrow_ring_rna.csv  (best)
  variants_bestrow_ring.csv      (acceptable only if it has explicit rna_* or ring_interaction_rows columns)

Example:
python select_alphafold_candidates_v4.py \
  --input variants_bestrow_ring_rna.csv \
  --outdir alphafold_candidate_selection_v2 \
  --top-n 50 \
  --require-direct-rna-contact
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TRUE_VALUES = {"true", "t", "1", "yes", "y", "si", "sí"}
FALSE_VALUES = {"false", "f", "0", "no", "n", "nan", "none", "", "na", "n/a"}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def colmap(df: pd.DataFrame) -> dict[str, str]:
    return {norm_col(c): c for c in df.columns}


def get_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = colmap(df)
    for cand in candidates:
        key = norm_col(cand)
        if key in lookup:
            return lookup[key]
    return None


def existing_cols(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    lookup = colmap(df)
    found = []
    for cand in candidates:
        key = norm_col(cand)
        if key in lookup and lookup[key] not in found:
            found.append(lookup[key])
    return found


def as_single_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return one clean Series even if the input table has duplicate column names.

    Some intermediate thesis CSVs can contain repeated headers after merges.
    In pandas, df[col] then returns a DataFrame, which can break reductions.
    This helper keeps the first duplicate column and preserves the original index.
    """
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def numeric(df: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    col = get_col(df, candidates)
    if col is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(as_single_series(df, col), errors="coerce")


def numeric_max(df: pd.DataFrame, candidates: Iterable[str], default=0) -> pd.Series:
    """Row-wise max across candidate numeric columns, without pandas DataFrame reductions.

    Some pandas versions can recurse when DataFrame.max() is applied to messy
    object blocks or duplicate-column outputs. This implementation converts each
    candidate column to a plain NumPy float array and uses np.nanmax directly.
    """
    cols = existing_cols(df, candidates)
    if not cols:
        return pd.Series(default, index=df.index, dtype="float64")

    arrays = []
    for c in cols:
        s = pd.to_numeric(as_single_series(df, c), errors="coerce")
        arrays.append(np.asarray(s, dtype="float64"))

    if not arrays:
        return pd.Series(default, index=df.index, dtype="float64")

    mat = np.vstack(arrays).T  # shape: n_rows x n_candidate_columns
    with np.errstate(all="ignore"):
        out = np.nanmax(mat, axis=1)
    out = np.where(np.isfinite(out), out, default)
    return pd.Series(out, index=df.index, dtype="float64")


def text(df: pd.DataFrame, candidates: Iterable[str], default="") -> pd.Series:
    col = get_col(df, candidates)
    if col is None:
        return pd.Series(default, index=df.index, dtype="object")
    return as_single_series(df, col).astype("string").fillna(default).astype(str)


def boolean(df: pd.DataFrame, candidates: Iterable[str], default=False) -> pd.Series:
    col = get_col(df, candidates)
    if col is None:
        return pd.Series(default, index=df.index, dtype="bool")
    s = as_single_series(df, col)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(1 if default else 0).astype(float).ne(0)
    return s.astype(str).str.strip().str.lower().map(
        lambda x: True if x in TRUE_VALUES else (False if x in FALSE_VALUES else default)
    )


def nonempty(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").astype(str).str.strip()
    return (~s.str.lower().isin(FALSE_VALUES)) & s.ne("")


def percentile_score(x: pd.Series, higher_is_better=True) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(0.0, index=x.index)
    r = x.rank(pct=True, na_option="bottom")
    return r if higher_is_better else 1 - r


def contains_any(series: pd.Series, patterns: Iterable[str]) -> pd.Series:
    pat = "|".join(patterns)
    return series.astype(str).str.contains(pat, case=False, na=False, regex=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="variants_bestrow_ring_rna.csv, variants_bestrow_ring.csv, or equivalent table")
    ap.add_argument("--outdir", default="alphafold_candidate_selection_v2")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--max-iupred", type=float, default=0.5)
    ap.add_argument("--require-direct-rna-contact", action="store_true", help="Keep only candidates with explicit RNA-contact evidence")
    ap.add_argument("--consurf-high-is-conserved", action="store_true", default=True, help="Assume higher ConSurf/color means more conserved. Use --consurf-low-is-conserved to invert.")
    ap.add_argument("--consurf-low-is-conserved", action="store_true", help="Assume lower ConSurf/rate4site means more conserved.")
    args = ap.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_table(inpath)
    original_cols = list(df.columns)

    out = pd.DataFrame(index=df.index)

    out["pdb_id"] = text(df, ["pdb_id_variant", "pdb_id_ringpair", "pdb_id", "entry_id", "PDB ID"])
    out["uniprot_id"] = text(df, ["uniprot_id", "uniprot", "accession", "UniProt"])
    out["gene"] = text(df, ["gene", "gene_name", "Gene Name", "Gene_Symbol"])
    out["protein_chain"] = text(df, ["protein_chain", "chain_id", "auth_asym_id", "label_asym_id"])
    out["uniprot_pos"] = numeric(df, ["uniprot_pos", "uniprot_position", "protein_pos", "pos", "position"])
    out["variant_id"] = text(df, ["variant_id", "variant", "mutation", "hgvsp", "HGVS", "variant_name"])
    out["wild_type"] = text(df, ["wild_type", "wt", "reference_aa", "ref_aa", "protein_resname", "resname"])
    out["mutated_type"] = text(df, ["mutated_type", "mut", "alt_aa", "mutant", "alternate_aa"])

    for c in ["wild_type", "mutated_type"]:
        out[c] = out[c].replace(AA3_TO_1)

    clinical_text = text(df, ["clinical_significance", "ClinicalSignificance", "clinical", "disease", "Disease Association", "disease_association", "source", "phenotype"])
    out["clinical_text"] = clinical_text
    out["is_pathogenic_text"] = contains_any(clinical_text, [r"pathogenic", r"likely pathogenic", r"disease", r"association"])
    out["is_benign_text"] = contains_any(clinical_text, [r"benign", r"likely benign"])
    out["disease_association"] = boolean(df, ["Disease Association", "disease_association", "yA", "has_disease_association"], default=False)
    out["bad_mutation"] = boolean(df, ["bad_mutation", "bad_mutations", "yB"], default=False)
    out["polyphen_damaging"] = contains_any(text(df, ["Prediction_PolyPhen", "polyphen", "PolyPhen"]), [r"damaging", r"probably", r"possibly"])
    out["sift_deleterious"] = contains_any(text(df, ["SIFT", "Prediction_SIFT", "sift_prediction"]), [r"deleterious"])

    out["is_pathogenic_or_damaging"] = (
        out["is_pathogenic_text"] | out["disease_association"] | out["bad_mutation"] |
        out["polyphen_damaging"] | out["sift_deleterious"]
    ) & ~out["is_benign_text"]

    out["is_interface"] = boolean(df, ["is_interface_abs", "is_interface", "interface"], default=False)
    out["is_strong_interface"] = boolean(df, ["is_strong_interface", "strong_interface", "delta_sasa_percentile_global"], default=False)
    out["is_core_hotspot"] = boolean(df, ["is_core_hotspot", "core_hotspot", "hotspot"], default=False)

    # Explicit RNA/RING contact signals. Do not use broad/generic has_contact here.
    out["ring_interaction_count"] = numeric_max(
        df,
        [
            "rna_contact_count",
            "rna_unique_nucleotides",
            "ring_interaction_rows",
            "ring_interaction_count",
            "n_rna_contacts",
            "protein_rna_contact_count",
        ],
        default=0,
    )
    out["rna_unique_nucleotides"] = numeric_max(df, ["rna_unique_nucleotides", "rna_unique_nt", "rna_residue_count", "n_unique_rna_residues"], default=0)

    out["rna_chain"] = text(df, ["rna_chain", "rna_chain_id", "rna_auth_asym_id", "rna_label_asym_id"])
    out["rna_resnum"] = numeric(df, ["rna_resnum", "rna_resnum_used", "rna_pos", "query_pos"])
    out["rna_nt"] = text(df, ["rna_nt", "nt", "rna_resname", "resname_y"])
    out["ring_interaction_types"] = text(df, ["ring_interaction_types", "interaction_type", "interaction_types"])

    explicit_ring_flag = boolean(df, ["has_ring_interface", "ring_interface", "has_rna_contact", "has_protein_rna_contact"], default=False)
    has_rna_identity = nonempty(out["rna_chain"]) | out["rna_resnum"].notna() | nonempty(out["rna_nt"])
    has_ring_types = nonempty(out["ring_interaction_types"])
    out["has_direct_rna_contact"] = out["ring_interaction_count"].gt(0) | has_rna_identity | has_ring_types | explicit_ring_flag
    out["has_ring_interface"] = out["has_direct_rna_contact"]

    out["delta_sasa"] = numeric(df, ["delta_sasa", "delta_sasa_abs", "bsa", "buried_surface_area"])
    out["buried_fraction_residue"] = numeric(df, ["buried_fraction_residue", "buried_fraction", "fraction_buried"])
    out["consurf_score"] = numeric(df, ["consurf_score", "consurf", "color", "conservation", "rate4site_score"])
    out["iupred_score"] = numeric(df, ["iupred_score", "iupred", "iupred_long", "disorder_score"])
    out["rna_shannon"] = numeric(df, ["rna_shannon", "shannon", "shannon_conservation", "mean_shannon"])

    out["is_disordered"] = out["iupred_score"].fillna(0).ge(args.max_iupred)
    out["score_delta_sasa_pct"] = percentile_score(out["delta_sasa"], higher_is_better=True)
    out["score_buried_pct"] = percentile_score(out["buried_fraction_residue"], higher_is_better=True)
    consurf_higher = not args.consurf_low_is_conserved
    out["score_consurf_pct"] = percentile_score(out["consurf_score"], higher_is_better=consurf_higher)
    out["score_rna_shannon_pct"] = percentile_score(out["rna_shannon"], higher_is_better=True)

    out["alphafold_candidate_score"] = 0.0
    out.loc[out["is_pathogenic_or_damaging"], "alphafold_candidate_score"] += 4
    out.loc[out["is_interface"], "alphafold_candidate_score"] += 3
    out.loc[out["has_direct_rna_contact"], "alphafold_candidate_score"] += 4
    out.loc[out["is_core_hotspot"], "alphafold_candidate_score"] += 3
    out.loc[out["is_strong_interface"], "alphafold_candidate_score"] += 1.5
    out.loc[out["ring_interaction_count"].ge(2), "alphafold_candidate_score"] += 1.5
    out.loc[out["ring_interaction_count"].ge(3), "alphafold_candidate_score"] += 1.0
    out["alphafold_candidate_score"] += 1.5 * out["score_consurf_pct"]
    out["alphafold_candidate_score"] += 1.0 * out["score_delta_sasa_pct"]
    out["alphafold_candidate_score"] += 1.0 * out["score_buried_pct"]
    out["alphafold_candidate_score"] += 0.5 * out["score_rna_shannon_pct"]
    out.loc[out["is_disordered"], "alphafold_candidate_score"] -= 2
    out.loc[out["is_benign_text"], "alphafold_candidate_score"] -= 3
    out.loc[~out["has_direct_rna_contact"], "alphafold_candidate_score"] -= 5

    out["candidate_tier"] = "C_review"
    tier_a = (
        out["is_pathogenic_or_damaging"] &
        out["is_interface"] &
        out["has_direct_rna_contact"] &
        (out["is_core_hotspot"] | out["ring_interaction_count"].ge(2))
    )
    tier_b = (
        out["is_pathogenic_or_damaging"] &
        out["is_interface"] &
        out["has_direct_rna_contact"]
    ) & ~tier_a
    out.loc[tier_a, "candidate_tier"] = "A_best"
    out.loc[tier_b, "candidate_tier"] = "B_good"

    if args.require_direct_rna_contact:
        out = out[out["has_direct_rna_contact"]].copy()

    out_sorted = out.sort_values("alphafold_candidate_score", ascending=False)
    dedup_keys = ["uniprot_id", "uniprot_pos", "wild_type", "mutated_type"]
    nonempty_dedup = out_sorted[dedup_keys].replace("", np.nan).notna().all(axis=1)
    if nonempty_dedup.any():
        candidates = out_sorted.drop_duplicates(subset=dedup_keys, keep="first")
    else:
        candidates = out_sorted

    first_cols = [
        "candidate_tier", "alphafold_candidate_score", "pdb_id", "uniprot_id", "gene",
        "protein_chain", "uniprot_pos", "variant_id", "wild_type", "mutated_type",
        "clinical_text", "is_pathogenic_or_damaging", "is_interface", "is_strong_interface",
        "is_core_hotspot", "has_direct_rna_contact", "ring_interaction_count", "rna_unique_nucleotides",
        "delta_sasa", "buried_fraction_residue", "consurf_score", "iupred_score",
        "rna_chain", "rna_resnum", "rna_nt", "rna_shannon", "ring_interaction_types",
    ]
    first_cols = [c for c in first_cols if c in candidates.columns]
    candidates = candidates[first_cols + [c for c in candidates.columns if c not in first_cols]]

    out_sorted.to_csv(outdir / "alphafold_candidates_all_scored.csv", index=False)
    candidates.to_csv(outdir / "alphafold_candidates_deduplicated.csv", index=False)
    candidates.head(args.top_n).to_csv(outdir / f"alphafold_candidates_top{args.top_n}.csv", index=False)
    for tier in ["A_best", "B_good", "C_review"]:
        candidates[candidates["candidate_tier"].eq(tier)].to_csv(outdir / f"alphafold_candidates_{tier}.csv", index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Rows after optional RNA-contact filter: {len(out):,}")
    print(f"Deduplicated candidate mutations: {len(candidates):,}")
    print("\nCandidate tiers:")
    if len(candidates):
        print(candidates["candidate_tier"].value_counts(dropna=False).to_string())
    else:
        print("No candidates retained.")
    print("\nDirect RNA-contact evidence:")
    print(f"Rows with direct RNA contact in original input: {int((out_sorted['has_direct_rna_contact']).sum()) if len(out_sorted) else 0:,}")
    print("\nTop candidates:")
    cols = [c for c in ["candidate_tier", "alphafold_candidate_score", "pdb_id", "gene", "uniprot_id", "uniprot_pos", "wild_type", "mutated_type", "ring_interaction_count", "rna_chain", "rna_resnum", "rna_nt", "delta_sasa", "consurf_score"] if c in candidates.columns]
    if len(candidates):
        print(candidates[cols].head(args.top_n).to_string(index=False))
    print(f"\nWrote outputs to: {outdir.resolve()}")
    print("\nOriginal columns found:")
    print(", ".join(original_cols))


if __name__ == "__main__":
    main()
