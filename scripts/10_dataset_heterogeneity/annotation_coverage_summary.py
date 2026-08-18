#!/usr/bin/env python3

"""
Chapter 10.3 - Annotation coverage audit

Quantifies the effective coverage of the main annotations integrated into the
RNA-protein interface dataset. This script does NOT regenerate annotations.
It reads final/intermediate outputs produced by the corresponding pipelines
and reports their representation at PDB, chain, biological-entity and
residue/position levels.

The script is deliberately conservative:
- missing annotation != biological absence;
- NaN values are treated as unavailable annotations;
- structural and annotation universes are reported separately;
- files that cannot be identified unambiguously are reported as unavailable
  instead of silently substituting another dataset.

Author: Abigail Stricker
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------------
# Expected final values obtained during the thesis analysis.
# These are NOT used to calculate the results.
# They are only shown as QC references.
# -------------------------------------------------------------------------

EXPECTED = {
    "backbone_rows": 5_792_320,

    "pdb_uniprot_chains": 29_881,
    "pdb_uniprot_pdbs": 1_370,
    "pdb_uniprot_uniprots": 1_598,

    "pfam_mapped_chains": 9_198,
    "pfam_fraction": 0.3078210233927914,
    "pfam_families": 1_181,

    "consurf_residues": 112_135,
    "consurf_fraction": 0.0194,

    "iupred_residues": 5_462_482,
    "iupred_fraction": 0.9431,
    "iupred_noninterface_fraction": 0.9132,
    "iupred_interface_fraction": 0.9411,

    "rfam_total_chains": 2_369,
    "rfam_hit_chains": 1_540,
    "rfam_chain_fraction": 0.6501,
    "rfam_unique_total": 1_034,
    "rfam_unique_hits": 566,
    "rfam_unique_fraction": 0.5474,
    "rfam_queries": 849,
    "rfam_query_hits": 566,
    "rfam_conditional_fraction": 0.6667,

    "rna_shannon_rows": 1_304_459,
    "rna_shannon_chains": 1_529,
    "rna_shannon_pdbs": 583,
    "rna_shannon_families": 42,

    "shannon_protein_positions": 75_177,
    "shannon_protein_chains": 9_135,
    "shannon_protein_pdbs": 502,
    "shannon_contacts": 82_537,

    "variant_rows": 135_526,
    "variant_fraction": 0.0234,
    "protein_variants_structural": 278_222,
    "protein_variant_model_subset": 93_700,
    "protein_variant_model_uniprots": 670,

    "rna_pathology_entities": 147,
    "rna_pathology_total_entities": 386,
}


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def norm(s):
    return str(s).strip().lower()


def pct(n, d):
    if not d:
        return float("nan")
    return 100.0 * n / d


def first_existing(root: Path, names):
    """Return first exact filename found recursively."""
    for name in names:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def header_columns(path: Path):
    """Read only the CSV header."""
    try:
        return set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return set()


def find_csv_by_columns(root: Path, required, preferred_terms=()):
    """
    Find CSV containing all required columns.
    Preference is given to filenames containing preferred_terms.
    """
    required = set(required)
    candidates = []

    for p in root.rglob("*.csv"):
        try:
            cols = header_columns(p)
        except Exception:
            continue

        if required.issubset(cols):
            score = sum(term.lower() in p.name.lower()
                        for term in preferred_terms)
            candidates.append((score, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return candidates[0][1]


def load_csv(path, usecols=None):
    if path is None:
        return None
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}", file=sys.stderr)
        return None


def unique_chain_count(df):
    """
    Count unique PDB-chain combinations using the best available chain ID.
    """
    if df is None or "pdb_id" not in df.columns:
        return None

    for chain_col in (
        "auth_asym_id",
        "label_asym_id",
        "chain_id",
        "chain",
        "asym_id",
    ):
        if chain_col in df.columns:
            return (
                df[["pdb_id", chain_col]]
                .dropna()
                .drop_duplicates()
                .shape[0]
            )

    return None


def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_metric(label, value, expected=None):
    if expected is None:
        print(f"{label}: {value}")
        return

    if isinstance(value, float):
        print(f"{label}: {value:.6g}   [reference: {expected}]")
    else:
        print(f"{label}: {value}   [reference: {expected}]")


def report_file(label, path):
    print(f"{label}: {path if path else 'NOT FOUND'}")


# -------------------------------------------------------------------------
# Main analyses
# -------------------------------------------------------------------------

def backbone_coverage(base: Path):
    print_header("STRUCTURAL BACKBONE / INTEGRATED PROTEIN DATASET")

    master = first_existing(
        base,
        [
            "interface_analysis_dataset_residue_level.csv",
            "MASTER_interface_dataset_PER_CHAIN_WITH_VARIANTS.csv",
            "MASTER_interface_dataset.csv",
        ],
    )

    report_file("Backbone file", master)

    if master is None:
        return None

    needed = [
        "pdb_id",
        "consurf_score",
        "iupred_score",
        "is_interface_abs",
        "is_interface",
        "uniprot_id",
        "uniprot_pos",
    ]

    available = header_columns(master)
    usecols = [x for x in needed if x in available]

    df = load_csv(master, usecols=usecols)
    if df is None:
        return None

    n = len(df)
    print_metric("Rows", n, EXPECTED["backbone_rows"])

    if "pdb_id" in df:
        print(f"Unique PDB: {df['pdb_id'].nunique()}")

    if "uniprot_id" in df:
        print(f"Unique UniProt IDs: {df['uniprot_id'].dropna().nunique()}")

    # ConSurf
    if "consurf_score" in df:
        n_consurf = df["consurf_score"].notna().sum()
        print()
        print("ConSurf")
        print_metric(
            "Residues with ConSurf",
            n_consurf,
            EXPECTED["consurf_residues"],
        )
        print_metric(
            "Coverage",
            n_consurf / n if n else float("nan"),
            EXPECTED["consurf_fraction"],
        )

    # IUPred
    if "iupred_score" in df:
        n_iupred = df["iupred_score"].notna().sum()

        print()
        print("IUPred")
        print_metric(
            "Residues with IUPred",
            n_iupred,
            EXPECTED["iupred_residues"],
        )
        print_metric(
            "Coverage",
            n_iupred / n if n else float("nan"),
            EXPECTED["iupred_fraction"],
        )

        interface_col = None
        if "is_interface_abs" in df:
            interface_col = "is_interface_abs"
        elif "is_interface" in df:
            interface_col = "is_interface"

        if interface_col:
            flag = df[interface_col]

            # Normalize booleans/numeric strings.
            if flag.dtype == object:
                flag = (
                    flag.astype(str)
                    .str.lower()
                    .map({
                        "true": 1,
                        "false": 0,
                        "1": 1,
                        "0": 0,
                    })
                )

            interface = df.loc[flag == 1]
            noninterface = df.loc[flag == 0]

            if len(interface):
                print_metric(
                    "IUPred interface coverage",
                    interface["iupred_score"].notna().mean(),
                    EXPECTED["iupred_interface_fraction"],
                )

            if len(noninterface):
                print_metric(
                    "IUPred non-interface coverage",
                    noninterface["iupred_score"].notna().mean(),
                    EXPECTED["iupred_noninterface_fraction"],
                )

    return df


def pdb_uniprot_pfam_coverage(base: Path):
    print_header("UNIPROT / PFAM COVERAGE")

    # Try known/common mapping names first.
    mapping = first_existing(
        base,
        [
            "pdb_uniprot_position_map.csv",
            "pdb_uniprot_map.csv",
            "pdb_chain_uniprot_map.csv",
        ],
    )

    # Position-level map may enormously overcount chains, so look for a
    # chain-level mapping if one can be identified.
    chain_map = find_csv_by_columns(
        base,
        required=["pdb_id", "uniprot_id"],
        preferred_terms=("mapping", "uniprot", "chain"),
    )

    if chain_map is not None:
        mapping = chain_map

    report_file("PDB-UniProt mapping candidate", mapping)

    if mapping:
        cols = header_columns(mapping)
        useful = [
            x for x in (
                "pdb_id",
                "uniprot_id",
                "auth_asym_id",
                "label_asym_id",
                "chain_id",
                "chain",
                "asym_id",
                "uniprot_pos",
            )
            if x in cols
        ]

        m = load_csv(mapping, usecols=useful)

        if m is not None:
            chains = unique_chain_count(m)
            if chains is not None:
                print_metric(
                    "Protein PDB-chains",
                    chains,
                    EXPECTED["pdb_uniprot_chains"],
                )

            if "pdb_id" in m:
                print_metric(
                    "PDBs",
                    m["pdb_id"].dropna().nunique(),
                    EXPECTED["pdb_uniprot_pdbs"],
                )

            if "uniprot_id" in m:
                print_metric(
                    "Unique UniProt IDs",
                    m["uniprot_id"].dropna().nunique(),
                    EXPECTED["pdb_uniprot_uniprots"],
                )

    # Pfam mapping
    pfam_file = None

    candidates = []
    for p in base.rglob("*.csv"):
        cols = header_columns(p)
        pfam_cols = [
            c for c in cols
            if "pfam" in norm(c)
        ]
        if "pdb_id" in cols and pfam_cols:
            score = (
                2 * ("pfam" in p.name.lower())
                + ("mapping" in p.name.lower())
            )
            candidates.append((score, p, pfam_cols))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
        _, pfam_file, pfam_cols = candidates[0]

        report_file("Pfam mapping candidate", pfam_file)

        cols = header_columns(pfam_file)
        useful = [
            x for x in (
                "pdb_id",
                "auth_asym_id",
                "label_asym_id",
                "chain_id",
                "chain",
                "asym_id",
                "uniprot_id",
            )
            if x in cols
        ] + pfam_cols

        p = load_csv(pfam_file, usecols=list(dict.fromkeys(useful)))

        if p is not None:
            pfam_col = pfam_cols[0]

            covered = p[p[pfam_col].notna()].copy()

            chains = unique_chain_count(covered)
            if chains is not None:
                print_metric(
                    "Mapped protein chains with >=1 Pfam",
                    chains,
                    EXPECTED["pfam_mapped_chains"],
                )

            families = (
                covered[pfam_col]
                .dropna()
                .astype(str)
                .nunique()
            )

            print_metric(
                "Unique Pfam families",
                families,
                EXPECTED["pfam_families"],
            )
    else:
        print("Pfam mapping candidate: NOT FOUND")


def rfam_coverage(base: Path):
    print_header("RFAM COVERAGE")

    candidates = []

    for p in base.rglob("*.csv"):
        cols = header_columns(p)

        if not any("rfam" in norm(c) for c in cols):
            continue

        score = (
            2 * ("rfam" in p.name.lower())
            + ("annot" in p.name.lower())
            + ("chain" in p.name.lower())
        )
        candidates.append((score, p))

    if not candidates:
        print("Rfam annotation file: NOT FOUND")
        return

    candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
    rfam_file = candidates[0][1]

    report_file("Rfam annotation candidate", rfam_file)

    r = load_csv(rfam_file)
    if r is None:
        return

    rfam_cols = [c for c in r.columns if "rfam" in norm(c)]

    family_col = None
    for c in rfam_cols:
        if any(k in norm(c) for k in ("family", "accession", "acc", "rfam")):
            family_col = c
            break

    if family_col is None and rfam_cols:
        family_col = rfam_cols[0]

    chain_count = unique_chain_count(r)

    if chain_count is not None:
        print(f"RNA PDB-chains in selected file: {chain_count}")

    if family_col:
        hit = r[r[family_col].notna()]
        hit_chains = unique_chain_count(hit)

        if hit_chains is not None:
            print_metric(
                "Rfam hit chains",
                hit_chains,
                EXPECTED["rfam_hit_chains"],
            )

        print(f"Unique Rfam annotations: "
              f"{hit[family_col].dropna().astype(str).nunique()}")


def shannon_coverage(base: Path):
    print_header("RNA SHANNON COVERAGE")

    shannon = first_existing(
        base,
        [
            "RNA_SHANNON_MASTER.csv",
            "rna_shannon_master.csv",
        ],
    )

    report_file("RNA Shannon dataset", shannon)

    if shannon:
        s = load_csv(shannon)

        if s is not None:
            print_metric(
                "Rows",
                len(s),
                EXPECTED["rna_shannon_rows"],
            )

            if "pdb_id" in s:
                print_metric(
                    "Unique PDB",
                    s["pdb_id"].nunique(),
                    EXPECTED["rna_shannon_pdbs"],
                )

            chains = unique_chain_count(s)
            if chains is not None:
                print_metric(
                    "RNA PDB-chains",
                    chains,
                    EXPECTED["rna_shannon_chains"],
                )

            rfam_cols = [
                c for c in s.columns
                if "rfam" in norm(c)
            ]
            if rfam_cols:
                print_metric(
                    "Unique Rfam families",
                    s[rfam_cols[0]].dropna().nunique(),
                    EXPECTED["rna_shannon_families"],
                )

            entropy_cols = [
                c for c in s.columns
                if "shannon" in norm(c)
                or norm(c) == "entropy"
            ]

            if entropy_cols:
                c = entropy_cols[0]
                print(
                    f"Rows with Shannon: "
                    f"{s[c].notna().sum()}/{len(s)} "
                    f"({pct(s[c].notna().sum(), len(s)):.2f}%)"
                )

    # Protein-contact transfer.
    candidates = []

    conservation_dir = base / "conservacion"
    search_root = conservation_dir if conservation_dir.exists() else base

    for p in search_root.rglob("*.csv"):
        cols = header_columns(p)

        has_protein_position = (
            "uniprot_pos" in cols
            or "protein_pos" in cols
        )
        has_shannon = any("shannon" in norm(c) for c in cols)

        if has_protein_position and has_shannon and "pdb_id" in cols:
            score = (
                2 * ("protein" in p.name.lower())
                + 2 * ("shannon" in p.name.lower())
                + ("merge" in p.name.lower())
            )
            candidates.append((score, p))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
        transferred = candidates[0][1]

        report_file("Shannon -> protein candidate", transferred)

        t = load_csv(transferred)

        if t is not None:
            shannon_cols = [
                c for c in t.columns
                if "shannon" in norm(c)
            ]

            if shannon_cols:
                valid = t[t[shannon_cols[0]].notna()].copy()
            else:
                valid = t

            if "pdb_id" in valid:
                print_metric(
                    "Unique PDB",
                    valid["pdb_id"].nunique(),
                    EXPECTED["shannon_protein_pdbs"],
                )

            chains = unique_chain_count(valid)
            if chains is not None:
                print_metric(
                    "Protein PDB-chains",
                    chains,
                    EXPECTED["shannon_protein_chains"],
                )

            poscols = [
                c for c in ("uniprot_id", "uniprot_pos")
                if c in valid.columns
            ]

            if len(poscols) == 2:
                npos = valid[poscols].dropna().drop_duplicates().shape[0]
                print_metric(
                    "Unique protein positions linked to RNA Shannon",
                    npos,
                    EXPECTED["shannon_protein_positions"],
                )
    else:
        print("Shannon -> protein dataset: NOT FOUND")


def variant_coverage(base: Path, backbone_df):
    print_header("VARIANT COVERAGE")

    if backbone_df is not None:
        cols = backbone_df.columns

        variant_cols = [
            c for c in cols
            if any(
                term in norm(c)
                for term in (
                    "variant",
                    "mutated",
                    "mutation",
                    "polyphen",
                    "sift",
                    "pathogenic",
                )
            )
        ]

        if variant_cols:
            mask = pd.Series(False, index=backbone_df.index)

            for c in variant_cols:
                mask |= backbone_df[c].notna()

            n = int(mask.sum())

            print_metric(
                "Backbone rows with variant fields",
                n,
                EXPECTED["variant_rows"],
            )

            print_metric(
                "Fraction",
                n / len(backbone_df),
                EXPECTED["variant_fraction"],
            )
        else:
            print(
                "Variant columns not loaded from backbone; "
                "using dedicated variant datasets below."
            )

    bestrow = first_existing(
        base,
        [
            "variants_bestrow.csv",
            "variants_bestrow_ring.csv",
        ],
    )

    if bestrow:
        report_file("Structural protein variants dataset", bestrow)
        v = load_csv(bestrow)

        if v is not None:
            keys = [
                c for c in (
                    "uniprot_id",
                    "uniprot_pos",
                    "wild_type",
                    "mutated_type",
                )
                if c in v.columns
            ]

            if len(keys) >= 3:
                nvar = v[keys].drop_duplicates().shape[0]
            else:
                nvar = len(v)

            print_metric(
                "Structurally represented protein variants",
                nvar,
                EXPECTED["protein_variants_structural"],
            )

    # Search explicitly for the final modeled subset (~93,700 rows)
    modeled_candidates = []

    for p in base.rglob("*.csv"):
        name = p.name.lower()

        if not any(
            term in name
            for term in ("variant", "model", "bestrow")
        ):
            continue

        try:
            nrows = sum(1 for _ in open(
                p,
                "r",
                encoding="utf-8",
                errors="ignore"
            )) - 1
        except Exception:
            continue

        modeled_candidates.append((abs(nrows - 93_700), nrows, p))

    if modeled_candidates:
        modeled_candidates.sort(key=lambda x: x[0])
        diff, nrows, p = modeled_candidates[0]

        if diff < 5_000:
            report_file("Likely final variant-model subset", p)
            print_metric(
                "Rows",
                nrows,
                EXPECTED["protein_variant_model_subset"],
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate Chapter 10.3 annotation coverage summary."
    )

    parser.add_argument(
        "--base",
        default="/mnt/e/Tesis_Maestria/15_abr_26",
        help="Root directory containing the final thesis analysis files.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output TXT file.",
    )

    args = parser.parse_args()

    base = Path(args.base)

    if not base.exists():
        raise SystemExit(f"Base directory does not exist: {base}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        original_stdout = sys.stdout
        fh = out.open("w", encoding="utf-8")
        sys.stdout = fh
    else:
        original_stdout = None
        fh = None

    try:
        print("CHAPTER 10.3 — ANNOTATION COVERAGE")
        print(f"Base directory: {base}")

        backbone = backbone_coverage(base)
        pdb_uniprot_pfam_coverage(base)
        rfam_coverage(base)
        shannon_coverage(base)
        variant_coverage(base, backbone)

        print_header("INTERPRETATION")
        print(
            "Coverage values refer to the effective universe available for "
            "each annotation. Missing values represent unavailable annotations "
            "unless the corresponding pipeline explicitly evaluated the "
            "position and assigned a negative result."
        )

        print(
            "\nValues marked '[reference: ...]' are previously obtained thesis "
            "results included only for quality control. They are not used to "
            "derive the reported values."
        )

    finally:
        if fh:
            fh.close()
            sys.stdout = original_stdout

    if args.output:
        print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
