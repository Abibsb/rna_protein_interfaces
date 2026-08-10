#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"No encontré ninguna de estas columnas: {candidates}\n"
            f"Columnas disponibles:\n{list(df.columns)}"
        )
    return None


def boolify(s):
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "yes", "y", "t"])
    )


def main():
    parser = argparse.ArgumentParser(
        description="Merge clean IUPred output with residue-level interface/SASA table."
    )

    parser.add_argument("--iupred", required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--out", default="interface_with_iupred.csv")
    parser.add_argument("--interface-chain-col", default=None)
    parser.add_argument("--interface-pos-col", default=None)
    parser.add_argument("--protein-only", action="store_true")

    args = parser.parse_args()

    print("Reading IUPred...")
    iupred = pd.read_csv(args.iupred, low_memory=False)

    print("Reading interface table...")
    interface = pd.read_csv(args.interface, low_memory=False)

    pdb_col = find_col(interface, ["pdb_id", "PDBID", "pdb", "PDB_ID"])

    chain_col = args.interface_chain_col or find_col(
        interface,
        ["auth_asym_id", "auth_chain", "chain_id", "chain", "label_asym_id"]
    )

    pos_col = args.interface_pos_col or find_col(
        interface,
        ["auth_seq_id", "resnum_used", "label_seq_id", "residue_number", "resi"]
    )

    interface_col = find_col(
        interface,
        ["is_interface_abs", "is_interface", "interface"]
    )

    delta_col = find_col(
        interface,
        ["delta_sasa", "Delta_SASA", "bsa", "buried_area"],
        required=False
    )

    strong_col = find_col(
        interface,
        ["is_strong_interface", "strong_interface"],
        required=False
    )

    hotspot_col = find_col(
        interface,
        ["is_core_hotspot", "core_hotspot", "is_hotspot"],
        required=False
    )

    ring_col = find_col(
        interface,
        ["ring_interaction_count", "n_ring_contacts", "ring_contacts"],
        required=False
    )

    polymer_col = find_col(
        interface,
        ["polymer_class", "polymer_type"],
        required=False
    )

    print("\nInterface columns used:")
    print(f"  pdb:        {pdb_col}")
    print(f"  chain:      {chain_col}")
    print(f"  position:   {pos_col}")
    print(f"  interface:  {interface_col}")
    print(f"  delta_sasa: {delta_col}")
    print(f"  strong:     {strong_col}")
    print(f"  hotspot:    {hotspot_col}")
    print(f"  ring:       {ring_col}")
    print(f"  polymer:    {polymer_col}")

    if args.protein_only and polymer_col:
        before = len(interface)
        interface = interface[
            interface[polymer_col]
            .astype(str)
            .str.lower()
            .str.contains("protein|polypeptide", regex=True)
        ].copy()
        print(f"\nProtein filter: {before} -> {len(interface)} rows")

    # IUPred keys
    iupred["merge_pdb"] = iupred["pdb_id"].astype(str).str.upper()
    iupred["merge_chain"] = (
        iupred["chain_id"]
        .astype(str)
        .str.replace("]", "", regex=False)
        .str.replace("[", "", regex=False)
        .str.strip()
    )
    iupred["merge_pos"] = pd.to_numeric(iupred["iupred_pos"], errors="coerce")

    # Interface keys
    interface["merge_pdb"] = interface[pdb_col].astype(str).str.upper()
    interface["merge_chain"] = (
        interface[chain_col]
        .astype(str)
        .str.replace("]", "", regex=False)
        .str.replace("[", "", regex=False)
        .str.strip()
    )
    interface["merge_pos"] = pd.to_numeric(interface[pos_col], errors="coerce")

    iupred = iupred.dropna(subset=["merge_pos"]).copy()
    interface = interface.dropna(subset=["merge_pos"]).copy()

    iupred["merge_pos"] = iupred["merge_pos"].astype(int)
    interface["merge_pos"] = interface["merge_pos"].astype(int)

    # Evitar duplicados en IUPred
    iupred_unique = (
        iupred.sort_values(["merge_pdb", "merge_chain", "merge_pos"])
        .drop_duplicates(subset=["merge_pdb", "merge_chain", "merge_pos"])
        .copy()
    )

    merged = interface.merge(
        iupred_unique[
            [
                "merge_pdb",
                "merge_chain",
                "merge_pos",
                "residue",
                "iupred_score",
                "is_disordered",
                "source_file",
            ]
        ],
        on=["merge_pdb", "merge_chain", "merge_pos"],
        how="left"
    )

    merged[interface_col] = boolify(merged[interface_col])

    if strong_col:
        merged[strong_col] = boolify(merged[strong_col])

    if hotspot_col:
        merged[hotspot_col] = boolify(merged[hotspot_col])

    if ring_col:
        merged[ring_col] = pd.to_numeric(merged[ring_col], errors="coerce").fillna(0)

    if delta_col:
        merged[delta_col] = pd.to_numeric(merged[delta_col], errors="coerce")

    merged["iupred_score"] = pd.to_numeric(merged["iupred_score"], errors="coerce")

    merged.to_csv(args.out, index=False)

    print("\nMerge summary:")
    print(f"Interface rows: {len(interface)}")
    print(f"IUPred unique residues: {len(iupred_unique)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Rows with IUPred: {merged['iupred_score'].notna().sum()}")
    print(f"IUPred coverage: {merged['iupred_score'].notna().mean():.4f}")
    print(f"Output: {args.out}")

    cov = (
        merged.assign(has_iupred=merged["iupred_score"].notna())
        .groupby(interface_col)["has_iupred"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "n_rows", "sum": "n_with_iupred", "mean": "coverage"})
    )

    cov_path = Path(args.out).with_suffix(".coverage_by_interface.csv")
    cov.to_csv(cov_path, index=False)
    print(f"Coverage by interface: {cov_path}")


if __name__ == "__main__":
    main()