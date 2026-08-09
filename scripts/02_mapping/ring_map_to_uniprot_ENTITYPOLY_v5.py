#!/usr/bin/env python3

import argparse
import pandas as pd
import gemmi
import sys
from pathlib import Path

# ------------------------------------------------------------
# Import RING parser
# ------------------------------------------------------------

SCRIPT_DIR = Path("/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/01_feb_26")
sys.path.insert(0, str(SCRIPT_DIR))

from map_iupred_to_ring_interface_ENTITYPOLY_separate import parse_ring_table


# ------------------------------------------------------------
# CIF helpers
# ------------------------------------------------------------

def get_protein_auth_chains(cif_path):
    doc = gemmi.cif.read_file(cif_path)
    block = doc.sole_block()

    ent_poly = block.find_mmcif_category("_entity_poly")
    if ent_poly is None:
        raise RuntimeError("No _entity_poly category found")

    protein_entity_ids = {
        row["entity_id"]
        for row in ent_poly
        if "polypeptide" in row["type"].lower()
    }

    if not protein_entity_ids:
        raise RuntimeError("No protein entities found")

    struct_asym = block.find_mmcif_category("_struct_asym")
    if struct_asym is None:
        raise RuntimeError("No _struct_asym category found")

    protein_chains = {
        row["id"]
        for row in struct_asym
        if row["entity_id"] in protein_entity_ids
    }

    if not protein_chains:
        raise RuntimeError("Protein entities found but no chains resolved")

    return sorted(protein_chains)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(cif_path, ring_csv, up_map, out_prefix):

    pdb_id = Path(cif_path).stem.upper()
    print(f"🔹 PDB ID: {pdb_id}")

    protein_chains = get_protein_auth_chains(cif_path)
    print(f"🔹 Protein auth chains from CIF: {protein_chains}")

    iface = parse_ring_table(
        ring_csv_path=ring_csv,
        pdbid_filter=pdb_id
    )

    rows = []

    if pdb_id in iface:
        for chain, residue_container in iface[pdb_id].items():

            if chain not in protein_chains:
                continue

            # Old parser: set(resnum)
            if isinstance(residue_container, set):
                for resnum in residue_container:
                    rows.append({
                        "pdb_id": pdb_id,
                        "auth_asym_id": chain,
                        "auth_seq_id": resnum,
                        "ring_interaction_count": 1,
                        "ring_interaction_rows": 1
                    })

            # Future parser: dict(resnum -> n_rows)
            elif isinstance(residue_container, dict):
                for resnum, n_rows in residue_container.items():
                    rows.append({
                        "pdb_id": pdb_id,
                        "auth_asym_id": chain,
                        "auth_seq_id": resnum,
                        "ring_interaction_count": 1,
                        "ring_interaction_rows": int(n_rows)
                    })

            else:
                raise TypeError(
                    f"Unsupported residue container type: {type(residue_container)}"
                )

    iface_df = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"🔹 Interface residues (protein chains only): {len(iface_df)}")

    out_csv = f"{out_prefix}_interface_residues_with_uniprot.csv"

    # --------------------------------------------------------
    # EARLY EXIT: no interface residues
    # --------------------------------------------------------

    if iface_df.empty:
        empty_cols = [
            "pdb_id",
            "auth_asym_id",
            "auth_seq_id",
            "ring_interaction_count",
            "ring_interaction_rows",
            "uniprot_id",
            "uniprot_pos",
        ]

        pd.DataFrame(columns=empty_cols).to_csv(out_csv, index=False)

        print("⚠ No protein–RNA interface residues found")
        print(f"✔ Output written to: {out_csv}")
        return

    # --------------------------------------------------------
    # Merge with UniProt mapping
    # --------------------------------------------------------

    merged = iface_df.merge(
        up_map,
        how="left",
        on=["pdb_id", "auth_asym_id", "auth_seq_id"]
    )

    merged.to_csv(out_csv, index=False)

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    total = len(merged)
    mapped = merged["uniprot_pos"].notna().sum()

    print("\n=== SANITY CHECKS ===")
    print(f"Mapped residues: {mapped}/{total} ({mapped/total:.1%})")

    multi = (
        merged
        .groupby(["pdb_id", "auth_asym_id", "auth_seq_id"])
        ["uniprot_id"]
        .nunique()
    )

    print(f"Residues mapping to >1 UniProt: {(multi > 1).sum()}")
    print(f"✔ Output written to: {out_csv}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--ring", required=True)
    ap.add_argument("--uniprot_map", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    # Load UniProt map ONCE
    up_map = pd.read_csv(args.uniprot_map, engine="python")
    up_map.columns = up_map.columns.str.strip()
    up_map["pdb_id"] = up_map["pdb_id"].str.upper()

    main(args.cif, args.ring, up_map, args.out_prefix)
