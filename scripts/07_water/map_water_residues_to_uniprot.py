#!/usr/bin/env python3

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Map water-mediated protein residues to UniProt positions"
)
parser.add_argument(
    "--water_json_dir", required=True,
    help="Directory with per-PDB water JSON files"
)
parser.add_argument(
    "--pdb_uniprot_map", required=True,
    help="CSV: pdb_id,auth_asym_id,auth_seq_id,uniprot_id,uniprot_pos"
)
parser.add_argument(
    "--out_csv", required=True,
    help="Output CSV file"
)
args = parser.parse_args()

water_json_dir = Path(args.water_json_dir)
pdb_uniprot_map = Path(args.pdb_uniprot_map)

# ------------------------------------------------------------
# Load PDB → UniProt residue map
# ------------------------------------------------------------
map_rows = {}
with open(pdb_uniprot_map) as f:
    reader = csv.DictReader(f)
    for r in reader:
        key = (
            r["pdb_id"].upper(),
            r["auth_asym_id"],
            int(r["auth_seq_id"])
        )
        map_rows[key] = {
            "uniprot_id": r["uniprot_id"],
            "uniprot_pos": int(r["uniprot_pos"])
        }

# ------------------------------------------------------------
# Accumulator:
# (pdb, chain, resnum) → counts
# ------------------------------------------------------------
residue_data = defaultdict(lambda: {
    "resname": None,
    "n_bridging_waters": 0,
    "n_bridging_waters_angle_filtered": 0
})

# ------------------------------------------------------------
# Parse water JSONs
# ------------------------------------------------------------
for json_file in water_json_dir.glob("*.json"):
    pdb_id = json_file.stem.upper()

    with open(json_file) as f:
        data = json.load(f)

    # ---------- distance-only waters ----------
    for water_id, partners in data.get("bridging_water_partners", {}).items():
        for resname, auth_resnum in partners.get("protein_residues", []):
            # NOTE: chain is not encoded in JSON residue tuple
            # If your JSONs include chain, replace accordingly
            # Here we assume 1 protein chain per JSON
            # or that auth_asym_id is implicit in JSON naming
            key = (pdb_id, None, int(auth_resnum))
            residue_data[key]["resname"] = resname
            residue_data[key]["n_bridging_waters"] += 1

    # ---------- angle-filtered waters ----------
    for water_id, partners in data.get("bridging_water_partners_angle_filtered", {}).items():
        for resname, auth_resnum in partners.get("protein_residues", []):
            key = (pdb_id, None, int(auth_resnum))
            residue_data[key]["resname"] = resname
            residue_data[key]["n_bridging_waters_angle_filtered"] += 1

# ------------------------------------------------------------
# Write output with UniProt mapping
# ------------------------------------------------------------
with open(args.out_csv, "w", newline="") as out:
    fieldnames = [
        "pdb_id",
        "auth_asym_id",
        "auth_seq_id",
        "resname",
        "uniprot_id",
        "uniprot_pos",
        "n_bridging_waters",
        "n_bridging_waters_angle_filtered"
    ]
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()

    for (pdb_id, chain, auth_resnum), info in sorted(residue_data.items()):
        # try all chains for this residue (safe, CIF-derived map)
        matched = False
        for (mpdb, mchain, mresnum), umap in map_rows.items():
            if mpdb == pdb_id and mresnum == auth_resnum:
                writer.writerow({
                    "pdb_id": pdb_id,
                    "auth_asym_id": mchain,
                    "auth_seq_id": auth_resnum,
                    "resname": info["resname"],
                    "uniprot_id": umap["uniprot_id"],
                    "uniprot_pos": umap["uniprot_pos"],
                    "n_bridging_waters": info["n_bridging_waters"],
                    "n_bridging_waters_angle_filtered": info["n_bridging_waters_angle_filtered"]
                })
                matched = True

        # optional: keep unmapped residues for QC
        if not matched:
            pass

print("Done.")