#!/usr/bin/env python3

import json
import glob
import os
import csv

JSON_DIR = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/29_jul_25/output_uniprot/"   # carpeta con todos los *.json
OUT_CSV = "pdb_uniprot_position_map.csv"

rows = []

for json_file in glob.glob(os.path.join(JSON_DIR, "*.json")):
    pdb_id = os.path.basename(json_file).replace(".json", "").upper()

    with open(json_file) as f:
        data = json.load(f)

    root_key = pdb_id.lower()
    if root_key not in data:
        print(f"⚠️ {pdb_id}: root key not found")
        continue

    uniprot_block = data[root_key].get("UniProt", {})

    for uniprot_id, info in uniprot_block.items():
        for m in info.get("mappings", []):

            auth_chain = m.get("chain_id")
            start = m["start"]["residue_number"]
            end = m["end"]["residue_number"]
            unp_start = m["unp_start"]
            unp_end = m["unp_end"]

            length = min(end - start, unp_end - unp_start)

            for i in range(length + 1):
                rows.append({
                    "pdb_id": pdb_id,
                    "auth_asym_id": auth_chain,
                    "auth_seq_id": start + i,
                    "uniprot_id": uniprot_id,
                    "uniprot_pos": unp_start + i
                })

print("Total mappings:", len(rows))

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "pdb_id",
            "auth_asym_id",
            "auth_seq_id",
            "uniprot_id",
            "uniprot_pos"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("✔ Saved", OUT_CSV)
