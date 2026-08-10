#!/usr/bin/env python3
from Bio import SeqIO
from pathlib import Path
import sys
import re

# ---------------------------
# Check input arguments
# ---------------------------
if len(sys.argv) < 2:
    print("Usage: python3 filter_protein_msa_folder.py <folder_with_fastas>")
    sys.exit(1)

folder = Path(sys.argv[1])
valid_aa = set("ACDEFGHIKLMNPQRSTVWY-")

# ---------------------------
# Create output folder
# ---------------------------
out_dir = folder / "filtered_msas"
out_dir.mkdir(exist_ok=True)

# ---------------------------
# Process each MSA file
# ---------------------------
for msa_file in folder.glob("*_padded_wrapped.aln"):
    # Attempt to extract PF and PDB info to create a short filename
    match = re.match(r"(PF\d+)_([0-9A-Z]+)_", msa_file.name)
    if match:
        pfam_id, pdbid = match.groups()
        out_name = f"{pfam_id}_{pdbid}_prot_only.fasta"
    else:
        # Fallback: take first 50 characters of stem
        out_name = msa_file.stem[:50] + "_prot_only.fasta"

    out_file = out_dir / out_name
    count_removed = 0

    # Parse and filter sequences
    with msa_file.open() as handle_in, out_file.open("w") as handle_out:
        for record in SeqIO.parse(handle_in, "fasta"):
            seq_letters = set(str(record.seq).upper())
            if seq_letters.issubset(valid_aa):
                SeqIO.write(record, handle_out, "fasta")
            else:
                count_removed += 1

    print(f"✅ {msa_file.name}: removed {count_removed} non-protein sequences → {out_file.name}")

print(f"\n✅ All filtered MSAs are saved in: {out_dir}")

