#!/usr/bin/env python3
from pathlib import Path
from Bio import SeqIO
import re

INPUT_DIR = Path("pdb_fastas")
OUTPUT_DIR = Path("pdb_fastas_protein")
OUTPUT_DIR.mkdir(exist_ok=True)

# Regex: protein sequences only (no U, no ambiguous RNA)
protein_re = re.compile("^[ACDEFGHIKLMNPQRSTVWYBXZ]+$", re.IGNORECASE)

for fasta in INPUT_DIR.glob("*.fasta"):
    pdb_id = fasta.stem

    for record in SeqIO.parse(fasta, "fasta"):
        seq = str(record.seq).upper().replace("X", "X")

        # Remove gaps and whitespace
        seq = seq.replace("-", "").replace(" ", "")

        # Skip empty
        if not seq:
            continue

        # Skip RNA/DNA (contains U or only ACGT)
        if "U" in seq:
            continue

        # Ensure it looks like protein
        if not protein_re.match(seq):
            continue

        # Extract chain ID if present
        header = record.description
        chain = "UNK"
        if "Chain" in header:
            parts = header.split("|")
            for p in parts:
                if p.strip().startswith("Chain"):
                    chain = p.strip().split()[-1]

        out_fasta = OUTPUT_DIR / f"{pdb_id}_chain{chain}.fasta"

        with open(out_fasta, "w") as out:
            out.write(f">{pdb_id}|chain {chain}\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i+60] + "\n")

print("✅ Protein-only FASTAs written to:", OUTPUT_DIR)
