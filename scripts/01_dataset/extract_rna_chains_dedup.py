#!/usr/bin/env python3

from Bio.PDB import MMCIFParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import sys
import os
import glob

# =========================
# Configuration
# =========================
CIF_DIR = sys.argv[1]              # directory with mmCIF files
OUT_FASTA = sys.argv[2]            # output fasta

parser = MMCIFParser(QUIET=True)

# =========================
# RNA residue mapping
# =========================
# Canonical + common modified nucleotides → RNA base
RNA_MAP = {
    # canonical
    "A": "A", "C": "C", "G": "G", "U": "U", "I": "G",

    # common modified bases
    "PSU": "U", "H2U": "U",
    "1MA": "A", "2MA": "A", "6MA": "A",
    "M2G": "G", "7MG": "G", "OMG": "G",
    "5MC": "C", "OMC": "C",
    "YG": "G"
}

def is_rna_residue(resname):
    return resname.strip() in RNA_MAP

def residue_to_base(resname):
    return RNA_MAP.get(resname.strip())

# =========================
# Main logic
# =========================
records = []
seen = set()   # (pdb_id, chain_id, sequence)

cif_files = sorted(glob.glob(os.path.join(CIF_DIR, "*.cif")))

for cif_file in cif_files:
    pdb_id = os.path.basename(cif_file).split(".")[0]

    try:
        structure = parser.get_structure(pdb_id, cif_file)
    except Exception as e:
        print(f"[WARN] Failed to parse {pdb_id}: {e}", file=sys.stderr)
        continue

    # -------- FIX 1: only first model --------
    model = next(structure.get_models())

    for chain in model:
        chain_id = chain.id
        seq = []

        for residue in chain:
            resname = residue.get_resname()
            if is_rna_residue(resname):
                base = residue_to_base(resname)
                if base:
                    seq.append(base)

        if not seq:
            continue

        sequence = "".join(seq)

        # -------- FIX 2: deduplicate --------
        key = (pdb_id, chain_id, sequence)
        if key in seen:
            continue
        seen.add(key)

        record_id = f"{pdb_id}_{chain_id}"
        description = "RNA chain (deduplicated, model 0)"

        records.append(
            SeqRecord(
                Seq(sequence),
                id=record_id,
                description=description
            )
        )

# =========================
# Write output
# =========================
SeqIO.write(records, OUT_FASTA, "fasta")

print(f"✔ Written {len(records)} unique RNA sequences to {OUT_FASTA}")