#!/usr/bin/env python3
import sys
from pathlib import Path
import gemmi

AA_RESNAMES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
    "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
    "THR","TRP","TYR","VAL"
}

def is_protein_chain(chain):
    aa_count = 0
    for res in chain:
        if res.name in AA_RESNAMES:
            aa_count += 1
    return aa_count > 0

def resolve_chain(cif_path, chain_x, chain_y):
    cif_path = Path(cif_path)
    structure = gemmi.read_structure(str(cif_path))

    protein_chains = []
    for model in structure:
        for chain in model:
            if len(chain) == 0:
                continue
            if is_protein_chain(chain):
                protein_chains.append(chain)

    if not protein_chains:
        return None

    # 1) Exact match with Chain_ID_x
    for c in protein_chains:
        if c.name == chain_x:
            return c.name

    # 2) If only one protein chain, take it
    if len(protein_chains) == 1:
        return protein_chains[0].name

    # 3) Otherwise, take the longest protein chain
    protein_chains.sort(key=lambda c: len(c), reverse=True)
    return protein_chains[0].name

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)

    cif, x, y = sys.argv[1:]
    resolved = resolve_chain(cif, x, y)

    if resolved:
        print(resolved)
        sys.exit(0)

    sys.exit(1)
