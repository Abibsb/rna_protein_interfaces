import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.Align import PairwiseAligner
import os

# Load mapping and UniProt sequences
mapping = pd.read_csv("pdb_chain_candidates.csv")
uniprot_seqs = pd.read_csv("uniprot_sequences.csv").set_index("UniProt_ID")["Sequence"].to_dict()

parser = MMCIFParser(QUIET=True)
ppb = PPBuilder()

aligner = PairwiseAligner()
aligner.mode = "global"

pdb_dir = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/27_ago_25/dataset_complejos_sapiens"

results = []

for (pdb, uni), subdf in mapping.groupby(["PDBID", "UniProt_ID"]):
    uniprot_seq = uniprot_seqs.get(uni)
    if not uniprot_seq:
        continue

    cif_path = os.path.join(pdb_dir, f"{pdb}.cif")
    if not os.path.exists(cif_path):
        # Try lowercase fallback if needed
        cif_path = os.path.join(pdb_dir, f"{pdb.lower()}.cif")
        if not os.path.exists(cif_path):
            continue

    try:
        structure = parser.get_structure(pdb, cif_path)
    except Exception:
        continue

    best_chain = None
    best_score = -1

    for chain_id in subdf["Chain"].unique():
        try:
            chain = structure[0][chain_id]
        except KeyError:
            continue

        peptides = ppb.build_peptides(chain)
        if not peptides:
            continue
        
        chain_seq = str(peptides[0].get_sequence())

        # Alignment score
        score = aligner.score(uniprot_seq, chain_seq)

        if score > best_score:
            best_score = score
            best_chain = chain_id

    if best_chain:
        results.append([pdb, best_chain, uni])

df_out = pd.DataFrame(results, columns=["PDBID", "Chain", "UniProt_ID"])
df_out.to_csv("pdb_chain_onechain_correct.csv", index=False)

print("✅ Saved: pdb_chain_onechain_correct.csv")
print(f"Remaining PDB+chain entries: {len(df_out)}")
print(f"Unique UniProts: {df_out['UniProt_ID'].nunique()}")
print(f"Unique PDBs: {df_out['PDBID'].nunique()}")

