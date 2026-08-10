import pandas as pd
import sys

master_file = sys.argv[1]
shannon_protein_file = sys.argv[2]
outfile = sys.argv[3]

master = pd.read_csv(master_file, low_memory=False)
sh = pd.read_csv(shannon_protein_file, low_memory=False)

master["pdb_id"] = master["pdb_id"].astype(str).str.upper()
sh["pdb_id"] = sh["pdb_id"].astype(str).str.upper()

# adaptar nombres para mergear con master per-chain
sh = sh.rename(columns={
    "protein_chain": "auth_asym_id",
    "protein_resnum": "auth_seq_id"
})

master["auth_asym_id"] = master["auth_asym_id"].astype(str)
sh["auth_asym_id"] = sh["auth_asym_id"].astype(str)

master["auth_seq_id"] = pd.to_numeric(master["auth_seq_id"], errors="coerce")
sh["auth_seq_id"] = pd.to_numeric(sh["auth_seq_id"], errors="coerce")

merged = master.merge(
    sh,
    on=["pdb_id", "auth_asym_id", "auth_seq_id"],
    how="left"
)

print("Master:", len(master))
print("Con RNA Shannon:", merged["shannon_mean"].notna().sum())

merged.to_csv(outfile, index=False)
print("Guardado:", outfile)