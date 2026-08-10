import pandas as pd
import sys

master_file = sys.argv[1]
shannon_file = sys.argv[2]
outfile = sys.argv[3]

master = pd.read_csv(master_file, low_memory=False)
shannon = pd.read_csv(shannon_file, low_memory=False)

# =========================
# NORMALIZAR
# =========================

master["pdb_id"] = master["pdb_id"].astype(str).str.upper()
shannon["pdb_id"] = shannon["pdb_id"].astype(str).str.upper()

# columnas de shannon
shannon = shannon.rename(columns={
    "chain_id": "auth_asym_id",
    "query_pos": "auth_seq_id"
})

# tipos
master["auth_asym_id"] = master["auth_asym_id"].astype(str)
shannon["auth_asym_id"] = shannon["auth_asym_id"].astype(str)

master["auth_seq_id"] = pd.to_numeric(master["auth_seq_id"], errors="coerce")
shannon["auth_seq_id"] = pd.to_numeric(shannon["auth_seq_id"], errors="coerce")

# =========================
# FILTRAR RNA
# =========================

rna_master = master[
    master["polymer_class"].astype(str).str.contains("rna", case=False, na=False)
].copy()

# =========================
# MERGE
# =========================

merged = rna_master.merge(
    shannon,
    on=["pdb_id", "auth_asym_id", "auth_seq_id"],
    how="inner"
)

print("RNA master:", len(rna_master))
print("Shannon:", len(shannon))
print("Merge:", len(merged))

merged.to_csv(outfile, index=False)
print("Guardado en:", outfile)