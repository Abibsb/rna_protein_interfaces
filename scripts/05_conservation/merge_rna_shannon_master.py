import pandas as pd
import sys

master_file = sys.argv[1]
shannon_file = sys.argv[2]
outfile = sys.argv[3]

master = pd.read_csv(master_file, low_memory=False)
shannon = pd.read_csv(shannon_file, low_memory=False)

# =========================
# LIMPIEZA
# =========================

shannon["pdb_id"] = shannon["pdb_id"].astype(str).str.upper()
master["pdb_id"] = master["pdb_id"].astype(str).str.upper()

# estandarizar nombres
if "chain_id" in shannon.columns:
    shannon = shannon.rename(columns={"chain_id": "auth_asym_id"})

if "query_pos" in shannon.columns:
    shannon = shannon.rename(columns={"query_pos": "auth_seq_id"})

# =========================
# FILTRAR SOLO RNA EN MASTER
# =========================

rna_master = master[
    master["polymer_class"].astype(str).str.contains("rna", case=False, na=False)
].copy()

# =========================
# MERGE
# =========================

merged = pd.merge(
    rna_master,
    shannon,
    on=["pdb_id", "auth_asym_id", "auth_seq_id"],
    how="inner"
)

print("Filas RNA en master:", len(rna_master))
print("Filas con Shannon:", len(merged))

merged.to_csv(outfile, index=False)
print("Guardado en:", outfile)