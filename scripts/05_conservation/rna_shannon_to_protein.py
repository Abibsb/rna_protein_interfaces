import pandas as pd
import sys

contacts_file = sys.argv[1]
shannon_file = sys.argv[2]
outfile = sys.argv[3]

contacts = pd.read_csv(contacts_file, low_memory=False)
shannon = pd.read_csv(shannon_file, low_memory=False)

contacts["pdb_id"] = contacts["pdb_id"].astype(str).str.upper()
shannon["pdb_id"] = shannon["pdb_id"].astype(str).str.upper()

shannon = shannon.rename(columns={
    "chain_id": "rna_chain",
    "query_pos": "rna_resnum"
})

contacts["rna_chain"] = contacts["rna_chain"].astype(str)
shannon["rna_chain"] = shannon["rna_chain"].astype(str)

contacts["rna_resnum"] = pd.to_numeric(contacts["rna_resnum"], errors="coerce")
shannon["rna_resnum"] = pd.to_numeric(shannon["rna_resnum"], errors="coerce")
contacts["protein_resnum"] = pd.to_numeric(contacts["protein_resnum"], errors="coerce")

shannon["shannon_conservation"] = pd.to_numeric(
    shannon["shannon_conservation"],
    errors="coerce"
)

merged = contacts.merge(
    shannon,
    on=["pdb_id", "rna_chain", "rna_resnum"],
    how="inner"
)

print("Contactos totales:", len(contacts))
print("Filas Shannon:", len(shannon))
print("Contactos mergeados por posición:", len(merged))
print("Contactos con Shannon no-NA:", merged["shannon_conservation"].notna().sum())

merged_valid = merged.dropna(subset=["shannon_conservation"]).copy()

prot = merged_valid.groupby(
    ["pdb_id", "protein_chain", "protein_resnum", "protein_resname"],
    dropna=False
).agg(
    shannon_mean=("shannon_conservation", "mean"),
    shannon_median=("shannon_conservation", "median"),
    shannon_min=("shannon_conservation", "min"),
    shannon_max=("shannon_conservation", "max"),
    shannon_contacts=("shannon_conservation", "count"),
    rna_nt_contacted=("rna_resname", lambda x: ",".join(sorted(set(x.dropna().astype(str))))),
    rfam_hits=("rfam_acc", lambda x: ",".join(sorted(set(x.dropna().astype(str)))))
).reset_index()

prot.to_csv(outfile, index=False)

print("Residuos proteicos con RNA-Shannon:", len(prot))
print("Guardado:", outfile)