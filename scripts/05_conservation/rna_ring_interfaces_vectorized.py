import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
RING_CSV = "/media/usuario/Seagate Basic1/Tesis_Maestria/01_feb_26/maxitabla_dataset_actualizado.csv"
OUT = "protein_rna_residue_contacts.csv"

# --------------------------------------------------
# Residue typing (RELAXED PIPELINE LOGIC)
# --------------------------------------------------

AA_SET = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "SEC","PYL"
}

RNA_SET = {
    "A","U","G","C","I",
    "ADE","URA","GUA","CYT"
}

# --------------------------------------------------
# Load RING table (FAST + SAFE)
# --------------------------------------------------

df = pd.read_csv(
    RING_CSV,
    sep="\t",
    engine="python",
    dtype=str,
    skipinitialspace=True
)

# Drop completely empty columns (phantom first column)
df = df.loc[:, df.columns.notna()]
df = df.dropna(axis=1, how="all")

df.columns = df.columns.str.strip()

required = {"PDBID","NodeId1","NodeId2"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing expected columns: {missing}")

# --------------------------------------------------
# SAFE VECTORISED NodeId parsing (ROBUST VERSION)
# --------------------------------------------------

split1 = df["NodeId1"].str.split(":", expand=True)
split2 = df["NodeId2"].str.split(":", expand=True)

# Ensure at least 4 columns exist by reindexing
split1 = split1.reindex(columns=range(4))
split2 = split2.reindex(columns=range(4))

df["chain1"]  = split1[0]
df["resnum1"] = split1[1]
df["res1"]    = split1[3]

df["chain2"]  = split2[0]
df["resnum2"] = split2[1]
df["res2"]    = split2[3]

# Drop rows where residue names could not be parsed
df = df.dropna(subset=["res1","res2"]).copy()


# --------------------------------------------------
# Vectorised residue classification
# --------------------------------------------------

df["type1"] = "other"
df.loc[df["res1"].isin(AA_SET), "type1"] = "protein"
df.loc[df["res1"].isin(RNA_SET), "type1"] = "RNA"

df["type2"] = "other"
df.loc[df["res2"].isin(AA_SET), "type2"] = "protein"
df.loc[df["res2"].isin(RNA_SET), "type2"] = "RNA"

# --------------------------------------------------
# VECTORISED protein ↔ RNA pairing
# --------------------------------------------------

mask_pr = (df["type1"] == "protein") & (df["type2"] == "RNA")
mask_rp = (df["type1"] == "RNA") & (df["type2"] == "protein")

# protein on side 1
contacts_pr = df.loc[mask_pr, [
    "PDBID","chain1","resnum1","res1",
    "chain2","resnum2","res2"
]].copy()

contacts_pr.columns = [
    "pdb_id",
    "protein_chain","protein_resnum","protein_resname",
    "rna_chain","rna_resnum","rna_resname"
]

# protein on side 2 (swap columns)
contacts_rp = df.loc[mask_rp, [
    "PDBID","chain2","resnum2","res2",
    "chain1","resnum1","res1"
]].copy()

contacts_rp.columns = contacts_pr.columns

# Combine
contacts_df = (
    pd.concat([contacts_pr, contacts_rp], ignore_index=True)
    .drop_duplicates()
    .sort_values(["pdb_id","protein_chain","protein_resnum","rna_chain","rna_resnum"])
)

# --------------------------------------------------
# Save
# --------------------------------------------------

contacts_df.to_csv(OUT, index=False)

print("✔ Vectorized extraction complete.")
print(f"Rows written: {len(contacts_df)}")