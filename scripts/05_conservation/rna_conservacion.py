import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load datasets
# -----------------------------
shannon = pd.read_csv("/media/usuario/Seagate Basic1/Tesis_Maestria/11_feb_26/rfam_seed_shannon_out/shannon/ALL_queries.mapped_merged.csv", low_memory=False)
contacts = pd.read_csv("/media/usuario/Seagate Basic1/Tesis_Maestria/23_feb_26/protein_rna_residue_contacts_with_pdb.csv", low_memory=False)

# -----------------------------
# 2. Normalize keys
# -----------------------------
shannon["pdb_id"] = shannon["pdb_id"].astype(str).str.upper().str.strip()
shannon["chain_id"] = shannon["chain_id"].astype(str).str.strip()
shannon["query_pos"] = shannon["query_pos"].astype(str).str.strip()

contacts["pdb_id"] = contacts["pdb_id"].astype(str).str.upper().str.strip()
contacts["rna_chain"] = contacts["rna_chain"].astype(str).str.strip()
contacts["rna_resnum"] = contacts["rna_resnum"].astype(str).str.strip()

# -----------------------------
# 3. Aggregate RNA interface metrics
#    at the nucleotide level
# -----------------------------
rna_interface = (
    contacts
    .groupby(["pdb_id", "rna_chain", "rna_resnum"], as_index=False)
    .agg(
        contact_count=("protein_resnum", "count"),
        unique_protein_residues=("protein_resnum", "nunique"),
        unique_protein_chains=("protein_chain", "nunique")
    )
)

rna_interface["is_rna_interface"] = rna_interface["contact_count"] > 0

# -----------------------------
# 4. Merge Shannon + RNA interface
# -----------------------------
df = shannon.merge(
    rna_interface,
    left_on=["pdb_id", "chain_id", "query_pos"],
    right_on=["pdb_id", "rna_chain", "rna_resnum"],
    how="left"
)

# -----------------------------
# 5. Fill missing values
# -----------------------------
df["contact_count"] = df["contact_count"].fillna(0).astype(int)
df["unique_protein_residues"] = df["unique_protein_residues"].fillna(0).astype(int)
df["unique_protein_chains"] = df["unique_protein_chains"].fillna(0).astype(int)
df["is_rna_interface"] = df["is_rna_interface"].fillna(False)

# -----------------------------
# 6. Optional filtering
#    Keep only usable conservation positions
# -----------------------------
df = df.dropna(subset=["shannon_conservation"]).copy()

# Optional: remove very gappy positions
# Uncomment if you want stricter filtering
# df = df[df["gap_fraction"] < 0.5].copy()

# -----------------------------
# 7. Save merged dataset
# -----------------------------
out_csv = "RNA_shannon_with_interface.csv"
df.to_csv(out_csv, index=False)

print("Saved merged dataset:", out_csv)
print("Total RNA residues:", len(df))
print("RNA interface residues:", df["is_rna_interface"].sum())
print("RNA non-interface residues:", (~df["is_rna_interface"]).sum())

# -----------------------------
# 8. Graph 1: interface vs non-interface
# -----------------------------
non_interface = df.loc[~df["is_rna_interface"], "shannon_conservation"]
interface = df.loc[df["is_rna_interface"], "shannon_conservation"]

plt.figure(figsize=(6, 5))
plt.boxplot(
    [non_interface, interface],
    labels=["Non-interface RNA", "Interface RNA"],
    showfliers=False
)

plt.ylabel("Shannon conservation")
plt.title("RNA conservation in protein–RNA interfaces")
plt.tight_layout()
plt.savefig("rna_conservation_interface_vs_non.png", dpi=300)
plt.close()

# -----------------------------
# 9. Graph 2: conservation vs contact count
# -----------------------------
df["contact_bin"] = pd.cut(
    df["contact_count"],
    bins=[-1, 0, 2, float("inf")],
    labels=["0 contacts", "1–2 contacts", "≥3 contacts"]
)

groups = [
    df.loc[df["contact_bin"] == "0 contacts", "shannon_conservation"],
    df.loc[df["contact_bin"] == "1–2 contacts", "shannon_conservation"],
    df.loc[df["contact_bin"] == "≥3 contacts", "shannon_conservation"]
]

plt.figure(figsize=(7, 5))
plt.boxplot(
    groups,
    labels=["0 contacts", "1–2 contacts", "≥3 contacts"],
    showfliers=False
)

plt.ylabel("Shannon conservation")
plt.title("RNA conservation vs protein contact density")
plt.tight_layout()
plt.savefig("rna_conservation_by_contacts.png", dpi=300)
plt.close()

print("Saved figures:")
print("- rna_conservation_interface_vs_non.png")
print("- rna_conservation_by_contacts.png")