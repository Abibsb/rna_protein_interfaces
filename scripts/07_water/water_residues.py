import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and clean
# -----------------------------
df = pd.read_csv("/media/usuario/Seagate Basic1/Tesis_Maestria/23_feb_26/MASTER_interface_dataset.csv", low_memory=False)

df = df[df["polymer_class"] == "protein"].copy()
df = df.dropna(subset=["resname_x"]).copy()

# one row per residue
df = df.drop_duplicates(subset=["pdb_id", "uniprot_id", "uniprot_pos"])

# water flag
df["has_water"] = df["has_bridging_water"].fillna(False).astype(bool)

# -----------------------------
# 2. Proportion of water by residue type
# -----------------------------
summary = (
    df.groupby("resname_x")["has_water"]
    .mean()
    .sort_values(ascending=False) * 100
)

# -----------------------------
# 3. Plot
# -----------------------------
plt.figure(figsize=(10, 5))
ax = summary.plot(kind="bar")

plt.ylabel("Residues with bridging water (%)")
plt.xlabel("Residue type")
plt.title("Bridging water frequency by residue type")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("water_by_residue_type_barplot.png", dpi=300)
plt.close()

print("Saved: water_by_residue_type_barplot.png")