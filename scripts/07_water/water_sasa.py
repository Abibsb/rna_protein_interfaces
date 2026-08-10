import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and clean
# -----------------------------
df = pd.read_csv("/media/usuario/Seagate Basic1/Tesis_Maestria/23_feb_26/MASTER_interface_dataset.csv", low_memory=False)

df = df[df["polymer_class"] == "protein"].copy()
df = df.dropna(subset=["interface_class"]).copy()

# one row per residue
df = df.drop_duplicates(subset=["pdb_id", "uniprot_id", "uniprot_pos"])

# water flag
df["has_water"] = df["has_bridging_water"].fillna(False).astype(bool)

class_order = ["non_interface", "interface", "strong_interface", "core_hotspot"]
df = df[df["interface_class"].isin(class_order)].copy()

# -----------------------------
# 2. Proportion by class
# -----------------------------
summary = (
    df.groupby("interface_class")["has_water"]
    .mean()
    .reindex(class_order) * 100
)

summary.index = ["Non-interface", "Interface", "Strong interface", "Core hotspot"]

# -----------------------------
# 3. Plot
# -----------------------------
plt.figure(figsize=(8, 5))
ax = summary.plot(kind="bar")

plt.ylabel("Residues with bridging water (%)")
plt.xlabel("Interface class")
plt.title("Bridging water frequency across interface classes")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("water_by_interface_class_barplot.png", dpi=300)
plt.close()

print("Saved: water_by_interface_class_barplot.png")

df = df[df["polymer_class"] == "protein"].copy()
df = df.dropna(subset=["interface_class"]).copy()

# one row per residue
df = df.drop_duplicates(subset=["pdb_id", "uniprot_id", "uniprot_pos"])

# water flag
df["has_water"] = df["has_bridging_water"].fillna(False).astype(bool)

class_order = ["non_interface", "interface", "strong_interface", "core_hotspot"]
df = df[df["interface_class"].isin(class_order)].copy()

# -----------------------------
# 2. Proportion by class
# -----------------------------
summary = (
    df.groupby("interface_class")["has_water"]
    .mean()
    .reindex(class_order) * 100
)

summary.index = ["Non-interface", "Interface", "Strong interface", "Core hotspot"]

# -----------------------------
# 3. Plot
# -----------------------------
plt.figure(figsize=(8, 5))
ax = summary.plot(kind="bar")

plt.ylabel("Residues with bridging water (%)")
plt.xlabel("Interface class")
plt.title("Bridging water frequency across interface classes")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("water_by_interface_class_barplot.png", dpi=300)
plt.close()

print("Saved: water_by_interface_class_barplot.png")