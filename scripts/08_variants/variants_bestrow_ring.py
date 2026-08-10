import pandas as pd

df = pd.read_csv(
    "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/23_feb_26/MASTER_interface_dataset_PER_CHAIN_WITH_VARIANTS.csv",
    low_memory=False
)

# ===== FILTRAR SOLO VARIANTES REALES =====
df = df[
    df["Wild_Type"].notna() &
    df["Mutated_Type"].notna()
].copy()

# Crear variant_id
df["variant_id"] = (
    df["uniprot_id"].astype(str) + "_" +
    df["uniprot_pos"].astype(str) + "_" +
    df["Wild_Type"].astype(str) + ">" +
    df["Mutated_Type"].astype(str)
)

df["has_ring_interface"] = df["has_ring_interface"].astype(int)
df["buried_fraction_residue"] = pd.to_numeric(df["buried_fraction_residue"], errors="coerce")

# Orden estructural prioritario
df = df.sort_values(
    by=["variant_id", "has_ring_interface", "buried_fraction_residue"],
    ascending=[True, False, False]
)

best = df.groupby("variant_id", as_index=False).first()

best.to_csv("variants_bestrow_ring.csv", index=False)

print("Wrote:", len(best), "rows")