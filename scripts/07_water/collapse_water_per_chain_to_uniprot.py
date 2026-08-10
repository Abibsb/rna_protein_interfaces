import pandas as pd

IN = "/media/usuario/Seagate Basic1/Tesis_Maestria/07_feb_26/water_mediated_residues_uniprot.csv"
OUT = "water_mediated_residues_chain_uniprot_COLLAPSED.csv"

df = pd.read_csv(IN, dtype=str)

df["n_bridging_waters"] = df["n_bridging_waters"].astype(int)
df["n_bridging_waters_angle_filtered"] = df["n_bridging_waters_angle_filtered"].astype(int)

# --------------------------------------------------
# Collapse but keep chain metadata
# --------------------------------------------------
grp = (
    df.groupby(["pdb_id","uniprot_id","uniprot_pos"], as_index=False)
      .agg({
          "resname":"first",
          "n_bridging_waters":"max",
          "n_bridging_waters_angle_filtered":"max",
          "auth_asym_id":lambda x: ";".join(sorted(set(x)))
      })
)

grp["n_chains_with_water"] = grp["auth_asym_id"].str.count(";") + 1

grp["has_bridging_water"] = grp["n_bridging_waters"] > 0
grp["has_angle_filtered_water"] = grp["n_bridging_waters_angle_filtered"] > 0
grp["is_recurrent_water_site"] = grp["n_bridging_waters"] >= 2

grp.rename(columns={"auth_asym_id":"chains_with_water"}, inplace=True)

grp.to_csv(OUT,index=False)

print("✔ Collapsed with chain metadata")