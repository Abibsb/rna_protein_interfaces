import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv(
"/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/23_feb_26/MASTER_VIEW_B_PER_CHAIN_with_bad_mutation_flag.csv",
low_memory=False
)

# convertir booleanos
for c in ["has_pathogenic","has_variant","is_interface","has_bridging_water"]:
    df[c] = df[c].astype(int)

# solo posiciones con variantes
df_sub = df[df["has_variant"] == 1]

# eliminar NA
df_sub = df_sub.dropna(subset=[
    "consurf_score",
    "delta_sasa_percentile_within_chain",
    "iupred_score",
    "uniprot_id"
])

# mantener solo proteínas con ambas clases
counts = df_sub.groupby("uniprot_id")["has_pathogenic"].nunique()

valid_proteins = counts[counts == 2].index

df_sub = df_sub[df_sub["uniprot_id"].isin(valid_proteins)]

# modelo controlando por proteína
model = smf.logit(
    "has_pathogenic ~ consurf_score + delta_sasa_percentile_within_chain + iupred_score + C(uniprot_id)",
    data=df_sub
).fit()

print(model.summary())