import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/02_mar_26/3_rna/variants_bestrow_ring_rna.csv", low_memory=False)

# target
da = df["Disease_Association"]
df["yA"] = (
    da.notna() &
    (da.astype(str).str.strip() != "") &
    (da.astype(str).str.lower().str.strip() != "nan")
).astype(int)

# variables
df["rna_contact"] = (pd.to_numeric(df["rna_contact_count"], errors="coerce").fillna(0) > 0).astype(int)
df["basic_residue"] = df["resname"].isin(["ARG","LYS","HIS"]).astype(int)

for c in ["consurf_score","delta_sasa_norm","iupred_score","is_core_hotspot"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["is_core_hotspot"] = df["is_core_hotspot"].fillna(0).astype(int)

# filtro mínimo
m = df[
    df["consurf_score"].notna() &
    df["iupred_score"].notna()
].copy()

m = m.dropna(subset=[
    "yA",
    "consurf_score",
    "delta_sasa_norm",
    "iupred_score",
    "uniprot_id"
])

print("Rows:", len(m))
print("Unique proteins:", m["uniprot_id"].nunique())
print("RNA contact fraction:", m["rna_contact"].mean())

features = [
    "consurf_score",
    "delta_sasa_norm",
    "iupred_score",
    "basic_residue",
    "rna_contact",
    "is_core_hotspot"
]

X = m[features]
X = sm.add_constant(X)

y = m["yA"]

model = sm.Logit(y, X).fit(
    disp=False,
    cov_type="cluster",
    cov_kwds={"groups": m["uniprot_id"]}
)

print(model.summary())

params = model.params
conf = model.conf_int()

or_table = pd.DataFrame({
    "OR": np.exp(params),
    "CI_low": np.exp(conf[0]),
    "CI_high": np.exp(conf[1]),
    "p_value": model.pvalues
})

print("\nOdds ratios")
print(or_table.sort_values("p_value"))