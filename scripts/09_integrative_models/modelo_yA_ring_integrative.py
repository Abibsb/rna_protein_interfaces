import pandas as pd
import numpy as np
import statsmodels.api as sm

INFILE = "/media/usuario/Seagate Basic1/Tesis_Maestria/02_mar_26/3_rna/variants_bestrow_ring_rna.csv"

df = pd.read_csv(INFILE, low_memory=False)

# ======================
# Target yA
# ======================
da = df["Disease_Association"]
df["yA"] = (
    da.notna() &
    (da.astype(str).str.strip() != "") &
    (da.astype(str).str.lower().str.strip() != "nan")
).astype(int)

# ======================
# Variables
# ======================
df["rna_contact"] = (pd.to_numeric(df["rna_contact_count"], errors="coerce").fillna(0) > 0).astype(int)
df["basic_residue"] = df["resname"].isin(["ARG", "LYS", "HIS"]).astype(int)
df["is_core_hotspot"] = pd.to_numeric(df["is_core_hotspot"], errors="coerce").fillna(0).astype(int)

for c in ["consurf_score", "delta_sasa_norm", "iupred_score", "has_ring_interface"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ======================
# Filtering
# ======================
m = df[
    df["consurf_score"].notna() &
    df["iupred_score"].notna()
].copy()

m["is_low_conf"] = m["is_low_conf"].astype(str).str.lower()
m = m[~m["is_low_conf"].isin(["true", "1", "yes"])]

# opcional: trabajar solo en interfaz RNA estructural
m = m[m["has_ring_interface"] == 1].copy()

m = m.dropna(subset=[
    "yA",
    "consurf_score",
    "delta_sasa_norm",
    "iupred_score",
    "uniprot_id"
])

print("Final N:", len(m))
print("Unique proteins:", m["uniprot_id"].nunique())
print("Positive fraction yA:", m["yA"].mean())

# ======================
# Modelo
# ======================
features = [
    "consurf_score",
    "delta_sasa_norm",
    "iupred_score",
    "basic_residue",
    "rna_contact",
    "is_core_hotspot"
]

X = m[features].copy()
X = sm.add_constant(X)
y = m["yA"].astype(int)

model = sm.Logit(y, X).fit(
    disp=False,
    cov_type="cluster",
    cov_kwds={"groups": m["uniprot_id"]}
)

print(model.summary())

# ======================
# Odds ratios
# ======================
params = model.params
conf = model.conf_int()
conf.columns = ["2.5%", "97.5%"]

or_table = pd.DataFrame({
    "OR": np.exp(params),
    "CI_low": np.exp(conf["2.5%"]),
    "CI_high": np.exp(conf["97.5%"]),
    "p_value": model.pvalues
}).sort_values("p_value")

print("\nOdds ratios:")
print(or_table)