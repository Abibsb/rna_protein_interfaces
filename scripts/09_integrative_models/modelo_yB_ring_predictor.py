import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("variants_bestrow_ring.csv", low_memory=False)

# ===== Target =====
pp = df["Prediction_PolyPhen"].astype(str).str.lower()
sf = df["Prediction_SIFT"].astype(str).str.lower()
damaging_pp = pp.str.contains("damaging", regex=True, na=False)
deleterious_sf = sf.str.contains("deleterious", regex=True, na=False)
df["yB"] = (damaging_pp & deleterious_sf).astype(int)

# ===== Filtrado estructural-evolutivo =====
m = df[
    df["consurf_score"].notna() &
    df["iupred_score"].notna()
].copy()

# Forzar numéricos
for c in ["consurf_score",
          "buried_fraction_residue",
          "iupred_score"]:
    m[c] = pd.to_numeric(m[c], errors="coerce")

m = m.dropna(subset=[
    "consurf_score",
    "buried_fraction_residue",
    "iupred_score"
])

# Limpiar low confidence
m["is_low_conf"] = m["is_low_conf"].astype(str).str.lower()
m = m[~m["is_low_conf"].isin(["true","1","yes"])]

# Convertir RING
m["has_ring_interface"] = m["has_ring_interface"].astype(int)

# ===== Features =====
features = [
    "consurf_score",
    "buried_fraction_residue",
    "has_ring_interface",
    "iupred_score"
]

# Estandarizar continuas
continuous = [
    "consurf_score",
    "buried_fraction_residue",
    "iupred_score"
]

for c in continuous:
    m[c] = (m[c] - m[c].mean()) / (m[c].std() + 1e-9)

X = m[features].astype(float)
X = sm.add_constant(X)

y = m["yB"].astype(int)

print("Final N:", len(m))
print("Positive fraction:", y.mean())

model = sm.Logit(y, X).fit(
    disp=0,
    cov_type="cluster",
    cov_kwds={"groups": m["uniprot_id"]}
)

print(model.summary())

# Odds ratios
params = model.params
conf = model.conf_int()
conf.columns = ["2.5%","97.5%"]

or_table = pd.DataFrame({
    "OR": np.exp(params),
    "CI_low": np.exp(conf["2.5%"]),
    "CI_high": np.exp(conf["97.5%"]),
    "p_value": model.pvalues
}).sort_values("p_value")

print("\nOdds ratios:")
print(or_table)