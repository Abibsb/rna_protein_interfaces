import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("variants_bestrow_ring_rna.csv", low_memory=False)

# ---- yA definition ----
da = df["Disease_Association"]

df["yA"] = (
    da.notna() &
    (da.astype(str).str.strip() != "")
).astype(int)

# ---- same filtering as before ----
m = df[df["consurf_score"].notna() & df["iupred_score"].notna()].copy()

m["is_low_conf"] = m["is_low_conf"].astype(str).str.lower()
m = m[~m["is_low_conf"].isin(["true","1","yes"])]

m["has_ring_interface"] = pd.to_numeric(m["has_ring_interface"], errors="coerce")
m = m[m["has_ring_interface"] == 1].copy()

m["rna_contact_count"] = pd.to_numeric(m["rna_contact_count"], errors="coerce")

m = m.dropna(subset=[
    "rna_contact_count",
    "consurf_score",
    "delta_sasa_norm",
    "iupred_score",
    "yA"
])

print("Final rows:", len(m))
print("Unique proteins:", m["uniprot_id"].nunique())

# ---- predictors ----
X = m[[
    "consurf_score",
    "delta_sasa_norm",
    "rna_contact_count",
    "iupred_score"
]]

X = sm.add_constant(X)

y = m["yA"]

# ---- logistic regression with cluster SE ----
model = sm.Logit(y, X)

res = model.fit(
    disp=False,
    cov_type="cluster",
    cov_kwds={"groups": m["uniprot_id"]}
)

print(res.summary())

# ---- odds ratios ----
import numpy as np

params = res.params
conf = res.conf_int()

or_table = pd.DataFrame({
    "OR": np.exp(params),
    "CI_low": np.exp(conf[0]),
    "CI_high": np.exp(conf[1]),
    "p_value": res.pvalues
})

print("\nOdds ratios:")
print(or_table)