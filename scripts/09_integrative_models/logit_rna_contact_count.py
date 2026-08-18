import pandas as pd
import statsmodels.api as sm
import numpy as np

INFILE = "variants_bestrow_ring_rna.csv"

# ======================
# Load data
# ======================
df = pd.read_csv(INFILE, low_memory=False)

# ======================
# Targets
# ======================

# yA: disease association
da = df["Disease_Association"].astype(str).str.strip().str.lower()
df["yA"] = ((da != "") & (da != "nan")).astype(int)

# yB: predicted damaging
poly = df["Prediction_PolyPhen"].astype(str).str.lower()
sift = df["Prediction_SIFT"].astype(str).str.lower()

df["yB"] = (
    poly.str.contains("damaging|deleterious", na=False)
    | sift.str.contains("deleterious", na=False)
).astype(int)

# ======================
# Filtering (same as previous analyses)
# ======================

m = df[df["consurf_score"].notna() & df["iupred_score"].notna()].copy()

m["is_low_conf"] = m["is_low_conf"].astype(str).str.lower()
m = m[~m["is_low_conf"].isin(["true","1","yes"])]

m["has_ring_interface"] = pd.to_numeric(m["has_ring_interface"], errors="coerce")
m = m[m["has_ring_interface"] == 1]

m["rna_contact_count"] = pd.to_numeric(m["rna_contact_count"], errors="coerce")

m = m.dropna(subset=["rna_contact_count","uniprot_id"])

print("Final N:", len(m))
print("Unique proteins:", m["uniprot_id"].nunique())

# ======================
# Function to run model
# ======================

def run_logit(target):

    y = m[target]
    X = sm.add_constant(m[["rna_contact_count"]])

    model = sm.Logit(y, X)

    res = model.fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": m["uniprot_id"]}
    )

    print("\n====================")
    print("Target:", target)
    print(res.summary())

    params = res.params
    conf = res.conf_int()

    OR = np.exp(params)
    CI_low = np.exp(conf[0])
    CI_high = np.exp(conf[1])

    or_table = pd.DataFrame({
        "OR": OR,
        "CI_low": CI_low,
        "CI_high": CI_high,
        "p_value": res.pvalues
    })

    print("\nOdds ratios:")
    print(or_table)
# ======================
# Run models
# ======================

run_logit("yA")
run_logit("yB")