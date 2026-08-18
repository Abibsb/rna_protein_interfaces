import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

INFILE = "variants_bestrow_ring_rna.csv"
N_SPLITS = 5

df = pd.read_csv(INFILE, low_memory=False)

# ======================
# TARGETS
# ======================

da = df["Disease_Association"].astype(str).str.strip().str.lower()
df["yA"] = df["Disease_Association"].notna().astype(int)

poly = df["Prediction_PolyPhen"].astype(str).str.lower()
sift = df["Prediction_SIFT"].astype(str).str.lower()

df["yB"] = (
    poly.str.contains("damaging|deleterious", na=False)
    | sift.str.contains("deleterious", na=False)
).astype(int)

# ======================
# FILTERING
# ======================

m = df[df["consurf_score"].notna() & df["iupred_score"].notna()].copy()

# numeric coercion
for c in ["consurf_score", "rna_contact_count"]:
    m[c] = pd.to_numeric(m[c], errors="coerce")

m["is_low_conf"] = m["is_low_conf"].astype(str).str.lower()
m = m[~m["is_low_conf"].isin(["true", "1", "yes"])]

m["is_core_hotspot"] = pd.to_numeric(m["is_core_hotspot"], errors="coerce").fillna(0).astype(int)
m["has_ring_interface"] = pd.to_numeric(m["has_ring_interface"], errors="coerce")

# restrict to interface residues
m = m[m["has_ring_interface"] == 1]

# drop rows with missing features
m = m.dropna(subset=["consurf_score", "rna_contact_count", "uniprot_id"])

# ======================
# CRITICAL FIX: CLEAN TARGETS
# ======================

for target in ["yA", "yB"]:
    m[target] = pd.to_numeric(m[target], errors="coerce")
    m = m[m[target].isin([0, 1])]
    m[target] = m[target].astype(int)

print("Final N:", len(m))

# grouping
groups = m["uniprot_id"].astype(str).values


# ======================
# MODEL FUNCTION
# ======================

def run_model(target, features):

    y = m[target].values
    X = m[features].values

    gkf = GroupKFold(n_splits=N_SPLITS)

    roc_scores = []
    pr_scores = []
    sizes = []

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",           # FIX: more stable solver
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    for train_idx, test_idx in gkf.split(X, y, groups):

        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        # FIX: ensure both train and test have both classes
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue

        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]

        roc_scores.append(roc_auc_score(yte, p))
        pr_scores.append(average_precision_score(yte, p))
        sizes.append(len(yte))

    if len(roc_scores) == 0:
        return np.nan, np.nan

    roc = np.average(roc_scores, weights=sizes)
    pr = np.average(pr_scores, weights=sizes)

    return roc, pr


# ======================
# MODELS
# ======================

features_base = ["consurf_score", "is_core_hotspot"]
features_rna = ["consurf_score", "is_core_hotspot", "rna_contact_count"]

for target in ["yA", "yB"]:

    print("\n====================")
    print("TARGET:", target)

    print("Class distribution:")
    print(m[target].value_counts())

    roc1, pr1 = run_model(target, features_base)
    roc2, pr2 = run_model(target, features_rna)

    print("\nBase ROC:", roc1)
    print("Base PR:", pr1)

    print("\nRNA ROC:", roc2)
    print("RNA PR:", pr2)

    print("\nΔPR:", pr2 - pr1)