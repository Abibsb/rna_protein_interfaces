#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# PATHS
# ============================================================

INFILE = (
    "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/"
    "Tesis_Maestria/01_mar_26/variants_bestrow_ring.csv"
)

OUTFILE = (
    "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/"
    "Tesis_Maestria/02_ago_26_ultimos_retoques/"
    "cv_structural_current_FINAL.txt"
)

N_SPLITS = 5


# ============================================================
# FEATURES
# ============================================================

FEATURES_CONSURF = [
    "consurf_score",
]

FEATURES_FULL = [
    "consurf_score",
    "buried_fraction_residue",
    "is_core_hotspot",
    "has_bridging_water",
    "iupred_score",
]


# ============================================================
# HELPERS
# ============================================================

def weighted_average(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    return np.sum(values * weights) / np.sum(weights)


def evaluate_model(df, features, target, label):
    X = df[features].astype(float).values
    y = df[target].astype(int).values
    groups = df["uniprot_id"].astype(str).values

    gkf = GroupKFold(n_splits=N_SPLITS)

    roc_aucs = []
    pr_aucs = []
    fold_sizes = []

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    ])

    print(f"\n=== {target} — {label} ===")

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups),
        start=1,
    ):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        if len(np.unique(y_train)) < 2:
            print(
                f"Fold {fold}: omitido "
                "(una sola clase en entrenamiento)"
            )
            continue

        if len(np.unique(y_test)) < 2:
            print(
                f"Fold {fold}: omitido "
                "(una sola clase en test)"
            )
            continue

        pipe.fit(X_train, y_train)

        probabilities = pipe.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(
            y_test,
            probabilities,
        )

        pr = average_precision_score(
            y_test,
            probabilities,
        )

        roc_aucs.append(roc)
        pr_aucs.append(pr)
        fold_sizes.append(len(y_test))

        print(
            f"Fold {fold}: "
            f"N={len(y_test)} "
            f"ROC={roc:.6f} "
            f"PR={pr:.6f}"
        )

    if not roc_aucs:
        raise RuntimeError(
            f"No hubo folds evaluables para {target} / {label}"
        )

    roc_mean = weighted_average(
        roc_aucs,
        fold_sizes,
    )

    pr_mean = weighted_average(
        pr_aucs,
        fold_sizes,
    )

    print("\nResumen:")
    print(
        "Folds usados:",
        len(roc_aucs),
        "/",
        N_SPLITS,
    )

    print(
        "ROC AUC weighted mean:",
        roc_mean,
    )

    print(
        "PR AUC / AP weighted mean:",
        pr_mean,
    )

    print(
        "ROC AUC por fold:",
        [round(x, 4) for x in roc_aucs],
    )

    print(
        "PR AUC por fold:",
        [round(x, 4) for x in pr_aucs],
    )

    return roc_mean, pr_mean


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INFILE,
    low_memory=False,
)


# ============================================================
# TARGETS
# ============================================================

# yA:
# Disease_Association presente y no vacía
da = df["Disease_Association"]

df["yA"] = (
    da.notna()
    & da.astype("string").str.strip().ne("")
).astype(int)


# yB:
# PolyPhen damaging AND SIFT deleterious
pp = (
    df["Prediction_PolyPhen"]
    .astype("string")
    .str.lower()
)

sf = (
    df["Prediction_SIFT"]
    .astype("string")
    .str.lower()
)

df["yB"] = (
    pp.str.contains(
        "damaging",
        regex=True,
        na=False,
    )
    &
    sf.str.contains(
        "deleterious",
        regex=True,
        na=False,
    )
).astype(int)


# ============================================================
# FILTERING
# ============================================================

m = df[
    df["consurf_score"].notna()
    &
    df["iupred_score"].notna()
].copy()


numeric_columns = [
    "consurf_score",
    "buried_fraction_residue",
    "iupred_score",
    "has_ring_interface",
    "is_core_hotspot",
    "has_bridging_water",
]

for column in numeric_columns:
    m[column] = pd.to_numeric(
        m[column],
        errors="coerce",
    )


# Excluir low confidence
m["is_low_conf"] = (
    m["is_low_conf"]
    .astype(str)
    .str.lower()
)

m = m[
    ~m["is_low_conf"].isin(
        ["true", "1", "yes"]
    )
].copy()


# Binarias
m["is_core_hotspot"] = (
    m["is_core_hotspot"]
    .fillna(0)
    .astype(int)
)

m["has_bridging_water"] = (
    m["has_bridging_water"]
    .fillna(0)
    .astype(int)
)


# Solo variantes con contacto RING
m = m[
    m["has_ring_interface"] == 1
].copy()


# Dataset común para todos los modelos
m = m.dropna(
    subset=[
        "consurf_score",
        "buried_fraction_residue",
        "iupred_score",
        "uniprot_id",
    ]
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("=" * 70)
print("CURRENT STRUCTURAL CV DATASET")
print("=" * 70)

print("Input:", INFILE)
print("Final N:", len(m))
print(
    "Unique proteins:",
    m["uniprot_id"].nunique(),
)

print("\nyA distribution:")
print(
    m["yA"].value_counts().sort_index()
)

print(
    "yA positive fraction:",
    m["yA"].mean(),
)

print("\nyB distribution:")
print(
    m["yB"].value_counts().sort_index()
)

print(
    "yB positive fraction:",
    m["yB"].mean(),
)


# ============================================================
# EVALUATION
# ============================================================

results = {}

for target in ["yA", "yB"]:

    print("\n")
    print("=" * 70)
    print("TARGET:", target)
    print("=" * 70)

    roc_base, pr_base = evaluate_model(
        m,
        FEATURES_CONSURF,
        target,
        "ConSurf only",
    )

    roc_full, pr_full = evaluate_model(
        m,
        FEATURES_FULL,
        target,
        "Full structural + evolutionary",
    )

    delta_roc = roc_full - roc_base
    delta_pr = pr_full - pr_base

    print("\n=== Incremental performance ===")
    print("Base ROC:", roc_base)
    print("Base PR:", pr_base)
    print("Full ROC:", roc_full)
    print("Full PR:", pr_full)
    print("Delta ROC:", delta_roc)
    print("Delta PR:", delta_pr)

    results[target] = {
        "base_roc": roc_base,
        "base_pr": pr_base,
        "full_roc": roc_full,
        "full_pr": pr_full,
        "delta_roc": delta_roc,
        "delta_pr": delta_pr,
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for target, res in results.items():
    print(f"\n{target}")
    print(
        f"ConSurf only: "
        f"ROC={res['base_roc']:.6f}, "
        f"PR={res['base_pr']:.6f}"
    )
    print(
        f"Full model:   "
        f"ROC={res['full_roc']:.6f}, "
        f"PR={res['full_pr']:.6f}"
    )
    print(
        f"Delta:        "
        f"ROC={res['delta_roc']:.6f}, "
        f"PR={res['delta_pr']:.6f}"
    )
