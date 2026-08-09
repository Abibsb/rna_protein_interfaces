import pandas as pd
from scipy.stats import spearmanr

INPUT = "dataset_sasa_ring.csv"

df = pd.read_csv(INPUT)

# Normalización mínima
df["ring_interaction_count"] = df["ring_interaction_count"].fillna(0)
df["delta_sasa"] = pd.to_numeric(df["delta_sasa"], errors="coerce")

# Definir clases
interface_col = "is_interface_abs" if "is_interface_abs" in df.columns else "is_interface"

df["interface_class"] = "non_interface"
df.loc[df[interface_col] == 1, "interface_class"] = "interface"
df.loc[df["is_strong_interface"] == 1, "interface_class"] = "strong_interface"
df.loc[df["is_core_hotspot"] == 1, "interface_class"] = "core_hotspot"

classes = ["non_interface", "interface", "strong_interface", "core_hotspot"]

results = []

for cls in classes:
    sub = df[df["interface_class"] == cls].copy()

    # --- TODOS ---
    sub_all = sub[["delta_sasa", "ring_interaction_count"]].dropna()

    if len(sub_all) > 10:
        rho_all, p_all = spearmanr(
            sub_all["delta_sasa"],
            sub_all["ring_interaction_count"]
        )
    else:
        rho_all, p_all = None, None

    # --- SOLO CON CONTACTO ---
    sub_contact = sub[sub["ring_interaction_count"] > 0][
        ["delta_sasa", "ring_interaction_count"]
    ].dropna()

    if len(sub_contact) > 10:
        rho_contact, p_contact = spearmanr(
            sub_contact["delta_sasa"],
            sub_contact["ring_interaction_count"]
        )
    else:
        rho_contact, p_contact = None, None

    results.append({
        "class": cls,
        "n_total": len(sub),
        "n_with_contact": len(sub_contact),
        "rho_all": rho_all,
        "p_all": p_all,
        "rho_contact_only": rho_contact,
        "p_contact_only": p_contact
    })

results_df = pd.DataFrame(results)

print("\n=== Spearman por clase ===\n")
print(results_df)

results_df.to_csv("spearman_by_hotspot.csv", index=False)

print("\nGuardado en spearman_by_hotspot.csv")