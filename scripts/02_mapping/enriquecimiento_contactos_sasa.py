import pandas as pd
from scipy.stats import fisher_exact

# -----------------------------
# 1. cargar dataset
# -----------------------------
csv_file = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/15_abr_26/ring_sasa/clean_resname/merged_final.csv"
df = pd.read_csv(csv_file, low_memory=False)

print("Columnas disponibles:")
print(df.columns.tolist())

# -----------------------------
# 2. limpiar y tipar
# -----------------------------
df["delta_sasa"] = pd.to_numeric(df["delta_sasa"], errors="coerce")
df["ring_interaction_count"] = pd.to_numeric(df["ring_interaction_count"], errors="coerce")

# normalizar is_interface_abs
df["is_interface_abs"] = (
    df["is_interface_abs"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"true": True, "false": False, "1": True, "0": False})
)

# eliminar filas con NA en lo importante
df = df.dropna(subset=["delta_sasa", "ring_interaction_count", "is_interface_abs"]).copy()

# -----------------------------
# 3. restringir a residuos de interfaz
# -----------------------------
df = df[df["is_interface_abs"] == True].copy()

print("\nN residuos de interfaz:", len(df))

# -----------------------------
# 4. definir contacto
# -----------------------------
df["contact"] = df["ring_interaction_count"] >= 1

# también podemos mirar contacto fuerte
df["contact_ge2"] = df["ring_interaction_count"] >= 2
df["contact_ge3"] = df["ring_interaction_count"] >= 3

# -----------------------------
# 5. definir alta ΔSASA
#    opción A: mediana
# -----------------------------
median_threshold = df["delta_sasa"].median()
df["high_sasa_median"] = df["delta_sasa"] >= median_threshold

# opción B: top 25%
q75_threshold = df["delta_sasa"].quantile(0.75)
df["high_sasa_q75"] = df["delta_sasa"] >= q75_threshold

print("\nThresholds:")
print("median delta_sasa =", median_threshold)
print("q75 delta_sasa    =", q75_threshold)

# -----------------------------
# 6. función para OR
# -----------------------------
def run_or(df, high_col, contact_col, label):
    a = ((df[high_col]) & (df[contact_col])).sum()
    b = ((df[high_col]) & (~df[contact_col])).sum()
    c = ((~df[high_col]) & (df[contact_col])).sum()
    d = ((~df[high_col]) & (~df[contact_col])).sum()

    print(f"\n=== {label} ===")
    print("Contingency table:")
    print(f"high_sasa & contact     = {a}")
    print(f"high_sasa & no_contact  = {b}")
    print(f"low_sasa  & contact     = {c}")
    print(f"low_sasa  & no_contact  = {d}")

    oddsratio, pvalue = fisher_exact([[a, b], [c, d]])

    print("Odds ratio =", oddsratio)
    print("Fisher p   =", pvalue)

    contact_rate_high = a / (a + b) if (a + b) > 0 else float("nan")
    contact_rate_low = c / (c + d) if (c + d) > 0 else float("nan")

    print("P(contact | high_sasa) =", contact_rate_high)
    print("P(contact | low_sasa)  =", contact_rate_low)

# -----------------------------
# 7. correr ORs
# -----------------------------
run_or(df, "high_sasa_median", "contact", "Median threshold, contact >=1")
run_or(df, "high_sasa_q75", "contact", "Q75 threshold, contact >=1")

run_or(df, "high_sasa_median", "contact_ge2", "Median threshold, contact >=2")
run_or(df, "high_sasa_q75", "contact_ge2", "Q75 threshold, contact >=2")

run_or(df, "high_sasa_median", "contact_ge3", "Median threshold, contact >=3")
run_or(df, "high_sasa_q75", "contact_ge3", "Q75 threshold, contact >=3")