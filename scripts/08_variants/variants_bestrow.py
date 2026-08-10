import pandas as pd
import numpy as np

INFILE = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/23_feb_26/MASTER_interface_dataset_PER_CHAIN_WITH_VARIANTS.csv"
OUTFILE = "variants_bestrow.csv"

usecols = [
    "polymer_class","uniprot_id","uniprot_pos",
    "Wild_Type","Mutated_Type",
    "consurf_score","msa_coverage","is_low_conf",
    "buried_fraction_residue","delta_sasa","is_interface_abs","interface_class","is_core_hotspot",
    "iupred_score","is_disordered",
    "has_bridging_water",
    "Disease_Association","Prediction_PolyPhen","Prediction_SIFT"
]

best = {}  # variant_id -> (score, row_dict)

chunksize = 400_000  # ajustable
for chunk in pd.read_csv(INFILE, usecols=usecols, low_memory=False, chunksize=chunksize):
    # Solo proteína para semana 1 (si querés RNA después lo habilitamos)
    chunk = chunk[chunk["polymer_class"] == "protein"]

    # Necesarios para identificar variante
    chunk = chunk.dropna(subset=["uniprot_id","uniprot_pos","Wild_Type","Mutated_Type"])

    # uniprot_pos a int cuando se pueda
    chunk["uniprot_pos"] = pd.to_numeric(chunk["uniprot_pos"], errors="coerce")
    chunk = chunk.dropna(subset=["uniprot_pos"])
    chunk["uniprot_pos"] = chunk["uniprot_pos"].astype(int)

    # variant_id
    chunk["variant_id"] = (
        chunk["uniprot_id"].astype(str) + "_" +
        chunk["uniprot_pos"].astype(str) + "_" +
        chunk["Wild_Type"].astype(str) + ">" +
        chunk["Mutated_Type"].astype(str)
    )

    # score para elegir “mejor representante”
    # - interfaz pesa mucho
    # - enterramiento y delta_sasa suman
    # - baja confianza penaliza
    iface = chunk["is_interface_abs"].astype(int).fillna(0)
    buried = chunk["buried_fraction_residue"].fillna(-1.0)
    dsasa = chunk["delta_sasa"].fillna(-1.0)
    lowconf = chunk["is_low_conf"].astype(str).str.lower().isin(["true","1","yes"])
    penalty = lowconf.astype(int) * 5  # penaliza fuerte

    chunk["pick_score"] = (10*iface) + (3*buried) + (0.01*dsasa) - penalty

    # iterar filas (sí, es más lento, pero mantiene RAM estable)
    for row in chunk.itertuples(index=False):
        vid = row.variant_id
        score = row.pick_score
        prev = best.get(vid)
        if (prev is None) or (score > prev[0]):
            best[vid] = (score, row._asdict())

# armar DF final
rows = [v[1] for v in best.values()]
out = pd.DataFrame(rows)

# limpiar columnas auxiliares si quedaron
if "pick_score" in out.columns:
    out = out.drop(columns=["pick_score"], errors="ignore")

out.to_csv(OUTFILE, index=False)
print("Wrote:", OUTFILE, "rows:", len(out), "unique variants:", out["variant_id"].nunique())