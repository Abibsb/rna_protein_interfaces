#!/usr/bin/env python3

from pathlib import Path
import json
import re
import pandas as pd

root = Path("/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/28_jun_26/muestreo_fulldataset/alphafold_fragment_workflow_stratified_60")
#root = Path("/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/22_jun_26/alphafold_fragment_workflow_wf1")<

rows = []

for case_dir in sorted(root.iterdir()):
    if not case_dir.is_dir():
        continue

    selected = case_dir / "alphafold_selected"

    for condition in ["CIF_WT", "CIF_MUT"]:
        summaries = sorted(selected.glob(f"{condition}_best_summary_confidences_*.json"))

        for sf in summaries:
            m = re.search(r"_(\d+)\.json$", sf.name)
            model_idx = m.group(1) if m else ""

            data = json.loads(sf.read_text())

            row = {
                "case_id": case_dir.name,
                "condition": condition,
                "best_model_idx": model_idx,
                "best_summary_file": str(sf),
                "best_model_file": str(selected / f"{condition}_best_model_{model_idx}.cif"),
            }

            for k, v in data.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row[k] = v

            rows.append(row)

df = pd.DataFrame(rows)

out = root / "workflow1_afserver_ingested_best_models_REBUILT.tsv"
df.to_csv(out, sep="\t", index=False)

print(df.to_string(index=False))
print("\nWrote:", out)
