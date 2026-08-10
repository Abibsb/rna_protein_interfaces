#!/usr/bin/env python3
"""
Descripción de nucleótidos asociados a aguas puente.

Unidad de análisis:
    par único (pdb_id, water_id, nucleotide_name, nucleotide_number)

Se generan conteos y porcentajes para:
    1) aguas puente definidas por distancia;
    2) aguas puente que además superan el filtro angular.

Uso:
    python water_nucleotide_composition.py \
        --json_dir /ruta/a/jsons \
        --out_prefix resultados_water_nt
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


NT_MAP = {
    "A": "A", "DA": "A",
    "C": "C", "DC": "C",
    "G": "G", "DG": "G",
    "U": "U", "DT": "U",
}
NT_ORDER = ["A", "C", "G", "U"]


def collect_pairs(json_dir: Path, key: str, mode: str) -> pd.DataFrame:
    rows = []

    for json_file in sorted(json_dir.glob("*.json")):
        pdb_id = json_file.stem.upper()

        try:
            with json_file.open() as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"Advertencia: no se pudo leer {json_file.name}: {exc}")
            continue

        partners_by_water = data.get(key, {})

        for water_id, partners in partners_by_water.items():
            for residue in partners.get("rna_residues", []):
                if len(residue) < 2:
                    continue

                raw_name = str(residue[0]).upper()
                residue_number = residue[1]
                nucleotide = NT_MAP.get(raw_name)

                rows.append({
                    "pdb_id": pdb_id,
                    "water_id": str(water_id),
                    "raw_resname": raw_name,
                    "nucleotide": nucleotide,
                    "rna_resnum": residue_number,
                    "mode": mode,
                    "is_standard": nucleotide is not None,
                })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Un mismo nucleótido se cuenta una sola vez por agua dentro de cada PDB.
    return df.drop_duplicates(
        subset=["pdb_id", "water_id", "raw_resname", "rna_resnum", "mode"]
    )


def summarize(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({
            "mode": [mode] * len(NT_ORDER),
            "nucleotide": NT_ORDER,
            "n_water_nucleotide_pairs": [0] * len(NT_ORDER),
            "percentage": [0.0] * len(NT_ORDER),
        })

    standard = df[df["is_standard"]].copy()
    counts = (
        standard.groupby("nucleotide")
        .size()
        .reindex(NT_ORDER, fill_value=0)
    )

    total = int(counts.sum())
    percentages = counts / total * 100 if total else counts.astype(float)

    return pd.DataFrame({
        "mode": mode,
        "nucleotide": NT_ORDER,
        "n_water_nucleotide_pairs": counts.values,
        "percentage": percentages.values,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True, type=Path)
    parser.add_argument("--out_prefix", default="water_nucleotide")
    args = parser.parse_args()

    if not args.json_dir.is_dir():
        raise SystemExit(f"No existe el directorio: {args.json_dir}")

    distance_df = collect_pairs(
        args.json_dir,
        key="bridging_water_partners",
        mode="distance_only"
    )
    angle_df = collect_pairs(
        args.json_dir,
        key="bridging_water_partners_angle_filtered",
        mode="angle_filtered"
    )

    all_pairs = pd.concat([distance_df, angle_df], ignore_index=True)
    all_pairs.to_csv(f"{args.out_prefix}_pairs.csv", index=False)

    summary = pd.concat([
        summarize(distance_df, "distance_only"),
        summarize(angle_df, "angle_filtered"),
    ], ignore_index=True)
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    qc_rows = []
    for mode, df in [("distance_only", distance_df), ("angle_filtered", angle_df)]:
        qc_rows.append({
            "mode": mode,
            "n_pairs_total": len(df),
            "n_pairs_standard": int(df["is_standard"].sum()) if not df.empty else 0,
            "n_pairs_nonstandard": int((~df["is_standard"]).sum()) if not df.empty else 0,
            "n_unique_pdb": df["pdb_id"].nunique() if not df.empty else 0,
            "n_unique_waters": (
                df[["pdb_id", "water_id"]].drop_duplicates().shape[0]
                if not df.empty else 0
            ),
        })

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(f"{args.out_prefix}_qc.csv", index=False)

    plot_df = summary.pivot(
        index="nucleotide",
        columns="mode",
        values="percentage"
    ).reindex(NT_ORDER)

    ax = plot_df.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("Nucleótido")
    ax.set_ylabel("Pares agua–nucleótido (%)")
    ax.set_title("Composición de nucleótidos asociados a aguas puente")
    ax.legend(title="Definición")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_barplot.png", dpi=300)
    plt.close()

    print("\nResumen:")
    print(summary.to_string(index=False))
    print(f"\nArchivos escritos con prefijo: {args.out_prefix}")


if __name__ == "__main__":
    main()
