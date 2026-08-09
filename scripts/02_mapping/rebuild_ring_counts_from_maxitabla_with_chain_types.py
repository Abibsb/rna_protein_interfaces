#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reconstruye conteos reales de contactos proteína-RNA a partir de maxitabla,
usando chain_types.csv para identificar correctamente qué cadenas son proteína
y cuáles son RNA.

Qué hace:
1. Lee maxitabla_dataset_actualizado.csv/tsv
2. Parsea NodeId1 y NodeId2
3. Usa chain_types.csv para decidir si cada cadena es proteína o RNA
4. Se queda solo con interacciones proteína-RNA
5. Resume por residuo proteína:
   - ring_row_count: cantidad real de filas RING
   - unique_rna_residue_contacts: cantidad de residuos de RNA distintos tocados
   - unique_interaction_types: cantidad de tipos de interacción distintos
6. Mergea con interface_areas_per_residue_normalized.csv
7. Escribe distribuciones y tablas de salida

Uso:
python rebuild_ring_counts_from_maxitabla_with_chain_types.py \
  --maxitabla "/ruta/maxitabla_dataset_actualizado.csv" \
  --chain_types "/ruta/chain_types.csv" \
  --interface_csv "/ruta/interface_areas_per_residue_normalized.csv" \
  --outdir "/ruta/salida"
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


NODE_RE = re.compile(r"^([^:]+):([^:]+):([^:]*):([^:]+)$")


def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruye conteos RING proteína-RNA usando chain_types.csv"
    )
    p.add_argument("--maxitabla", required=True,
                   help="Ruta a maxitabla_dataset_actualizado.csv o .tsv")
    p.add_argument("--chain_types", required=True,
                   help="Ruta a chain_types.csv")
    p.add_argument("--interface_csv", required=True,
                   help="Ruta a interface_areas_per_residue_normalized.csv")
    p.add_argument("--outdir", required=True,
                   help="Directorio de salida")
    return p.parse_args()


def read_table_flexible(path: str) -> pd.DataFrame:
    """
    Intenta leer primero como TSV y luego como CSV.
    """
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path)
    return df


def parse_node(node: str):
    """
    Parsea strings tipo:
    B:13:_:TYR
    Q:10:_:C
    R:664:_:C
    """
    if pd.isna(node):
        return None

    node = str(node).strip()
    m = NODE_RE.match(node)
    if not m:
        return None

    chain, resnum, insertion, resname = m.groups()

    try:
        resnum = int(resnum)
    except ValueError:
        return None

    return {
        "chain": chain,
        "resnum": resnum,
        "insertion": insertion,
        "resname": resname
    }


def load_chain_types(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    needed = ["pdb_id", "auth_asym_id", "polymer_type"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en chain_types.csv: {missing}")

    df["pdb_id"] = df["pdb_id"].astype(str).str.upper().str.strip()
    df["auth_asym_id"] = df["auth_asym_id"].astype(str).str.strip()
    df["polymer_type"] = df["polymer_type"].astype(str).str.strip()

    return df[["pdb_id", "auth_asym_id", "polymer_type"]].drop_duplicates()


def is_protein(polymer_type: str) -> bool:
    return polymer_type == "polypeptide(L)"


def is_rna(polymer_type: str) -> bool:
    return polymer_type == "polyribonucleotide"


def build_protein_rna_contacts(maxi: pd.DataFrame, chain_map: dict) -> pd.DataFrame:
    parsed_rows = []

    skipped_parse = 0
    skipped_chain_map = 0
    skipped_non_prot_rna = 0

    for _, row in maxi.iterrows():
        pdb_id = str(row["PDBID"]).upper().strip()

        n1 = parse_node(row["NodeId1"])
        n2 = parse_node(row["NodeId2"])

        if n1 is None or n2 is None:
            skipped_parse += 1
            continue

        t1 = chain_map.get((pdb_id, n1["chain"]), "unknown")
        t2 = chain_map.get((pdb_id, n2["chain"]), "unknown")

        if t1 == "unknown" or t2 == "unknown":
            skipped_chain_map += 1
            continue

        if is_protein(t1) and is_rna(t2):
            prot = n1
            nt = n2
            prot_polymer_type = t1
            rna_polymer_type = t2
        elif is_protein(t2) and is_rna(t1):
            prot = n2
            nt = n1
            prot_polymer_type = t2
            rna_polymer_type = t1
        else:
            skipped_non_prot_rna += 1
            continue

        parsed_rows.append({
            "pdb_id": pdb_id,
            "auth_asym_id": prot["chain"],
            "auth_seq_id": prot["resnum"],
            "protein_resname": prot["resname"],
            "protein_polymer_type": prot_polymer_type,
            "rna_chain": nt["chain"],
            "rna_resnum": nt["resnum"],
            "rna_resname": nt["resname"],
            "rna_polymer_type": rna_polymer_type,
            "Interaction": row.get("Interaction", np.nan),
            "Distance": row.get("Distance", np.nan),
            "Angle": row.get("Angle", np.nan),
            "Energy": row.get("Energy", np.nan),
            "Atom1": row.get("Atom1", np.nan),
            "Atom2": row.get("Atom2", np.nan),
        })

    prot_contacts = pd.DataFrame(parsed_rows)

    debug = {
        "total_rows_input": len(maxi),
        "protein_rna_rows_detected": len(prot_contacts),
        "skipped_parse": skipped_parse,
        "skipped_chain_map": skipped_chain_map,
        "skipped_non_protein_rna": skipped_non_prot_rna,
    }

    return prot_contacts, debug


def summarize_counts(prot_contacts: pd.DataFrame) -> pd.DataFrame:
    if prot_contacts.empty:
        return pd.DataFrame(columns=[
            "pdb_id", "auth_asym_id", "auth_seq_id",
            "ring_row_count",
            "unique_rna_residue_contacts",
            "unique_interaction_types"
        ])

    tmp = prot_contacts.copy()
    tmp["rna_residue_id"] = (
        tmp["rna_chain"].astype(str) + ":" +
        tmp["rna_resnum"].astype(str) + ":" +
        tmp["rna_resname"].astype(str)
    )

    summary = (
        tmp.groupby(["pdb_id", "auth_asym_id", "auth_seq_id"], as_index=False)
           .agg(
               ring_row_count=("Interaction", "size"),
               unique_rna_residue_contacts=("rna_residue_id", "nunique"),
               unique_interaction_types=("Interaction", "nunique")
           )
    )

    return summary


def load_interface(interface_csv: str) -> pd.DataFrame:
    df = pd.read_csv(interface_csv)

    needed = ["pdb_id", "auth_asym_id", "auth_seq_id", "delta_sasa", "is_interface_abs"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en interface CSV: {missing}")

    df["pdb_id"] = df["pdb_id"].astype(str).str.upper().str.strip()
    df["auth_asym_id"] = df["auth_asym_id"].astype(str).str.strip()
    df["auth_seq_id"] = pd.to_numeric(df["auth_seq_id"], errors="coerce")
    df["delta_sasa"] = pd.to_numeric(df["delta_sasa"], errors="coerce")

    df = df.dropna(subset=["auth_seq_id"]).copy()
    df["auth_seq_id"] = df["auth_seq_id"].astype(int)

    df["is_interface_abs"] = (
        df["is_interface_abs"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )

    return df


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Leyendo maxitabla...")
    maxi = read_table_flexible(args.maxitabla)

    required = ["PDBID", "NodeId1", "NodeId2"]
    missing = [c for c in required if c not in maxi.columns]
    if missing:
        print("Columnas detectadas en maxitabla:", list(maxi.columns))
        raise ValueError(f"Faltan columnas en maxitabla: {missing}")

    maxi["PDBID"] = maxi["PDBID"].astype(str).str.upper().str.strip()

    print("[2/7] Leyendo chain_types...")
    chain_types = load_chain_types(args.chain_types)
    chain_map = {
        (row.pdb_id, row.auth_asym_id): row.polymer_type
        for row in chain_types.itertuples(index=False)
    }

    print("[3/7] Parseando interacciones proteína-RNA...")
    prot_contacts, debug = build_protein_rna_contacts(maxi, chain_map)
    prot_contacts.to_csv(outdir / "protein_rna_ring_rows_from_maxitabla.csv", index=False)

    print(f"    Total filas RING: {debug['total_rows_input']}")
    print(f"    Filas proteína-RNA detectadas: {debug['protein_rna_rows_detected']}")
    print(f"    Filas salteadas por parseo: {debug['skipped_parse']}")
    print(f"    Filas salteadas por falta en chain_map: {debug['skipped_chain_map']}")
    print(f"    Filas salteadas por no ser proteína-RNA: {debug['skipped_non_protein_rna']}")

    print("[4/7] Resumiendo conteos por residuo proteína...")
    counts = summarize_counts(prot_contacts)
    counts.to_csv(outdir / "protein_rna_ring_counts_from_maxitabla.csv", index=False)

    print("[5/7] Leyendo interfaz SASA...")
    interface_df = load_interface(args.interface_csv)

    print("[6/7] Mergeando SASA + conteos reales de RING...")
    merged = interface_df.merge(
        counts,
        how="left",
        on=["pdb_id", "auth_asym_id", "auth_seq_id"]
    )

    for col in ["ring_row_count", "unique_rna_residue_contacts", "unique_interaction_types"]:
        merged[col] = merged[col].fillna(0).astype(int)

    merged["has_ring_row_contact"] = merged["ring_row_count"] > 0
    merged["has_multi_ring_rows"] = merged["ring_row_count"] >= 2
    merged["has_multi_rna_residue_contacts"] = merged["unique_rna_residue_contacts"] >= 2

    merged.to_csv(outdir / "merged_sasa_with_real_ring_counts.csv", index=False)

    print("[7/7] Escribiendo distribuciones y debug...")
    with open(outdir / "ring_count_distributions.txt", "w", encoding="utf-8") as fh:
        fh.write("DEBUG\n")
        fh.write("=====\n")
        for k, v in debug.items():
            fh.write(f"{k}: {v}\n")
        fh.write("\n")

        fh.write("ring_row_count distribution\n")
        fh.write("===========================\n")
        fh.write(merged["ring_row_count"].value_counts().sort_index().to_string())
        fh.write("\n\n")

        fh.write("unique_rna_residue_contacts distribution\n")
        fh.write("=======================================\n")
        fh.write(merged["unique_rna_residue_contacts"].value_counts().sort_index().to_string())
        fh.write("\n\n")

        fh.write("unique_interaction_types distribution\n")
        fh.write("===================================\n")
        fh.write(merged["unique_interaction_types"].value_counts().sort_index().to_string())
        fh.write("\n")

    print("Listo.")
    print(f"Salida en: {outdir}")


if __name__ == "__main__":
    main()