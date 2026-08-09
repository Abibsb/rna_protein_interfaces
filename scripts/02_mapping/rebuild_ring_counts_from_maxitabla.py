#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


NODE_RE = re.compile(r"^([^:]+):([^:]+):([^:]*):([^:]+)$")


def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruye ring_interaction_count real por residuo a partir de maxitabla_dataset_actualizado.csv"
    )
    p.add_argument("--maxitabla", required=True,
                   help="Ruta a maxitabla_dataset_actualizado.csv")
    p.add_argument("--interface_csv", required=True,
                   help="Ruta a interface_areas_per_residue_normalized.csv")
    p.add_argument("--outdir", required=True,
                   help="Directorio de salida")
    return p.parse_args()


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


def infer_polymer_type(resname: str):
    if pd.isna(resname):
        return "unknown"

    r = str(resname).upper().strip()

    # aminoácidos (más amplio)
    if len(r) == 3:
        return "protein"

    # nucleótidos (incluye variantes)
    if r in {"A","C","G","U","T","I"}:
        return "nucleotide"

    # fallback: nucleótidos largos tipo DA, DG, etc.
    if r.startswith("D") and len(r) == 2:
        return "nucleotide"

    return "unknown"


def build_protein_side_contacts(maxi: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de NodeId1/NodeId2 identifica qué lado del par es proteína
    y construye una tabla por fila con el residuo proteína.
    """

    parsed_rows = []

    for _, row in maxi.iterrows():
        pdb_id = str(row["PDBID"]).upper().strip()

        n1 = parse_node(row["NodeId1"])
        n2 = parse_node(row["NodeId2"])

        if n1 is None or n2 is None:
            continue

        t1 = infer_polymer_type(n1["resname"])
        t2 = infer_polymer_type(n2["resname"])

        # Queremos interacciones proteína-nucleótido
        # y nos quedamos con el lado proteína
        if t1 == "protein" and t2 == "nucleotide":
            prot = n1
            nt = n2
        elif t2 == "protein" and t1 == "nucleotide":
            prot = n2
            nt = n1
        else:
            continue

        parsed_rows.append({
            "pdb_id": pdb_id,
            "auth_asym_id": prot["chain"],
            "auth_seq_id": prot["resnum"],
            "protein_resname": prot["resname"],
            "rna_chain": nt["chain"],
            "rna_resnum": nt["resnum"],
            "rna_resname": nt["resname"],
            "Interaction": row.get("Interaction", np.nan),
            "Distance": row.get("Distance", np.nan),
            "Angle": row.get("Angle", np.nan),
            "Energy": row.get("Energy", np.nan),
            "Atom1": row.get("Atom1", np.nan),
            "Atom2": row.get("Atom2", np.nan),
        })

    return pd.DataFrame(parsed_rows)


def summarize_counts(prot_contacts: pd.DataFrame) -> pd.DataFrame:
    """
    Genera varios conteos útiles por residuo:
    - ring_row_count: cantidad real de filas RING
    - unique_rna_residue_contacts: cuántos residuos de RNA distintos toca
    - unique_interaction_types: cuántos tipos de interacción distintos tiene
    """

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

    # normalizar bool
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

    print("[1/6] Leyendo maxitabla...")
    maxi = pd.read_csv(args.maxitabla, sep="\t")

    required = ["PDBID", "NodeId1", "NodeId2"]
    missing = [c for c in required if c not in maxi.columns]
    if missing:
        raise ValueError(f"Faltan columnas en maxitabla: {missing}")

    print("[2/6] Parseando interacciones proteína-RNA...")
    prot_contacts = build_protein_side_contacts(maxi)
    prot_contacts.to_csv(outdir / "protein_rna_ring_rows_from_maxitabla.csv", index=False)

    print("[3/6] Resumiendo conteos por residuo proteína...")
    counts = summarize_counts(prot_contacts)
    counts.to_csv(outdir / "protein_rna_ring_counts_from_maxitabla.csv", index=False)

    print("[4/6] Leyendo interfaz SASA...")
    interface_df = load_interface(args.interface_csv)

    print("[5/6] Mergeando SASA + conteos reales de RING...")
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

    print("[6/6] Escribiendo distribuciones...")
    with open(outdir / "ring_count_distributions.txt", "w", encoding="utf-8") as fh:
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


print("Total filas RING:", len(maxi))
print("Filas proteína-RNA detectadas:", len(prot_contacts))