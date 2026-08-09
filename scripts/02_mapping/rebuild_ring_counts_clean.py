#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--maxitabla", required=True)
    p.add_argument("--interface_csv", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def read_maxitabla(path):
    df = pd.read_csv(path, sep="\t", index_col=False)

    needed = {"PDBID", "NodeId1", "NodeId2"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"No pude leer correctamente maxitabla. Columnas detectadas: {list(df.columns)}"
        )
    return df


def parse_node(node):
    if pd.isna(node):
        return None

    parts = str(node).strip().split(":")
    # Esperado: chain:resnum:insertion:resname
    if len(parts) < 4:
        return None

    chain = parts[0]
    try:
        resnum = int(parts[1])
    except Exception:
        return None
    resname = parts[3].upper().strip()

    return chain, resnum, resname


def infer_polymer_type(resname):
    aa3 = {
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        "SEC","PYL","ASX","GLX","XLE"
    }
    nt = {"A","C","G","U","T","I"}

    if resname in aa3:
        return "protein"
    if resname in nt:
        return "nucleotide"
    return "unknown"


def build_counts(maxi):
    records = []

    skipped_parse = 0
    skipped_unknown = 0
    skipped_non_pr = 0

    for _, row in maxi.iterrows():
        pdb = str(row["PDBID"]).upper().strip()

        n1 = parse_node(row["NodeId1"])
        n2 = parse_node(row["NodeId2"])

        if n1 is None or n2 is None:
            skipped_parse += 1
            continue

        c1, r1, res1 = n1
        c2, r2, res2 = n2

        t1 = infer_polymer_type(res1)
        t2 = infer_polymer_type(res2)

        if t1 == "unknown" or t2 == "unknown":
            skipped_unknown += 1
            continue

        if t1 == "protein" and t2 == "nucleotide":
            prot_chain, prot_res = c1, r1
        elif t2 == "protein" and t1 == "nucleotide":
            prot_chain, prot_res = c2, r2
        else:
            skipped_non_pr += 1
            continue

        records.append((pdb, prot_chain, prot_res))

    df = pd.DataFrame(records, columns=["pdb_id", "auth_asym_id", "auth_seq_id"])

    debug = {
        "total_rows": len(maxi),
        "parsed_rows": len(df),
        "skipped_parse": skipped_parse,
        "skipped_unknown_resname": skipped_unknown,
        "skipped_non_protein_rna": skipped_non_pr,
    }

    return df, debug


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1] loading maxitabla")
    maxi = read_maxitabla(args.maxitabla)
    print(maxi.head(3).to_string())

    print("[2] building counts")
    contacts, debug = build_counts(maxi)

    print("DEBUG:")
    for k, v in debug.items():
        print(k, v)

    print("[3] counting per residue")
    counts = (
        contacts
        .groupby(["pdb_id", "auth_asym_id", "auth_seq_id"])
        .size()
        .reset_index(name="ring_interaction_count")
    )
    counts.to_csv(outdir / "ring_counts_per_residue.csv", index=False)

    print("[4] merging with SASA")
    sasa = pd.read_csv(args.interface_csv)
    sasa["pdb_id"] = sasa["pdb_id"].astype(str).str.upper().str.strip()
    sasa["auth_asym_id"] = sasa["auth_asym_id"].astype(str).str.strip()
    sasa["auth_seq_id"] = pd.to_numeric(sasa["auth_seq_id"], errors="coerce")
    sasa = sasa.dropna(subset=["auth_seq_id"]).copy()
    sasa["auth_seq_id"] = sasa["auth_seq_id"].astype(int)

    merged = sasa.merge(
        counts,
        how="left",
        on=["pdb_id", "auth_asym_id", "auth_seq_id"]
    )
    merged["ring_interaction_count"] = merged["ring_interaction_count"].fillna(0).astype(int)
    merged.to_csv(outdir / "merged_final.csv", index=False)

    dist = merged["ring_interaction_count"].value_counts().sort_index()

    with open(outdir / "ring_count_distribution.txt", "w", encoding="utf-8") as fh:
        fh.write("DEBUG\n")
        fh.write("=====\n")
        for k, v in debug.items():
            fh.write(f"{k}: {v}\n")
        fh.write("\nring_interaction_count distribution\n")
        fh.write("==================================\n")
        fh.write(dist.to_string())
        fh.write("\n")

    print("[5] distribution")
    print(dist.head(30))
    print("DONE")
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()