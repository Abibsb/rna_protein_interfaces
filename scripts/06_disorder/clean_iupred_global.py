#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Clean parsed IUPred global dataset."
    )

    parser.add_argument(
        "input_csv",
        help="iupred_long_all_residues.csv"
    )

    parser.add_argument(
        "--output",
        default="iupred_long_all_residues.clean.csv"
    )

    parser.add_argument(
        "--min-chain-length",
        type=int,
        default=50
    )

    parser.add_argument(
        "--max-fraction-x",
        type=float,
        default=0.1
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Limpieza de IDs
    df["pdb_id"] = df["pdb_id"].astype(str).str.upper()
    df["chain_id"] = (
        df["chain_id"]
        .astype(str)
        .str.replace("]", "", regex=False)
        .str.replace("[", "", regex=False)
        .str.strip()
    )

    df["residue"] = df["residue"].astype(str).str.upper()
    df["iupred_score"] = pd.to_numeric(df["iupred_score"], errors="coerce")

    # Convertir booleano
    df["is_disordered"] = (
        df["is_disordered"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )

    df = df.dropna(subset=["iupred_score"]).copy()

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    # QC por cadena antes de remover X
    chain_qc = (
        df.groupby(["pdb_id", "chain_id"])
        .agg(
            raw_chain_length=("iupred_score", "size"),
            n_X=("residue", lambda x: (x == "X").sum()),
            fraction_X=("residue", lambda x: (x == "X").mean()),
        )
        .reset_index()
    )

    chain_qc["pass_x_filter"] = chain_qc["fraction_X"] <= args.max_fraction_x

    # Remover X y aminoácidos no estándar
    clean = df[df["residue"].isin(valid_aa)].copy()

    # Longitud válida por cadena después de remover X
    valid_lengths = (
        clean.groupby(["pdb_id", "chain_id"])
        .size()
        .reset_index(name="valid_chain_length")
    )

    chain_qc = chain_qc.merge(
        valid_lengths,
        on=["pdb_id", "chain_id"],
        how="left"
    )

    chain_qc["valid_chain_length"] = chain_qc["valid_chain_length"].fillna(0).astype(int)
    chain_qc["pass_length_filter"] = chain_qc["valid_chain_length"] >= args.min_chain_length
    chain_qc["keep_chain"] = chain_qc["pass_x_filter"] & chain_qc["pass_length_filter"]

    keep = chain_qc[chain_qc["keep_chain"]][["pdb_id", "chain_id"]]

    clean = clean.merge(
        keep.assign(keep_chain=True),
        on=["pdb_id", "chain_id"],
        how="left"
    )

    clean = clean[clean["keep_chain"] == True].copy()
    clean = clean.drop(columns=["keep_chain"])

    chain_qc.to_csv("iupred_chain_qc.csv", index=False)
    clean.to_csv(args.output, index=False)

    print("Done")
    print(f"Original rows: {len(df)}")
    print(f"Rows after removing non-standard residues and filtering chains: {len(clean)}")
    print(f"Original chains: {df[['pdb_id', 'chain_id']].drop_duplicates().shape[0]}")
    print(f"Kept chains: {clean[['pdb_id', 'chain_id']].drop_duplicates().shape[0]}")
    print(f"Output: {args.output}")
    print("QC table: iupred_chain_qc.csv")


if __name__ == "__main__":
    main()