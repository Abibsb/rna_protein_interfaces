#!/usr/bin/env python3

from pathlib import Path
import argparse
import re
import pandas as pd


def parse_filename(path):
    """
    Extrae PDBID, chain y modo desde nombres tipo:
    3ADL_chainA_long.txt
    """

    pattern = r"^(?P<pdb_id>[A-Za-z0-9]{4})_chain(?P<chain_id>.+?)_(?P<mode>long|short|glob)\.txt$"
    match = re.match(pattern, path.name)

    if match is None:
        return None, None, None

    pdb_id = match.group("pdb_id").upper()
    chain_id = match.group("chain_id")
    mode = match.group("mode")

    return pdb_id, chain_id, mode


def parse_iupred_file(path):
    pdb_id, chain_id, mode = parse_filename(path)

    if pdb_id is None:
        print(f"WARNING: nombre no reconocido, se saltea: {path.name}")
        return pd.DataFrame()

    rows = []

    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            try:
                pos = int(parts[0])
                residue = parts[1]
                score = float(parts[2])
            except ValueError:
                continue

            rows.append({
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "iupred_mode": mode,
                "iupred_pos": pos,
                "residue": residue,
                "iupred_score": score,
                "source_file": path.name,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Parse IUPred3 long output files into one CSV."
    )

    parser.add_argument(
        "--input-dir",
        default="/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/31_ago_25/iupred_per_chain/long/",
        help="Directorio con archivos *_long.txt de IUPred."
    )

    parser.add_argument(
        "--output",
        default="iupred_long_all_residues.csv",
        help="CSV de salida."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral para definir desorden. Default: 0.5"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    files = sorted(input_dir.glob("*_long.txt"))

    if not files:
        raise FileNotFoundError(f"No encontré archivos *_long.txt en {input_dir}")

    dfs = []

    for file in files:
        df = parse_iupred_file(file)

        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError("No se pudo parsear ningún archivo válido.")

    out = pd.concat(dfs, ignore_index=True)

    out["is_disordered"] = out["iupred_score"] > args.threshold

    out.to_csv(args.output, index=False)

    print("Listo.")
    print(f"Archivos leídos: {len(files)}")
    print(f"Residuos parseados: {len(out)}")
    print(f"PDBs únicos: {out['pdb_id'].nunique()}")
    print(f"Cadenas únicas: {out[['pdb_id', 'chain_id']].drop_duplicates().shape[0]}")
    print(f"Output: {args.output}")
    print()
    print("Resumen IUPred:")
    print(out["iupred_score"].describe())
    print()
    print("Fracción desordenada global:")
    print(out["is_disordered"].mean())


if __name__ == "__main__":
    main()