#!/usr/bin/env python3
"""
shannon_from_stockholm.py

Purpose
-------
Given a Stockholm alignment produced by Infernal (typically: RFAM SEED + 1 query),
compute per-alignment-column Shannon-based conservation, then map those column
scores onto each query's ungapped positions (query_pos).

Option 1 support (your choice):
-------------------------------
Add `--score_from seed` to compute conservation using ONLY the RFAM seed sequences
(i.e., excluding the query sequence(s) listed in --queries_fasta). This makes the
conservation independent of your query (and independent across queries).

Default remains `--score_from all` for backwards compatibility.

Outputs
-------
1) <outdir>/<rfam_acc>.per_column.csv
2) <outdir>/<rfam_acc>.<query>.mapped.csv (one per query in --queries_fasta)
"""

import argparse
import math
from collections import Counter, defaultdict
import os
import pandas as pd


# ------------------------------------------------------------
# I/O PARSERS
# ------------------------------------------------------------

def read_stockholm(sto_path: str):
    """
    Parse a Stockholm alignment into:
      seqs: dict[name] -> full aligned sequence string
      aln_len: alignment length (all sequences must match)
    Notes:
      - Skips comment/annotation lines starting with '#'
      - Stops at the Stockholm terminator '//'
      - Supports multi-block Stockholm (concatenates fragments per sequence name)
    """
    seqs = defaultdict(list)
    with open(sto_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.strip() == "//":
                break

            parts = line.split()
            if len(parts) < 2:
                continue
            name, frag = parts[0], parts[1]
            seqs[name].append(frag)

    seqs = {k: "".join(v) for k, v in seqs.items()}
    if not seqs:
        raise ValueError(f"No sequences parsed from {sto_path}")

    lengths = {len(v) for v in seqs.values()}
    if len(lengths) != 1:
        raise ValueError(f"Alignment lengths differ: {sorted(lengths)}")

    aln_len = len(next(iter(seqs.values())))
    return seqs, aln_len


def fasta_headers(fa_path: str):
    """
    Return sequence IDs from a FASTA file (first token after '>').
    These are treated as the query IDs you want to map.
    """
    names = []
    with open(fa_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                names.append(line[1:].strip().split()[0])
    return names


# ------------------------------------------------------------
# CONSERVATION METRIC
# ------------------------------------------------------------

def shannon_conservation(col_chars):
    """
    Compute Shannon-based conservation for one alignment column.

    We ignore gaps ('-' and '.') and ignore ambiguous characters.
    We treat T as U to be robust to DNA-like alphabets.

    Returns:
      cons (float|None): 1 - H/Hmax over A,C,G,U
                         None if no usable nucleotides in column
      n (int): number of usable A/C/G/U characters contributing to score
    """
    filtered = []
    for c in col_chars:
        c = c.upper()
        if c in "-.":
            continue
        if c == "T":
            c = "U"
        if c in ("A", "C", "G", "U"):
            filtered.append(c)

    n = len(filtered)
    if n == 0:
        return None, 0

    counts = Counter(filtered)
    H = 0.0
    for v in counts.values():
        p = v / n
        H -= p * math.log2(p)

    Hmax = 2.0  # log2(4)
    cons = 1.0 - (H / Hmax)
    return cons, n


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sto", required=True, help="Stockholm alignment from cmalign")
    ap.add_argument("--csv_annotations", required=True, help="rna_rfam_annotations.csv")
    ap.add_argument("--rfam_acc", required=True, help="RFxxxxx")
    ap.add_argument("--queries_fasta", required=True,
                    help="FASTA containing the query sequence(s) included in this alignment")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--score_from", choices=["all", "seed"], default="all",
                    help="Compute conservation from: "
                         "'all' = seed+query (legacy), "
                         "'seed' = seed only (Option 1, query-independent).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load alignment
    seqs, aln_len = read_stockholm(args.sto)

    # Query IDs to map (from FASTA headers)
    query_ids = sorted(set(fasta_headers(args.queries_fasta)))

    # Decide which sequences define conservation
    if args.score_from == "seed":
        # Seed = everything in Stockholm that's NOT one of the query IDs
        score_names = [n for n in seqs.keys() if n not in set(query_ids)]
    else:
        # Legacy: score from all sequences in Stockholm (seed + queries)
        score_names = list(seqs.keys())

    if not score_names:
        raise ValueError(
            "No sequences available to score conservation. "
            "If using --score_from seed, ensure the Stockholm file contains RFAM seed "
            "sequences in addition to your query."
        )

    # Precompute per-column conservation + QC stats
    cons_by_col = [None] * aln_len
    gapfrac_by_col = [0.0] * aln_len
    n_nongap_by_col = [0] * aln_len

    denom = len(score_names)
    for i in range(aln_len):
        col = [seqs[n][i] for n in score_names]
        gaps = sum(1 for c in col if c in "-.")
        gapfrac_by_col[i] = gaps / denom
        cons, n = shannon_conservation(col)
        cons_by_col[i] = cons
        n_nongap_by_col[i] = n

    # Write per-column table
    percol_path = os.path.join(args.outdir, f"{args.rfam_acc}.per_column.csv")
    with open(percol_path, "w", encoding="utf-8") as out:
        out.write("rfam_acc,aln_col,shannon_conservation,gap_fraction,n_nongap_ACGU\n")
        for i in range(aln_len):
            c = cons_by_col[i]
            c_str = "" if c is None else f"{c:.6f}"
            out.write(
                f"{args.rfam_acc},{i+1},{c_str},{gapfrac_by_col[i]:.6f},{n_nongap_by_col[i]}\n"
            )

    # Load annotations for joining metadata (pdb_id, chain_id, evalue, score)
    ann = pd.read_csv(args.csv_annotations)
    ann = ann[ann["rfam_acc"] == args.rfam_acc].copy()
    ann = ann.drop_duplicates(subset=["query"], keep="first")

    # Map each query: alignment columns -> query positions (ungapped)
    for q in query_ids:
        if q not in seqs:
            # If Infernal changed the name, you'll see it here (rare but possible).
            continue

        aligned = seqs[q]
        pos = 0
        rows = []

        for i, ch in enumerate(aligned):
            if ch in "-.":
                continue
            pos += 1
            nt = ch.upper().replace("T", "U")
            cons = cons_by_col[i]

            rows.append({
                "rfam_acc": args.rfam_acc,
                "query": q,
                "query_pos": pos,
                "nt": nt,
                "aln_col": i + 1,
                "shannon_conservation": cons,
                "gap_fraction": gapfrac_by_col[i],
                "n_nongap_ACGU": n_nongap_by_col[i],
            })

        df = pd.DataFrame(rows)
        if df.empty:
            continue

        # Attach metadata if present
        if not ann.empty and "query" in ann.columns:
            df = df.merge(
                ann[["query", "pdb_id", "chain_id", "evalue", "score"]],
                on="query",
                how="left",
            )

        out_path = os.path.join(args.outdir, f"{args.rfam_acc}.{q}.mapped.csv")
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()