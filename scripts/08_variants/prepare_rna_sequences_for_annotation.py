#!/usr/bin/env python3

import argparse
from pathlib import Path
from collections import defaultdict
import hashlib
import re
import pandas as pd


MODIFIED_NT_MAP = {
    # Common modified RNA residues converted to canonical bases when possible
    "A": "A", "C": "C", "G": "G", "U": "U", "T": "U",
    "I": "G",  # inosine approximation, imperfect but useful for search
    "N": "N",
}


def parse_fasta(path):
    header = None
    seq_chunks = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)

        if header is not None:
            yield header, "".join(seq_chunks)


def standardize_seq(seq):
    seq = seq.upper().replace("T", "U")
    cleaned = []
    for char in seq:
        if char in {"A", "C", "G", "U"}:
            cleaned.append(char)
        else:
            cleaned.append("N")
    return "".join(cleaned)


def is_low_complexity(seq):
    if not seq:
        return True

    n = len(seq)
    counts = {b: seq.count(b) for b in "ACGUN"}
    max_fraction = max(counts.values()) / n

    # Mostly one nucleotide, e.g. AAAAAAAA
    if max_fraction >= 0.80:
        return True

    # Very N-rich
    if counts["N"] / n >= 0.20:
        return True

    # Simple repeats
    for k in [1, 2, 3]:
        motif = seq[:k]
        if motif * (n // k) + motif[: n % k] == seq:
            return True

    return False


def parse_header(header):
    """
    Expected header examples:
    >1A9N_Q RNA chain (deduplicated, model 0)
    """
    first = header.split()[0]
    if "_" in first:
        pdb_id, chain_id = first.split("_", 1)
    else:
        pdb_id, chain_id = first[:4], ""

    return pdb_id.upper(), chain_id


def md5_short(seq):
    return hashlib.md5(seq.encode()).hexdigest()[:12]


def write_fasta(records, path):
    with open(path, "w") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Clean, filter, and deduplicate RNA sequences extracted from mmCIF/PDB."
    )
    parser.add_argument("--input", required=True, help="Input RNA FASTA")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--min-length", type=int, default=15, help="Minimum sequence length to keep")
    parser.add_argument("--rfam-min-length", type=int, default=25, help="Minimum length for Rfam/cmscan FASTA")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    seq_to_members = defaultdict(list)

    for header, raw_seq in parse_fasta(args.input):
        pdb_id, chain_id = parse_header(header)
        seq_std = standardize_seq(raw_seq)
        length = len(seq_std)

        low_complexity = is_low_complexity(seq_std)
        too_short = length < args.min_length
        usable = (not low_complexity) and (not too_short)

        rows.append({
            "original_header": header,
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "raw_sequence": raw_seq,
            "sequence_acgu": seq_std,
            "length": length,
            "n_A": seq_std.count("A"),
            "n_C": seq_std.count("C"),
            "n_G": seq_std.count("G"),
            "n_U": seq_std.count("U"),
            "n_N": seq_std.count("N"),
            "gc_fraction": (seq_std.count("G") + seq_std.count("C")) / length if length else 0,
            "low_complexity": low_complexity,
            "too_short": too_short,
            "usable_for_annotation": usable,
        })

        if usable:
            seq_to_members[seq_std].append((header, pdb_id, chain_id))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "rna_sequences_cleaned_all_chains.csv", index=False)

    unique_rows = []
    fasta_records_all = []
    fasta_records_rfam = []

    for idx, (seq, members) in enumerate(sorted(seq_to_members.items(), key=lambda x: (-len(x[1]), -len(x[0]))), start=1):
        seq_id = f"RNAuniq_{idx:05d}_{md5_short(seq)}"
        pdb_chains = ";".join([f"{pdb}_{chain}" for _, pdb, chain in members])

        unique_rows.append({
            "unique_rna_id": seq_id,
            "sequence_acgu": seq,
            "length": len(seq),
            "n_members": len(members),
            "pdb_chains": pdb_chains,
            "example_header": members[0][0],
            "gc_fraction": (seq.count("G") + seq.count("C")) / len(seq) if seq else 0,
        })

        fasta_records_all.append((f"{seq_id} n_members={len(members)}", seq))

        if len(seq) >= args.rfam_min_length:
            fasta_records_rfam.append((f"{seq_id} n_members={len(members)}", seq))

    unique_df = pd.DataFrame(unique_rows)
    unique_df.to_csv(outdir / "rna_unique_sequences.csv", index=False)

    write_fasta(fasta_records_all, outdir / "rna_unique_sequences.fasta")
    write_fasta(fasta_records_rfam, outdir / "rna_unique_sequences_for_rfam.fasta")

    summary = pd.DataFrame([{
        "n_total_chains": len(df),
        "n_usable_chains": int(df["usable_for_annotation"].sum()),
        "n_too_short": int(df["too_short"].sum()),
        "n_low_complexity": int(df["low_complexity"].sum()),
        "n_unique_usable_sequences": len(unique_df),
        "n_unique_for_rfam": len(fasta_records_rfam),
        "min_length": args.min_length,
        "rfam_min_length": args.rfam_min_length,
    }])

    summary.to_csv(outdir / "summary_sequence_preannotation.csv", index=False)

    print("Done.")
    print(f"Total chains: {len(df)}")
    print(f"Usable chains: {int(df['usable_for_annotation'].sum())}")
    print(f"Unique usable sequences: {len(unique_df)}")
    print(f"Unique sequences for Rfam: {len(fasta_records_rfam)}")
    print("\nOutputs:")
    print(outdir / "rna_sequences_cleaned_all_chains.csv")
    print(outdir / "rna_unique_sequences.csv")
    print(outdir / "rna_unique_sequences.fasta")
    print(outdir / "rna_unique_sequences_for_rfam.fasta")
    print(outdir / "summary_sequence_preannotation.csv")


if __name__ == "__main__":
    main()