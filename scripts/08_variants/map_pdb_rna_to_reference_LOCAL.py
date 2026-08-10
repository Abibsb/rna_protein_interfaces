#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
from Bio.Align import PairwiseAligner


def parse_fasta(path):
    records = {}
    header = None
    seqs = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    key = header.split("|")[0].strip()
                    records[key] = {
                        "header": header,
                        "sequence": "".join(seqs).upper().replace("T", "U")
                    }
                header = line[1:]
                seqs = []
            else:
                seqs.append(line)

        if header is not None:
            key = header.split("|")[0].strip()
            records[key] = {
                "header": header,
                "sequence": "".join(seqs).upper().replace("T", "U")
            }

    return records


def align_local_and_map(query_seq, ref_seq):
    aligner = PairwiseAligner()
    aligner.mode = "local"

    # Strict-ish scoring to avoid pathological gappy alignments
    aligner.match_score = 2
    aligner.mismatch_score = -2
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1

    aln = aligner.align(query_seq, ref_seq)[0]

    # aligned[0] = query blocks, aligned[1] = reference blocks
    q_blocks = aln.aligned[0]
    r_blocks = aln.aligned[1]

    rows = []
    matches = 0
    aligned_pairs = 0

    for (q_start, q_end), (r_start, r_end) in zip(q_blocks, r_blocks):
        q_len = q_end - q_start
        r_len = r_end - r_start

        # Only map ungapped aligned blocks of equal length
        if q_len != r_len:
            continue

        for i in range(q_len):
            q_pos = q_start + i + 1
            r_pos = r_start + i + 1
            q_nt = query_seq[q_start + i]
            r_nt = ref_seq[r_start + i]

            aligned_pairs += 1
            is_match = q_nt == r_nt
            if is_match:
                matches += 1

            rows.append({
                "pdb_seq_pos": q_pos,
                "pdb_nt": q_nt,
                "reference_pos": r_pos,
                "reference_nt": r_nt,
                "match_status": "match" if is_match else "mismatch",
            })

    if rows:
        map_df = pd.DataFrame(rows)
        min_ref = int(map_df["reference_pos"].min())
        max_ref = int(map_df["reference_pos"].max())
        ref_span = max_ref - min_ref + 1
        reference_span_density = aligned_pairs / ref_span if ref_span else 0
        min_query = int(map_df["pdb_seq_pos"].min())
        max_query = int(map_df["pdb_seq_pos"].max())
        query_span = max_query - min_query + 1
        query_span_density = aligned_pairs / query_span if query_span else 0
    else:
        map_df = pd.DataFrame(columns=[
            "pdb_seq_pos", "pdb_nt", "reference_pos",
            "reference_nt", "match_status"
        ])
        min_ref = max_ref = ref_span = None
        min_query = max_query = query_span = None
        reference_span_density = 0
        query_span_density = 0

    identity = matches / aligned_pairs if aligned_pairs else 0
    query_coverage = aligned_pairs / len(query_seq) if query_seq else 0
    reference_coverage = aligned_pairs / len(ref_seq) if ref_seq else 0

    stats = {
        "alignment_score": float(aln.score),
        "aligned_pairs": aligned_pairs,
        "identity": identity,
        "query_coverage": query_coverage,
        "reference_coverage": reference_coverage,
        "min_ref_pos": min_ref,
        "max_ref_pos": max_ref,
        "reference_span_density": reference_span_density,
        "min_query_pos": min_query,
        "max_query_pos": max_query,
        "query_span_density": query_span_density,
        "n_aligned_blocks": len(q_blocks),
    }

    if (
        identity >= 0.95
        and query_coverage >= 0.80
        and reference_span_density >= 0.80
        and query_span_density >= 0.95
    ):
        stats["mapping_quality"] = "high"
    elif (
        identity >= 0.90
        and query_coverage >= 0.50
        and reference_span_density >= 0.50
    ):
        stats["mapping_quality"] = "medium"
    else:
        stats["mapping_quality"] = "low"

    return map_df, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(args.targets)
    refs = parse_fasta(args.references)

    all_maps = []
    all_stats = []

    for _, row in targets.iterrows():
        gene_field = str(row.get("candidate_gene_symbol", "")).split("/")[0].strip()
        seq = str(row.get("sequence_acgu", "")).upper().replace("T", "U")

        if not gene_field or gene_field not in refs or not seq:
            continue

        ref = refs[gene_field]
        mapping, stats = align_local_and_map(seq, ref["sequence"])

        for col in [
            "unique_rna_id", "pdb_chains", "n_members", "target_name",
            "target_accession", "candidate_gene_symbol", "candidate_rna_name",
            "positional_tier"
        ]:
            mapping[col] = row.get(col, "")

        mapping["reference_key"] = gene_field
        mapping["reference_header"] = ref["header"]

        all_maps.append(mapping)

        stat_row = {
            "unique_rna_id": row.get("unique_rna_id", ""),
            "pdb_chains": row.get("pdb_chains", ""),
            "n_members": row.get("n_members", ""),
            "target_name": row.get("target_name", ""),
            "candidate_gene_symbol": row.get("candidate_gene_symbol", ""),
            "reference_key": gene_field,
            "reference_header": ref["header"],
        }
        stat_row.update(stats)
        all_stats.append(stat_row)

    if all_maps:
        pd.concat(all_maps, ignore_index=True).to_csv(
            outdir / "pdb_rna_to_reference_position_map_LOCAL.csv",
            index=False
        )
    else:
        pd.DataFrame().to_csv(
            outdir / "pdb_rna_to_reference_position_map_LOCAL.csv",
            index=False
        )

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(
        outdir / "pdb_rna_to_reference_alignment_stats_LOCAL.csv",
        index=False
    )

    print("Done.")
    print(outdir / "pdb_rna_to_reference_position_map_LOCAL.csv")
    print(outdir / "pdb_rna_to_reference_alignment_stats_LOCAL.csv")


if __name__ == "__main__":
    main()