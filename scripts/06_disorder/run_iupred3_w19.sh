#!/usr/bin/env bash

FASTA_DIR="/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/31_ago_25/pdb_fastas_protein"
OUTPUT_DIR="/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/31_ago_25/iupred_per_chain"
IUPRED="/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/04_jun_25/iupred3/iupred3.py"

MIN_LEN=19

mkdir -p "$OUTPUT_DIR/long" "$OUTPUT_DIR/short" "$OUTPUT_DIR/glob"
echo -e "file\tlength" > "$OUTPUT_DIR/skipped_short_chains.tsv"

run_iupred () {
    mode=$1
    outdir=$2

    for f in "$FASTA_DIR"/*.fasta; do
        len=$(awk '
            /^>/ {next}
            {l += length($0)}
            END {print l}
        ' "$f")

        if [ "$len" -lt "$MIN_LEN" ]; then
            echo -e "$(basename "$f")\t$len" >> "$OUTPUT_DIR/skipped_short_chains.tsv"
            continue
        fi

        python3 "$IUPRED" "$f" "$mode" > "$outdir/$(basename "${f%.fasta}_${mode}.txt")"
    done
}

run_iupred long  "$OUTPUT_DIR/long"
run_iupred short "$OUTPUT_DIR/short"
run_iupred glob  "$OUTPUT_DIR/glob"
