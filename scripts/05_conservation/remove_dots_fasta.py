#!/usr/bin/env python3
import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq

def remove_dots_from_fasta(input_path, output_path):
    records = []
    for record in SeqIO.parse(input_path, "fasta"):
        # Remove all dots from the sequence
        clean_seq = str(record.seq).replace('.', '')
        record.seq = Seq(clean_seq)
        records.append(record)
    SeqIO.write(records, output_path, "fasta")

def main(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".fasta") or filename.endswith(".afa"):
            in_file = os.path.join(input_folder, filename)
            out_file = os.path.join(output_folder, filename)
            print(f"Processing {filename}...")
            remove_dots_from_fasta(in_file, out_file)

    print(f"✅ Done! Cleaned FASTA files saved in: {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_dots_fasta.py <input_folder> <output_folder>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

