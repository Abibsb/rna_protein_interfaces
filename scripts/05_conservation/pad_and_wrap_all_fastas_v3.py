import os
import textwrap
from pathlib import Path
from datetime import datetime

input_folder = "per_sequence_output"
output_folder = "fastas_padded_v3"
log_file = "padding_log.txt"
os.makedirs(output_folder, exist_ok=True)

def read_fasta(filepath):
    """Reads fasta/MSA file safely, even with wrapped lines or blank spaces."""
    sequences = {}
    header = None
    seq_lines = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    sequences[header] = "".join(seq_lines).replace(" ", "").replace("\r", "")
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header:
            sequences[header] = "".join(seq_lines).replace(" ", "").replace("\r", "")
    return sequences

def pad_and_wrap_fasta(infile, outfile, log):
    sequences = read_fasta(infile)
    lengths = [len(seq) for seq in sequences.values()]
    max_len = max(lengths)
    unique_lengths = sorted(set(lengths))
    min_len = min(lengths)
    consistent = len(unique_lengths) == 1

    if not consistent:
        msg = f"⚠️  {infile.name}: sequence lengths vary ({unique_lengths[:10]}...)"
        print(msg)
        log.write(f"{msg}\n")
    else:
        msg = f"✅ {infile.name}: all sequences = {max_len} residues"
        print(msg)
        log.write(f"{msg}\n")

    with open(outfile, "w") as out:
        for header, seq in sequences.items():
            padded = seq.ljust(max_len, "-")
            wrapped = "\n".join(textwrap.wrap(padded, width=60))
            out.write(f"{header}\n{wrapped}\n")

    log.write(f"  → Saved as: {outfile.name}\n")
    log.write(f"  Total sequences: {len(sequences)}\n")
    log.write(f"  Min length: {min_len}, Max length: {max_len}\n")
    log.write(f"  {'-'*70}\n")

def main():
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "w") as log:
        log.write(f"Padding and wrapping log — started {start_time}\n")
        log.write("="*70 + "\n\n")

        for file in Path(input_folder).glob("*"):
            if file.suffix.lower() in [".fasta", ".fa", ".aln", ".msa"]:
                outpath = Path(output_folder) / f"{file.stem}_padded_wrapped.aln"
                pad_and_wrap_fasta(file, outpath, log)

        log.write("\nProcess complete.\n")

if __name__ == "__main__":
    main()

