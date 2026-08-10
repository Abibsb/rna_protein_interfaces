#!/usr/bin/env bash

mkdir -p pdb_fastas

while read -r pdb; do
    pdb_lower=$(echo "$pdb" | tr 'A-Z' 'a-z')
    wget -q \
        -O "pdb_fastas/${pdb}.fasta.gz" \
        "ftp://ftp.rcsb.org/pub/pdb/derived_data/pdb_seqres_fasta/pdb${pdb_lower}.ent.gz"
done < pdb_ids.txt

gunzip pdb_fastas/*.gz
