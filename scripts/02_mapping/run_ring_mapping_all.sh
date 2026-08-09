#!/usr/bin/env bash
set -uo pipefail

########################################
# USER-EDITABLE PATHS
########################################

# Folder containing all CIF files
CIF_DIR="/media/usuario/Seagate Basic1/Tesis_Maestria/27_ago_25/dataset_complejos_sapiens"

# Main script
SCRIPT="/media/usuario/Seagate Basic1/Tesis_Maestria/23_feb_26/ring_map_to_uniprot_ENTITYPOLY_v4_relaxed.py"

# Shared input tables
RING_CSV="/media/usuario/Seagate Basic1/Tesis_Maestria/01_feb_26/maxitabla_dataset_actualizado.csv"
UNIPROT_MAP="/media/usuario/Seagate Basic1/Tesis_Maestria/01_feb_26/pdb_uniprot_position_map.csv"

# Output folder
OUT_DIR="/media/usuario/Seagate Basic1/Tesis_Maestria/23_feb_26/ring_outputs"

########################################
# SETUP
########################################

mkdir -p "$OUT_DIR"

LOG_OK="$OUT_DIR/run_ok.log"
LOG_FAIL="$OUT_DIR/run_failed.log"

echo "=== RING mapping batch started ===" > "$LOG_OK"
echo "=== RING mapping failures ===" > "$LOG_FAIL"

########################################
# MAIN LOOP
########################################

for cif in "$CIF_DIR"/*.cif; do

    pdb=$(basename "$cif" .cif)
    out_csv="$OUT_DIR/${pdb}_RING_interface_residues_with_uniprot.csv"

    # Skip if already processed
    if [[ -f "$out_csv" ]]; then
        echo "✔ Skipping $pdb (already processed)" | tee -a "$LOG_OK"
        continue
    fi

    echo "▶ Processing $pdb"

    if python "$SCRIPT" \
        --cif "$cif" \
        --ring "$RING_CSV" \
        --uniprot_map "$UNIPROT_MAP" \
        --out_prefix "$OUT_DIR/$pdb"
    then
        echo "✔ SUCCESS: $pdb" | tee -a "$LOG_OK"
    else
        echo "❌ FAILED: $pdb" | tee -a "$LOG_FAIL"
    fi

done

echo "=== RING mapping batch finished ===" >> "$LOG_OK"
