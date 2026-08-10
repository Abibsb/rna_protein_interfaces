#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# USER SETTINGS
# ============================================================

CSV="/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/11_feb_26/rna_rfam_annotations.csv"
EV_CUTOFF="1e-5"

RFAM_SEED="/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/09_feb_26/rna_central_script/Rfam.seed"
RFAM_CM_DB="/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/09_feb_26/databases/Rfam/Rfam.cm"

QUERY_FASTA="/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/11_feb_26/rna_sequences.fasta"
OUTDIR="rfam_seed_shannon_out"

CMALIGN_MXSIZE="4096"

# ============================================================
# CHECKS
# ============================================================

command -v cmfetch >/dev/null 2>&1 || { echo "ERROR: cmfetch not found"; exit 1; }
command -v cmalign >/dev/null 2>&1 || { echo "ERROR: cmalign not found"; exit 1; }

[ -s "$CSV" ] || { echo "ERROR: CSV not found: $CSV"; exit 1; }
[ -s "$RFAM_SEED" ] || { echo "ERROR: Rfam.seed not found"; exit 1; }
[ -s "$RFAM_CM_DB" ] || { echo "ERROR: Rfam.cm not found"; exit 1; }
[ -s "$QUERY_FASTA" ] || { echo "ERROR: query fasta file not found: $QUERY_FASTA"; exit 1; }

mkdir -p "$OUTDIR"/{seed,cm,aligned,shannon,logs,tmp}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -f "$SCRIPT_DIR/shannon_from_stockholm_per_sequence.py" "$OUTDIR/"

# ============================================================
# STEP 1 — GET RFAMs PASSING EVALUE
# ============================================================

echo "[1/5] Finding RFAM families..."

RFAMS=$(CSV="$CSV" EV_CUTOFF="$EV_CUTOFF" python - <<'PY'
import pandas as pd, os
df=pd.read_csv(os.environ["CSV"])
ev=float(os.environ["EV_CUTOFF"])
keep=df[df["evalue"]<=ev]
print("\n".join(sorted(keep["rfam_acc"].dropna().unique())))
PY
)

[ -z "$RFAMS" ] && { echo "No families pass cutoff."; exit 0; }

# ============================================================
# STEP 2 — PROCESS EACH RFAM + EACH QUERY INDEPENDENTLY
# ============================================================

echo "[2/5] Processing families..."

for RF in $RFAMS; do
  echo "== $RF =="

  SEED_STO="$OUTDIR/seed/${RF}.seed.sto"
  CM_FILE="$OUTDIR/cm/${RF}.cm"

  # ----------------------------------------------------------
  # Extract RFAM seed alignment block (valid Stockholm)
  # ----------------------------------------------------------

  awk -v RF="$RF" '
  /^# STOCKHOLM 1.0/ {capture=1; buf=$0 "\n"; next}
  capture {buf = buf $0 "\n"}
  /^\/\// {
    if (buf ~ "#=GF AC[[:space:]]+"RF) {
        printf "%s", buf
        exit
    }
    capture=0; buf=""
  }' "$RFAM_SEED" > "$SEED_STO"

  [ -s "$SEED_STO" ] || { echo "Seed extraction failed for $RF"; continue; }

  # ----------------------------------------------------------
  # Fetch CM model
  # ----------------------------------------------------------

  RFAM_CM_DIR="$(dirname "$RFAM_CM_DB")"
  RFAM_CM_NAME="$(basename "$RFAM_CM_DB")"

  pushd "$RFAM_CM_DIR" >/dev/null
  cmfetch "$RFAM_CM_NAME" "$RF" > "$OLDPWD/$CM_FILE"
  popd >/dev/null

  [ -s "$CM_FILE" ] || { echo "cmfetch failed for $RF"; continue; }

  # ----------------------------------------------------------
  # Get query list for THIS RFAM
  # ----------------------------------------------------------

  QUERY_LIST=$(CSV="$CSV" RF="$RF" EV_CUTOFF="$EV_CUTOFF" python - <<'PY'
import pandas as pd, os
df=pd.read_csv(os.environ["CSV"])
rf=os.environ["RF"]
ev=float(os.environ["EV_CUTOFF"])
sub=df[(df["rfam_acc"]==rf)&(df["evalue"]<=ev)]
for q in sorted(sub["query"].dropna().unique()):
    print(q)
PY
)

  [ -z "$QUERY_LIST" ] && { echo "No queries for $RF"; continue; }

  # ==========================================================
  # 🔥 PROCESS EACH QUERY INDEPENDENTLY
  # ==========================================================

  for Q in $QUERY_LIST; do

    echo "   -> $Q"

    QFA="$OUTDIR/tmp/${RF}.${Q}.fasta"
    ALN_STO="$OUTDIR/aligned/${RF}.${Q}.seed_plus_query.sto"
    LOG="$OUTDIR/logs/${RF}.${Q}.cmalign.log"

    # Extract single FASTA entry
    awk -v q="$Q" '
      /^>/ {keep=($0~">"q"($|[[:space:]])")}
      keep {print}
    ' "$QUERY_FASTA" > "$QFA"

    [ -s "$QFA" ] || { echo "Missing FASTA for $Q"; continue; }

    cmalign --mxsize "$CMALIGN_MXSIZE" \
            -o "$ALN_STO" \
            --mapali "$SEED_STO" \
            "$CM_FILE" "$QFA" \
            > "$LOG" 2>&1

    [ -s "$ALN_STO" ] || { echo "cmalign failed for $Q"; continue; }

    python "$OUTDIR/shannon_from_stockholm_per_sequence.py" \
        --sto "$ALN_STO" \
        --csv_annotations "$CSV" \
        --rfam_acc "$RF" \
        --queries_fasta "$QFA" \
        --outdir "$OUTDIR/shannon" \
        --score_from seed

  done
done

# ============================================================
# STEP 3 — MERGE RESULTS
# ============================================================

echo "[3/5] Merging outputs..."

OUTDIR="$OUTDIR" python - <<'PY'
import os, glob, pandas as pd
outdir=os.environ["OUTDIR"]

parts=glob.glob(os.path.join(outdir,"shannon","*.mapped.csv"))
if not parts:
    print("No mapped outputs found.")
    raise SystemExit(0)

dfs=[pd.read_csv(p) for p in parts]
merged=pd.concat(dfs, ignore_index=True)
merged.to_csv(os.path.join(outdir,"shannon","ALL_queries.mapped_merged.csv"), index=False)
print("Merged:",len(merged))
PY

echo "[4/5] Done."
echo "[5/5] Finished."