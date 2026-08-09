#!/usr/bin/env python3
"""
map_iupred_to_ring_interface_ENTITYPOLY.py

Take:
- IUPred output for ONE protein sequence (from your per-chain FASTA split)
- mmCIF structure
- RING interface table (maxitabla_dataset_actualizado.csv OR .cif_ringEdges parsed style)
and produce:
- disorder annotation for residues that are in the RNA-protein interface
BUT: interface membership is defined ONLY by RING.

Key improvement:
Instead of trusting the user-provided chain label (which may be CIF label_asym_id),
we map the input sequence to CIF _entity_poly and extract the auth/PDB chain IDs
(e.g. "B,D"). Then we run the mapping separately for each auth chain and write
separate output files.

Example:
  python3 map_iupred_to_ring_interface_ENTITYPOLY.py \
    --iupred 1A9N_chainD]_long.txt \
    --cif 1A9N.cif \
    --ring maxitabla_dataset_actualizado.csv \
    --out_prefix 1A9N_U2B_interface_disorder \
    --cutoff 0.5

Outputs:
  1A9N_U2B_interface_disorder_authB.csv
  1A9N_U2B_interface_disorder_authD.csv
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import gemmi


def read_iupred(iupred_txt):
    """
    Parse IUPred output.
    Returns dict:
      seq (str)
      scores: {pos(int 1-based): float}
      aas: {pos: aa}
    """
    scores = {}
    aas = {}
    seq = []
    with open(iupred_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # typical: pos aa score
            if len(parts) < 3:
                continue
            if not parts[0].isdigit():
                continue
            pos = int(parts[0])
            aa = parts[1].upper()
            try:
                score = float(parts[2])
            except ValueError:
                continue
            scores[pos] = score
            aas[pos] = aa
            seq.append(aa)
    if not scores:
        raise RuntimeError(f"Could not parse IUPred file: {iupred_txt}")
    return {"seq": "".join(seq), "scores": scores, "aas": aas}


def safe_row_get(row, colname):
    try:
        return row[colname]
    except Exception:
        return ""


def get_entity_poly_entries(cif_path):
    doc = gemmi.cif.read_file(cif_path)
    block = doc.sole_block()

    cat = block.find_mmcif_category("_entity_poly")
    if cat is None:
        return []

    entries = []
    for row in cat:
        entity_id = safe_row_get(row, "entity_id")
        poly_type = safe_row_get(row, "type")

        seqs = []
        for col in ("pdbx_seq_one_letter_code_can", "pdbx_seq_one_letter_code"):
            s = safe_row_get(row, col)
            if s:
                seqs.append(s.replace("\n", "").replace(" ", "").strip().upper())

        chains = ""
        for col in ("pdbx_strand_id", "pdbx_target_identifier"):
            v = safe_row_get(row, col)
            if v:
                chains = v.strip()
                break

        entries.append({
            "entity_id": entity_id,
            "poly_type": poly_type,
            "seqs": [s for s in seqs if s],
            "chains": chains,
        })

    return entries


def best_sliding_identity(shorter, longer):
    if not shorter or not longer:
        return 0.0, 0
    if len(shorter) > len(longer):
        shorter, longer = longer, shorter

    m = len(shorter)
    best = 0.0
    best_off = 0
    for off in range(0, len(longer) - m + 1):
        win = longer[off:off + m]
        matches = sum(1 for i in range(m) if shorter[i] == win[i])
        pid = matches / m * 100.0
        if pid > best:
            best = pid
            best_off = off
    return best, best_off


def map_seq_to_entity_poly_chains(query_seq, cif_path, require_polypeptide=True):
    """
    Return list of auth/PDB chain IDs (strings) for the best matching entity.
    """
    entities = get_entity_poly_entries(cif_path)
    if not entities:
        raise RuntimeError("No _entity_poly entries found in CIF.")

    scored = []
    for ent in entities:
        if not ent["seqs"]:
            continue
        if require_polypeptide and "polypeptide" not in ent["poly_type"].lower():
            continue

        best_pid = 0.0
        best_seq = ""
        for s in ent["seqs"]:
            shorter, longer = (query_seq, s) if len(query_seq) <= len(s) else (s, query_seq)
            pid, _off = best_sliding_identity(shorter, longer)
            if pid > best_pid:
                best_pid = pid
                best_seq = s

        contains = (query_seq in best_seq) or (best_seq in query_seq)

        scored.append((contains, best_pid, ent["chains"], ent["entity_id"], ent["poly_type"]))

    if not scored:
        raise RuntimeError("No candidate entity_poly entries matched the query sequence.")

    scored.sort(reverse=True)
    contains, pid, chains, entity_id, poly_type = scored[0]

    if not chains:
        raise RuntimeError(f"Best entity match (entity_id={entity_id}) has no chains listed in CIF.")

    chain_list = [c.strip() for c in chains.split(",") if c.strip()]
    return chain_list, {"entity_id": entity_id, "poly_type": poly_type, "best_pid": pid, "contains": contains, "chains": chains}


def parse_ring_table(ring_csv_path, pdbid_filter=None):
    """
    Parse your maxitabla_dataset_actualizado.csv-like file.

    Returns:
      iface_res_by_chain[pdbid][auth_chain] = set(resseq_int)

    Interface definition:
      - KEEP LIG_SC and LIG_MC (RNA–protein)
      - DROP LIG_LIG
    """
    iface = defaultdict(lambda: defaultdict(set))

    with open(ring_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t") if "\t" in open(ring_csv_path).readline() else csv.DictReader(f)

        for row in reader:
            pdbid = row.get("PDBID") or row.get("pdbid") or row.get("Pdbid")
            if not pdbid:
                continue
            pdbid = pdbid.strip().upper()
            if pdbid_filter and pdbid != pdbid_filter:
                continue

            n1 = (row.get("NodeId1") or "").strip()
            n2 = (row.get("NodeId2") or "").strip()
            inter = (row.get("Interaction") or "").strip()

            if not n1 or not n2 or not inter:
                continue

            # ----------------------------------------
            # NEW: Interaction filtering
            # ----------------------------------------
            # Interaction examples:
            #   VDW:LIG_SC
            #   HBOND:LIG_MC
            #   VDW:LIG_LIG

            itype = inter.split(":")[-1] if ":" in inter else ""

            # Accept ONLY RNA–protein contacts
            if itype not in {"LIG_SC", "LIG_MC"}:
                continue

            # ----------------------------------------
            # Parse Node IDs
            # ----------------------------------------
            m1 = re.match(r"^([A-Za-z0-9]+):(\d+):_:(\w+)$", n1)
            m2 = re.match(r"^([A-Za-z0-9]+):(\d+):_:(\w+)$", n2)
            if not (m1 and m2):
                continue

            chain1, res1, name1 = m1.group(1), int(m1.group(2)), m1.group(3)
            chain2, res2, name2 = m2.group(1), int(m2.group(2)), m2.group(3)

            # crude AA/nuc test
            is_aa = len(name1) == 3
            is_nuc = len(name2) == 1

            if is_aa and is_nuc:
                iface[pdbid][chain1].add(res1)

    return iface


def write_output(out_csv, pdbid, auth_chain, iface_residues, iupred_data, cutoff):
    """
    Write per-interface residue disorder scores.
    """
    seq = iupred_data["seq"]
    scores = iupred_data["scores"]

    rows = []
    for resi in sorted(iface_residues):
        # resi in RING is residue number (auth seq id)
        # We do NOT have direct mapping from resi -> iupred position if numbering differs.
        # BUT in your dataset, you split sequences from RCSB FASTA, and IUPred is 1..N.
        # RING residue numbers are PDB residue numbers, usually starting at 1.
        # We'll try direct mapping: position == resi.
        pos = resi
        score = scores.get(pos, None)
        disorder = (score is not None and score >= cutoff)

        rows.append({
            "PDBID": pdbid,
            "auth_chain": auth_chain,
            "ring_resi": resi,
            "iupred_pos": pos,
            "iupred_score": score if score is not None else "",
            "is_disordered": int(disorder) if score is not None else "",
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["PDBID","auth_chain","ring_resi","iupred_pos","iupred_score","is_disordered"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(iupred_txt, cif_path, ring_csv, out_prefix, cutoff, pdbid=None):
    pdbid = pdbid or os.path.basename(cif_path).split(".")[0].upper()

    iup = read_iupred(iupred_txt)
    query_seq = iup["seq"]

    auth_chains, info = map_seq_to_entity_poly_chains(query_seq, cif_path, require_polypeptide=True)
    print(f"ℹ️ Best entity_poly match: entity_id={info['entity_id']} chains={info['chains']} pid={info['best_pid']:.1f} contains={info['contains']}")

    iface = parse_ring_table(ring_csv, pdbid_filter=pdbid)

    for ch in auth_chains:
        iface_res = iface.get(pdbid, {}).get(ch, set())
        out_csv = f"{out_prefix}_auth{ch}.csv"
        write_output(out_csv, pdbid, ch, iface_res, iup, cutoff)
        print(f"✅ Wrote: {out_csv}")
        print(f"ℹ️ Interface residues from RING for {pdbid} auth chain {ch}: {len(iface_res)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iupred", required=True, help="IUPred output file (*_long.txt)")
    ap.add_argument("--cif", required=True, help="mmCIF file")
    ap.add_argument("--ring", required=True, help="RING maxitabla_dataset_actualizado.csv")
    ap.add_argument("--out_prefix", required=True, help="Prefix for output CSV files")
    ap.add_argument("--cutoff", type=float, default=0.5, help="IUPred disorder cutoff (default 0.5)")
    ap.add_argument("--pdbid", default=None, help="Override PDBID (default derived from CIF filename)")
    args = ap.parse_args()

    main(args.iupred, args.cif, args.ring, args.out_prefix, args.cutoff, args.pdbid)
