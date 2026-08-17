#!/usr/bin/env python3
"""
Prepare AlphaFold/ChimeraX case folders using:
  - canonical UniProt WT/MUT protein sequence when available
  - observed mmCIF protein-chain WT/MUT sequence when valid
  - local RNA fragment extracted from mmCIF around RNA residues proximal to the candidate residue

Designed for RNA-protein variant candidates where candidates are defined by:
  pdb_id, uniprot_id, uniprot_pos, wild_type, mutated_type, protein_chain/auth_asym_id

Outputs one folder per candidate with FASTA files, metadata tables, and a ChimeraX template.

Requirements:
  conda install -c conda-forge biopython pandas numpy
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1

PROTEIN_3TO1 = {k.upper(): v for k, v in protein_letters_3to1.items()}
# Common modified residues seen in PDB/mmCIF. Extend if needed.
PROTEIN_3TO1.update({
    "MSE": "M", "SEC": "U", "PYL": "O", "HSD": "H", "HSE": "H", "HSP": "H",
})

RNA_CODES = {
    "A": "A", "C": "C", "G": "G", "U": "U", "I": "I",
    "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U", "URI": "U",
    # frequent modified bases -> approximate canonical identity where obvious
    "PSU": "U", "H2U": "U", "5MU": "U", "4SU": "U", "OMC": "C", "OMG": "G",
    "1MA": "A", "2MG": "G", "M2G": "G", "7MG": "G", "5MC": "C", "5MU": "U",
}


def clean_id(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def aa3to1(resname: str) -> str:
    return PROTEIN_3TO1.get(str(resname).strip().upper(), "X")


def nt3to1(resname: str) -> str:
    return RNA_CODES.get(str(resname).strip().upper(), "N")


def is_protein_residue(res) -> bool:
    hetflag, _, _ = res.id
    if hetflag != " ":
        return False
    return aa3to1(res.get_resname()) != "X"


def is_rna_residue(res) -> bool:
    name = str(res.get_resname()).strip().upper()
    return name in RNA_CODES


def read_fasta(path: Path) -> Dict[str, str]:
    records = {}
    if not path or not path.exists():
        return records
    header = None
    seq_parts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records[header] = "".join(seq_parts).replace(" ", "").upper()
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        records[header] = "".join(seq_parts).replace(" ", "").upper()
    return records


def write_fasta(path: Path, header: str, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")


def mutate_sequence(seq: str, pos_1based: int, wt: str, mut: str) -> Tuple[str, str]:
    obs = seq[pos_1based - 1]
    if obs != wt:
        raise ValueError(f"Expected {wt} at sequence position {pos_1based}, observed {obs}")
    return seq[:pos_1based - 1] + mut + seq[pos_1based:], obs


def find_cif(cif_root: Path, pdb_id: str) -> Optional[Path]:
    pdb = pdb_id.upper()
    patterns = [f"{pdb}.cif", f"{pdb.lower()}.cif", f"{pdb}.mmcif", f"{pdb.lower()}.mmcif"]
    for pat in patterns:
        matches = list(cif_root.rglob(pat))
        if matches:
            # Prefer not Raw Data if multiple? Keep shortest path as stable heuristic.
            return sorted(matches, key=lambda p: (len(str(p)), str(p)))[0]
    return None


def get_model(cif_path: Path):
    parser = MMCIFParser(QUIET=True)
    st = parser.get_structure(cif_path.stem.upper(), str(cif_path))
    return next(st.get_models())


def extract_protein_chain(model, chain_id: str):
    if chain_id not in [c.id for c in model]:
        return None, pd.DataFrame()
    chain = model[chain_id]
    seq = []
    rows = []
    for res in chain.get_residues():
        if not is_protein_residue(res):
            continue
        hetflag, resseq, icode = res.id
        aa = aa3to1(res.get_resname())
        seq.append(aa)
        rows.append({
            "seq_index": len(seq),
            "auth_seq_id": int(resseq),
            "icode": str(icode).strip() if str(icode).strip() else "",
            "resname": res.get_resname(),
            "aa": aa,
        })
    return "".join(seq), pd.DataFrame(rows)


def residue_min_distance(res1, res2) -> float:
    min_d = math.inf
    for a in res1.get_atoms():
        c1 = a.coord
        for b in res2.get_atoms():
            d = float(np.linalg.norm(c1 - b.coord))
            if d < min_d:
                min_d = d
    return min_d


def rna_chain_sequence_and_map(chain) -> Tuple[str, pd.DataFrame]:
    seq = []
    rows = []
    for res in chain.get_residues():
        if not is_rna_residue(res):
            continue
        hetflag, resseq, icode = res.id
        nt = nt3to1(res.get_resname())
        seq.append(nt)
        rows.append({
            "rna_seq_index": len(seq),
            "rna_chain": chain.id,
            "rna_auth_seq_id": int(resseq),
            "rna_icode": str(icode).strip() if str(icode).strip() else "",
            "rna_resname": res.get_resname(),
            "rna_nt": nt,
        })
    return "".join(seq), pd.DataFrame(rows)


def extract_rna_fragment_near_residue(model, protein_res, flank: int = 20, distance_cutoff: float = 5.0):
    contact_rows = []
    chain_maps = {}
    chain_seqs = {}

    for chain in model:
        rna_seq, rna_map = rna_chain_sequence_and_map(chain)
        if not rna_seq:
            continue
        chain_maps[chain.id] = rna_map
        chain_seqs[chain.id] = rna_seq
        for res in chain.get_residues():
            if not is_rna_residue(res):
                continue
            d = residue_min_distance(protein_res, res)
            if d <= distance_cutoff:
                hetflag, resseq, icode = res.id
                # get rna_seq_index
                hit = rna_map[rna_map["rna_auth_seq_id"] == int(resseq)]
                if hit.empty:
                    continue
                idx = int(hit.iloc[0]["rna_seq_index"])
                contact_rows.append({
                    "rna_chain": chain.id,
                    "rna_auth_seq_id": int(resseq),
                    "rna_seq_index": idx,
                    "rna_resname": res.get_resname(),
                    "rna_nt": nt3to1(res.get_resname()),
                    "min_atom_distance_A": d,
                })

    contacts = pd.DataFrame(contact_rows).sort_values(["min_atom_distance_A", "rna_chain", "rna_auth_seq_id"]) if contact_rows else pd.DataFrame()
    fragments = []

    if contacts.empty:
        return contacts, pd.DataFrame(), {}

    for rna_chain, grp in contacts.groupby("rna_chain"):
        seq = chain_seqs[rna_chain]
        rmap = chain_maps[rna_chain]
        min_idx = int(grp["rna_seq_index"].min())
        max_idx = int(grp["rna_seq_index"].max())
        start = max(1, min_idx - flank)
        end = min(len(seq), max_idx + flank)
        frag_seq = seq[start - 1:end]
        frag_map = rmap[(rmap["rna_seq_index"] >= start) & (rmap["rna_seq_index"] <= end)].copy()
        fragments.append({
            "rna_chain": rna_chain,
            "fragment_start_seq_index": start,
            "fragment_end_seq_index": end,
            "fragment_length": len(frag_seq),
            "n_contact_nts_in_fragment": len(grp),
            "fragment_sequence": frag_seq,
        })

    frag_df = pd.DataFrame(fragments).sort_values(["n_contact_nts_in_fragment", "fragment_length"], ascending=[False, True])
    fragment_sequences = {row["rna_chain"]: row["fragment_sequence"] for _, row in frag_df.iterrows()}
    return contacts, frag_df, fragment_sequences


def load_canonical_sequence(uniprot_id: str, canonical_fasta: Optional[Path], canonical_dir: Optional[Path]) -> Optional[Tuple[str, str]]:
    # Returns header, seq
    if canonical_fasta and canonical_fasta.exists():
        for h, s in read_fasta(canonical_fasta).items():
            if uniprot_id in h or h.split()[0] == uniprot_id:
                return h, s
    if canonical_dir and canonical_dir.exists():
        candidates = [
            canonical_dir / f"{uniprot_id}.fasta",
            canonical_dir / f"{uniprot_id}.fa",
            canonical_dir / f"{uniprot_id}.faa",
        ]
        candidates += list(canonical_dir.rglob(f"*{uniprot_id}*.fasta"))[:10]
        candidates += list(canonical_dir.rglob(f"*{uniprot_id}*.fa"))[:10]
        for p in candidates:
            if p.exists():
                recs = read_fasta(p)
                if recs:
                    h, s = next(iter(recs.items()))
                    return h, s
    return None


def choose_mapping_row(mapping: pd.DataFrame, cand: pd.Series) -> Optional[pd.Series]:
    pdb = clean_id(cand.get("pdb_id")).upper()
    uniprot = clean_id(cand.get("uniprot_id"))
    pos = int(float(cand.get("uniprot_pos")))
    chain_candidate = clean_id(cand.get("protein_chain")) or clean_id(cand.get("auth_asym_id")) or clean_id(cand.get("label_asym_id"))

    sub = mapping[
        (mapping["pdb_id"].astype(str).str.upper() == pdb) &
        (mapping["uniprot_id"].astype(str) == uniprot) &
        (pd.to_numeric(mapping["uniprot_pos"], errors="coerce") == pos)
    ].copy()
    if sub.empty:
        return None
    if chain_candidate and "auth_asym_id" in sub.columns:
        sub2 = sub[sub["auth_asym_id"].astype(str) == chain_candidate]
        if not sub2.empty:
            sub = sub2
    return sub.iloc[0]


def sanitize_case_id(pdb_id: str, uniprot_id: str, pos: str, wt: str, mut: str) -> str:
    raw = f"{pdb_id}_{uniprot_id}_{wt}{pos}{mut}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Scored candidate CSV")
    ap.add_argument("--mapping", required=True, help="pdb_uniprot_position_map.csv")
    ap.add_argument("--cif-root", required=True, help="Root directory to search for mmCIF files")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--flank-nt", type=int, default=20)
    ap.add_argument("--distance-cutoff", type=float, default=5.0)
    ap.add_argument("--canonical-fasta", default=None, help="Optional multi-FASTA with canonical UniProt sequences")
    ap.add_argument("--canonical-fasta-dir", default=None, help="Optional directory with UniProt FASTAs")
    ap.add_argument("--only-complete", action="store_true", help="Keep only candidates where observed PDB aa matches variant WT")
    ap.add_argument("--variant-id", default=None, help="Optional exact variant_id to process")
    ap.add_argument("--pdb-id", default=None, help="Optional PDB id filter")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates, low_memory=False)
    mapping = pd.read_csv(args.mapping, low_memory=False)
    cif_root = Path(args.cif_root)
    canonical_fasta = Path(args.canonical_fasta) if args.canonical_fasta else None
    canonical_dir = Path(args.canonical_fasta_dir) if args.canonical_fasta_dir else None

    # Harmonize candidate column names from raw or scored files
    rename = {
        "Wild_Type": "wild_type",
        "Mutated_Type": "mutated_type",
        "auth_asym_id": "protein_chain",
    }
    for old, new in rename.items():
        if old in candidates.columns and new not in candidates.columns:
            candidates[new] = candidates[old]

    if args.variant_id:
        candidates = candidates[candidates["variant_id"].astype(str) == args.variant_id]
    if args.pdb_id:
        candidates = candidates[candidates["pdb_id"].astype(str).str.upper() == args.pdb_id.upper()]

    # Sort by score if present
    if "alphafold_candidate_score" in candidates.columns:
        candidates = candidates.sort_values("alphafold_candidate_score", ascending=False)
    elif "delta_sasa" in candidates.columns:
        candidates = candidates.sort_values("delta_sasa", ascending=False)

    results = []
    processed = 0

    for _, cand in candidates.iterrows():
        if processed >= args.top_n:
            break
        required = ["pdb_id", "uniprot_id", "uniprot_pos", "wild_type", "mutated_type", "variant_id"]
        if any(c not in candidates.columns for c in required):
            raise SystemExit(f"Missing required columns. Need: {required}. Found: {list(candidates.columns)}")

        pdb_id = clean_id(cand["pdb_id"]).upper()
        uniprot_id = clean_id(cand["uniprot_id"])
        uniprot_pos = int(float(cand["uniprot_pos"]))
        wt = clean_id(cand["wild_type"]).upper()
        mut = clean_id(cand["mutated_type"]).upper()
        variant_id = clean_id(cand["variant_id"])

        maprow = choose_mapping_row(mapping, cand)
        if maprow is None:
            results.append({"variant_id": variant_id, "pdb_id": pdb_id, "status": "NO_MAPPING"})
            continue
        chain_id = clean_id(maprow["auth_asym_id"])
        auth_seq_id = int(float(maprow["auth_seq_id"]))

        cif = find_cif(cif_root, pdb_id)
        if cif is None:
            results.append({"variant_id": variant_id, "pdb_id": pdb_id, "status": "NO_CIF"})
            continue

        try:
            model = get_model(cif)
            protein_seq, prot_map = extract_protein_chain(model, chain_id)
            if not protein_seq:
                results.append({"variant_id": variant_id, "pdb_id": pdb_id, "status": "NO_PROTEIN_CHAIN", "chain": chain_id})
                continue
            residue_hit = prot_map[prot_map["auth_seq_id"] == auth_seq_id]
            if residue_hit.empty:
                results.append({"variant_id": variant_id, "pdb_id": pdb_id, "status": "PROTEIN_RES_NOT_FOUND", "chain": chain_id, "auth_seq_id": auth_seq_id})
                continue
            residue_hit = residue_hit.iloc[0]
            seq_index = int(residue_hit["seq_index"])
            observed_aa = clean_id(residue_hit["aa"])
            observed_matches = (observed_aa == wt)

            if args.only_complete and not observed_matches:
                results.append({"variant_id": variant_id, "pdb_id": pdb_id, "status": "SKIPPED_OBSERVED_MISMATCH", "observed_aa": observed_aa, "wt": wt})
                continue

            # find protein residue object
            chain = model[chain_id]
            protein_res = None
            for res in chain.get_residues():
                hetflag, resseq, icode = res.id
                if hetflag == " " and int(resseq) == auth_seq_id:
                    protein_res = res
                    break
            if protein_res is None:
                results.append({"variant_id": variant_id, "pdb_id": pdb_id, "status": "PROTEIN_RES_OBJ_NOT_FOUND"})
                continue

            contacts, frag_df, frag_sequences = extract_rna_fragment_near_residue(
                model, protein_res, flank=args.flank_nt, distance_cutoff=args.distance_cutoff
            )
            if contacts.empty:
                results.append({"variant_id": variant_id, "pdb_id": pdb_id, "status": "NO_RNA_NEARBY", "observed_aa": observed_aa, "wt": wt})
                continue

            case_id = sanitize_case_id(pdb_id, uniprot_id, str(uniprot_pos), wt, mut)
            case_dir = outdir / case_id
            seq_dir = case_dir / "sequences"
            meta_dir = case_dir / "metadata"
            exp_dir = case_dir / "experimental"
            af_dir = case_dir / "alphafold_inputs"
            cx_dir = case_dir / "chimerax"
            for d in [seq_dir, meta_dir, exp_dir, af_dir, cx_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # Symlink/copy note for experimental cif
            (exp_dir / "cif_path.txt").write_text(str(cif) + "\n")

            # Write CIF WT and maybe CIF MUT
            write_fasta(seq_dir / "protein_cif_WT.fasta", f"{pdb_id}_{chain_id}_CIF_WT_{uniprot_id}_{wt}{uniprot_pos}", protein_seq)
            cif_mut_written = False
            if observed_matches:
                protein_cif_mut, _ = mutate_sequence(protein_seq, seq_index, wt, mut)
                write_fasta(seq_dir / "protein_cif_MUT.fasta", f"{pdb_id}_{chain_id}_CIF_MUT_{uniprot_id}_{wt}{uniprot_pos}{mut}", protein_cif_mut)
                cif_mut_written = True

            # Write canonical WT/MUT if available
            canon = load_canonical_sequence(uniprot_id, canonical_fasta, canonical_dir)
            canon_status = "NOT_PROVIDED"
            canon_mut_written = False
            if canon is not None:
                ch, cseq = canon
                write_fasta(seq_dir / "protein_canonical_WT.fasta", f"{uniprot_id}_CANON_WT", cseq)
                if len(cseq) >= uniprot_pos and cseq[uniprot_pos - 1] == wt:
                    cseq_mut, _ = mutate_sequence(cseq, uniprot_pos, wt, mut)
                    write_fasta(seq_dir / "protein_canonical_MUT.fasta", f"{uniprot_id}_CANON_MUT_{wt}{uniprot_pos}{mut}", cseq_mut)
                    canon_status = "OK"
                    canon_mut_written = True
                else:
                    obs_can = cseq[uniprot_pos - 1] if len(cseq) >= uniprot_pos else "OUT_OF_RANGE"
                    canon_status = f"WT_MISMATCH_OR_OUT_OF_RANGE observed={obs_can} expected={wt}"

            # Write RNA fragments. If multiple RNA chains, write each. AF jobs can choose one or combine.
            contacts.to_csv(meta_dir / "rna_contacts_by_distance.tsv", sep="\t", index=False)
            frag_df.to_csv(meta_dir / "rna_fragments.tsv", sep="\t", index=False)
            prot_map.to_csv(meta_dir / "protein_cif_residue_map.tsv", sep="\t", index=False)
            pd.DataFrame([maprow.to_dict()]).to_csv(meta_dir / "pdb_uniprot_mapping_row.tsv", sep="\t", index=False)

            for rna_chain, frag_seq in frag_sequences.items():
                write_fasta(seq_dir / f"rna_fragment_chain_{rna_chain}.fasta", f"{pdb_id}_RNA_chain_{rna_chain}_fragment_flank{args.flank_nt}", frag_seq)

            # Choose best/first RNA chain for input templates
            primary_rna_chain = frag_df.iloc[0]["rna_chain"]
            primary_rna_file = seq_dir / f"rna_fragment_chain_{primary_rna_chain}.fasta"

            def make_af_job(job_name: str, protein_fasta: str):
                job = af_dir / job_name
                job.mkdir(exist_ok=True)
                (job / "README.txt").write_text(
                    "Use these sequences as separate chains in AlphaFold/AlphaFold Server/AF3.\n"
                    f"Protein FASTA: ../../sequences/{protein_fasta}\n"
                    f"RNA fragment FASTA: ../../sequences/{primary_rna_file.name}\n"
                    "Keep the RNA sequence identical across WT/MUT comparisons.\n"
                )

            make_af_job("CIF_WT", "protein_cif_WT.fasta")
            if cif_mut_written:
                make_af_job("CIF_MUT", "protein_cif_MUT.fasta")
            if canon is not None:
                make_af_job("CANON_WT", "protein_canonical_WT.fasta")
                if canon_mut_written:
                    make_af_job("CANON_MUT", "protein_canonical_MUT.fasta")

            # Write metadata summary
            meta = {
                "case_id": case_id,
                "variant_id": variant_id,
                "pdb_id": pdb_id,
                "cif_path": str(cif),
                "uniprot_id": uniprot_id,
                "uniprot_pos": uniprot_pos,
                "wild_type": wt,
                "mutated_type": mut,
                "auth_asym_id": chain_id,
                "auth_seq_id": auth_seq_id,
                "cif_seq_index": seq_index,
                "observed_pdb_aa": observed_aa,
                "observed_matches_variant_wt": observed_matches,
                "cif_mut_written": cif_mut_written,
                "canonical_status": canon_status,
                "canonical_mut_written": canon_mut_written,
                "primary_rna_chain": str(primary_rna_chain),
                "n_rna_contacts_distance_based": int(len(contacts)),
                "flank_nt": args.flank_nt,
                "distance_cutoff_A": args.distance_cutoff,
            }
            # Add selected original candidate cols
            for col in ["alphafold_candidate_score", "is_core_hotspot", "ring_interaction_count", "rna_unique_nucleotides", "delta_sasa", "buried_fraction_residue", "consurf_score", "iupred_score"]:
                if col in cand.index:
                    val = cand[col]
                    try:
                        if pd.isna(val):
                            val = None
                    except Exception:
                        pass
                    meta[col] = val.item() if hasattr(val, "item") else val
            (meta_dir / "case_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            pd.DataFrame([meta]).to_csv(meta_dir / "case_metadata.tsv", sep="\t", index=False)

            # ChimeraX template
            (cx_dir / "compare_template.cxc").write_text(
                f"# ChimeraX template for {case_id}\n"
                f"open {cif}\n"
                "# open ../alphafold_outputs/CANON_WT/model.cif\n"
                "# open ../alphafold_outputs/CANON_MUT/model.cif\n"
                "# open ../alphafold_outputs/CIF_WT/model.cif\n"
                "# open ../alphafold_outputs/CIF_MUT/model.cif\n"
                "hide atoms\nshow cartoons\n"
                f"select /{chain_id}:{auth_seq_id}\nshow sel atoms\nstyle sel stick\ncolor sel red\n"
                f"# Distance-based RNA contacts were written to metadata/rna_contacts_by_distance.tsv\n"
                "# After opening AF models, use matchmaker to align, then inspect local interface.\n"
            )

            results.append({**meta, "status": "OK", "case_dir": str(case_dir)})
            processed += 1
        except Exception as e:
            results.append({"variant_id": variant_id, "pdb_id": pdb_id, "status": "ERROR", "error": repr(e)})
            continue

    summary = pd.DataFrame(results)
    summary.to_csv(outdir / "prepared_cases_summary.tsv", sep="\t", index=False)
    print(f"Wrote summary: {outdir / 'prepared_cases_summary.tsv'}")
    if not summary.empty:
        cols = [c for c in ["status", "case_id", "variant_id", "pdb_id", "uniprot_id", "uniprot_pos", "wild_type", "mutated_type", "observed_pdb_aa", "observed_matches_variant_wt", "n_rna_contacts_distance_based", "case_dir", "error"] if c in summary.columns]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
