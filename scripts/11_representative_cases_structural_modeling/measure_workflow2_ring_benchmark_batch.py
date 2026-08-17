#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import math
import re
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser


RNA_BASES = {"A", "U", "G", "C"}

AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Workflow 2: evaluate AlphaFold WT/MUT against experimental RING residue-RNA contacts."
    )
    ap.add_argument("--root", required=True, help="alphafold_fragment_workflow_wf1 root")
    ap.add_argument("--case-ids-file", required=True)
    ap.add_argument("--maxitabla", required=True, help="RING maxitabla_dataset_actualizado.csv/tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--af-protein-chain", default="A")
    ap.add_argument("--af-rna-chain", default="B")
    ap.add_argument("--tolerance", type=float, default=1.0)
    return ap.parse_args()


def read_fasta(path):
    header = None
    seq = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
            else:
                seq.append(line.upper())
    return header, "".join(seq)


def parse_node(node):
    parts = str(node).strip().split(":")
    if len(parts) < 4:
        return None
    try:
        resnum = int(parts[1])
    except Exception:
        return None
    return {
        "chain": parts[0],
        "resnum": resnum,
        "icode": parts[2],
        "resname": parts[3],
    }


def is_rna_node(node):
    p = parse_node(node)
    return p is not None and p["resname"] in RNA_BASES


def atom_distance(a1, a2):
    return math.sqrt(sum((a1.coord[i] - a2.coord[i]) ** 2 for i in range(3)))


def get_model(cif_path, name):
    parser = MMCIFParser(QUIET=True)
    st = parser.get_structure(name, str(cif_path))
    return next(st.get_models())


def get_chain(model, chain_id):
    for ch in model:
        if ch.id == str(chain_id):
            return ch
    raise ValueError(f"Chain {chain_id} not found. Available: {[c.id for c in model]}")


def get_residue_by_number(chain, resnum):
    for res in chain:
        hetflag, rnum, icode = res.id
        if hetflag == " " and int(rnum) == int(resnum):
            return res
    raise ValueError(f"Residue {resnum} not found in chain {chain.id}")


def min_distance_residue_to_rna_nt(cif_path, model_name, protein_chain_id, protein_resnum, rna_chain_id, rna_resnum):
    model = get_model(cif_path, model_name)

    p_chain = get_chain(model, protein_chain_id)
    r_chain = get_chain(model, rna_chain_id)

    p_res = get_residue_by_number(p_chain, protein_resnum)
    r_res = get_residue_by_number(r_chain, rna_resnum)

    min_d = float("inf")
    closest = None

    for pa in p_res.get_atoms():
        for ra in r_res.get_atoms():
            d = atom_distance(pa, ra)
            if d < min_d:
                min_d = d
                closest = {
                    "protein_resname": p_res.get_resname(),
                    "protein_atom": pa.name,
                    "rna_resname": r_res.get_resname(),
                    "rna_atom": ra.name,
                }

    return min_d, closest


def read_case_core_metadata(case_dir):
    case_dir = Path(case_dir)

    maprow = pd.read_csv(case_dir / "metadata" / "pdb_uniprot_mapping_row.tsv", sep="\t").iloc[0]

    pdb_id = str(maprow["pdb_id"]).upper()
    protein_chain_exp = str(maprow["auth_asym_id"])
    protein_resnum_exp = int(maprow["auth_seq_id"])

    resmap = pd.read_csv(case_dir / "metadata" / "protein_cif_residue_map.tsv", sep="\t")

    if "auth_seq_id" not in resmap.columns or "seq_index" not in resmap.columns:
        raise ValueError(
            f"Bad protein_cif_residue_map.tsv columns in {case_dir}: {resmap.columns.tolist()}"
        )

    hit = resmap[
        pd.to_numeric(resmap["auth_seq_id"], errors="coerce") == protein_resnum_exp
    ].copy()

    if hit.empty:
        raise ValueError(f"Cannot map protein residue {protein_chain_exp}:{protein_resnum_exp} in {case_dir}")

    aa1 = str(hit.iloc[0].get("aa", "")).strip()
    resname3 = str(hit.iloc[0].get("resname", "")).strip()

    if not resname3 or resname3 == "nan":
        resname3 = AA1_TO_AA3.get(aa1, "")

    if not resname3:
        raise ValueError(f"Cannot determine 3-letter residue name for {case_dir}")

    af_protein_resnum = int(pd.to_numeric(hit.iloc[0]["seq_index"], errors="coerce"))

    exp_cif = Path((case_dir / "experimental" / "cif_path.txt").read_text().strip())

    wt_models = sorted((case_dir / "alphafold_selected").glob("CIF_WT_best_model_*.cif"))
    mut_models = sorted((case_dir / "alphafold_selected").glob("CIF_MUT_best_model_*.cif"))

    if not wt_models or not mut_models:
        raise FileNotFoundError(f"Missing selected AF models in {case_dir}/alphafold_selected")

    ring_node = f"{protein_chain_exp}:{protein_resnum_exp}:_:{resname3}"

    return {
        "pdb_id": pdb_id,
        "protein_chain_exp": protein_chain_exp,
        "protein_resnum_exp": protein_resnum_exp,
        "protein_resname_exp": resname3,
        "ring_node": ring_node,
        "af_protein_resnum": af_protein_resnum,
        "exp_cif": exp_cif,
        "wt_cif": wt_models[0],
        "mut_cif": mut_models[0],
    }


def extract_ring_contacts_linewise(maxitabla, pdb_id, ring_node):
    rows = []

    with open(maxitabla, errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")

        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < len(header):
                parts = parts + [""] * (len(header) - len(parts))

            row = dict(zip(header, parts))

            row_pdb = str(row.get("PDBID", "")).strip().replace("\ufeff", "").upper()
            if row_pdb != pdb_id.upper():
                continue

            n1 = str(row.get("NodeId1", "")).strip()
            n2 = str(row.get("NodeId2", "")).strip()

            if n1 != ring_node and n2 != ring_node:
                continue

            if n1 == ring_node and is_rna_node(n2):
                rna = parse_node(n2)
                rows.append({
                    "PDBID": row_pdb,
                    "protein_ring_node": n1,
                    "rna_ring_node": n2,
                    "exp_rna_chain": rna["chain"],
                    "exp_rna_resnum": rna["resnum"],
                    "exp_rna_nt": rna["resname"],
                    "ring_interaction": row.get("Interaction", ""),
                    "ring_distance_A": pd.to_numeric(row.get("Distance", ""), errors="coerce"),
                    "ring_angle": pd.to_numeric(row.get("Angle", ""), errors="coerce"),
                    "ring_energy": pd.to_numeric(row.get("Energy", ""), errors="coerce"),
                    "ring_atom_protein": row.get("Atom1", ""),
                    "ring_atom_rna": row.get("Atom2", ""),
                    "ring_donor": row.get("Donor", ""),
                })

            elif n2 == ring_node and is_rna_node(n1):
                rna = parse_node(n1)
                rows.append({
                    "PDBID": row_pdb,
                    "protein_ring_node": n2,
                    "rna_ring_node": n1,
                    "exp_rna_chain": rna["chain"],
                    "exp_rna_resnum": rna["resnum"],
                    "exp_rna_nt": rna["resname"],
                    "ring_interaction": row.get("Interaction", ""),
                    "ring_distance_A": pd.to_numeric(row.get("Distance", ""), errors="coerce"),
                    "ring_angle": pd.to_numeric(row.get("Angle", ""), errors="coerce"),
                    "ring_energy": pd.to_numeric(row.get("Energy", ""), errors="coerce"),
                    "ring_atom_protein": row.get("Atom2", ""),
                    "ring_atom_rna": row.get("Atom1", ""),
                    "ring_donor": row.get("Donor", ""),
                })

    return pd.DataFrame(rows)


def get_ordered_rna_residues_from_cif(cif_path, chain_id):
    model = get_model(cif_path, "experimental")
    chain = get_chain(model, chain_id)

    rows = []
    for res in chain:
        hetflag, rnum, icode = res.id
        resname = res.get_resname().strip()

        if hetflag != " ":
            continue

        if resname in RNA_BASES:
            rows.append({
                "exp_rna_chain": chain_id,
                "exp_rna_resnum": int(rnum),
                "nt": resname,
            })

    if not rows:
        raise ValueError(f"No standard RNA residues found in chain {chain_id} of {cif_path}")

    return rows


def map_ring_nt_to_af_fragment(case_dir, exp_cif_path, ring_df):
    case_dir = Path(case_dir)

    fasta_files = sorted((case_dir / "sequences").glob("rna_fragment_chain_*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No rna_fragment_chain_*.fasta found in {case_dir / 'sequences'}")

    ring_chains = sorted(ring_df["exp_rna_chain"].dropna().astype(str).unique())

    mappings = []

    for ring_chain in ring_chains:
        matching_fastas = [f for f in fasta_files if f"chain_{ring_chain}.fasta" in f.name]

        if matching_fastas:
            fasta_path = matching_fastas[0]
        elif len(fasta_files) == 1:
            fasta_path = fasta_files[0]
        else:
            continue

        header, frag_seq = read_fasta(fasta_path)

        exp_rows = get_ordered_rna_residues_from_cif(exp_cif_path, ring_chain)
        full_seq = "".join(r["nt"] for r in exp_rows)

        starts = []
        start = full_seq.find(frag_seq)
        while start != -1:
            starts.append(start)
            start = full_seq.find(frag_seq, start + 1)

        if not starts:
            raise ValueError(
                f"Could not find RNA fragment sequence inside experimental chain {ring_chain} for {case_dir}"
            )

        target_resnums = set(
            ring_df.loc[ring_df["exp_rna_chain"] == ring_chain, "exp_rna_resnum"]
            .astype(int)
            .tolist()
        )

        chosen_start = None
        for s in starts:
            window = exp_rows[s:s + len(frag_seq)]
            window_resnums = {r["exp_rna_resnum"] for r in window}
            if target_resnums.issubset(window_resnums):
                chosen_start = s
                break

        if chosen_start is None:
            chosen_start = starts[0]

        window = exp_rows[chosen_start:chosen_start + len(frag_seq)]

        for i, r in enumerate(window, start=1):
            mappings.append({
                "exp_rna_chain": ring_chain,
                "exp_rna_resnum": int(r["exp_rna_resnum"]),
                "exp_rna_nt_mapped": r["nt"],
                "af_rna_chain": "B",
                "af_rna_resnum": i,
                "rna_fragment_fasta": str(fasta_path),
            })

    return pd.DataFrame(mappings)


def classify_case(n_mapped, wt_preserved, mut_preserved):
    if n_mapped == 0:
        return "ring_nt_not_in_af_fragment"
    if wt_preserved == 0:
        return "AF_WT_not_recovered"
    if mut_preserved == 0:
        return "strong_ring_contact_loss"
    if mut_preserved < wt_preserved:
        return "partial_ring_contact_loss"
    return "no_ring_benchmark_loss"


def main():
    args = parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    case_ids = [
        line.strip()
        for line in Path(args.case_ids_file).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    all_rows = []
    summaries = []
    failures = []

    for case_id in case_ids:
        case_dir = root / case_id
        print(f"\n### {case_id}")

        try:
            meta = read_case_core_metadata(case_dir)

            ring_df = extract_ring_contacts_linewise(
                args.maxitabla,
                meta["pdb_id"],
                meta["ring_node"],
            )

            if ring_df.empty:
                summaries.append({
                    "case_id": case_id,
                    "pdb_id": meta["pdb_id"],
                    "ring_node": meta["ring_node"],
                    "workflow2_class": "no_experimental_ring_contacts",
                    "n_ring_rows": 0,
                    "n_ring_rows_mapped_to_af": 0,
                    "n_wt_preserved": 0,
                    "n_mut_preserved": 0,
                    "n_lost_rows": 0,
                    "tolerance_A": args.tolerance,
                })
                print("No RING contacts found.")
                continue

            mapping = map_ring_nt_to_af_fragment(case_dir, meta["exp_cif"], ring_df)

            bench = ring_df.merge(
                mapping,
                on=["exp_rna_chain", "exp_rna_resnum"],
                how="left",
            )

            mapped = bench.dropna(subset=["af_rna_resnum"]).copy()

            case_rows = []

            for _, row in mapped.iterrows():
                af_rna_resnum = int(row["af_rna_resnum"])
                ring_distance = float(row["ring_distance_A"])
                threshold = ring_distance + args.tolerance

                wt_min, wt_closest = min_distance_residue_to_rna_nt(
                    meta["wt_cif"],
                    "AF_WT",
                    args.af_protein_chain,
                    meta["af_protein_resnum"],
                    args.af_rna_chain,
                    af_rna_resnum,
                )

                mut_min, mut_closest = min_distance_residue_to_rna_nt(
                    meta["mut_cif"],
                    "AF_MUT",
                    args.af_protein_chain,
                    meta["af_protein_resnum"],
                    args.af_rna_chain,
                    af_rna_resnum,
                )

                wt_preserved = wt_min <= threshold
                mut_preserved = mut_min <= threshold

                case_rows.append({
                    "case_id": case_id,
                    "pdb_id": meta["pdb_id"],
                    "ring_node": meta["ring_node"],
                    "protein_af_resnum": meta["af_protein_resnum"],
                    "rna_ring_node": row["rna_ring_node"],
                    "ring_interaction": row["ring_interaction"],
                    "ring_distance_A": ring_distance,
                    "preservation_threshold_A": threshold,
                    "exp_rna_chain": row["exp_rna_chain"],
                    "exp_rna_resnum": int(row["exp_rna_resnum"]),
                    "exp_rna_nt": row["exp_rna_nt"],
                    "af_rna_chain": args.af_rna_chain,
                    "af_rna_resnum": af_rna_resnum,
                    "af_rna_nt": row["exp_rna_nt_mapped"],
                    "AF_WT_min_distance_to_ring_nt_A": round(wt_min, 3),
                    "AF_WT_closest_pair": (
                        f"{wt_closest['protein_resname']}:{wt_closest['protein_atom']}--"
                        f"{wt_closest['rna_resname']}:{wt_closest['rna_atom']}"
                    ),
                    "AF_WT_preserves_ring_contact": wt_preserved,
                    "AF_MUT_min_distance_to_ring_nt_A": round(mut_min, 3),
                    "AF_MUT_closest_pair": (
                        f"{mut_closest['protein_resname']}:{mut_closest['protein_atom']}--"
                        f"{mut_closest['rna_resname']}:{mut_closest['rna_atom']}"
                    ),
                    "AF_MUT_preserves_ring_contact": mut_preserved,
                    "ring_contact_lost_in_mutant": wt_preserved and not mut_preserved,
                })

            case_df = pd.DataFrame(case_rows)
            all_rows.extend(case_rows)

            n_mapped = len(case_df)
            wt_preserved = int(case_df["AF_WT_preserves_ring_contact"].sum()) if n_mapped else 0
            mut_preserved = int(case_df["AF_MUT_preserves_ring_contact"].sum()) if n_mapped else 0
            lost_rows = int(case_df["ring_contact_lost_in_mutant"].sum()) if n_mapped else 0

            workflow2_class = classify_case(n_mapped, wt_preserved, mut_preserved)

            summaries.append({
                "case_id": case_id,
                "pdb_id": meta["pdb_id"],
                "ring_node": meta["ring_node"],
                "workflow2_class": workflow2_class,
                "n_ring_rows": len(ring_df),
                "n_ring_rows_mapped_to_af": n_mapped,
                "n_wt_preserved": wt_preserved,
                "n_mut_preserved": mut_preserved,
                "n_lost_rows": lost_rows,
                "tolerance_A": args.tolerance,
            })

            print(
                f"{workflow2_class}: RING rows={len(ring_df)}, "
                f"mapped={n_mapped}, WT preserved={wt_preserved}, "
                f"MUT preserved={mut_preserved}, lost={lost_rows}"
            )

        except Exception as e:
            print(f"FAIL {case_id}: {e}")
            failures.append({
                "case_id": case_id,
                "error": str(e),
            })

    all_rows_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries)
    failed_df = pd.DataFrame(failures)

    all_rows_path = outdir / "workflow2_ring_benchmark_all_rows.tsv"
    summary_path = outdir / "workflow2_ring_benchmark_case_summary.tsv"
    failed_path = outdir / "workflow2_ring_benchmark_failed.tsv"

    all_rows_df.to_csv(all_rows_path, sep="\t", index=False)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    failed_df.to_csv(failed_path, sep="\t", index=False)

    print("\nWrote:")
    print(all_rows_path)
    print(summary_path)
    print(failed_path)

    if not summary_df.empty:
        print("\nSummary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()