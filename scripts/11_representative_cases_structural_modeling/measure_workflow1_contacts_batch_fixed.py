#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser


def atom_distance(a1, a2):
    return math.sqrt(sum((a1.coord[i] - a2.coord[i]) ** 2 for i in range(3)))


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


def get_residues_by_numbers(chain, resnums):
    wanted = set(int(x) for x in resnums)
    out = []
    for res in chain:
        hetflag, rnum, icode = res.id
        if hetflag == " " and int(rnum) in wanted:
            out.append(res)
    return out


def parse_structure(cif_path, name):
    parser = MMCIFParser(QUIET=True)
    st = parser.get_structure(name, str(cif_path))
    return next(st.get_models())


def measure(model_name, cif_path, protein_chain, protein_resnum, rna_chain, rna_resnums=None):
    model = parse_structure(cif_path, model_name)

    p_chain = get_chain(model, protein_chain)
    r_chain = get_chain(model, rna_chain)
    p_res = get_residue_by_number(p_chain, protein_resnum)

    if rna_resnums is None:
        rna_residues = [r for r in r_chain if r.id[0] == " "]
        rna_scope = "full_chain"
    else:
        rna_residues = get_residues_by_numbers(r_chain, rna_resnums)
        rna_scope = ",".join(map(str, sorted(set(map(int, rna_resnums)))))

    if not rna_residues:
        raise ValueError(f"No RNA residues found for {model_name}, chain {rna_chain}, scope {rna_scope}")

    rows = []
    min_dist = float("inf")
    closest = None

    for pa in p_res.get_atoms():
        for rr in rna_residues:
            for ra in rr.get_atoms():
                d = atom_distance(pa, ra)
                if d < min_dist:
                    min_dist = d
                    closest = (pa.name, rr.get_resname(), rr.id[1], ra.name)
                rows.append({
                    "model": model_name,
                    "protein_chain": protein_chain,
                    "protein_resnum": protein_resnum,
                    "protein_resname": p_res.get_resname(),
                    "protein_atom": pa.name,
                    "rna_chain": rna_chain,
                    "rna_resnum": rr.id[1],
                    "rna_resname": rr.get_resname(),
                    "rna_atom": ra.name,
                    "distance_A": d,
                    "rna_scope": rna_scope,
                })

    df = pd.DataFrame(rows)

    summary = {
        "model": model_name,
        "protein_residue": f"{protein_chain}:{protein_resnum}:{p_res.get_resname()}",
        "rna_chain": rna_chain,
        "rna_scope": rna_scope,
        "min_distance_A": round(min_dist, 3),
        "closest_pair": (
            f"{protein_chain}:{protein_resnum}:{p_res.get_resname()}:{closest[0]}--"
            f"{rna_chain}:{closest[2]}:{closest[1]}:{closest[3]}"
        ),
        "n_atom_pairs_lt_4A": int((df["distance_A"] < 4.0).sum()),
        "n_atom_pairs_lt_5A": int((df["distance_A"] < 5.0).sum()),
        "n_unique_rna_nt_lt_4A": int(df.loc[df["distance_A"] < 4.0, "rna_resnum"].nunique()),
        "n_unique_rna_nt_lt_5A": int(df.loc[df["distance_A"] < 5.0, "rna_resnum"].nunique()),
        "rna_nts_lt_5A": ",".join(map(str, sorted(df.loc[df["distance_A"] < 5.0, "rna_resnum"].unique()))),
    }

    contacts = df[df["distance_A"] < 5.0].sort_values("distance_A")
    return summary, contacts


def read_case_metadata(case_dir):
    case_dir = Path(case_dir)

    # =========================
    # Protein mapping: experimental PDB/mmCIF residue
    # =========================
    maprow = pd.read_csv(
        case_dir / "metadata" / "pdb_uniprot_mapping_row.tsv",
        sep="\t",
    ).iloc[0]

    protein_chain = str(maprow["auth_asym_id"])
    protein_resnum = int(maprow["auth_seq_id"])

    # =========================
    # Protein residue map:
    # maps experimental auth_seq_id -> AlphaFold fragment seq_index
    # =========================
    resmap_path = case_dir / "metadata" / "protein_cif_residue_map.tsv"
    resmap = pd.read_csv(resmap_path, sep="\t")

    if "auth_seq_id" not in resmap.columns:
        raise ValueError(
            f"Missing auth_seq_id column in {resmap_path}. "
            f"Available columns: {resmap.columns.tolist()}"
        )

    if "seq_index" not in resmap.columns:
        raise ValueError(
            f"Missing seq_index column in {resmap_path}. "
            f"Available columns: {resmap.columns.tolist()}"
        )

    hit = resmap[
        pd.to_numeric(resmap["auth_seq_id"], errors="coerce") == protein_resnum
    ].copy()

    if hit.empty:
        raise ValueError(
            f"Could not find auth_seq_id={protein_resnum} in {resmap_path}"
        )

    af_resnum = int(pd.to_numeric(hit.iloc[0]["seq_index"], errors="coerce"))

    # =========================
    # RNA contacts used to define local experimental neighborhood
    # =========================
    rna_contacts_path = case_dir / "metadata" / "rna_contacts_by_distance.tsv"
    rna_contacts = pd.read_csv(rna_contacts_path, sep="\t")

    if "rna_chain" not in rna_contacts.columns:
        raise ValueError(
            f"Missing rna_chain column in {rna_contacts_path}. "
            f"Available columns: {rna_contacts.columns.tolist()}"
        )

    rna_chain = str(rna_contacts.iloc[0]["rna_chain"])

    if "rna_auth_seq_id" in rna_contacts.columns:
        rna_resnums = (
            pd.to_numeric(rna_contacts["rna_auth_seq_id"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
    elif "rna_resnum" in rna_contacts.columns:
        rna_resnums = (
            pd.to_numeric(rna_contacts["rna_resnum"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
    elif "rna_seq_index" in rna_contacts.columns:
        rna_resnums = (
            pd.to_numeric(rna_contacts["rna_seq_index"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
    else:
        raise ValueError(
            f"Cannot find RNA residue number column in {rna_contacts_path}. "
            f"Available columns: {rna_contacts.columns.tolist()}"
        )

    # Remove duplicates while preserving order
    rna_resnums = list(dict.fromkeys(rna_resnums))

    # =========================
    # Experimental CIF and selected AlphaFold models
    # =========================
    exp_cif = Path(
        (case_dir / "experimental" / "cif_path.txt").read_text().strip()
    )

    wt_models = sorted(
        (case_dir / "alphafold_selected").glob("CIF_WT_best_model_*.cif")
    )
    mut_models = sorted(
        (case_dir / "alphafold_selected").glob("CIF_MUT_best_model_*.cif")
    )

    if not wt_models or not mut_models:
        raise FileNotFoundError(
            f"Missing selected AF models in {case_dir}/alphafold_selected"
        )

    return {
        "exp_cif": exp_cif,
        "protein_chain": protein_chain,
        "protein_resnum": protein_resnum,
        "af_resnum": af_resnum,
        "rna_chain": rna_chain,
        "rna_resnums": rna_resnums,
        "wt_cif": wt_models[0],
        "mut_cif": mut_models[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--prepared-summary", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    prepared = Path(args.prepared_summary) if args.prepared_summary else root / "prepared_cases_summary.tsv"
    cases = pd.read_csv(prepared, sep="\t")
    cases = cases[cases["status"].astype(str).str.upper().eq("OK")].copy()

    all_summaries = []
    failed = []

    for _, row in cases.iterrows():
        case_id = row["case_id"]
        case_dir = Path(row["case_dir"])

        # Skip cases whose AlphaFold WT/MUT outputs have not been ingested yet.
        # This is useful for partial runs after only one AlphaFold Server batch.
        selected_dir = case_dir / "alphafold_selected"
        has_wt = bool(list(selected_dir.glob("CIF_WT_best_model_*.cif")))
        has_mut = bool(list(selected_dir.glob("CIF_MUT_best_model_*.cif")))

        if not (has_wt and has_mut):
            continue

        try:
            meta = read_case_metadata(case_dir)

            summaries = []
            contacts_all = []

            s, c = measure(
                "experimental_local",
                meta["exp_cif"],
                meta["protein_chain"],
                meta["protein_resnum"],
                meta["rna_chain"],
                meta["rna_resnums"],
            )
            summaries.append(s)
            contacts_all.append(c)

            s, c = measure(
                "AF_CIF_WT",
                meta["wt_cif"],
                "A",
                meta["af_resnum"],
                "B",
                None,
            )
            summaries.append(s)
            contacts_all.append(c)

            s, c = measure(
                "AF_CIF_MUT",
                meta["mut_cif"],
                "A",
                meta["af_resnum"],
                "B",
                None,
            )
            summaries.append(s)
            contacts_all.append(c)

            outdir = case_dir / "contact_measurements"
            outdir.mkdir(exist_ok=True)

            sdf = pd.DataFrame(summaries)
            sdf.insert(0, "case_id", case_id)
            sdf.to_csv(outdir / "workflow1_residue_rna_contact_summary.tsv", sep="\t", index=False)

            cdf = pd.concat(contacts_all, ignore_index=True) if contacts_all else pd.DataFrame()
            cdf.insert(0, "case_id", case_id)
            cdf.to_csv(outdir / "workflow1_residue_rna_contacts_lt5A.tsv", sep="\t", index=False)

            all_summaries.append(sdf)

            print(f"OK {case_id}")

        except Exception as e:
            print(f"FAIL {case_id}: {e}")
            failed.append({"case_id": case_id, "case_dir": str(case_dir), "error": str(e)})

    final = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out, sep="\t", index=False)

    failed_df = pd.DataFrame(failed)
    failed_path = str(Path(args.out).with_suffix("")) + "_failed.tsv"
    failed_df.to_csv(failed_path, sep="\t", index=False)

    print(f"\nWrote: {args.out}")
    print(f"Wrote: {failed_path}")


if __name__ == "__main__":
    main()