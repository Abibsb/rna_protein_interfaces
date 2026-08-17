from pathlib import Path
import pandas as pd
import argparse


STRONG_CASES = [
    "6NU3_Q9BYC8_R85C",
    "6VMI_Q96GC5_Y48C",
    "6VMI_Q96GC5_Y48D",
    "6W6L_P62910_R48I",
]


def parse_ring_node(node):
    # ej: 0:85:_:ARG
    parts = str(node).split(":")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse ring node: {node}")
    return {
        "chain": parts[0],
        "resnum": parts[1],
        "icode": parts[2],
        "resname": parts[3],
    }


def parse_closest_pair(s):
    # ej: ARG:NH2--U:O3'
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None, None
    try:
        left, right = s.split("--")
        prot_atom = left.split(":")[-1]
        rna_atom = right.split(":")[-1]
        return prot_atom, rna_atom
    except Exception:
        return None, None


def unique_in_order(vals):
    out = []
    seen = set()
    for v in vals:
        if pd.isna(v):
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_residue_list(chain, resnums):
    resnums = unique_in_order(resnums)
    return f"/{chain}:{','.join(str(x) for x in resnums)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="alphafold_fragment_workflow_stratified_60 root")
    ap.add_argument("--all-rows", required=True, help="workflow2_ring_benchmark_all_rows.tsv")
    args = ap.parse_args()

    root = Path(args.root)
    df = pd.read_csv(args.all_rows, sep="\t")

    df = df[df["case_id"].isin(STRONG_CASES)].copy()

    # Solo filas que realmente se pierden en el mutante
    lost = df[df["ring_contact_lost_in_mutant"] == True].copy()

    if lost.empty:
        print("No lost benchmark rows found.")
        return

    for case_id, sub in lost.groupby("case_id"):
        case_dir = root / case_id
        chim_dir = case_dir / "chimerax"
        chim_dir.mkdir(parents=True, exist_ok=True)

        exp_cif = Path((case_dir / "experimental" / "cif_path.txt").read_text().strip())
        wt_cif = sorted((case_dir / "alphafold_selected").glob("CIF_WT_best_model_*.cif"))[0]
        mut_cif = sorted((case_dir / "alphafold_selected").glob("CIF_MUT_best_model_*.cif"))[0]

        ring_info = parse_ring_node(sub.iloc[0]["ring_node"])
        exp_prot_chain = ring_info["chain"]
        exp_prot_resnum = str(ring_info["resnum"])

        exp_rna_chain = str(sub.iloc[0]["exp_rna_chain"])
        exp_rna_resnums = unique_in_order(sub["exp_rna_resnum"].astype(str).tolist())

        af_prot_chain = "A"
        af_prot_resnum = str(int(sub.iloc[0]["protein_af_resnum"]))

        af_rna_chain = str(sub.iloc[0]["af_rna_chain"])
        af_rna_resnums = unique_in_order(sub["af_rna_resnum"].astype(str).tolist())

        summary_rows = []

        cxc_lines = []
        cxc_lines.append(f"open {exp_cif}")
        cxc_lines.append(f"open {wt_cif}")
        cxc_lines.append(f"open {mut_cif}")
        cxc_lines.append("rename #1 EXP")
        cxc_lines.append("rename #2 WT")
        cxc_lines.append("rename #3 MUT")
        cxc_lines.append("tile")
        cxc_lines.append("set bgColor white")
        cxc_lines.append("lighting soft")
        cxc_lines.append("hide atoms")
        cxc_lines.append("cartoon #1-3")
        cxc_lines.append("color #1-3 lightgray")

        # Resaltar experimental
        cxc_lines.append(f"show #1/{exp_prot_chain}:{exp_prot_resnum} atoms")
        cxc_lines.append(f"style #1/{exp_prot_chain}:{exp_prot_resnum} stick")
        cxc_lines.append(f"color #1/{exp_prot_chain}:{exp_prot_resnum} hotpink")

        cxc_lines.append(f"show #1/{exp_rna_chain}:{','.join(exp_rna_resnums)} atoms")
        cxc_lines.append(f"style #1/{exp_rna_chain}:{','.join(exp_rna_resnums)} stick")
        cxc_lines.append(f"color #1/{exp_rna_chain}:{','.join(exp_rna_resnums)} gold")

        # WT
        cxc_lines.append(f"show #2/{af_prot_chain}:{af_prot_resnum} atoms")
        cxc_lines.append(f"style #2/{af_prot_chain}:{af_prot_resnum} stick")
        cxc_lines.append(f"color #2/{af_prot_chain}:{af_prot_resnum} forestgreen")

        cxc_lines.append(f"show #2/{af_rna_chain}:{','.join(af_rna_resnums)} atoms")
        cxc_lines.append(f"style #2/{af_rna_chain}:{','.join(af_rna_resnums)} stick")
        cxc_lines.append(f"color #2/{af_rna_chain}:{','.join(af_rna_resnums)} gold")

        # MUT
        cxc_lines.append(f"show #3/{af_prot_chain}:{af_prot_resnum} atoms")
        cxc_lines.append(f"style #3/{af_prot_chain}:{af_prot_resnum} stick")
        cxc_lines.append(f"color #3/{af_prot_chain}:{af_prot_resnum} red")

        cxc_lines.append(f"show #3/{af_rna_chain}:{','.join(af_rna_resnums)} atoms")
        cxc_lines.append(f"style #3/{af_rna_chain}:{','.join(af_rna_resnums)} stick")
        cxc_lines.append(f"color #3/{af_rna_chain}:{','.join(af_rna_resnums)} gold")

        # Etiquetas
        cxc_lines.append(f"label #1/{exp_prot_chain}:{exp_prot_resnum} residues")
        cxc_lines.append(f"label #2/{af_prot_chain}:{af_prot_resnum} residues")
        cxc_lines.append(f"label #3/{af_prot_chain}:{af_prot_resnum} residues")
        cxc_lines.append(f"label #1/{exp_rna_chain}:{','.join(exp_rna_resnums)} residues")
        cxc_lines.append(f"label #2/{af_rna_chain}:{','.join(af_rna_resnums)} residues")
        cxc_lines.append(f"label #3/{af_rna_chain}:{','.join(af_rna_resnums)} residues")

        # Distancias exactas WT/MUT para cada fila perdida
        dist_idx = 1
        for _, row in sub.iterrows():
            wt_prot_atom, wt_rna_atom = parse_closest_pair(row["AF_WT_closest_pair"])
            mut_prot_atom, mut_rna_atom = parse_closest_pair(row["AF_MUT_closest_pair"])

            af_rna_resnum = str(int(row["af_rna_resnum"]))

            if wt_prot_atom and wt_rna_atom:
                cxc_lines.append(
                    f'distance "#2/{af_prot_chain}:{af_prot_resnum}@{wt_prot_atom}" '
                    f'"#2/{af_rna_chain}:{af_rna_resnum}@{wt_rna_atom}" color lime radius 0.12'
                )

            if mut_prot_atom and mut_rna_atom:
                cxc_lines.append(
                    f'distance "#3/{af_prot_chain}:{af_prot_resnum}@{mut_prot_atom}" '
                    f'"#3/{af_rna_chain}:{af_rna_resnum}@{mut_rna_atom}" color red radius 0.12'
                )

            summary_rows.append({
                "case_id": row["case_id"],
                "pdb_id": row["pdb_id"],
                "ring_node": row["ring_node"],
                "rna_ring_node": row["rna_ring_node"],
                "ring_interaction": row["ring_interaction"],
                "ring_distance_A": row["ring_distance_A"],
                "preservation_threshold_A": row["preservation_threshold_A"],
                "exp_rna_chain": row["exp_rna_chain"],
                "exp_rna_resnum": row["exp_rna_resnum"],
                "exp_rna_nt": row["exp_rna_nt"],
                "af_rna_chain": row["af_rna_chain"],
                "af_rna_resnum": row["af_rna_resnum"],
                "af_rna_nt": row["af_rna_nt"],
                "AF_WT_min_distance_to_ring_nt_A": row["AF_WT_min_distance_to_ring_nt_A"],
                "AF_WT_closest_pair": row["AF_WT_closest_pair"],
                "AF_MUT_min_distance_to_ring_nt_A": row["AF_MUT_min_distance_to_ring_nt_A"],
                "AF_MUT_closest_pair": row["AF_MUT_closest_pair"],
            })
            dist_idx += 1

        cxc_lines.append("view")

        out_cxc = chim_dir / "workflow2_lost_contacts.cxc"
        out_cxc.write_text("\n".join(cxc_lines) + "\n")

        out_tsv = chim_dir / "workflow2_lost_contacts_summary.tsv"
        pd.DataFrame(summary_rows).to_csv(out_tsv, sep="\t", index=False)

        print(f"Wrote: {out_cxc}")
        print(f"Wrote: {out_tsv}")


if __name__ == "__main__":
    main()