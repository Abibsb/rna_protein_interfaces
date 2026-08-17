#!/usr/bin/env python3

import argparse
import copy
import json
import re
from pathlib import Path
import pandas as pd


PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
RNA_ALPHABET = set("ACGU")


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


def load_template(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        if not data:
            raise ValueError("Template JSON list is empty")
        return data[0]
    return data


def set_job_name(obj, name):
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            lk = k.lower()
            if lk in {"name", "jobname", "job_name", "title"} and isinstance(obj[k], str):
                obj[k] = name
        for v in obj.values():
            set_job_name(v, name)
    elif isinstance(obj, list):
        for v in obj:
            set_job_name(v, name)


def replace_sequences(obj, protein_seq, rna_seq):
    """
    Robust replacement for common AlphaFold Server job_request schemas.
    It tries to replace sequence fields in protein/rna entities while preserving the rest of the template.
    """

    replaced = {"protein": 0, "rna": 0}

    def rec(x, parent_keys=()):
        if isinstance(x, dict):
            keys_lower = {str(k).lower(): k for k in x.keys()}

            # Case 1: {"protein": {"sequence": "..."}}
            if "protein" in keys_lower:
                pkey = keys_lower["protein"]
                if isinstance(x[pkey], dict):
                    for sk in ["sequence", "proteinSequence"]:
                        if sk in x[pkey]:
                            x[pkey][sk] = protein_seq
                            replaced["protein"] += 1

            # Case 2: {"rna": {"sequence": "..."}}
            if "rna" in keys_lower:
                rkey = keys_lower["rna"]
                if isinstance(x[rkey], dict):
                    for sk in ["sequence", "rnaSequence"]:
                        if sk in x[rkey]:
                            x[rkey][sk] = rna_seq
                            replaced["rna"] += 1

            # Case 3: {"type": "protein", "sequence": "..."}
            typ = str(x.get("type", x.get("moleculeType", ""))).lower()
            if typ == "protein" and "sequence" in x:
                x["sequence"] = protein_seq
                replaced["protein"] += 1
            if typ == "rna" and "sequence" in x:
                x["sequence"] = rna_seq
                replaced["rna"] += 1

            for k, v in x.items():
                rec(v, parent_keys + (str(k),))

        elif isinstance(x, list):
            for v in x:
                rec(v, parent_keys)

    rec(obj)

    # If template has direct sequence strings only, try a conservative fallback:
    if replaced["protein"] == 0 or replaced["rna"] == 0:
        def fallback(x):
            if isinstance(x, dict):
                for k, v in list(x.items()):
                    if isinstance(v, str):
                        seq = v.replace("\n", "").replace(" ", "").upper()
                        if len(seq) >= 4 and set(seq).issubset(PROTEIN_ALPHABET) and replaced["protein"] == 0:
                            x[k] = protein_seq
                            replaced["protein"] += 1
                        elif len(seq) >= 4 and set(seq).issubset(RNA_ALPHABET) and replaced["rna"] == 0:
                            x[k] = rna_seq
                            replaced["rna"] += 1
                    else:
                        fallback(v)
            elif isinstance(x, list):
                for v in x:
                    fallback(v)

        fallback(obj)

    if replaced["protein"] == 0 or replaced["rna"] == 0:
        raise ValueError(
            f"Could not replace sequences in template. Replaced={replaced}. "
            "Use a job_request.json exported from a protein+RNA AlphaFold Server job."
        )

    return replaced


def clean_job_name(s):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="alphafold_fragment_workflow directory")
    ap.add_argument("--prepared-summary", default=None)
    ap.add_argument("--template-json", required=True, help="A downloaded AlphaFold Server *_job_request.json from a protein+RNA job")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-jobs-per-batch", type=int, default=30)
    ap.add_argument("--only-wt-matches", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prepared = Path(args.prepared_summary) if args.prepared_summary else root / "prepared_cases_summary.tsv"
    summary = pd.read_csv(prepared, sep="\t")

    if args.only_wt_matches and "observed_matches_variant_wt" in summary.columns:
        summary = summary[summary["observed_matches_variant_wt"].astype(str).str.lower().isin(["true", "1"])]

    summary = summary[summary["status"].astype(str).str.upper().eq("OK")].copy()

    template = load_template(args.template_json)

    jobs = []
    manifest = []

    for _, row in summary.iterrows():
        case_id = row["case_id"]
        case_dir = Path(row["case_dir"])

        wt_fasta = case_dir / "sequences" / "protein_cif_WT.fasta"
        mut_fasta = case_dir / "sequences" / "protein_cif_MUT.fasta"
        rna_fastas = sorted((case_dir / "sequences").glob("rna_fragment_chain_*.fasta"))

        if not wt_fasta.exists() or not mut_fasta.exists() or not rna_fastas:
            print(f"SKIP missing files: {case_id}")
            continue

        _, wt_seq = read_fasta(wt_fasta)
        _, mut_seq = read_fasta(mut_fasta)
        _, rna_seq = read_fasta(rna_fastas[0])

        for condition, protein_seq in [("CIF_WT", wt_seq), ("CIF_MUT", mut_seq)]:
            job_name = clean_job_name(f"{case_id}__{condition}")

            job = copy.deepcopy(template)
            set_job_name(job, job_name)
            replace_sequences(job, protein_seq, rna_seq)

            jobs.append(job)
            manifest.append({
                "job_name": job_name,
                "case_id": case_id,
                "condition": condition,
                "case_dir": str(case_dir),
                "protein_fasta": str(wt_fasta if condition == "CIF_WT" else mut_fasta),
                "rna_fasta": str(rna_fastas[0]),
                "n_protein_aa": len(protein_seq),
                "n_rna_nt": len(rna_seq),
                "tokens_approx": len(protein_seq) + len(rna_seq),
            })

    manifest_df = pd.DataFrame(manifest)
    manifest_path = outdir / "afserver_jobs_manifest.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)

    for i in range(0, len(jobs), args.max_jobs_per_batch):
        batch_no = i // args.max_jobs_per_batch + 1
        batch_jobs = jobs[i:i + args.max_jobs_per_batch]
        batch_manifest = manifest_df.iloc[i:i + args.max_jobs_per_batch].copy()

        json_path = outdir / f"afserver_import_batch_{batch_no:03d}.json"
        man_path = outdir / f"afserver_import_batch_{batch_no:03d}_manifest.tsv"

        json_path.write_text(json.dumps(batch_jobs, indent=2))
        batch_manifest.to_csv(man_path, sep="\t", index=False)

        print(f"Wrote {json_path} with {len(batch_jobs)} jobs")
        print(f"Wrote {man_path}")

    print(f"\nTotal jobs: {len(jobs)}")
    print(f"Total cases: {manifest_df['case_id'].nunique() if len(manifest_df) else 0}")
    print(f"Wrote manifest: {manifest_path}")
    print("\nWith 30 jobs/day: 2 jobs per case = 15 cases/day.")


if __name__ == "__main__":
    main()