#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path
import pandas as pd


def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def unpack_zips(downloads_dir):
    downloads_dir = Path(downloads_dir)
    for z in downloads_dir.glob("*.zip"):
        out = downloads_dir / z.stem
        if out.exists():
            continue
        print(f"Unzipping {z} -> {out}")
        safe_mkdir(out)
        with zipfile.ZipFile(z) as zipf:
            zipf.extractall(out)


def find_job_request(folder):
    files = list(Path(folder).rglob("*_job_request.json"))
    return files[0] if files else None


def load_job_name(job_request):
    data = json.loads(Path(job_request).read_text())

    def find_name(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k).lower() in {"name", "jobname", "job_name", "title"} and isinstance(v, str):
                    return v
            for v in x.values():
                ans = find_name(v)
                if ans:
                    return ans
        elif isinstance(x, list):
            for v in x:
                ans = find_name(v)
                if ans:
                    return ans
        return None

    return find_name(data)


def parse_model_idx(path):
    m = re.search(r"_summary_confidences_(\d+)\.json$", Path(path).name)
    if m:
        return m.group(1)
    return None


def read_summary_confidences(folder):
    rows = []
    for f in sorted(Path(folder).glob("*summary_confidences_*.json")):
        idx = parse_model_idx(f)
        data = json.loads(f.read_text())
        row = {"model_idx": idx, "summary_file": str(f)}
        for k, v in data.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_model(folder):
    df = read_summary_confidences(folder)
    if df.empty:
        return None, None

    sort_cols = [c for c in ["ranking_score", "iptm", "ptm", "mean_plddt"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)

    best = df.iloc[0].to_dict()
    idx = str(best["model_idx"])
    model_files = list(Path(folder).glob(f"*_model_{idx}.cif"))
    model_file = model_files[0] if model_files else None

    return best, model_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    downloads = Path(args.downloads)
    root = Path(args.root)

    unpack_zips(downloads)

    manifest = pd.read_csv(args.manifest, sep="\t")
    by_job = {r["job_name"]: r for _, r in manifest.iterrows()}

    result_rows = []

    folders = [p for p in downloads.iterdir() if p.is_dir()]

    for folder in folders:
        jr = find_job_request(folder)
        if not jr:
            continue

        job_name = load_job_name(jr)
        if job_name not in by_job:
            print(f"SKIP unknown job name: {job_name} in {folder}")
            continue

        meta = by_job[job_name]
        case_dir = Path(meta["case_dir"])
        condition = meta["condition"]

        outdir = case_dir / "alphafold_outputs" / condition
        selected = case_dir / "alphafold_selected"
        safe_mkdir(outdir)
        safe_mkdir(selected)

        # Copy full downloaded folder contents
        for item in folder.iterdir():
            dest = outdir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        best, model_file = select_best_model(outdir)
        if best is None:
            print(f"No summary_confidences for {job_name}")
            continue

        idx = str(best["model_idx"])

        if model_file:
            dest_model = selected / f"{condition}_best_model_{idx}.cif"
            shutil.copy2(model_file, dest_model)
        else:
            dest_model = ""

        summary_src = Path(best["summary_file"])
        dest_summary = selected / f"{condition}_best_summary_confidences_{idx}.json"
        shutil.copy2(summary_src, dest_summary)

        row = {
            "job_name": job_name,
            "case_id": meta["case_id"],
            "condition": condition,
            "case_dir": str(case_dir),
            "best_model_idx": idx,
            "best_model_file": str(dest_model),
            "best_summary_file": str(dest_summary),
        }
        for k, v in best.items():
            if k not in row:
                row[k] = v

        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    out = root / "workflow1_afserver_ingested_best_models.tsv"
    results.to_csv(out, sep="\t", index=False)

    print(results.to_string(index=False))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()