#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import time
import urllib.request
import urllib.error
import urllib.parse
import pandas as pd


INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api"

# Mantengamos primero sólo estas dos, que son las más útiles y limpias.
ENTRY_DATABASES = ["interpro", "pfam"]


def fetch_json(url, cache_file=None, sleep=0.2):
    if cache_file and cache_file.exists() and cache_file.stat().st_size > 0:
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "workflow2-interpro-annotation/2.0",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode("utf-8")
            data = json.loads(txt)
    except urllib.error.HTTPError as e:
        print(f"WARNING HTTP {e.code}: {url}")
        return None
    except Exception as e:
        print(f"WARNING fetch failed: {url} -> {e}")
        return None

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, indent=2))

    time.sleep(sleep)
    return data


def fetch_paginated_results(url, cache_dir, cache_prefix, sleep=0.2):
    """
    InterPro API puede responder de dos formas:

    1) Directo con results:
       {"count": ..., "results": [...]}

    2) Indirecto:
       {"metadata": ..., "entries_url": "https://..."}

    En el segundo caso hay que abrir entries_url.
    """
    first_cache = cache_dir / f"{cache_prefix}.initial.json"
    data = fetch_json(url, cache_file=first_cache, sleep=sleep)

    if not data:
        return []

    # Caso indirecto: seguir entries_url
    if "results" not in data and data.get("entries_url"):
        next_url = data["entries_url"]
        if next_url.startswith("/"):
            next_url = "https://www.ebi.ac.uk" + next_url
    else:
        next_url = url

    all_results = []
    page = 1

    while next_url:
        page_cache = cache_dir / f"{cache_prefix}.page{page}.json"
        page_data = fetch_json(next_url, cache_file=page_cache, sleep=sleep)

        if not page_data:
            break

        results = page_data.get("results", [])
        if isinstance(results, list):
            all_results.extend(results)

        next_url = page_data.get("next")
        page += 1

        if page > 100:
            print(f"WARNING: too many pages for {cache_prefix}; stopping")
            break

    return all_results


def collect_fragments(obj):
    """
    Busca coordenadas start/end de forma recursiva.
    En InterPro suelen estar en:
    proteins -> entry_protein_locations -> fragments -> start/end
    """
    fragments = []

    if isinstance(obj, dict):
        if "start" in obj and "end" in obj:
            try:
                start = int(obj["start"])
                end = int(obj["end"])
                if start > 0 and end > 0:
                    fragments.append((start, end))
            except Exception:
                pass

        for v in obj.values():
            fragments.extend(collect_fragments(v))

    elif isinstance(obj, list):
        for item in obj:
            fragments.extend(collect_fragments(item))

    # deduplicar
    out = []
    seen = set()
    for f in fragments:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def extract_metadata(result, db):
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}

    accession = metadata.get("accession", "") or ""
    name = metadata.get("name", "") or ""
    source_database = metadata.get("source_database", db) or db
    entry_type = metadata.get("type", "") or ""
    integrated = metadata.get("integrated", "") or ""

    return {
        "interpro_source_db": db,
        "interpro_accession": accession,
        "interpro_name": name,
        "interpro_type": entry_type,
        "interpro_integrated_accession": integrated,
    }


def hits_for_position(result, db, uniprot_id, pos):
    meta = extract_metadata(result, db)
    fragments = collect_fragments(result)

    hits = []
    for start, end in fragments:
        if start <= pos <= end:
            hits.append({
                "uniprot_id": uniprot_id,
                **meta,
                "interpro_feature_start": start,
                "interpro_feature_end": end,
                "interpro_feature_length": end - start + 1,
                "distance_to_interpro_start": pos - start,
                "distance_to_interpro_end": end - pos,
                "residue_inside_interpro_feature": True,
            })

    return hits


def join_unique(x):
    vals = []
    for v in x:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            vals.append(s)
    return "; ".join(sorted(set(vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    cache_dir = Path(args.cache_dir)

    required = ["case_id", "uniprot_id", "uniprot_pos", "workflow2_class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Input rows:", len(df))
    print("Unique UniProt IDs:", df["uniprot_id"].nunique())

    db_results = {}

    for acc in sorted(df["uniprot_id"].dropna().astype(str).unique()):
        db_results[acc] = {}

        for db in ENTRY_DATABASES:
            url = f"{INTERPRO_BASE}/protein/uniprot/{urllib.parse.quote(acc)}/entry/{db}/"
            prefix = f"{acc}.{db}"

            print(f"Fetching/reading InterPro: {acc} / {db}")
            results = fetch_paginated_results(url, cache_dir, prefix, sleep=args.sleep)
            db_results[acc][db] = results
            print(f"  results: {len(results)}")

    detailed_rows = []

    for _, row in df.iterrows():
        acc = str(row["uniprot_id"])
        pos = int(row["uniprot_pos"])
        base = row.to_dict()

        hits = []
        for db in ENTRY_DATABASES:
            for result in db_results.get(acc, {}).get(db, []):
                hits.extend(hits_for_position(result, db, acc, pos))

        # deduplicar hits
        unique_hits = []
        seen = set()
        for h in hits:
            key = (
                h["interpro_source_db"],
                h["interpro_accession"],
                h["interpro_feature_start"],
                h["interpro_feature_end"],
            )
            if key not in seen:
                seen.add(key)
                unique_hits.append(h)

        if unique_hits:
            for h in unique_hits:
                detailed_rows.append({
                    **base,
                    **h,
                    "n_overlapping_interpro_features": len(unique_hits),
                })
        else:
            detailed_rows.append({
                **base,
                "interpro_source_db": "",
                "interpro_accession": "",
                "interpro_name": "",
                "interpro_type": "",
                "interpro_integrated_accession": "",
                "interpro_feature_start": "",
                "interpro_feature_end": "",
                "interpro_feature_length": "",
                "distance_to_interpro_start": "",
                "distance_to_interpro_end": "",
                "residue_inside_interpro_feature": False,
                "n_overlapping_interpro_features": 0,
            })

    detailed = pd.DataFrame(detailed_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(out, sep="\t", index=False)

    summary = (
        detailed
        .groupby("case_id", as_index=False)
        .agg(
            workflow2_class=("workflow2_class", "first"),
            uniprot_id=("uniprot_id", "first"),
            uniprot_pos=("uniprot_pos", "first"),
            variant_id=("variant_id", "first"),
            clinical_text=("clinical_text", "first"),
            is_pathogenic_or_damaging=("is_pathogenic_or_damaging", "first"),
            n_ring_rows=("n_ring_rows", "first"),
            n_wt_preserved=("n_wt_preserved", "first"),
            n_mut_preserved=("n_mut_preserved", "first"),
            n_lost_rows=("n_lost_rows", "first"),
            residue_inside_interpro_feature=("residue_inside_interpro_feature", "max"),
            n_overlapping_interpro_features=("n_overlapping_interpro_features", "max"),
            interpro_source_dbs=("interpro_source_db", join_unique),
            interpro_accessions=("interpro_accession", join_unique),
            interpro_names=("interpro_name", join_unique),
            interpro_types=("interpro_type", join_unique),
            interpro_integrated_accessions=("interpro_integrated_accession", join_unique),
        )
    )

    ranges = []
    for case_id, sub in detailed.groupby("case_id"):
        vals = []
        for _, r in sub.iterrows():
            start = str(r.get("interpro_feature_start", "")).strip()
            end = str(r.get("interpro_feature_end", "")).strip()
            acc = str(r.get("interpro_accession", "")).strip()
            name = str(r.get("interpro_name", "")).strip()
            db = str(r.get("interpro_source_db", "")).strip()

            if start and end and start.lower() != "nan" and end.lower() != "nan":
                label = acc if acc else name
                vals.append(f"{db}:{label}:{start}-{end}")

        ranges.append((case_id, "; ".join(sorted(set(vals)))))

    ranges = pd.DataFrame(ranges, columns=["case_id", "interpro_feature_ranges"])
    summary = summary.merge(ranges, on="case_id", how="left")

    summary_out = Path(args.summary_out)
    summary.to_csv(summary_out, sep="\t", index=False)

    print("\nWrote detailed:", out)
    print("Wrote summary:", summary_out)

    print("\nSummary by workflow2_class and residue_inside_interpro_feature:")
    print(
        summary
        .groupby(["workflow2_class", "residue_inside_interpro_feature"], dropna=False)
        .size()
        .reset_index(name="n")
        .to_string(index=False)
    )

    print("\nCases with overlapping InterPro/Pfam features:")
    cols = [
        "case_id",
        "workflow2_class",
        "variant_id",
        "clinical_text",
        "interpro_source_dbs",
        "interpro_accessions",
        "interpro_names",
        "interpro_types",
        "interpro_integrated_accessions",
        "interpro_feature_ranges",
    ]
    print(
        summary.loc[
            summary["residue_inside_interpro_feature"] == True,
            [c for c in cols if c in summary.columns],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
