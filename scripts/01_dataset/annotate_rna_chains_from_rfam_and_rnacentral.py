#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    requests = None


# -----------------------------
# Helpers
# -----------------------------
def md5_dnaseq_from_rna(seq: str) -> str:
    """
    RNAcentral API md5 convention (DNA alphabet):
      - uppercase
      - replace U with T
    :contentReference[oaicite:2]{index=2}
    """
    s = "".join(seq.split()).upper().replace("U", "T")
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def read_fasta_as_dict(fasta_path: Path) -> dict:
    seqs = {}
    header = None
    chunks = []
    with fasta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seqs[header] = "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            seqs[header] = "".join(chunks)
    return seqs


def parse_rfam_tblout(tblout_path: Path) -> pd.DataFrame:
    """
    Parses Infernal cmscan --tblout output.

    Your tblout lines look like:
    target_name target_acc query_name ... bit_score evalue ... description...

    Example (from you):
    U2 RF00004 7evo_H - cm ... 112.1 1.8e-24 ! U2 spliceosomal RNA

    We'll parse defensively:
      - ignore comments (#)
      - split on whitespace
      - take:
          rfam_name = fields[0]
          rfam_acc  = fields[1]
          query     = fields[2]
          bit_score = float(fields[-5])   (in your example: 112.1)
          evalue    = float(fields[-4])   (in your example: 1.8e-24)
      - keep only hits marked '!' (significant) when present
    """
    rows = []
    with tblout_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 10:
                continue

            rfam_name = fields[0]
            rfam_acc = fields[1]
            query = fields[2]

            # Many tblout formats have "!" near the end; if present, keep only "!" hits.
            # In your sample it is the third token from the end.
            has_bang = "!" in fields
            if has_bang and fields[-3] != "!":
                # If it's not in the expected position, still allow but don't filter hard
                pass
            if has_bang and ("!" not in fields):
                continue

            # robust score/evalue parsing:
            # try common layout: ... <bit> <evalue> ! <desc...>
            bit_score = None
            evalue = None

            # try the most common positions
            for (b_idx, e_idx) in [(-5, -4), (-6, -5), (-4, -3)]:
                try:
                    bit_score = float(fields[b_idx])
                    evalue = float(fields[e_idx])
                    break
                except Exception:
                    continue

            if bit_score is None or evalue is None:
                continue

            rows.append(
                {
                    "query": query,
                    "rfam_acc": rfam_acc,
                    "rfam_name": rfam_name,
                    "bit_score": bit_score,
                    "evalue": evalue,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["query", "rfam_acc", "rfam_name", "bit_score", "evalue"])

    df = pd.DataFrame(rows)

    # Choose best hit per query:
    # priority: lowest evalue, then highest bit_score
    df = df.sort_values(["query", "evalue", "bit_score"], ascending=[True, True, False])
    df_best = df.drop_duplicates("query", keep="first").copy()
    return df_best


def infer_rna_type_from_rfam_name(rfam_name: str) -> str:
    """
    Same classification scheme you used:
      - tRNA
      - rRNA_5S, rRNA_5_8S, rRNA_SSU, rRNA_LSU
      - snRNA (U2 etc.)
    """
    n = rfam_name.lower()
    if n == "trna" or "trna" in n:
        return "tRNA"
    if n.startswith("5s_rrna") or n == "5s_rrna":
        return "rRNA_5S"
    if "5_8s" in n or "5.8s" in n or n.startswith("5_8s_rrna"):
        return "rRNA_5_8S"
    if n.startswith("ssu") or "small subunit" in n:
        return "rRNA_SSU"
    if n.startswith("lsu") or "large subunit" in n:
        return "rRNA_LSU"
    if n in {"u1", "u2", "u4", "u5", "u6"} or n.startswith("u") and "snrna" in n:
        return "snRNA"
    if "spliceosomal" in n or "snrna" in n:
        return "snRNA"
    return "other"


def split_query_to_pdb_chain(query: str):
    """
    Your FASTA headers are like: 9b0s_aP or 8any_Ax or 7evo_H
    Split on first underscore.
    """
    if "_" not in query:
        return (None, None)
    pdb_id, chain_id = query.split("_", 1)
    return (pdb_id, chain_id)


# -----------------------------
# RNAcentral exact match (md5 -> URS)
# -----------------------------
def rnacentral_lookup_by_md5(md5: str, timeout=30):
    """
    Uses RNAcentral REST API to fetch URS by md5.
    The docs show filtering exists for length/external_id; md5 filtering is commonly supported in practice.
    If md5 filter is unavailable in your deployment, this will return None gracefully.

    Base API info: :contentReference[oaicite:3]{index=3}
    """
    if requests is None:
        return None

    url = "https://rnacentral.org/api/v1/rna/"
    try:
        r = requests.get(url, params={"md5": md5, "page_size": 1}, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        results = data.get("results", [])
        if not results:
            return None
        return results[0].get("rnacentral_id")
    except Exception:
        return None


# -----------------------------
# RNAcentral alignment option (sequence similarity search API)
# -----------------------------
def rnacentral_similarity_search(sequence: str, timeout=60, poll_sleep=3, poll_tries=60):
    """
    RNAcentral sequence similarity search is nhmmer-powered. :contentReference[oaicite:4]{index=4}

    API is documented at: https://rnacentral.org/sequence-search/api :contentReference[oaicite:5]{index=5}
    BUT the Swagger is JS-based, so endpoints can vary.

    This function tries a small set of known/common endpoint patterns.
    If none work, it returns None with a helpful error message upstream.
    """
    if requests is None:
        raise RuntimeError("requests is not installed. Install with: pip install requests")

    base_candidates = [
        "https://rnacentral.org/sequence-search/api",
        "https://sequence-search.rnacentral.org/api",  # used by embeddable components (commonly)
    ]

    submit_paths = [
        "/search/",          # common REST
        "/search",           # common REST
        "/jobs/",            # job style
        "/jobs",             # job style
        "/v1/search/",       # versioned
        "/v1/search",
    ]

    payload_candidates = [
        {"sequence": sequence},
        {"query": sequence},
        {"fasta": f">query\n{sequence}\n"},
    ]

    # 1) submit
    last_err = None
    job_url = None
    for base in base_candidates:
        for sp in submit_paths:
            submit_url = base.rstrip("/") + sp
            for payload in payload_candidates:
                try:
                    r = requests.post(submit_url, json=payload, timeout=timeout)
                    if r.status_code in (200, 201, 202):
                        # try to extract job URL or job id
                        try:
                            data = r.json()
                        except Exception:
                            data = None

                        if isinstance(data, dict):
                            # common patterns
                            for key in ["url", "job_url", "result_url"]:
                                if key in data and isinstance(data[key], str):
                                    job_url = data[key]
                                    break
                            if job_url:
                                break
                            for key in ["id", "job_id", "uuid"]:
                                if key in data and isinstance(data[key], str):
                                    jid = data[key]
                                    job_url = base.rstrip("/") + f"/jobs/{jid}"
                                    break
                        if job_url:
                            break
                        # if server returns Location header for async job
                        loc = r.headers.get("Location")
                        if loc:
                            job_url = loc
                            break
                except Exception as e:
                    last_err = e
            if job_url:
                break
        if job_url:
            break

    if not job_url:
        raise RuntimeError(
            "Could not submit to RNAcentral sequence-search API. "
            "Endpoints may differ in your environment. "
            f"Last error: {last_err}\n"
            "Check the (JS) Swagger docs at https://rnacentral.org/sequence-search/api "
            "and adapt submit/poll URLs."
        )

    # 2) poll for results
    # try common result urls:
    poll_urls = [
        job_url,
        job_url.rstrip("/") + "/",
        job_url.rstrip("/") + "/result",
        job_url.rstrip("/") + "/results",
        job_url.rstrip("/") + "/status",
    ]

    for _ in range(poll_tries):
        for pu in poll_urls:
            try:
                rr = requests.get(pu, timeout=timeout)
                if rr.status_code != 200:
                    continue
                data = rr.json()
                # Common “finished” patterns:
                if isinstance(data, dict):
                    if data.get("status") in {"DONE", "FINISHED", "completed", "COMPLETE"} and "results" in data:
                        return data
                    if "results" in data:
                        # sometimes it just returns results directly
                        return data
            except Exception:
                continue
        time.sleep(poll_sleep)

    raise RuntimeError(f"Timed out polling RNAcentral sequence-search job: {job_url}")


def pick_best_similarity_hit(result_json: dict):
    """
    Try to pick the top hit URS + basic alignment metrics if present.
    Different servers return different keys, so parse defensively.
    """
    if not isinstance(result_json, dict):
        return None

    # Most likely: {"results":[{...hit...}, ...]}
    hits = result_json.get("results")
    if hits is None and "hits" in result_json:
        hits = result_json.get("hits")
    if not isinstance(hits, list) or len(hits) == 0:
        return None

    h0 = hits[0]
    if not isinstance(h0, dict):
        return None

    urs = h0.get("rnacentral_id") or h0.get("urs") or h0.get("id")

    # alignment-ish fields (best-effort)
    ident = h0.get("identity") or h0.get("pct_identity") or h0.get("percent_identity")
    cov = h0.get("coverage") or h0.get("query_coverage") or h0.get("pct_coverage")
    evalue = h0.get("evalue")
    score = h0.get("score") or h0.get("bit_score")

    return {"rnacentral_id": urs, "identity": ident, "coverage": cov, "evalue": evalue, "score": score}


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Merge Rfam cmscan annotations + RNAcentral ID lookup (exact + optional alignment) + pathogenic variants flag"
    )
    ap.add_argument("--fasta", required=True, help="FASTA of RNA chains (e.g., all_rna_sequences_dataset_sapiens.fasta)")
    ap.add_argument("--rfam_tblout", required=True, help="cmscan --tblout output (e.g., rfam.tblout)")
    ap.add_argument("--out_csv", default="rna_chain_annotations_final.csv", help="Output CSV")
    ap.add_argument("--variants_csv", default=None, help="Optional CSV containing pathogenic variants; must have pdb_id,chain_id and either pathogenic/is_pathogenic or clinical_significance")
    ap.add_argument("--use_alignment", action="store_true", help="If exact md5 match fails, try RNAcentral sequence similarity search (alignment option)")
    ap.add_argument("--min_identity", type=float, default=0.95, help="Minimum identity to accept similarity hit (if provided by API)")
    ap.add_argument("--min_coverage", type=float, default=0.70, help="Minimum query coverage to accept similarity hit (if provided by API)")

    args = ap.parse_args()

    fasta_path = Path(args.fasta)
    tblout_path = Path(args.rfam_tblout)

    if not fasta_path.exists():
        sys.exit(f"ERROR: FASTA not found: {fasta_path}")
    if not tblout_path.exists():
        sys.exit(f"ERROR: Rfam tblout not found: {tblout_path}")

    seqs = read_fasta_as_dict(fasta_path)

    df_rfam = parse_rfam_tblout(tblout_path)
    if df_rfam.empty:
        sys.exit("ERROR: No parsable hits in rfam.tblout (did cmscan run successfully?)")

    # add pdb/chain, rna_type
    df_rfam[["pdb_id", "chain_id"]] = df_rfam["query"].apply(
        lambda q: pd.Series(split_query_to_pdb_chain(q))
    )
    df_rfam["rna_type"] = df_rfam["rfam_name"].apply(infer_rna_type_from_rfam_name)

    # exact match via md5
    df_rfam["rnacentral_exact_id"] = ""
    df_rfam["md5_dnaseq"] = ""

    if requests is None:
        print("[WARN] 'requests' not installed; skipping RNAcentral exact lookup. Install: pip install requests", file=sys.stderr)
    else:
        for i, row in df_rfam.iterrows():
            q = row["query"]
            seq = seqs.get(q, "")
            if not seq:
                continue
            md5 = md5_dnaseq_from_rna(seq)
            df_rfam.at[i, "md5_dnaseq"] = md5
            urs = rnacentral_lookup_by_md5(md5)
            if urs:
                df_rfam.at[i, "rnacentral_exact_id"] = urs

    # optional: similarity search (alignment option) only when exact missing
    df_rfam["rnacentral_align_id"] = ""
    df_rfam["align_identity"] = ""
    df_rfam["align_coverage"] = ""
    df_rfam["align_evalue"] = ""
    df_rfam["align_score"] = ""

    if args.use_alignment:
        if requests is None:
            sys.exit("ERROR: --use_alignment requires 'requests'. Install: pip install requests")

        for i, row in df_rfam.iterrows():
            if row["rnacentral_exact_id"]:
                continue

            q = row["query"]
            seq = seqs.get(q, "")
            if not seq:
                continue

            try:
                res = rnacentral_similarity_search(seq)
                best = pick_best_similarity_hit(res)
            except Exception as e:
                print(f"[WARN] similarity search failed for {q}: {e}", file=sys.stderr)
                continue

            if not best or not best.get("rnacentral_id"):
                continue

            # Apply thresholds only if the API returns identity/coverage numbers
            ident = best.get("identity")
            cov = best.get("coverage")

            def to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            ident_f = to_float(ident)
            cov_f = to_float(cov)

            accept = True
            if ident_f is not None and ident_f < args.min_identity:
                accept = False
            if cov_f is not None and cov_f < args.min_coverage:
                accept = False

            if accept:
                df_rfam.at[i, "rnacentral_align_id"] = best.get("rnacentral_id") or ""
                df_rfam.at[i, "align_identity"] = "" if ident is None else str(ident)
                df_rfam.at[i, "align_coverage"] = "" if cov is None else str(cov)
                df_rfam.at[i, "align_evalue"] = "" if best.get("evalue") is None else str(best.get("evalue"))
                df_rfam.at[i, "align_score"] = "" if best.get("score") is None else str(best.get("score"))

    # pathogenic variants flag per chain (optional)
    df_rfam["has_pathogenic_variants"] = False
    if args.variants_csv:
        vpath = Path(args.variants_csv)
        if not vpath.exists():
            print(f"[WARN] variants_csv not found, skipping: {vpath}", file=sys.stderr)
        else:
            v = pd.read_csv(vpath)

            # Heuristics for “pathogenic” column names
            # You can adapt to your real column names.
            path_col = None
            for c in ["pathogenic", "is_pathogenic", "is_pathogenic_variant", "pathogenicity"]:
                if c in v.columns:
                    path_col = c
                    break

            if path_col is None and "clinical_significance" in v.columns:
                # treat strings containing Pathogenic/Likely_pathogenic as True
                v["_is_path"] = v["clinical_significance"].astype(str).str.contains("pathogenic", case=False, na=False)
                path_col = "_is_path"

            needed = {"pdb_id", "chain_id", path_col} if path_col else {"pdb_id", "chain_id"}
            if path_col is None or not needed.issubset(set(v.columns)):
                print(
                    "[WARN] variants_csv does not have required columns. Need pdb_id, chain_id, and a pathogenic flag column.",
                    file=sys.stderr,
                )
            else:
                v2 = v[v[path_col].astype(bool)].copy()
                flagged = set(zip(v2["pdb_id"].astype(str).str.lower(), v2["chain_id"].astype(str)))
                df_rfam["has_pathogenic_variants"] = df_rfam.apply(
                    lambda r: (str(r["pdb_id"]).lower(), str(r["chain_id"])) in flagged, axis=1
                )

    # final minimal table (requested)
    out = df_rfam[
        [
            "pdb_id",
            "chain_id",
            "rna_type",
            "rfam_name",
            "rnacentral_exact_id",
            "rnacentral_align_id",
            "align_identity",
            "align_coverage",
            "has_pathogenic_variants",
        ]
    ].drop_duplicates().sort_values(["pdb_id", "chain_id"])

    out.to_csv(args.out_csv, index=False)

    print(f"✔ Written {args.out_csv}")
    print("\nRNA type counts:")
    print(out["rna_type"].value_counts())

    # quick sanity on how many got IDs
    n_exact = (out["rnacentral_exact_id"].astype(str) != "").sum()
    n_aln = (out["rnacentral_align_id"].astype(str) != "").sum()
    print(f"\nRNAcentral IDs: exact={n_exact}, alignment={n_aln}")


if __name__ == "__main__":
    main()