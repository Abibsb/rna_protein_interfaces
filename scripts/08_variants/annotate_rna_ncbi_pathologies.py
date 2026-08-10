#!/usr/bin/env python3

import pandas as pd
import requests
import time
import json
import os

INPUT = "rna_metadata_pathogenicity.csv"
OUTPUT = "rna_metadata_ncbi_pathologies.csv"
CACHE  = "ncbi_gene_cache.json"

SLEEP = 0.34

# ------------------------------------------------------------
# load table
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# cache (important — avoids killing NCBI)
# ------------------------------------------------------------

if os.path.exists(CACHE):
    cache = json.load(open(CACHE))
else:
    cache = {}

# ------------------------------------------------------------
# NCBI fetch
# ------------------------------------------------------------

def fetch_gene_summary(gene_id):

    gene_id = str(int(gene_id))

    if gene_id in cache:
        return cache[gene_id]

    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=gene&id={gene_id}&retmode=json"
    )

    try:
        r = requests.get(url, timeout=60)
        js = r.json()

        rec = js["result"].get(gene_id)

        if not rec:
            cache[gene_id] = None
            return None

        out = {
            "name": rec.get("name"),
            "description": rec.get("description"),
            "summary": rec.get("summary"),
            "organism": rec.get("organism", {}).get("scientificname")
        }

        cache[gene_id] = out
        return out

    except Exception:
        cache[gene_id] = None
        return None


# ------------------------------------------------------------
# enrichment
# ------------------------------------------------------------

names = []
descriptions = []
summaries = []
organisms = []

print("\nFetching NCBI gene information...\n")

for _, row in df.iterrows():

    gene_id = row.get("ncbi_gene_id")

    if pd.isna(gene_id):
        names.append(None)
        descriptions.append(None)
        summaries.append(None)
        organisms.append(None)
        continue

    info = fetch_gene_summary(gene_id)

    if info:
        names.append(info["name"])
        descriptions.append(info["description"])
        summaries.append(info["summary"])
        organisms.append(info["organism"])
    else:
        names.append(None)
        descriptions.append(None)
        summaries.append(None)
        organisms.append(None)

    time.sleep(SLEEP)

# ------------------------------------------------------------
# add columns
# ------------------------------------------------------------

df["ncbi_gene_name"] = names
df["ncbi_gene_description"] = descriptions
df["ncbi_gene_summary"] = summaries
df["ncbi_gene_organism"] = organisms

df.to_csv(OUTPUT, index=False)

json.dump(cache, open(CACHE,"w"), indent=2)

print("\nDONE — saved:", OUTPUT)