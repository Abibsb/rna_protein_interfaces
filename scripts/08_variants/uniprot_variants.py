import requests
import csv

# Read UniProt IDs from the first column of uniprots.csv
uniprot_ids = []
with open("unique_uniprot_codes.csv", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if row:  # skip empty lines
            uniprot_ids.append(row[0].strip())

failed_ids = []  # to store failed UniProt IDs

for uid in uniprot_ids:
    url = f"https://www.ebi.ac.uk/proteins/api/variation/{uid}?format=json"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch data for {uid}")
        failed_ids.append(uid)
        continue

    data = r.json()
    variants = data.get("features", [])
    csv_file = f"{uid}_variants.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Position",
            "Wild_Type",
            "Mutated_Type",
            "Consequence",
            "Disease_Association",
            "Population_Frequencies",
            "Prediction_PolyPhen",
            "Prediction_SIFT",
            "References"
        ])

        for var in variants:
            begin = var.get("begin", "")
            end = var.get("end", "")
            pos = f"{begin}-{end}" if begin != end else str(begin)

            wt = var.get("wildType", "")
            mt = var.get("mutatedType", "")
            consequence = var.get("consequenceType", "")

            # Disease associations
            assoc = [a.get("name", "") for a in var.get("association", []) if a.get("disease", False)]
            disease_assoc = "; ".join(assoc)

            # Population frequencies
            pop_freqs = [f"{pf.get('populationName')}={pf.get('frequency')}" for pf in var.get("populationFrequencies", [])]
            pop_freq_str = "; ".join(pop_freqs)

            # Predictions
            polyphen = "; ".join([p.get("predictionValType", "") for p in var.get("predictions", []) if p.get("predAlgorithmNameType") == "PolyPhen"])
            sift = "; ".join([p.get("predictionValType", "") for p in var.get("predictions", []) if p.get("predAlgorithmNameType") == "SIFT"])

            # References
            refs = [f"{x.get('name')}:{x.get('id')}" for x in var.get("xrefs", [])]
            refs_str = "; ".join(refs)

            writer.writerow([pos, wt, mt, consequence, disease_assoc, pop_freq_str, polyphen, sift, refs_str])

    print(f"Saved {len(variants)} variants for {uid} → {csv_file}")

# Save failed IDs to a text file
if failed_ids:
    with open("failed_uniprots.txt", "w") as f:
        f.write("\n".join(failed_ids))
    print(f"\nSaved {len(failed_ids)} failed UniProt IDs to failed_uniprots.txt")
else:
    print("\nAll UniProt IDs processed successfully")

