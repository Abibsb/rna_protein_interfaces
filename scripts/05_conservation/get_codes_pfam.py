import json
import os
import pandas as pd

# Paths
csv_file = "pdbid_chains_combined.csv"  
json_folder = "output_jsons"
# Load CSV data
df = pd.read_csv(csv_file)

filtered_results = []

# Process each row in the CSV file
for _, row in df.iterrows():
    pdb_id = row["PDBID"]
    chains = set(row["Chains"].replace('"', '').split(","))  

    json_file = os.path.join(json_folder, f"{pdb_id}.json")  # Update if filename format is different
    if not os.path.exists(json_file):
        print(f"⚠ Warning: JSON file not found for {pdb_id}")
        continue

    # Load the JSON
    with open(json_file, "r") as f:
        json_data = json.load(f)

    # Convert PDB ID to lowercase to match JSON format
    json_pdb_key = pdb_id.lower()

    # Debug: Print available keys in JSON
    print(f"🔍 JSON keys in {json_file}: {json_data.keys()}")

    # Check if the PDB ID exists
    if json_pdb_key not in json_data:
        print(f"⚠ Warning: {pdb_id} (converted to {json_pdb_key}) not found in JSON file")
        continue

    # Check if Pfam key exists
    if "Pfam" not in json_data[json_pdb_key]:
        print(f"⚠ Warning: 'Pfam' key not found in JSON for {pdb_id}")
        continue

    # Extract Pfam mappings
    pfam_mappings = json_data[json_pdb_key]["Pfam"]
    print(f"✅ Found Pfam mappings for {pdb_id}")

    for pfam_id, details in pfam_mappings.items():
        for mapping in details["mappings"]:
            print(f"🔍 Checking Chain {mapping['chain_id']} -> Pfam {pfam_id}")
            if mapping["chain_id"] in chains:
                print(f"✅ Match found: PDB {pdb_id}, Chain {mapping['chain_id']}, Pfam {pfam_id}")
                filtered_results.append({
                    "PDBID": pdb_id,
                    "Chain_ID": mapping["chain_id"],
                    "Pfam_ID": pfam_id
                })

# Save to CSV
filtered_df = pd.DataFrame(filtered_results)

if not filtered_df.empty:
    filtered_df.to_csv("pdb_pfam_filtered.csv", index=False)
    print("✅ Filtered results saved to pdb_pfam_filtered.csv")
else:
    print("⚠ No matches found. The output CSV is empty.")
