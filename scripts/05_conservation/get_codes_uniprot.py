import json
import os
import pandas as pd

# Paths
csv_file = "pdbid_chains_combined.csv"  
json_folder = "output_uniprot"


#"uniprot_json_files"  # Folder containing UniProt JSON files

# Load CSV data
df = pd.read_csv(csv_file)

# Dictionary to store results
filtered_results = []

# Process each row in the CSV file
for _, row in df.iterrows():
    pdb_id = row["PDBID"]
    chains = set(row["Chains"].replace('"', '').split(","))  # Convert to a set for easy lookup

    json_file = os.path.join(json_folder, f"{pdb_id}.json")
    if not os.path.exists(json_file):
        print(f"Warning: JSON file not found for {pdb_id}")
        continue

    # Load the corresponding JSON file
    with open(json_file, "r") as f:
        json_data = json.load(f)

    # Extract UniProt mappings
    if pdb_id.lower() in json_data:
        uniprot_mappings = json_data[pdb_id.lower()].get("UniProt", {})

        for uniprot_id, details in uniprot_mappings.items():
            for mapping in details["mappings"]:
                if mapping["chain_id"] in chains:
                    filtered_results.append({
                        "PDBID": pdb_id,
                        "Chain_ID": mapping["chain_id"],
                        "UniProt_ID": uniprot_id
                    })

# Convert to DataFrame and save to CSV
filtered_df = pd.DataFrame(filtered_results)
filtered_df.to_csv("pdb_uniprot_filtered.csv", index=False)

print("Filtered results saved to pdb_uniprot_filtered.csv")

