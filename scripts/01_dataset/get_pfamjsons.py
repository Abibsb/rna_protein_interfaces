import requests
import json
import pandas as pd
import os

def fetch_pdb_data(pdb_csv, output_dir, failed_log):
    """Fetch InterPro mappings for PDB IDs, save successful responses as JSON, and log failures in a CSV."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file and extract PDB IDs from the first row
    pdb_data = pd.read_csv(pdb_csv, header=None)  # Read without headers
    pdb_ids = pdb_data.iloc[0].dropna().astype(str).str.strip().tolist()  # Extract PDB IDs from first row

    failed_pdbs = []

    for pdb_id in pdb_ids:
        pdb_id = pdb_id.upper()  # Ensure uppercase PDB IDs
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/pfam/{pdb_id}"
        response = requests.get(url)

        if response.status_code == 200:
            interpro_data = response.json()
            json_path = os.path.join(output_dir, f"{pdb_id}.json")

            with open(json_path, "w") as json_file:
                json.dump(interpro_data, json_file, indent=4)
            print(f"Saved: {json_path}")

        else:
            print(f"Failed: {pdb_id} (Status: {response.status_code})")
            failed_pdbs.append([pdb_id, response.status_code])

    # Save failed requests to CSV
    if failed_pdbs:
        failed_df = pd.DataFrame(failed_pdbs, columns=["PDB_ID", "Status_Code"])
        failed_df.to_csv(failed_log, index=False)
        print(f"Failed PDBs logged in: {failed_log}")

# Example usage:
pdb_csv = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/29_jul_25/pdbids.csv"  # Update with actual CSV path
output_dir = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/29_jul_25/output_pfam"  # Directory to save JSON files
failed_log = "failed_pdb_requests_uniprot.csv"  # CSV file for failed requests
fetch_pdb_data(pdb_csv, output_dir, failed_log)
