#!/usr/bin/env python3
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import argparse
import json
import os
import numpy as np

# --------------------
# Parameters
# --------------------
CUT_HBOND = 3.5      # Å distance cutoff
ANGLE_MIN = 120.0    # degrees, angle at water oxygen (proxy for HBPLUS geometry)
protein_resnames = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
                    "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
rna_resnames = ["A","U","G","C","DA","DT","DG","DC"]
water_resnames = ["HOH","WAT"]

# Only polar heavy atoms to reduce noise
polar_elements = {"N", "O"}

# --------------------
# Helper functions
# --------------------
def angle_between(v1, v2):
    """Return the angle (degrees) between two vectors."""
    v1u = v1 / np.linalg.norm(v1)
    v2u = v2 / np.linalg.norm(v2)
    cosang = np.clip(np.dot(v1u, v2u), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

# --------------------
# Core analysis
# --------------------
def analyze_structure(pdb_file):
    u = mda.Universe(pdb_file)

    protein = u.select_atoms(f"resname {' '.join(protein_resnames)} and (name N* or name O*)")
    rna     = u.select_atoms(f"resname {' '.join(rna_resnames)} and (name N* or name O*)")
    waters  = u.select_atoms(f"resname {' '.join(water_resnames)} and name O*")

    hydration_shell = set()
    bridging_waters = {}
    bridging_waters_angle_filtered = {}

    for water in waters.residues:
        wat_pos = water.atoms.positions[0]  # only O atom in water

        # Compute distances to protein and RNA polar atoms
        prot_dists = distances.distance_array(np.array([wat_pos]), protein.positions)[0]
        rna_dists  = distances.distance_array(np.array([wat_pos]), rna.positions)[0]

        near_prot_idx = np.where(prot_dists < CUT_HBOND)[0]
        near_rna_idx  = np.where(rna_dists < CUT_HBOND)[0]

        near_prot = len(near_prot_idx) > 0
        near_rna  = len(near_rna_idx) > 0

        if near_prot or near_rna:
            hydration_shell.add(int(water.resid))

        if near_prot and near_rna:
            prot_residues = {(protein[i].resname, int(protein[i].resid)) for i in near_prot_idx}
            rna_residues  = {(rna[j].resname, int(rna[j].resid)) for j in near_rna_idx}
            bridging_waters[int(water.resid)] = {
                "protein_residues": sorted(prot_residues),
                "rna_residues": sorted(rna_residues)
            }

            # --- HBPLUS-like angle filter ---
            angle_ok = False
            for i in near_prot_idx:
                for j in near_rna_idx:
                    v1 = protein[i].position - wat_pos
                    v2 = rna[j].position - wat_pos
                    ang = angle_between(v1, v2)
                    if ang >= ANGLE_MIN:
                        angle_ok = True
                        break
                if angle_ok:
                    break

            if angle_ok:
                bridging_waters_angle_filtered[int(water.resid)] = {
                    "protein_residues": sorted(prot_residues),
                    "rna_residues": sorted(rna_residues),
                    "angle_min_deg": ANGLE_MIN
                }

    return {
        "total_waters_in_shell": len(hydration_shell),
        "bridging_waters_count": len(bridging_waters),
        "bridging_waters": sorted(list(bridging_waters.keys())),
        "bridging_water_partners": bridging_waters,
        "bridging_waters_angle_filtered_count": len(bridging_waters_angle_filtered),
        "bridging_waters_angle_filtered": sorted(list(bridging_waters_angle_filtered.keys())),
        "bridging_water_partners_angle_filtered": bridging_waters_angle_filtered
    }

# --------------------
# Main loop
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder", required=True, help="Folder with PDBs")
    ap.add_argument("--output_folder", required=True, help="Folder for JSON results")
    args = ap.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    pdb_files = [f for f in os.listdir(args.input_folder) if f.endswith(".pdb")]
    failed_files = []

    for pdb_file in pdb_files:
        pdb_path = os.path.join(args.input_folder, pdb_file)
        print(f"Processing {pdb_file}...")

        try:
            result = analyze_structure(pdb_path)
            out_file = os.path.join(args.output_folder, pdb_file.replace(".pdb", ".json"))
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"⚠️ Skipping {pdb_file}: {e}")
            failed_files.append(pdb_file)
            continue

    # Save failed files log
    if failed_files:
        log_path = os.path.join(args.output_folder, "failed_files.log")
        with open(log_path, "w") as log:
            for f in failed_files:
                log.write(f + "\n")
        print(f"\n⚠️ Some files failed. See {log_path}")

    print("\n✅ Processing complete.")

if __name__ == "__main__":
    main()

