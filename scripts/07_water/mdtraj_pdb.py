import os
import glob
import mdtraj as md

# Input/output folders
CIF_FOLDER = "cifs"     # folder with your .cif files
PDB_FOLDER = "pdbs_mdtraj"     # folder to save .pdb files
os.makedirs(PDB_FOLDER, exist_ok=True)

def convert_cif_to_pdb(cif_file):
    """Convert a mmCIF file to PDB format using MDTraj."""
    try:
        traj = md.load(cif_file)  # MDTraj automatically detects .cif
        pdb_file = os.path.join(PDB_FOLDER, os.path.basename(cif_file).replace(".cif", ".pdb"))
        traj.save_pdb(pdb_file)
        print(f"Converted: {cif_file} -> {pdb_file}")
    except Exception as e:
        print(f"❌ Error converting {cif_file}: {e}")

if __name__ == "__main__":
    cif_files = glob.glob(os.path.join(CIF_FOLDER, "*.cif"))
    if not cif_files:
        print("⚠️ No CIF files found in", CIF_FOLDER)
    for cif_file in cif_files:
        convert_cif_to_pdb(cif_file)

