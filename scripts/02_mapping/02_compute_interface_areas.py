#!/usr/bin/env python3
import os
import csv
from collections import defaultdict

import freesasa
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


# =========================
# CONFIG
# =========================
cif_dir = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/27_ago_25/dataset_complejos_sapiens/"
chain_types_csv = os.path.join(cif_dir, "chain_types.csv")

out_complex = os.path.join(cif_dir, "interface_areas_per_complex.csv")
out_residue = os.path.join(cif_dir, "interface_areas_per_residue.csv")

PROTEIN_TYPES = {"polypeptide(L)"}
RNA_TYPES = {"polyribonucleotide"}

WATER_RESNAMES = {"HOH", "WAT", "H2O"}

# =========================
# Load chain_types.csv
# =========================
def load_chain_types(chain_csv_path):
    chain_map = defaultdict(lambda: {"by_asym": {}, "by_auth": {}})

    with open(chain_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row["pdb_id"].upper().strip()
            asym_id = row["asym_id"].strip()
            auth_id = row["auth_asym_id"].strip()
            polymer_type = row["polymer_type"].strip()

            chain_map[pdb_id]["by_asym"][asym_id] = {
                "auth_asym_id": auth_id,
                "polymer_type": polymer_type
            }
            chain_map[pdb_id]["by_auth"][auth_id] = {
                "asym_id": asym_id,
                "polymer_type": polymer_type
            }

    return chain_map


CHAIN_TYPES = load_chain_types(chain_types_csv)


def classify_chain(pdb_id, label_asym_id, auth_asym_id):
    """
    Decide polymer_class usando chain_types.csv.
    Probamos por label_asym_id y si no está, por auth_asym_id.
    """
    pdb_id = pdb_id.upper()

    info = CHAIN_TYPES.get(pdb_id, {}).get("by_asym", {}).get(label_asym_id)
    if info is None:
        info = CHAIN_TYPES.get(pdb_id, {}).get("by_auth", {}).get(auth_asym_id)

    if info is None:
        return "other", "unknown"

    poly_type = info["polymer_type"]

    if poly_type in PROTEIN_TYPES:
        return "protein", poly_type
    if poly_type in RNA_TYPES:
        return "rna", poly_type

    return "other", poly_type


# =========================
# Build freesasa structure from atom_site
# =========================
def build_freesasa_from_atom_site(mm, pdb_id, include_classes):
    """
    Returns:
      fs_struct: freesasa.Structure
      atom_meta: list of dict metadata per atom
    """
    fs_struct = freesasa.Structure()
    atom_meta = []

    # Required atom_site fields
    group_PDB = mm.get("_atom_site.group_PDB", [])
    label_asym_id = mm.get("_atom_site.label_asym_id", [])
    auth_asym_id = mm.get("_atom_site.auth_asym_id", label_asym_id)
    label_seq_id = mm.get("_atom_site.label_seq_id", [])
    auth_seq_id = mm.get("_atom_site.auth_seq_id", label_seq_id)

    label_comp_id = mm.get("_atom_site.label_comp_id", [])
    auth_comp_id = mm.get("_atom_site.auth_comp_id", label_comp_id)

    label_atom_id = mm.get("_atom_site.label_atom_id", [])
    auth_atom_id = mm.get("_atom_site.auth_atom_id", label_atom_id)

    Cartn_x = mm.get("_atom_site.Cartn_x", [])
    Cartn_y = mm.get("_atom_site.Cartn_y", [])
    Cartn_z = mm.get("_atom_site.Cartn_z", [])

    # Make sure they're lists
    if isinstance(label_asym_id, str):
        # Single-atom weird case
        label_asym_id = [label_asym_id]
        auth_asym_id = [auth_asym_id]
        label_seq_id = [label_seq_id]
        auth_seq_id = [auth_seq_id]
        label_comp_id = [label_comp_id]
        auth_comp_id = [auth_comp_id]
        label_atom_id = [label_atom_id]
        auth_atom_id = [auth_atom_id]
        Cartn_x = [Cartn_x]
        Cartn_y = [Cartn_y]
        Cartn_z = [Cartn_z]
        group_PDB = [group_PDB]

    for i in range(len(label_asym_id)):
        # Keep only ATOM (ignore HETATM ligands)
        if group_PDB and group_PDB[i] != "ATOM":
            continue

        lab_chain = str(label_asym_id[i]).strip()
        auth_chain = str(auth_asym_id[i]).strip()

        poly_class, poly_type = classify_chain(pdb_id, lab_chain, auth_chain)
        if poly_class not in include_classes:
            continue

        resname = str(label_comp_id[i]).strip().upper()

        if resname in WATER_RESNAMES:
            continue

        # residue number for freesasa: we use auth_seq_id if possible, else label_seq_id
        # freesasa needs an int residue number, so we must convert
        seq_raw = auth_seq_id[i] if auth_seq_id else label_seq_id[i]

        try:
            resnum = int(float(seq_raw))
        except Exception:
            # Some atoms may lack seq_id (rare), skip them
            continue

        atomname = str(label_atom_id[i]).strip()

        try:
            x = float(Cartn_x[i])
            y = float(Cartn_y[i])
            z = float(Cartn_z[i])
        except Exception:
            continue

        fs_struct.addAtom(
            atomname,
            resname,
            resnum,
            lab_chain,   # store label_asym_id as chain for freesasa indexing
            x, y, z
        )

        atom_meta.append({
            "polymer_class": poly_class,
            "polymer_type": poly_type,
            "label_asym_id": lab_chain,
            "auth_asym_id": auth_chain,
            "label_seq_id": str(label_seq_id[i]).strip() if label_seq_id else "",
            "auth_seq_id": str(auth_seq_id[i]).strip() if auth_seq_id else "",
            "resnum_used": resnum,
            "resname": resname
        })

    return fs_struct, atom_meta


def accumulate_residue_sasa(atom_areas, atom_meta):
    res_sasa = defaultdict(float)
    for area, meta in zip(atom_areas, atom_meta):
        key = (
            meta["polymer_class"],
            meta["label_asym_id"],
            meta["auth_asym_id"],
            meta["resnum_used"],
            meta["resname"],
            meta["label_seq_id"],
            meta["auth_seq_id"]
        )
        res_sasa[key] += area
    return res_sasa


# =========================
# Main loop
# =========================
complex_rows = []
residue_rows = []

for fname in sorted(os.listdir(cif_dir)):
    if not fname.lower().endswith(".cif"):
        continue

    pdb_id = os.path.splitext(fname)[0].upper()
    cif_path = os.path.join(cif_dir, fname)

    print(f"Processing {pdb_id} ...")

    try:
        mm = MMCIF2Dict(cif_path)
    except Exception as e:
        print(f"  Failed to read {fname}: {e}")
        continue

    S_prot, meta_prot = build_freesasa_from_atom_site(mm, pdb_id, {"protein"})
    S_rna,  meta_rna  = build_freesasa_from_atom_site(mm, pdb_id, {"rna"})
    S_full, meta_full = build_freesasa_from_atom_site(mm, pdb_id, {"protein", "rna"})

    print(f"DEBUG {pdb_id}: prot_atoms={S_prot.nAtoms()} rna_atoms={S_rna.nAtoms()} full_atoms={S_full.nAtoms()}")

    if S_prot.nAtoms() == 0 or S_rna.nAtoms() == 0:
        print(f"  Skipping {pdb_id} (missing protein or RNA atoms)")
        continue

    # SASA calculations
    R_prot = freesasa.calc(S_prot)
    R_rna  = freesasa.calc(S_rna)
    R_full = freesasa.calc(S_full)

    prot_atom_areas_unbound = [R_prot.atomArea(i) for i in range(S_prot.nAtoms())]
    rna_atom_areas_unbound  = [R_rna.atomArea(i)  for i in range(S_rna.nAtoms())]
    full_atom_areas         = [R_full.atomArea(i) for i in range(S_full.nAtoms())]

    sasa_prot_unbound = sum(prot_atom_areas_unbound)
    sasa_rna_unbound  = sum(rna_atom_areas_unbound)
    sasa_full         = sum(full_atom_areas)

    # bound SASA split by polymer_class using meta_full
    sasa_prot_bound = 0.0
    sasa_rna_bound  = 0.0
    for area, meta in zip(full_atom_areas, meta_full):
        if meta["polymer_class"] == "protein":
            sasa_prot_bound += area
        elif meta["polymer_class"] == "rna":
            sasa_rna_bound += area

    # BSA
    BSA_total = (sasa_prot_unbound + sasa_rna_unbound) - sasa_full
    BSA_prot  = sasa_prot_unbound - sasa_prot_bound
    BSA_rna   = sasa_rna_unbound  - sasa_rna_bound

    complex_rows.append({
        "pdb_id": pdb_id,
        "sasa_prot_unbound": sasa_prot_unbound,
        "sasa_rna_unbound": sasa_rna_unbound,
        "sasa_full": sasa_full,
        "sasa_prot_bound": sasa_prot_bound,
        "sasa_rna_bound": sasa_rna_bound,
        "bsa_total": BSA_total,
        "bsa_prot": BSA_prot,
        "bsa_rna": BSA_rna,
    })

    # Per-residue SASA
    prot_res_unbound = accumulate_residue_sasa(prot_atom_areas_unbound, meta_prot)
    rna_res_unbound  = accumulate_residue_sasa(rna_atom_areas_unbound,  meta_rna)

    prot_res_bound = defaultdict(float)
    rna_res_bound  = defaultdict(float)

    for area, meta in zip(full_atom_areas, meta_full):
        key = (
            meta["polymer_class"],
            meta["label_asym_id"],
            meta["auth_asym_id"],
            meta["resnum_used"],
            meta["resname"],
            meta["label_seq_id"],
            meta["auth_seq_id"]
        )
        if meta["polymer_class"] == "protein":
            prot_res_bound[key] += area
        elif meta["polymer_class"] == "rna":
            rna_res_bound[key] += area

    all_keys = set(prot_res_unbound) | set(rna_res_unbound) | set(prot_res_bound) | set(rna_res_bound)

    for key in sorted(all_keys):
        polymer_class, label_chain, auth_chain, resnum_used, resname, label_seq, auth_seq = key

        if polymer_class == "protein":
            sasa_unbound = prot_res_unbound.get(key, 0.0)
            sasa_bound   = prot_res_bound.get(key, 0.0)
        elif polymer_class == "rna":
            sasa_unbound = rna_res_unbound.get(key, 0.0)
            sasa_bound   = rna_res_bound.get(key, 0.0)
        else:
            continue

        delta_sasa = sasa_unbound - sasa_bound
        is_interface = int(delta_sasa > 0.1)

        residue_rows.append({
            "pdb_id": pdb_id,
            "polymer_class": polymer_class,
            "label_asym_id": label_chain,
            "auth_asym_id": auth_chain,
            "resnum_used": resnum_used,
            "resname": resname,
            "label_seq_id": label_seq,
            "auth_seq_id": auth_seq,
            "sasa_unbound": sasa_unbound,
            "sasa_bound": sasa_bound,
            "delta_sasa": delta_sasa,
            "is_interface": is_interface,
        })


# =========================
# Write outputs
# =========================
with open(out_complex, "w", newline="") as f:
    fieldnames = [
        "pdb_id",
        "sasa_prot_unbound",
        "sasa_rna_unbound",
        "sasa_full",
        "sasa_prot_bound",
        "sasa_rna_bound",
        "bsa_total",
        "bsa_prot",
        "bsa_rna",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(complex_rows)

with open(out_residue, "w", newline="") as f:
    fieldnames = [
        "pdb_id",
        "polymer_class",
        "label_asym_id",
        "auth_asym_id",
        "resnum_used",
        "resname",
        "label_seq_id",
        "auth_seq_id",
        "sasa_unbound",
        "sasa_bound",
        "delta_sasa",
        "is_interface",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(residue_rows)

print(f"Wrote per-complex results to: {out_complex}")
print(f"Wrote per-residue results to: {out_residue}")
