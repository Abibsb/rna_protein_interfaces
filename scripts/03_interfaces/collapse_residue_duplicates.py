import pandas as pd

df = pd.read_csv("interface_analysis_dataset.csv", low_memory=False)

print("Rows before:", len(df))


group_cols = [
    "pdb_id",
    "auth_asym_id",
    "resnum_used"
]

agg_dict = {
    "resname": "first",
    "is_interface_abs": "max",
    "is_strong_interface": "max",
    "delta_sasa": "first",
    "buried_fraction_residue": "first",
    "delta_sasa_percentile_global": "first",
    "consurf_score": "first",
    "iupred_score": "first",
    "is_disordered": "max",
    "n_bridging_waters": "max",
    "has_bridging_water": "max",
    "rna_contact_count": "max",
    "rna_unique_nucleotides": "max",
    "n_variants": "max",
    "n_pathogenic": "max",
    "has_variant": "max",
    "has_pathogenic": "max",
    "has_bad_mutation": "max",
    "is_interface": "max",
    "is_hotspot_candidate": "max"
}


df2 = (
    df.groupby(group_cols)
    .agg(agg_dict)
    .reset_index()
)

print("Rows after:", len(df2))

df2.to_csv("interface_analysis_dataset_residue_level.csv", index=False)

print("Saved: interface_analysis_dataset_residue_level.csv")