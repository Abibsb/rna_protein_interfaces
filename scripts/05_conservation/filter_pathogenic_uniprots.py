import pandas as pd
import glob

critical_polyphen = ["probably damaging", "possibly damaging"]
critical_sift = ["deleterious", "deleterious - low confidence"]

def is_pathogenic(row):
    # Disease association is explicitly stated
    if isinstance(row["Disease_Association"], str) and row["Disease_Association"].strip():
        return True
    
    # PolyPhen contains damaging prediction
    if isinstance(row["Prediction_PolyPhen"], str):
        if any(x in row["Prediction_PolyPhen"].lower() for x in critical_polyphen):
            return True
    
    # SIFT contains deleterious prediction
    if isinstance(row["Prediction_SIFT"], str):
        if any(x in row["Prediction_SIFT"].lower() for x in critical_sift):
            return True
    
    return False

pathogenic = []

for file in glob.glob("/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/20_ago_25//*_variants.csv"):
    df = pd.read_csv(file)
    if df.apply(is_pathogenic, axis=1).any():
        uniprot = file.split("/")[-1].replace("_variants.csv","")
        pathogenic.append(uniprot)

pd.Series(pathogenic).to_csv("uniprots_pathogenic.csv", index=False)
print(f"✅ Pathogenic UniProts: {len(pathogenic)} saved to uniprots_pathogenic.csv")

