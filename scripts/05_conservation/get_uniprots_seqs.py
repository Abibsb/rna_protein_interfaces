import pandas as pd
import requests

uniprots = pd.read_csv("pdb_chain_uniprot.csv")["UniProt_ID"].unique()
seqs = {}

for u in uniprots:
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{u}.fasta")
    if r.ok:
        seqs[u] = "".join(r.text.splitlines()[1:])
    else:
        seqs[u] = None

pd.DataFrame(list(seqs.items()), columns=["UniProt_ID","Sequence"]).to_csv("uniprot_sequences.csv", index=False)
print("✅ Saved uniprot_sequences.csv")

