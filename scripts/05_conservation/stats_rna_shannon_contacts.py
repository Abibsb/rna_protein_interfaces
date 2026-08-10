#!/usr/bin/env python3
import argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, spearmanr
from statsmodels.stats.multitest import multipletests

KEYS=["pdb_id","chain_id","query_pos"]

def args():
    p=argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("output_dir")
    p.add_argument("--max-gap-fraction",type=float,default=None)
    return p.parse_args()

def summ(x):
    x=pd.to_numeric(x,errors="coerce").dropna()
    return {"n":len(x),"mean":x.mean() if len(x) else np.nan,"median":x.median() if len(x) else np.nan,
            "q1":x.quantile(.25) if len(x) else np.nan,"q3":x.quantile(.75) if len(x) else np.nan}

def main():
    a=args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.input_csv,low_memory=False); d=raw.copy()
    need=set(KEYS+["shannon_conservation","contact_count"])
    miss=need-set(d.columns)
    if miss: raise ValueError(f"Faltan columnas: {sorted(miss)}")
    d["pdb_id"]=d["pdb_id"].astype(str).str.upper().str.strip(); d["chain_id"]=d["chain_id"].astype(str).str.strip()
    d["query_pos"]=pd.to_numeric(d["query_pos"],errors="coerce")
    for c in ["shannon_conservation","contact_count","unique_protein_residues","unique_protein_chains","gap_fraction"]:
        if c in d.columns: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["contact_count"]=d["contact_count"].fillna(0)
    d=d.dropna(subset=KEYS+["shannon_conservation"])
    agg={"shannon_conservation":"median","contact_count":"max"}
    for c in ["unique_protein_residues","unique_protein_chains"]:
        if c in d.columns: agg[c]="max"
    if "gap_fraction" in d.columns: agg["gap_fraction"]="median"
    if "nt" in d.columns: agg["nt"]="first"
    d=d.groupby(KEYS,as_index=False).agg(agg)
    if a.max_gap_fraction is not None:
        if "gap_fraction" not in d.columns: raise ValueError("No existe gap_fraction")
        d=d[d["gap_fraction"]<a.max_gap_fraction].copy()
    d["has_contact"]=d["contact_count"]>=1
    d.to_csv(out/"dataset_rna_contacts_deduplicado.csv",index=False)

    desc=pd.DataFrame([{"group":"sin_contacto",**summ(d.loc[~d.has_contact,"shannon_conservation"])},
                       {"group":"con_contacto",**summ(d.loc[d.has_contact,"shannon_conservation"])}])
    desc.to_csv(out/"resumen_contacto_binario.csv",index=False)
    x=d.loc[d.has_contact,"shannon_conservation"].dropna(); y=d.loc[~d.has_contact,"shannon_conservation"].dropna()
    if len(x) and len(y):
        u,p=mannwhitneyu(x,y,alternative="two-sided")
        mw=pd.DataFrame([{"comparison":"con_contacto_vs_sin_contacto","n_contact":len(x),"n_no_contact":len(y),
                          "median_contact":x.median(),"median_no_contact":y.median(),"U":u,"p_value":p,
                          "cliffs_delta_from_U":2*u/(len(x)*len(y))-1}])
    else: mw=pd.DataFrame()
    mw.to_csv(out/"mannwhitney_contacto_binario.csv",index=False)

    d["contact_bin"]=pd.cut(d["contact_count"],[-np.inf,0,2,np.inf],labels=["0","1-2",">=3"])
    bins=pd.DataFrame([{"contact_bin":g,**summ(d.loc[d.contact_bin.astype(str)==g,"shannon_conservation"])} for g in ["0","1-2",">=3"]])
    bins.to_csv(out/"resumen_bins_contactos.csv",index=False)
    present=[g for g in ["0","1-2",">=3"] if (d.contact_bin.astype(str)==g).sum()>0]
    groups=[d.loc[d.contact_bin.astype(str)==g,"shannon_conservation"].dropna() for g in present]
    if len(groups)>=2:
        h,p=kruskal(*groups); kw=pd.DataFrame([{"groups":";".join(present),"H":h,"p_value":p,"n_total":sum(map(len,groups))}])
    else: kw=pd.DataFrame()
    kw.to_csv(out/"kruskal_bins_contactos.csv",index=False)

    rows=[]
    for g1,g2 in combinations(present,2):
        a1=d.loc[d.contact_bin.astype(str)==g1,"shannon_conservation"].dropna(); a2=d.loc[d.contact_bin.astype(str)==g2,"shannon_conservation"].dropna()
        u,p=mannwhitneyu(a1,a2,alternative="two-sided")
        rows.append({"group_1":g1,"group_2":g2,"n_1":len(a1),"n_2":len(a2),"median_1":a1.median(),"median_2":a2.median(),"U":u,"p_value":p,"cliffs_delta_from_U":2*u/(len(a1)*len(a2))-1})
    pair=pd.DataFrame(rows)
    if not pair.empty: pair["p_adj_bh"]=multipletests(pair.p_value,method="fdr_bh")[1]
    pair.to_csv(out/"pairwise_bins_contactos.csv",index=False)

    corr=[]
    for v in [c for c in ["contact_count","unique_protein_residues","unique_protein_chains"] if c in d.columns]:
        s=d.dropna(subset=["shannon_conservation",v])
        if len(s)>1 and s[v].nunique()>1:
            rho,p=spearmanr(s.shannon_conservation,s[v]); corr.append({"variable":v,"n":len(s),"spearman_rho":rho,"p_value":p})
    corr=pd.DataFrame(corr)
    if not corr.empty: corr["p_adj_bh"]=multipletests(corr.p_value,method="fdr_bh")[1]
    corr.to_csv(out/"spearman_shannon_contactos.csv",index=False)

    with open(out/"reporte_rna_shannon_contacts.txt","w") as f:
        f.write("5.2.5 — CONSERVACIÓN DE ARN Y CONTACTOS PROTEICOS\n")
        f.write("Valores mayores de shannon_conservation = mayor conservación.\n\n")
        f.write(f"Filas originales: {len(raw)}\nNucleótidos deduplicados: {len(d)}\nFiltro gap_fraction: {a.max_gap_fraction}\n\n")
        f.write("Mann–Whitney:\n"+(mw.to_string(index=False) if not mw.empty else "Sin resultados")+"\n\n")
        f.write("Kruskal–Wallis:\n"+(kw.to_string(index=False) if not kw.empty else "Sin resultados")+"\n\n")
        f.write("Post hoc:\n"+(pair.to_string(index=False) if not pair.empty else "Sin resultados")+"\n\n")
        f.write("Spearman:\n"+(corr.to_string(index=False) if not corr.empty else "Sin resultados")+"\n")
    print("Listo:",out)
if __name__=="__main__": main()
