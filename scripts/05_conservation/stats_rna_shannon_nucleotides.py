#!/usr/bin/env python3
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

KEYS=["pdb_id","chain_id","query_pos"]; NTS=["A","C","G","U"]

def args():
    p=argparse.ArgumentParser(); p.add_argument("input_csv"); p.add_argument("output_dir")
    p.add_argument("--max-gap-fraction",type=float,default=None); p.add_argument("--contact-only",action="store_true")
    return p.parse_args()

def summ(x):
    x=pd.to_numeric(x,errors="coerce").dropna()
    return {"n":len(x),"mean":x.mean() if len(x) else np.nan,"median":x.median() if len(x) else np.nan,"q1":x.quantile(.25) if len(x) else np.nan,"q3":x.quantile(.75) if len(x) else np.nan}

def main():
    a=args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.input_csv,low_memory=False); d=raw.copy()
    need=set(KEYS+["shannon_conservation"]); miss=need-set(d.columns)
    if miss: raise ValueError(f"Faltan columnas: {sorted(miss)}")
    ntcol="nt" if "nt" in d.columns else ("rna_resname" if "rna_resname" in d.columns else None)
    if ntcol is None: raise ValueError("No existe columna nt ni rna_resname")
    d["pdb_id"]=d["pdb_id"].astype(str).str.upper().str.strip(); d["chain_id"]=d["chain_id"].astype(str).str.strip()
    d["query_pos"]=pd.to_numeric(d["query_pos"],errors="coerce"); d["shannon_conservation"]=pd.to_numeric(d["shannon_conservation"],errors="coerce")
    d["nt_norm"]=d[ntcol].astype(str).str.upper().str.strip().replace({"T":"U"})
    if "gap_fraction" in d.columns: d["gap_fraction"]=pd.to_numeric(d["gap_fraction"],errors="coerce")
    for c in ["contact_count"]:
        if c in d.columns: d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0)
    for c in ["has_contact","is_rna_interface"]:
        if c in d.columns: d[c]=d[c].astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(False)
    d=d[d.nt_norm.isin(NTS)].dropna(subset=KEYS+["shannon_conservation"])
    agg={"shannon_conservation":"median","nt_norm":"first"}
    if "gap_fraction" in d.columns: agg["gap_fraction"]="median"
    if "contact_count" in d.columns: agg["contact_count"]="max"
    if "has_contact" in d.columns: agg["has_contact"]="any"
    if "is_rna_interface" in d.columns: agg["is_rna_interface"]="any"
    d=d.groupby(KEYS,as_index=False).agg(agg)
    if a.max_gap_fraction is not None:
        if "gap_fraction" not in d.columns: raise ValueError("No existe gap_fraction")
        d=d[d.gap_fraction<a.max_gap_fraction].copy()
    if a.contact_only:
        if "has_contact" in d.columns: d=d[d.has_contact]
        elif "is_rna_interface" in d.columns: d=d[d.is_rna_interface]
        elif "contact_count" in d.columns: d=d[d.contact_count>=1]
        else: raise ValueError("--contact-only requiere has_contact, is_rna_interface o contact_count")
    d.to_csv(out/"dataset_rna_nucleotides_deduplicado.csv",index=False)
    desc=pd.DataFrame([{"nt":nt,**summ(d.loc[d.nt_norm==nt,"shannon_conservation"])} for nt in NTS])
    desc.to_csv(out/"resumen_shannon_por_nucleotido.csv",index=False)
    present=[nt for nt in NTS if (d.nt_norm==nt).sum()>0]; groups=[d.loc[d.nt_norm==nt,"shannon_conservation"].dropna() for nt in present]
    if len(groups)>=2:
        h,p=kruskal(*groups); kw=pd.DataFrame([{"groups":";".join(present),"H":h,"p_value":p,"n_total":sum(map(len,groups))}])
    else: kw=pd.DataFrame()
    kw.to_csv(out/"kruskal_nucleotidos.csv",index=False)
    rows=[]
    for a1,a2 in combinations(present,2):
        x=d.loc[d.nt_norm==a1,"shannon_conservation"].dropna(); y=d.loc[d.nt_norm==a2,"shannon_conservation"].dropna()
        u,p=mannwhitneyu(x,y,alternative="two-sided")
        rows.append({"nt_1":a1,"nt_2":a2,"n_1":len(x),"n_2":len(y),"median_1":x.median(),"median_2":y.median(),"U":u,"p_value":p,"cliffs_delta_from_U":2*u/(len(x)*len(y))-1})
    pair=pd.DataFrame(rows)
    if not pair.empty: pair["p_adj_bh"]=multipletests(pair.p_value,method="fdr_bh")[1]
    pair.to_csv(out/"pairwise_nucleotidos.csv",index=False)
    with open(out/"reporte_rna_shannon_nucleotides.txt","w") as f:
        f.write("5.2.6 — CONSERVACIÓN ENTRE TIPOS DE NUCLEÓTIDOS\n")
        f.write("Valores mayores de shannon_conservation = mayor conservación.\n\n")
        f.write(f"Filas originales: {len(raw)}\nNucleótidos deduplicados: {len(d)}\nFiltro gap_fraction: {a.max_gap_fraction}\nSolo contacto: {a.contact_only}\n\n")
        f.write("Resumen:\n"+desc.to_string(index=False)+"\n\n")
        f.write("Kruskal–Wallis:\n"+(kw.to_string(index=False) if not kw.empty else "Sin resultados")+"\n\n")
        f.write("Post hoc:\n"+(pair.to_string(index=False) if not pair.empty else "Sin resultados")+"\n")
    print("Listo:",out)
if __name__=="__main__": main()
