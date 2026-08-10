#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

KEYS = ["pdb_id", "uniprot_id", "uniprot_pos"]
CLASS_ORDER = {"non_interface":0,"interface":1,"strong_interface":2,"core_hotspot":3}

def args():
    p=argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("output_dir")
    p.add_argument("--exclude-low-confidence",action="store_true")
    return p.parse_args()

def to_bool(s):
    return s.astype(str).str.strip().str.lower().map({"true":True,"false":False,"1":True,"0":False})

def severe(x):
    vals=[v for v in x.dropna().astype(str) if v in CLASS_ORDER]
    return max(vals,key=lambda z:CLASS_ORDER[z]) if vals else np.nan

def dedup(df):
    req=KEYS+["polymer_class","interface_class","consurf_score","delta_sasa","buried_fraction_residue","is_low_conf"]
    miss=[c for c in req if c not in df.columns]
    if miss: raise ValueError(f"Faltan columnas: {miss}")
    d=df.copy()
    d["pdb_id"]=d["pdb_id"].astype(str).str.upper().str.strip()
    d["uniprot_id"]=d["uniprot_id"].astype(str).str.strip()
    d["uniprot_pos"]=pd.to_numeric(d["uniprot_pos"],errors="coerce")
    for c in ["consurf_score","delta_sasa","buried_fraction_residue"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d["is_low_conf"]=to_bool(d["is_low_conf"]).fillna(False)
    d=d[d["polymer_class"].astype(str).str.contains("protein",case=False,na=False)]
    d=d.dropna(subset=KEYS+["consurf_score","interface_class"])
    d=d.drop_duplicates(subset=KEYS+["interface_class","consurf_score","delta_sasa","buried_fraction_residue","is_low_conf"])
    return d.groupby(KEYS,as_index=False).agg(
        interface_class=("interface_class",severe),
        consurf_score=("consurf_score","median"),
        delta_sasa=("delta_sasa","median"),
        buried_fraction_residue=("buried_fraction_residue","median"),
        is_low_conf=("is_low_conf","any"),
        n_source_rows=("consurf_score","size"),
    )

def summary(df,col):
    out=[]
    for g,s in df.groupby(col,dropna=False):
        x=s["consurf_score"].dropna()
        if len(x):
            out.append({"group":g,"n":len(x),"mean_consurf":x.mean(),"median_consurf":x.median(),"q1_consurf":x.quantile(.25),"q3_consurf":x.quantile(.75)})
    return pd.DataFrame(out)

def mw(df,label,ma,mb,ga,gb):
    a=df.loc[ma,"consurf_score"].dropna(); b=df.loc[mb,"consurf_score"].dropna()
    if not len(a) or not len(b): return None
    u,p=mannwhitneyu(a,b,alternative="two-sided")
    return {"comparison":label,"group_a":ga,"group_b":gb,"n_a":len(a),"n_b":len(b),"median_a":a.median(),"median_b":b.median(),"U":u,"p_value":p}

def main():
    a=args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.input_csv,low_memory=False)
    df=dedup(raw)
    if a.exclude_low_confidence: df=df[~df["is_low_conf"]].copy()
    df.to_csv(out/"dataset_conservacion_deduplicado.csv",index=False)
    df["is_interface"]=df["interface_class"].ne("non_interface")
    df["is_strong_or_core"]=df["interface_class"].isin(["strong_interface","core_hotspot"])
    df["is_core"]=df["interface_class"].eq("core_hotspot")
    summary(df,"interface_class").to_csv(out/"resumen_grupos_interfaz.csv",index=False)
    rows=[
        mw(df,"interface_vs_non_interface",df["is_interface"],~df["is_interface"],"interface","non_interface"),
        mw(df,"strong_or_core_vs_interface_only",df["is_strong_or_core"],df["interface_class"].eq("interface"),"strong_or_core","interface_only"),
        mw(df,"core_vs_interface_non_core",df["is_core"],df["interface_class"].isin(["interface","strong_interface"]),"core_hotspot","interface_non_core")
    ]
    comp=pd.DataFrame([r for r in rows if r])
    if len(comp): comp["p_adj_bh"]=multipletests(comp["p_value"],method="fdr_bh")[1]
    comp.to_csv(out/"mannwhitney_comparaciones_interfaz.csv",index=False)
    present=[c for c in CLASS_ORDER if (df["interface_class"]==c).sum()>0]
    groups=[df.loc[df["interface_class"]==c,"consurf_score"].dropna() for c in present]
    kw=pd.DataFrame()
    if len(groups)>=2:
        h,p=kruskal(*groups); kw=pd.DataFrame([{"classes":";".join(present),"H":h,"p_value":p,"n_total":sum(map(len,groups))}])
    kw.to_csv(out/"kruskal_clases_interfaz.csv",index=False)
    corr=[]
    for var in ["delta_sasa","buried_fraction_residue"]:
        s=df.dropna(subset=["consurf_score",var])
        if len(s)>1:
            rho,p=spearmanr(s["consurf_score"],s[var]); corr.append({"variable":var,"n":len(s),"spearman_rho":rho,"p_value":p})
    corr=pd.DataFrame(corr)
    if len(corr): corr["p_adj_bh"]=multipletests(corr["p_value"],method="fdr_bh")[1]
    corr.to_csv(out/"spearman_conservacion_estructura.csv",index=False)
    with open(out/"reporte_conservacion_interfaz.txt","w") as f:
        f.write("Más negativo en consurf_score = mayor conservación.\n")
        f.write(f"Filas originales: {len(raw)}\nResiduos deduplicados: {len(df)}\n")
        f.write(f"Exclusión baja confianza: {a.exclude_low_confidence}\n\n")
        f.write("Mann-Whitney\n"+comp.to_string(index=False)+"\n\n")
        f.write("Kruskal-Wallis\n"+kw.to_string(index=False)+"\n\n")
        f.write("Spearman\n"+corr.to_string(index=False)+"\n")
    print(out.resolve())
if __name__=="__main__": main()
