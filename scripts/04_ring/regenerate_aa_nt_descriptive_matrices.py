#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

AA_ORDER = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
            "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
NT_ORDER = ["A","C","G","U"]


def matrix(df):
    return pd.crosstab(df["aa"], df["nt"]).reindex(
        index=AA_ORDER, columns=NT_ORDER, fill_value=0
    )


def save_group(df, label, outdir):
    counts = matrix(df)
    total = counts.values.sum()
    frac_total = counts / total if total else counts.astype(float)
    frac_by_nt = counts.div(counts.sum(axis=0).replace(0, pd.NA), axis=1)
    frac_by_aa = counts.div(counts.sum(axis=1).replace(0, pd.NA), axis=0)

    counts.to_csv(outdir / f"{label}_counts.csv")
    frac_total.to_csv(outdir / f"{label}_fraction_total.csv")
    frac_by_nt.to_csv(outdir / f"{label}_fraction_within_nt.csv")
    frac_by_aa.to_csv(outdir / f"{label}_fraction_within_aa.csv")

    top = counts.stack().rename("count").reset_index()
    top["pair"] = top["aa"] + "–" + top["nt"]
    top["fraction_total"] = top["count"] / total if total else 0
    top = top.sort_values("count", ascending=False)
    top[["pair","aa","nt","count","fraction_total"]].to_csv(
        outdir / f"{label}_top_pairs.csv", index=False
    )

    for kind, data, cbar in [
        ("counts", counts, "Contactos únicos"),
        ("fraction_total", frac_total, "Fracción del total"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 10))
        im = ax.imshow(data.astype(float).values, aspect="auto")
        ax.set_xticks(range(len(NT_ORDER)), NT_ORDER)
        ax.set_yticks(range(len(AA_ORDER)), AA_ORDER)
        ax.set_xlabel("Nucleótido")
        ax.set_ylabel("Aminoácido")
        ax.set_title(f"AA–NT {label.upper()} — {kind}")
        fig.colorbar(im, ax=ax, label=cbar)
        fig.tight_layout()
        fig.savefig(outdir / f"{label}_{kind}_heatmap.png", dpi=300)
        fig.savefig(outdir / f"{label}_{kind}_heatmap.pdf")
        plt.close(fig)

    return counts, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--outdir", default="aa_nt_descriptive_fixed")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    required = {"aa", "nt", "interaction_class"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Faltan columnas: {sorted(missing)}. Disponibles: {list(df.columns)}")

    df["aa"] = df["aa"].astype(str).str.upper().str.strip()
    df["nt"] = df["nt"].astype(str).str.upper().str.strip()
    df["interaction_class"] = df["interaction_class"].astype(str).str.upper().str.strip()
    df = df[df["aa"].isin(AA_ORDER) & df["nt"].isin(NT_ORDER) &
            df["interaction_class"].isin(["HBOND", "VDW"])].copy()

    global_counts, global_n = save_group(df, "global", outdir)
    hb_counts, hb_n = save_group(df[df["interaction_class"] == "HBOND"], "hbond", outdir)
    vdw_counts, vdw_n = save_group(df[df["interaction_class"] == "VDW"], "vdw", outdir)

    qc = [
        "QC matrices descriptivas AA–NT",
        "===============================",
        f"Contactos globales: {global_n}",
        f"HBOND: {hb_n}",
        f"VDW: {vdw_n}",
        f"Suma HBOND + VDW: {hb_n + vdw_n}",
        f"Coincide con global: {global_n == hb_n + vdw_n}",
        f"AA observados: {df['aa'].nunique()} / 20",
        f"NT observados: {df['nt'].nunique()} / 4",
        f"Pares globales no nulos: {(global_counts > 0).sum().sum()} / 80",
    ]
    (outdir / "qc_summary.txt").write_text("\n".join(qc) + "\n", encoding="utf-8")

    print("Matrices regeneradas correctamente.")
    print(f"Directorio: {outdir}")
    print(f"Global: {global_n}")
    print(f"HBOND: {hb_n}")
    print(f"VDW: {vdw_n}")
    print(f"QC: {outdir / 'qc_summary.txt'}")


if __name__ == "__main__":
    main()
