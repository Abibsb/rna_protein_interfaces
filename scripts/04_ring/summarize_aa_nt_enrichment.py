#!/usr/bin/env python3
"""
Post-análisis y controles de calidad para enriquecimiento HBOND vs VDW.

Entradas esperadas:
    1) Tabla RING corregida:
       maxitabla_dataset_actualizado_header_fixed.csv

    2) Resultados del script de enriquecimiento:
       <prefix>_aa_enrichment.csv
       <prefix>_nt_enrichment.csv
       <prefix>_pair_enrichment.csv

Ejemplo:
    python summarize_aa_nt_enrichment.py \
        --ring maxitabla_dataset_actualizado_header_fixed.csv \
        --prefix aa_nt_hbond_vs_vdw_fixed \
        --outdir aa_nt_hbond_vs_vdw_summary

Salidas:
    aa_enrichment_sorted.csv
    nt_enrichment_sorted.csv
    significant_pairs.csv
    significant_pairs_top30.csv
    gly_hbond_subtypes.csv
    gly_hbond_contacts.csv
    pro_hbond_contacts.csv
    pro_contact_summary.csv
    interaction_subtype_summary.csv
    pdb_contact_summary.csv
    analysis_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


VALID_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}
VALID_NT = {"A", "C", "G", "U"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume y controla resultados de enriquecimiento HBOND vs VDW."
    )
    parser.add_argument(
        "--ring",
        required=True,
        help="Tabla RING corregida con columnas PDBID, NodeId1, Interaction y NodeId2.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefijo de los archivos *_aa_enrichment.csv, *_nt_enrichment.csv y *_pair_enrichment.csv.",
    )
    parser.add_argument(
        "--outdir",
        default="aa_nt_hbond_vs_vdw_summary",
        help="Directorio de salida.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Umbral de significación ajustada. Default: 0.05.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Cantidad de pares significativos a guardar en el ranking principal.",
    )
    return parser.parse_args()


def read_csv_checked(path: Path, sep: str = ",") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    df = pd.read_csv(path, sep=sep, low_memory=False)
    if df.empty:
        raise ValueError(f"El archivo está vacío: {path}")
    return df


def extract_resname(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.split(":")
        .str[-1]
    )


def prepare_ring(df: pd.DataFrame) -> pd.DataFrame:
    required = {"PDBID", "NodeId1", "Interaction", "NodeId2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en la tabla RING: {sorted(missing)}")

    work = df[["PDBID", "NodeId1", "Interaction", "NodeId2"]].copy()
    work.columns = ["pdb_id", "node1", "interaction", "node2"]

    work["res1"] = extract_resname(work["node1"])
    work["res2"] = extract_resname(work["node2"])
    work["interaction_upper"] = work["interaction"].astype(str).str.upper()

    mask_1aa_2nt = work["res1"].isin(VALID_AA) & work["res2"].isin(VALID_NT)
    mask_1nt_2aa = work["res1"].isin(VALID_NT) & work["res2"].isin(VALID_AA)

    work["aa"] = np.where(
        mask_1aa_2nt,
        work["res1"],
        np.where(mask_1nt_2aa, work["res2"], None),
    )
    work["nt"] = np.where(
        mask_1aa_2nt,
        work["res2"],
        np.where(mask_1nt_2aa, work["res1"], None),
    )
    work["protein_node"] = np.where(
        mask_1aa_2nt,
        work["node1"],
        np.where(mask_1nt_2aa, work["node2"], None),
    )
    work["rna_node"] = np.where(
        mask_1aa_2nt,
        work["node2"],
        np.where(mask_1nt_2aa, work["node1"], None),
    )

    work["interaction_class"] = np.select(
        [
            work["interaction_upper"].str.contains("HBOND", na=False),
            work["interaction_upper"].str.contains("VDW", na=False),
        ],
        ["HBOND", "VDW"],
        default=None,
    )

    work = work[
        work["aa"].isin(VALID_AA)
        & work["nt"].isin(VALID_NT)
        & work["interaction_class"].notna()
    ].copy()

    work["aa_nt_pair"] = work["aa"] + "–" + work["nt"]

    # Contactos únicos por estructura, par de residuos y clase.
    unique = work.drop_duplicates(
        subset=["pdb_id", "protein_node", "rna_node", "interaction_class"]
    ).copy()

    return unique


def sort_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    expected = {
        "feature",
        "odds_ratio",
        "log2_odds_ratio_corrected",
        "p_adjust_bh",
        "significant_bh",
        "enrichment_direction",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en archivo de enriquecimiento: {sorted(missing)}")

    return df.sort_values(
        ["significant_bh", "p_adjust_bh", "total_with_feature"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def main() -> int:
    args = parse_args()

    try:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        ring_path = Path(args.ring)
        prefix = Path(args.prefix)

        ring = read_csv_checked(ring_path, sep="\t")
        unique_contacts = prepare_ring(ring)

        aa_path = Path(f"{prefix}_aa_enrichment.csv")
        nt_path = Path(f"{prefix}_nt_enrichment.csv")
        pair_path = Path(f"{prefix}_pair_enrichment.csv")

        aa = sort_enrichment(read_csv_checked(aa_path))
        nt = sort_enrichment(read_csv_checked(nt_path))
        pairs = sort_enrichment(read_csv_checked(pair_path))

        # Guardar tablas ordenadas.
        aa.to_csv(outdir / "aa_enrichment_sorted.csv", index=False)
        nt.to_csv(outdir / "nt_enrichment_sorted.csv", index=False)

        sig_pairs = pairs[
            (pairs["significant_bh"] == True)
            & (pairs["p_adjust_bh"] < args.alpha)
        ].copy()

        sig_pairs.to_csv(outdir / "significant_pairs.csv", index=False)
        sig_pairs.head(args.top_n).to_csv(
            outdir / f"significant_pairs_top{args.top_n}.csv",
            index=False,
        )

        # Control de glicina: subtipos HBOND.
        gly_hbond = unique_contacts[
            (unique_contacts["aa"] == "GLY")
            & (unique_contacts["interaction_class"] == "HBOND")
        ].copy()

        gly_hbond.to_csv(outdir / "gly_hbond_contacts.csv", index=False)

        gly_subtypes = (
            gly_hbond.groupby("interaction_upper", dropna=False)
            .size()
            .reset_index(name="n_unique_contacts")
            .sort_values("n_unique_contacts", ascending=False)
        )
        if len(gly_hbond):
            gly_subtypes["fraction"] = (
                gly_subtypes["n_unique_contacts"] / gly_subtypes["n_unique_contacts"].sum()
            )
        else:
            gly_subtypes["fraction"] = np.nan

        gly_subtypes.to_csv(outdir / "gly_hbond_subtypes.csv", index=False)

        # Control de prolina.
        pro_all = unique_contacts[unique_contacts["aa"] == "PRO"].copy()
        pro_hbond = pro_all[pro_all["interaction_class"] == "HBOND"].copy()
        pro_vdw = pro_all[pro_all["interaction_class"] == "VDW"].copy()

        pro_hbond.to_csv(outdir / "pro_hbond_contacts.csv", index=False)

        pro_summary = pd.DataFrame(
            [
                {
                    "residue": "PRO",
                    "total_unique_contacts": len(pro_all),
                    "hbond_unique_contacts": len(pro_hbond),
                    "vdw_unique_contacts": len(pro_vdw),
                    "fraction_hbond": len(pro_hbond) / len(pro_all) if len(pro_all) else np.nan,
                    "fraction_vdw": len(pro_vdw) / len(pro_all) if len(pro_all) else np.nan,
                }
            ]
        )
        pro_summary.to_csv(outdir / "pro_contact_summary.csv", index=False)

        # Resumen de subtipos globales.
        subtype_summary = (
            unique_contacts.groupby(
                ["interaction_class", "interaction_upper"],
                dropna=False,
            )
            .size()
            .reset_index(name="n_unique_contacts")
            .sort_values(
                ["interaction_class", "n_unique_contacts"],
                ascending=[True, False],
            )
        )
        subtype_summary["fraction_within_class"] = (
            subtype_summary["n_unique_contacts"]
            / subtype_summary.groupby("interaction_class")["n_unique_contacts"].transform("sum")
        )
        subtype_summary.to_csv(
            outdir / "interaction_subtype_summary.csv",
            index=False,
        )

        # Resumen por PDB para mostrar dependencia/heterogeneidad.
        pdb_summary = (
            unique_contacts.groupby(["pdb_id", "interaction_class"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        if "HBOND" not in pdb_summary.columns:
            pdb_summary["HBOND"] = 0
        if "VDW" not in pdb_summary.columns:
            pdb_summary["VDW"] = 0

        pdb_summary["total"] = pdb_summary["HBOND"] + pdb_summary["VDW"]
        pdb_summary["fraction_hbond"] = np.where(
            pdb_summary["total"] > 0,
            pdb_summary["HBOND"] / pdb_summary["total"],
            np.nan,
        )
        pdb_summary["fraction_vdw"] = np.where(
            pdb_summary["total"] > 0,
            pdb_summary["VDW"] / pdb_summary["total"],
            np.nan,
        )
        pdb_summary.sort_values("total", ascending=False).to_csv(
            outdir / "pdb_contact_summary.csv",
            index=False,
        )

        # Resúmenes de principales enriquecimientos.
        aa_hbond = aa[aa["log2_odds_ratio_corrected"] > 0].head(10)
        aa_vdw = aa[aa["log2_odds_ratio_corrected"] < 0].sort_values(
            "log2_odds_ratio_corrected"
        ).head(10)

        nt_hbond = nt[nt["log2_odds_ratio_corrected"] > 0].head(4)
        nt_vdw = nt[nt["log2_odds_ratio_corrected"] < 0].sort_values(
            "log2_odds_ratio_corrected"
        ).head(4)

        pair_hbond = sig_pairs[
            sig_pairs["log2_odds_ratio_corrected"] > 0
        ].sort_values("log2_odds_ratio_corrected", ascending=False).head(10)

        pair_vdw = sig_pairs[
            sig_pairs["log2_odds_ratio_corrected"] < 0
        ].sort_values("log2_odds_ratio_corrected", ascending=True).head(10)

        lines = [
            "Resumen del análisis HBOND vs VDW",
            "=================================",
            "",
            f"Archivo RING: {ring_path}",
            f"Prefijo de enriquecimiento: {prefix}",
            f"Contactos únicos analizados: {len(unique_contacts)}",
            f"HBOND: {(unique_contacts['interaction_class'] == 'HBOND').sum()}",
            f"VDW: {(unique_contacts['interaction_class'] == 'VDW').sum()}",
            f"PDB con contactos válidos: {unique_contacts['pdb_id'].nunique()}",
            "",
            "Control de glicina",
            "-------------------",
            f"Contactos HBOND únicos con GLY: {len(gly_hbond)}",
        ]

        if not gly_subtypes.empty:
            lines.append("Subtipos HBOND con GLY:")
            for _, row in gly_subtypes.iterrows():
                lines.append(
                    f"  {row['interaction_upper']}: "
                    f"{int(row['n_unique_contacts'])} "
                    f"({row['fraction']:.4f})"
                )
        else:
            lines.append("No se detectaron contactos HBOND con GLY.")

        lines.extend(
            [
                "",
                "Control de prolina",
                "-------------------",
                f"Contactos totales únicos con PRO: {len(pro_all)}",
                f"Contactos HBOND únicos con PRO: {len(pro_hbond)}",
                f"Contactos VDW únicos con PRO: {len(pro_vdw)}",
                "",
                "Top aminoácidos enriquecidos en HBOND",
                "--------------------------------------",
            ]
        )

        for _, row in aa_hbond.iterrows():
            lines.append(
                f"{row['feature']}: OR={row['odds_ratio']:.4g}, "
                f"log2OR={row['log2_odds_ratio_corrected']:.4f}, "
                f"p_adj={row['p_adjust_bh']:.4g}"
            )

        lines.extend(
            [
                "",
                "Top aminoácidos enriquecidos en VDW",
                "------------------------------------",
            ]
        )
        for _, row in aa_vdw.iterrows():
            lines.append(
                f"{row['feature']}: OR={row['odds_ratio']:.4g}, "
                f"log2OR={row['log2_odds_ratio_corrected']:.4f}, "
                f"p_adj={row['p_adjust_bh']:.4g}"
            )

        lines.extend(
            [
                "",
                "Nucleótidos enriquecidos en HBOND",
                "----------------------------------",
            ]
        )
        for _, row in nt_hbond.iterrows():
            lines.append(
                f"{row['feature']}: OR={row['odds_ratio']:.4g}, "
                f"log2OR={row['log2_odds_ratio_corrected']:.4f}, "
                f"p_adj={row['p_adjust_bh']:.4g}"
            )

        lines.extend(
            [
                "",
                "Nucleótidos enriquecidos en VDW",
                "--------------------------------",
            ]
        )
        for _, row in nt_vdw.iterrows():
            lines.append(
                f"{row['feature']}: OR={row['odds_ratio']:.4g}, "
                f"log2OR={row['log2_odds_ratio_corrected']:.4f}, "
                f"p_adj={row['p_adjust_bh']:.4g}"
            )

        lines.extend(
            [
                "",
                "Top pares enriquecidos en HBOND",
                "-------------------------------",
            ]
        )
        for _, row in pair_hbond.iterrows():
            lines.append(
                f"{row['feature']}: OR={row['odds_ratio']:.4g}, "
                f"log2OR={row['log2_odds_ratio_corrected']:.4f}, "
                f"p_adj={row['p_adjust_bh']:.4g}"
            )

        lines.extend(
            [
                "",
                "Top pares enriquecidos en VDW",
                "-----------------------------",
            ]
        )
        for _, row in pair_vdw.iterrows():
            lines.append(
                f"{row['feature']}: OR={row['odds_ratio']:.4g}, "
                f"log2OR={row['log2_odds_ratio_corrected']:.4f}, "
                f"p_adj={row['p_adjust_bh']:.4g}"
            )

        lines.extend(
            [
                "",
                "Notas metodológicas",
                "-------------------",
                "- Los contactos se deduplicaron por PDB, nodo proteico, nodo de ARN y clase de interacción.",
                "- Los valores de Fisher iguales a 0 en la salida representan subdesbordamiento numérico, no probabilidad literalmente nula.",
                "- Los odds ratios describen mejor la magnitud del efecto que los valores de p con este tamaño muestral.",
                "- Los contactos dentro de una misma estructura no son completamente independientes.",
            ]
        )

        (outdir / "analysis_summary.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        print("Post-análisis completado.")
        print(f"Directorio de salida: {outdir}")
        print(f"Contactos únicos analizados: {len(unique_contacts)}")
        print(f"HBOND con GLY: {len(gly_hbond)}")
        print(f"HBOND con PRO: {len(pro_hbond)}")
        print(f"Pares significativos: {len(sig_pairs)}")
        print(f"Resumen: {outdir / 'analysis_summary.txt'}")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
