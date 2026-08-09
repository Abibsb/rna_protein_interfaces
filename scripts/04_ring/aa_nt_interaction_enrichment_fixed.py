#!/usr/bin/env python3
"""
Enriquecimiento de aminoácidos, nucleótidos y pares AA–NT entre tipos de interacción RING.

Comparación principal:
    conjunto objetivo     = contactos HBOND
    conjunto de referencia = contactos VDW

Unidad de análisis por defecto:
    un par único de residuos por PDB y por clase de interacción.
Esto evita contar varias veces el mismo contacto por aparecer en múltiples pares atómicos.

Salidas:
    <prefix>_clean_contacts.csv
    <prefix>_aa_enrichment.csv
    <prefix>_nt_enrichment.csv
    <prefix>_pair_enrichment.csv
    <prefix>_pair_log2or_matrix.csv
    <prefix>_pair_log2or_heatmap.png
    <prefix>_qc_summary.txt

Ejemplo:
    python aa_nt_interaction_enrichment.py \
        -i maxitabla_dataset_actualizado.csv \
        -o aa_nt_hbond_vs_vdw

Dependencias:
    pandas, numpy, scipy, statsmodels, matplotlib
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


VALID_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}
VALID_NT = {"A", "C", "G", "U"}

NODE_RE = re.compile(
    r"^(?P<chain>[^:]+):(?P<position>-?\d+[A-Za-z]?):(?P<insertion>[^:]*):(?P<resname>[^:]+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula enriquecimiento de AA, NT y pares AA–NT en HBOND frente a VDW "
            "a partir de una tabla consolidada de RING."
        )
    )
    parser.add_argument("-i", "--input", required=True, help="Tabla de RING en TSV o CSV.")
    parser.add_argument(
        "-o",
        "--output-prefix",
        default="aa_nt_hbond_vs_vdw",
        help="Prefijo de los archivos de salida.",
    )
    parser.add_argument(
        "--pdb-column",
        default=None,
        help="Columna con el identificador PDB. Se autodetecta si se omite.",
    )
    parser.add_argument(
        "--node1-column",
        default=None,
        help="Columna del primer nodo. Se autodetecta si se omite.",
    )
    parser.add_argument(
        "--node2-column",
        default=None,
        help="Columna del segundo nodo. Se autodetecta si se omite.",
    )
    parser.add_argument(
        "--interaction-column",
        default=None,
        help="Columna con la etiqueta HBOND/VDW. Se autodetecta si se omite.",
    )
    parser.add_argument(
        "--count-mode",
        choices=("unique_pairs", "rows"),
        default="unique_pairs",
        help=(
            "unique_pairs: un contacto por PDB, par de residuos y clase; "
            "rows: usa cada fila atómica como observación. Default: unique_pairs."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Umbral de significación para p ajustado. Default: 0.05.",
    )
    parser.add_argument(
        "--min-total",
        type=int,
        default=10,
        help="Conteo mínimo total HBOND+VDW para marcar un resultado como interpretable.",
    )
    return parser.parse_args()


def read_table(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {file_path}")

    # sep=None usa el detector de delimitador de Python.
    df = pd.read_csv(file_path,sep="\t",index_col=False,low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError("La tabla de entrada está vacía.")
    return df


def find_column(
    df: pd.DataFrame,
    explicit: str | None,
    preferred_names: Iterable[str],
    content_test=None,
    description: str = "columna",
) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"No existe {description} '{explicit}'. Columnas disponibles: {df.columns.tolist()}"
            )
        return explicit

    lower_map = {c.lower(): c for c in df.columns}
    for name in preferred_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    if content_test is not None:
        best_col = None
        best_score = 0.0
        for col in df.columns:
            sample = df[col].dropna().astype(str)
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=1)
            score = float(content_test(sample).mean()) if len(sample) else 0.0
            if score > best_score:
                best_col = col
                best_score = score
        if best_col is not None and best_score > 0:
            return best_col

    raise ValueError(f"No pude autodetectar {description}.")


def node_like(series: pd.Series) -> pd.Series:
    return series.str.count(":").ge(3)


def interaction_like(series: pd.Series) -> pd.Series:
    upper = series.str.upper()
    return upper.str.contains("HBOND|VDW", regex=True, na=False)


def pdb_like(series: pd.Series) -> pd.Series:
    return series.str.fullmatch(r"[0-9][A-Za-z0-9]{3}", na=False)


def extract_resname(node: str) -> str | None:
    if pd.isna(node):
        return None
    text = str(node).strip().upper()
    match = NODE_RE.match(text)
    if match:
        return match.group("resname").strip().upper()

    # Fallback para formatos levemente diferentes.
    parts = text.split(":")
    if len(parts) >= 4:
        return parts[-1].strip().upper()
    return None


def normalize_interaction(value: str) -> str | None:
    text = str(value).upper()
    has_hbond = "HBOND" in text
    has_vdw = "VDW" in text

    # Una fila debería pertenecer a una única categoría.
    if has_hbond and not has_vdw:
        return "HBOND"
    if has_vdw and not has_hbond:
        return "VDW"
    return None


def clean_contacts(
    df: pd.DataFrame,
    pdb_col: str,
    node1_col: str,
    node2_col: str,
    interaction_col: str,
    count_mode: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = df[[pdb_col, node1_col, node2_col, interaction_col]].copy()
    work.columns = ["pdb_id", "node1", "node2", "interaction_raw"]

    qc = {"input_rows": len(work)}

    work["res1"] = work["node1"].map(extract_resname)
    work["res2"] = work["node2"].map(extract_resname)
    work["interaction_class"] = work["interaction_raw"].map(normalize_interaction)

    qc["rows_hbond_or_vdw"] = int(work["interaction_class"].notna().sum())

    # Detectar orientación sin asumir que proteína está siempre en NodeId1.
    mask_1aa_2nt = work["res1"].isin(VALID_AA) & work["res2"].isin(VALID_NT)
    mask_1nt_2aa = work["res1"].isin(VALID_NT) & work["res2"].isin(VALID_AA)

    work["aa"] = np.where(mask_1aa_2nt, work["res1"], np.where(mask_1nt_2aa, work["res2"], None))
    work["nt"] = np.where(mask_1aa_2nt, work["res2"], np.where(mask_1nt_2aa, work["res1"], None))
    work["protein_node"] = np.where(
        mask_1aa_2nt, work["node1"], np.where(mask_1nt_2aa, work["node2"], None)
    )
    work["rna_node"] = np.where(
        mask_1aa_2nt, work["node2"], np.where(mask_1nt_2aa, work["node1"], None)
    )

    work = work[
        work["interaction_class"].notna()
        & work["aa"].isin(VALID_AA)
        & work["nt"].isin(VALID_NT)
    ].copy()

    qc["valid_aa_nt_rows"] = len(work)
    qc["reversed_orientation_rows"] = int(mask_1nt_2aa.sum())

    # Normalizar campos usados para deduplicar.
    for col in ["pdb_id", "protein_node", "rna_node"]:
        work[col] = work[col].astype(str).str.strip()

    if count_mode == "unique_pairs":
        before = len(work)
        work = work.drop_duplicates(
            subset=["pdb_id", "protein_node", "rna_node", "interaction_class"]
        ).copy()
        qc["duplicates_removed"] = before - len(work)
    else:
        qc["duplicates_removed"] = 0

    qc["analysis_rows"] = len(work)
    qc["hbond_rows"] = int((work["interaction_class"] == "HBOND").sum())
    qc["vdw_rows"] = int((work["interaction_class"] == "VDW").sum())
    qc["unique_structural_pairs"] = int(
        work[["pdb_id", "protein_node", "rna_node"]].drop_duplicates().shape[0]
    )

    if qc["hbond_rows"] == 0 or qc["vdw_rows"] == 0:
        raise ValueError(
            "Después del filtrado no quedaron observaciones en ambas clases HBOND y VDW."
        )

    return work, qc


def fisher_enrichment(
    df: pd.DataFrame,
    feature_col: str,
    all_features: list[str] | None = None,
    alpha: float = 0.05,
    min_total: int = 10,
) -> pd.DataFrame:
    target = df["interaction_class"].eq("HBOND")
    features = all_features if all_features is not None else sorted(df[feature_col].dropna().unique())

    rows = []
    for feature in features:
        feat = df[feature_col].eq(feature)

        a = int((feat & target).sum())      # feature en HBOND
        b = int((~feat & target).sum())     # otros en HBOND
        c = int((feat & ~target).sum())     # feature en VDW
        d = int((~feat & ~target).sum())    # otros en VDW

        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")

        frac_hbond = a / (a + b) if (a + b) else np.nan
        frac_vdw = c / (c + d) if (c + d) else np.nan

        # Haldane–Anscombe sólo para obtener un log2 OR finito cuando hay ceros.
        log2_or_corrected = np.log2(
            ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        )

        rows.append(
            {
                "feature": feature,
                "hbond_with_feature": a,
                "hbond_without_feature": b,
                "vdw_with_feature": c,
                "vdw_without_feature": d,
                "total_with_feature": a + c,
                "fraction_within_hbond": frac_hbond,
                "fraction_within_vdw": frac_vdw,
                "odds_ratio": odds_ratio,
                "log2_odds_ratio_corrected": log2_or_corrected,
                "p_value": p_value,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_adjust_bh"] = multipletests(
        out["p_value"].fillna(1.0).to_numpy(),
        alpha=alpha,
        method="fdr_bh",
    )[1]
    out["significant_bh"] = out["p_adjust_bh"] < alpha
    out["enrichment_direction"] = np.select(
        [
            out["log2_odds_ratio_corrected"] > 0,
            out["log2_odds_ratio_corrected"] < 0,
        ],
        ["HBOND", "VDW"],
        default="neutral",
    )
    out["enough_observations"] = out["total_with_feature"] >= min_total

    return out.sort_values(
        ["p_adjust_bh", "total_with_feature"],
        ascending=[True, False],
    ).reset_index(drop=True)


def make_pair_heatmap(pair_results: pd.DataFrame, prefix: str) -> None:
    if pair_results.empty:
        return

    split = pair_results["feature"].str.split("–", n=1, expand=True)
    plot_df = pair_results.assign(AA=split[0], NT=split[1])

    matrix = (
        plot_df.pivot(index="AA", columns="NT", values="log2_odds_ratio_corrected")
        .reindex(index=sorted(VALID_AA), columns=["A", "C", "G", "U"])
    )
    matrix.to_csv(f"{prefix}_pair_log2or_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 11))
    im = ax.imshow(matrix.values, aspect="auto", vmin=-np.nanmax(np.abs(matrix.values)),
                   vmax=np.nanmax(np.abs(matrix.values)))

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Nucleótido")
    ax.set_ylabel("Aminoácido")
    ax.set_title("Enriquecimiento AA–NT: HBOND frente a VDW\n(log2 odds ratio corregido)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2 OR; positivo = HBOND, negativo = VDW")

    sig_map = (
        plot_df.pivot(index="AA", columns="NT", values="significant_bh")
        .reindex(index=matrix.index, columns=matrix.columns)
        .fillna(False)
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                marker = "*" if bool(sig_map.iloc[i, j]) else ""
                ax.text(j, i, f"{value:.2f}{marker}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{prefix}_pair_log2or_heatmap.png", dpi=300)
    plt.close(fig)


def write_qc(
    qc: dict[str, int],
    columns: dict[str, str],
    prefix: str,
    count_mode: str,
) -> None:
    lines = [
        "AA–NT interaction enrichment QC",
        "================================",
        f"count_mode: {count_mode}",
        "",
        "Detected columns:",
        f"pdb: {columns['pdb']}",
        f"node1: {columns['node1']}",
        f"node2: {columns['node2']}",
        f"interaction: {columns['interaction']}",
        "",
        "Counts:",
    ]
    lines.extend(f"{key}: {value}" for key, value in qc.items())
    Path(f"{prefix}_qc_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        df = read_table(args.input)

        pdb_col = find_column(
            df,
            args.pdb_column,
            preferred_names=["PDBID", "pdb_id", "Entry ID", "entry_id"],
            content_test=pdb_like,
            description="la columna PDB",
        )
        node1_col = find_column(
            df,
            args.node1_column,
            preferred_names=["NodeId1", "node_id_1", "node1"],
            content_test=node_like,
            description="la columna del nodo 1",
        )
        node2_col = find_column(
            df,
            args.node2_column,
            preferred_names=["NodeId2", "node_id_2", "node2"],
            content_test=node_like,
            description="la columna del nodo 2",
        )
        interaction_col = find_column(
            df,
            args.interaction_column,
            preferred_names=["Interaction", "interaction", "Interaction Type", "interaction_type"],
            content_test=interaction_like,
            description="la columna de interacción",
        )

        contacts, qc = clean_contacts(
            df=df,
            pdb_col=pdb_col,
            node1_col=node1_col,
            node2_col=node2_col,
            interaction_col=interaction_col,
            count_mode=args.count_mode,
        )

        prefix = args.output_prefix
        contacts.to_csv(f"{prefix}_clean_contacts.csv", index=False)

        contacts["aa_nt_pair"] = contacts["aa"] + "–" + contacts["nt"]

        aa_results = fisher_enrichment(
            contacts,
            feature_col="aa",
            all_features=sorted(VALID_AA),
            alpha=args.alpha,
            min_total=args.min_total,
        )
        nt_results = fisher_enrichment(
            contacts,
            feature_col="nt",
            all_features=["A", "C", "G", "U"],
            alpha=args.alpha,
            min_total=args.min_total,
        )
        pair_results = fisher_enrichment(
            contacts,
            feature_col="aa_nt_pair",
            all_features=[f"{aa}–{nt}" for aa in sorted(VALID_AA) for nt in ["A", "C", "G", "U"]],
            alpha=args.alpha,
            min_total=args.min_total,
        )

        aa_results.to_csv(f"{prefix}_aa_enrichment.csv", index=False)
        nt_results.to_csv(f"{prefix}_nt_enrichment.csv", index=False)
        pair_results.to_csv(f"{prefix}_pair_enrichment.csv", index=False)

        make_pair_heatmap(pair_results, prefix)

        write_qc(
            qc,
            columns={
                "pdb": pdb_col,
                "node1": node1_col,
                "node2": node2_col,
                "interaction": interaction_col,
            },
            prefix=prefix,
            count_mode=args.count_mode,
        )

        print("Análisis completado.")
        print(f"Contactos analizados: {len(contacts)}")
        print(f"HBOND: {(contacts['interaction_class'] == 'HBOND').sum()}")
        print(f"VDW: {(contacts['interaction_class'] == 'VDW').sum()}")
        print(f"Resultados AA: {prefix}_aa_enrichment.csv")
        print(f"Resultados NT: {prefix}_nt_enrichment.csv")
        print(f"Resultados pares: {prefix}_pair_enrichment.csv")
        print(f"QC: {prefix}_qc_summary.txt")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
