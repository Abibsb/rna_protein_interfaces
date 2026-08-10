#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import duckdb


KEY = [
    "pdb_id",
    "uniprot_id",
    "uniprot_pos",
]

# Estas columnas NO fueron resumidas con "first".
MUTATION_COLS = {
    "Wild_Type",
    "Mutated_Type",
    "Consequence",
    "Disease_Association",
    "Prediction_PolyPhen",
    "Prediction_SIFT",
}

MAX_COLS = {
    "has_bridging_water",
    "has_angle_filtered_water",
    "is_recurrent_water_site",
    "has_ring_interface",
    "ring_interaction_count",
}


def quote_identifier(name: str) -> str:
    """Escapa nombres de columnas para SQL."""
    return '"' + name.replace('"', '""') + '"'


def sql_string(value: str) -> str:
    """Escapa una ruta o cadena para SQL."""
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Comprueba si las columnas agregadas mediante 'first' "
            "son constantes dentro de cada residuo estructural."
        )
    )
    parser.add_argument(
        "input_csv",
        help="MASTER_interface_dataset_WITH_VARIANTS.csv",
    )
    parser.add_argument(
        "--out-prefix",
        default="viewB_first_check",
        help="Prefijo de los archivos de salida",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Número máximo de grupos conflictivos por columna",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"No se encontró: {input_csv}")

    out_prefix = Path(args.out_prefix)
    summary_file = out_prefix.with_name(out_prefix.name + "_summary.csv")
    examples_file = out_prefix.with_name(out_prefix.name + "_examples.csv")
    txt_file = out_prefix.with_name(out_prefix.name + "_summary.txt")

    con = duckdb.connect()

    # Permite que DuckDB use disco temporal cuando sea necesario.
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_viewb_check'")

    csv_path = sql_string(str(input_csv.resolve()))

    # all_varchar=True evita inconsistencias de inferencia de tipos.
    con.execute(
        f"""
        CREATE OR REPLACE VIEW src AS
        SELECT *
        FROM read_csv_auto(
            {csv_path},
            header=true,
            all_varchar=true,
            sample_size=-1
        )
        """
    )

    columns = [
        row[0]
        for row in con.execute("DESCRIBE src").fetchall()
    ]

    missing_key = [column for column in KEY if column not in columns]
    if missing_key:
        raise KeyError(
            "Faltan columnas de la clave: " + ", ".join(missing_key)
        )

    first_columns = [
        column
        for column in columns
        if column not in KEY
        and column not in MUTATION_COLS
        and column not in MAX_COLS
    ]

    key_sql = ", ".join(quote_identifier(column) for column in KEY)

    total_rows = con.execute(
        "SELECT COUNT(*) FROM src"
    ).fetchone()[0]

    total_groups = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT {key_sql}
            FROM src
            GROUP BY {key_sql}
        )
        """
    ).fetchone()[0]

    print("=" * 72)
    print("CONTROL DE COLUMNAS AGREGADAS CON FIRST")
    print("=" * 72)
    print(f"Archivo: {input_csv}")
    print(f"Filas: {total_rows:,}")
    print(f"Grupos residuo: {total_groups:,}")
    print(f"Columnas evaluadas: {len(first_columns)}")
    print()

    summary_rows = []
    example_rows = []

    for index, column in enumerate(first_columns, start=1):
        qcol = quote_identifier(column)

        # Convertimos cadenas vacías a NULL.
        clean_value = f"NULLIF(TRIM(CAST({qcol} AS VARCHAR)), '')"

        result = con.execute(
            f"""
            WITH grouped AS (
                SELECT
                    {key_sql},
                    COUNT(*) AS n_rows,
                    COUNT({clean_value}) AS n_nonmissing,
                    COUNT(DISTINCT {clean_value}) AS n_distinct
                FROM src
                GROUP BY {key_sql}
            )
            SELECT
                SUM(CASE WHEN n_distinct > 1 THEN 1 ELSE 0 END),
                SUM(
                    CASE
                        WHEN n_nonmissing > 0
                         AND n_nonmissing < n_rows
                        THEN 1 ELSE 0
                    END
                ),
                MAX(n_distinct)
            FROM grouped
            """
        ).fetchone()

        conflicting_groups = int(result[0] or 0)
        mixed_missing_groups = int(result[1] or 0)
        max_distinct = int(result[2] or 0)

        status = "OK" if conflicting_groups == 0 else "CONFLICT"

        summary_rows.append({
            "column": column,
            "status": status,
            "groups_with_multiple_nonmissing_values": conflicting_groups,
            "percent_conflicting_groups": (
                100 * conflicting_groups / total_groups
                if total_groups else 0
            ),
            "groups_with_mixed_missing_and_nonmissing": mixed_missing_groups,
            "max_distinct_values_in_one_group": max_distinct,
        })

        print(
            f"[{index:02d}/{len(first_columns):02d}] "
            f"{column}: {status}; "
            f"conflictos={conflicting_groups:,}; "
            f"mixtos NA/valor={mixed_missing_groups:,}"
        )

        if conflicting_groups > 0:
            examples = con.execute(
                f"""
                SELECT
                    {key_sql},
                    COUNT(*) AS n_rows,
                    COUNT(DISTINCT {clean_value}) AS n_distinct,
                    MIN({clean_value}) AS example_min,
                    MAX({clean_value}) AS example_max
                FROM src
                GROUP BY {key_sql}
                HAVING COUNT(DISTINCT {clean_value}) > 1
                ORDER BY n_distinct DESC, n_rows DESC
                LIMIT {int(args.examples)}
                """
            ).fetchall()

            for row in examples:
                example_rows.append({
                    "column": column,
                    "pdb_id": row[0],
                    "uniprot_id": row[1],
                    "uniprot_pos": row[2],
                    "n_rows": row[3],
                    "n_distinct": row[4],
                    "example_min": row[5],
                    "example_max": row[6],
                })

    with summary_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "column",
                "status",
                "groups_with_multiple_nonmissing_values",
                "percent_conflicting_groups",
                "groups_with_mixed_missing_and_nonmissing",
                "max_distinct_values_in_one_group",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with examples_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "column",
                "pdb_id",
                "uniprot_id",
                "uniprot_pos",
                "n_rows",
                "n_distinct",
                "example_min",
                "example_max",
            ],
        )
        writer.writeheader()
        writer.writerows(example_rows)

    conflicts = [
        row for row in summary_rows
        if row["groups_with_multiple_nonmissing_values"] > 0
    ]

    with txt_file.open("w", encoding="utf-8") as handle:
        handle.write("CONTROL DE COLUMNAS AGREGADAS CON FIRST\n")
        handle.write("=" * 72 + "\n")
        handle.write(f"Archivo: {input_csv}\n")
        handle.write(f"Filas: {total_rows:,}\n")
        handle.write(f"Grupos residuo: {total_groups:,}\n")
        handle.write(f"Columnas evaluadas: {len(first_columns)}\n")
        handle.write(f"Columnas con conflictos: {len(conflicts)}\n\n")

        for row in summary_rows:
            handle.write(
                f"{row['column']}: {row['status']}; "
                f"grupos conflictivos="
                f"{row['groups_with_multiple_nonmissing_values']:,}; "
                f"porcentaje="
                f"{row['percent_conflicting_groups']:.6f}%; "
                f"grupos con NA y valor="
                f"{row['groups_with_mixed_missing_and_nonmissing']:,}; "
                f"máximo de valores distintos="
                f"{row['max_distinct_values_in_one_group']}\n"
            )

    print()
    print("=" * 72)
    print(f"Columnas sin conflictos: {len(first_columns) - len(conflicts)}")
    print(f"Columnas con conflictos: {len(conflicts)}")
    print(f"Guardado: {summary_file}")
    print(f"Guardado: {examples_file}")
    print(f"Guardado: {txt_file}")


if __name__ == "__main__":
    main()