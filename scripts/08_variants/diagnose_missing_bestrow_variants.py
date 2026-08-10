#!/usr/bin/env python3

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compara las variantes únicas de la Vista A con variants_bestrow "
            "y diagnostica las variantes ausentes."
        )
    )
    parser.add_argument(
        "--view-a",
        required=True,
        help="MASTER_VIEW_A_interaction_variant_level.csv",
    )
    parser.add_argument(
        "--bestrow",
        required=True,
        help="variants_bestrow.csv",
    )
    parser.add_argument(
        "--bestrow-ring",
        required=True,
        help="variants_bestrow_ring.csv",
    )
    parser.add_argument(
        "--out-prefix",
        default="missing_bestrow_variants",
    )
    args = parser.parse_args()

    view_a = Path(args.view_a)
    bestrow = Path(args.bestrow)
    bestrow_ring = Path(args.bestrow_ring)

    for path in (view_a, bestrow, bestrow_ring):
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró: {path}")

    con = duckdb.connect()

    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_bestrow_check'")

    con.execute(
        f"""
        CREATE VIEW view_a AS
        SELECT *
        FROM read_csv_auto(
            {sql_path(view_a)},
            header=true,
            all_varchar=true,
            sample_size=-1
        )
        """
    )

    con.execute(
        f"""
        CREATE VIEW bestrow AS
        SELECT *
        FROM read_csv_auto(
            {sql_path(bestrow)},
            header=true,
            all_varchar=true,
            sample_size=-1
        )
        """
    )

    con.execute(
        f"""
        CREATE VIEW bestrow_ring AS
        SELECT *
        FROM read_csv_auto(
            {sql_path(bestrow_ring)},
            header=true,
            all_varchar=true,
            sample_size=-1
        )
        """
    )

    # Variantes completas de la Vista A.
    con.execute(
        """
        CREATE TABLE variants_view_a AS
        SELECT DISTINCT
            TRIM(uniprot_id) AS uniprot_id,
            TRIM(uniprot_pos) AS uniprot_pos,
            TRIM(Wild_Type) AS Wild_Type,
            TRIM(Mutated_Type) AS Mutated_Type
        FROM view_a
        WHERE NULLIF(TRIM(uniprot_id), '') IS NOT NULL
          AND NULLIF(TRIM(uniprot_pos), '') IS NOT NULL
          AND NULLIF(TRIM(Wild_Type), '') IS NOT NULL
          AND NULLIF(TRIM(Mutated_Type), '') IS NOT NULL
        """
    )

    # Variantes completas de variants_bestrow.csv.
    con.execute(
        """
        CREATE TABLE variants_bestrow AS
        SELECT DISTINCT
            TRIM(uniprot_id) AS uniprot_id,
            TRIM(uniprot_pos) AS uniprot_pos,
            TRIM(Wild_Type) AS Wild_Type,
            TRIM(Mutated_Type) AS Mutated_Type
        FROM bestrow
        WHERE NULLIF(TRIM(uniprot_id), '') IS NOT NULL
          AND NULLIF(TRIM(uniprot_pos), '') IS NOT NULL
          AND NULLIF(TRIM(Wild_Type), '') IS NOT NULL
          AND NULLIF(TRIM(Mutated_Type), '') IS NOT NULL
        """
    )

    # Variantes completas de variants_bestrow_ring.csv.
    con.execute(
        """
        CREATE TABLE variants_bestrow_ring AS
        SELECT DISTINCT
            TRIM(uniprot_id) AS uniprot_id,
            TRIM(uniprot_pos) AS uniprot_pos,
            TRIM(Wild_Type) AS Wild_Type,
            TRIM(Mutated_Type) AS Mutated_Type
        FROM bestrow_ring
        WHERE NULLIF(TRIM(uniprot_id), '') IS NOT NULL
          AND NULLIF(TRIM(uniprot_pos), '') IS NOT NULL
          AND NULLIF(TRIM(Wild_Type), '') IS NOT NULL
          AND NULLIF(TRIM(Mutated_Type), '') IS NOT NULL
        """
    )

    key_join = """
        a.uniprot_id = b.uniprot_id
        AND a.uniprot_pos = b.uniprot_pos
        AND a.Wild_Type = b.Wild_Type
        AND a.Mutated_Type = b.Mutated_Type
    """

    missing_bestrow = con.execute(
        f"""
        SELECT a.*
        FROM variants_view_a a
        LEFT JOIN variants_bestrow b
          ON {key_join}
        WHERE b.uniprot_id IS NULL
        ORDER BY
            a.uniprot_id,
            TRY_CAST(a.uniprot_pos AS INTEGER),
            a.Wild_Type,
            a.Mutated_Type
        """
    ).df()

    missing_ring = con.execute(
        f"""
        SELECT a.*
        FROM variants_view_a a
        LEFT JOIN variants_bestrow_ring b
          ON {key_join}
        WHERE b.uniprot_id IS NULL
        ORDER BY
            a.uniprot_id,
            TRY_CAST(a.uniprot_pos AS INTEGER),
            a.Wild_Type,
            a.Mutated_Type
        """
    ).df()

    print("=" * 72)
    print("RECUENTOS")
    print("=" * 72)

    for table in (
        "variants_view_a",
        "variants_bestrow",
        "variants_bestrow_ring",
    ):
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n:,}")

    print()
    print("Ausentes en variants_bestrow:", len(missing_bestrow))
    print("Ausentes en variants_bestrow_ring:", len(missing_ring))

    missing_bestrow.to_csv(
        f"{args.out_prefix}_vs_bestrow.csv",
        index=False,
    )
    missing_ring.to_csv(
        f"{args.out_prefix}_vs_bestrow_ring.csv",
        index=False,
    )

    # Si no hay ausentes, crear los archivos vacíos y finalizar.
    if missing_bestrow.empty:
        empty_details = pd.DataFrame()
        empty_diagnosis = pd.DataFrame(
            columns=[
                "uniprot_id",
                "uniprot_pos",
                "Wild_Type",
                "Mutated_Type",
                "n_rows",
                "polymer_classes",
                "n_protein_rows",
                "n_non_numeric_positions",
                "pdb_ids",
            ]
        )

        empty_details.to_csv(
            f"{args.out_prefix}_details.csv",
            index=False,
        )
        empty_diagnosis.to_csv(
            f"{args.out_prefix}_diagnosis.csv",
            index=False,
        )

        print()
        print("No se encontraron variantes ausentes.")
        print()
        print("Archivos generados:")
        print(f"  {args.out_prefix}_vs_bestrow.csv")
        print(f"  {args.out_prefix}_vs_bestrow_ring.csv")
        print(f"  {args.out_prefix}_details.csv")
        print(f"  {args.out_prefix}_diagnosis.csv")

        con.close()
        return

    # Registrar las claves ausentes como tabla temporal de DuckDB.
    con.register("missing_keys", missing_bestrow)

    # Recuperar todas las filas originales de las variantes ausentes.
    details = con.execute(
        """
        SELECT
            v.*
        FROM view_a v
        INNER JOIN missing_keys m
          ON TRIM(v.uniprot_id) = m.uniprot_id
         AND TRIM(v.uniprot_pos) = m.uniprot_pos
         AND TRIM(v.Wild_Type) = m.Wild_Type
         AND TRIM(v.Mutated_Type) = m.Mutated_Type
        ORDER BY
            TRIM(v.uniprot_id),
            TRY_CAST(TRIM(v.uniprot_pos) AS INTEGER),
            TRIM(v.Wild_Type),
            TRIM(v.Mutated_Type)
        """
    ).df()

    details.to_csv(
        f"{args.out_prefix}_details.csv",
        index=False,
    )

    # Diagnóstico resumido de cada variante ausente.
    diagnosis = con.execute(
        """
        SELECT
            TRIM(v.uniprot_id) AS uniprot_id,
            TRIM(v.uniprot_pos) AS uniprot_pos,
            TRIM(v.Wild_Type) AS Wild_Type,
            TRIM(v.Mutated_Type) AS Mutated_Type,

            COUNT(*) AS n_rows,

            STRING_AGG(
                DISTINCT COALESCE(
                    NULLIF(TRIM(v.polymer_class), ''),
                    '<NA>'
                ),
                ' | '
            ) AS polymer_classes,

            SUM(
                CASE
                    WHEN LOWER(
                        TRIM(COALESCE(v.polymer_class, ''))
                    ) = 'protein'
                    THEN 1
                    ELSE 0
                END
            ) AS n_protein_rows,

            SUM(
                CASE
                    WHEN TRY_CAST(
                        TRIM(v.uniprot_pos) AS INTEGER
                    ) IS NULL
                    THEN 1
                    ELSE 0
                END
            ) AS n_non_numeric_positions,

            STRING_AGG(
                DISTINCT COALESCE(
                    NULLIF(TRIM(v.pdb_id), ''),
                    '<NA>'
                ),
                ' | '
            ) AS pdb_ids

        FROM view_a v
        INNER JOIN missing_keys m
          ON TRIM(v.uniprot_id) = m.uniprot_id
         AND TRIM(v.uniprot_pos) = m.uniprot_pos
         AND TRIM(v.Wild_Type) = m.Wild_Type
         AND TRIM(v.Mutated_Type) = m.Mutated_Type

        GROUP BY
            TRIM(v.uniprot_id),
            TRIM(v.uniprot_pos),
            TRIM(v.Wild_Type),
            TRIM(v.Mutated_Type)

        ORDER BY
            TRIM(v.uniprot_id),
            TRY_CAST(TRIM(v.uniprot_pos) AS INTEGER),
            TRIM(v.Wild_Type),
            TRIM(v.Mutated_Type)
        """
    ).df()

    diagnosis.to_csv(
        f"{args.out_prefix}_diagnosis.csv",
        index=False,
    )

    print()
    print("=" * 72)
    print("VARIANTES AUSENTES")
    print("=" * 72)

    if diagnosis.empty:
        print("No se encontraron diferencias.")
    else:
        print(diagnosis.to_string(index=False))

    print()
    print("Archivos generados:")
    print(f"  {args.out_prefix}_vs_bestrow.csv")
    print(f"  {args.out_prefix}_vs_bestrow_ring.csv")
    print(f"  {args.out_prefix}_details.csv")
    print(f"  {args.out_prefix}_diagnosis.csv")

    con.close()


if __name__ == "__main__":
    main()