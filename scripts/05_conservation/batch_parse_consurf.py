#!/usr/bin/env python3

import os
import csv

CONSURF_ROOT = "/mnt/usb-Seagate_Basic_NABSSV5R-0:0-part1/Tesis_Maestria/20_ene_26/consurf_outputs_nozip"
OUT_CSV = "consurf_ALL.csv"


def parse_consurf_file(txt_path, pdbid):
    rows = []

    with open(txt_path, errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or not s[0].isdigit():
                continue

            parts = s.split()
            if len(parts) < 9:
                continue

            try:
                atom = parts[2]            # e.g. GLU:30:A
                score = float(parts[3])

                color_raw = parts[4]
                is_low_conf = "*" in color_raw
                color = int(color_raw.replace("*", ""))

                be = parts[6]
                fs = parts[7]
                msa = parts[8]
            except Exception:
                continue

            try:
                _, auth_resnum, atom_chain = atom.split(":")
                auth_resnum = int(auth_resnum)
            except ValueError:
                continue

            rows.append({
                "PDBID": pdbid,
                "Chain": atom_chain,        # ← SOURCE OF TRUTH
                "auth_seq_id": auth_resnum,
                "consurf_score": score,
                "consurf_color": color,
                "BE": be,
                "FS": fs,
                "msa_coverage": msa,
                "is_low_conf": is_low_conf
            })

    return rows


def main():
    all_rows = []
    missing = 0

    folders = [
        d for d in os.listdir(CONSURF_ROOT)
        if os.path.isdir(os.path.join(CONSURF_ROOT, d))
    ]

    total = len(folders)
    print(f"[INFO] Found {total} ConSurf folders")

    for i, d in enumerate(folders, 1):
        if i % 50 == 0:
            print(f"[INFO] Processed {i}/{total} folders")

        pdbid = d.split("_", 1)[0]

        txt_file = os.path.join(
            CONSURF_ROOT, d, f"{pdbid}_consurf_grades.txt"
        )

        if not os.path.exists(txt_file):
            missing += 1
            continue

        rows = parse_consurf_file(txt_file, pdbid)
        all_rows.extend(rows)

    print(f"[INFO] Missing grades files: {missing}")
    print(f"[INFO] Writing {len(all_rows)} rows to {OUT_CSV}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "PDBID", "Chain", "auth_seq_id",
                "consurf_score", "consurf_color",
                "BE", "FS", "msa_coverage", "is_low_conf"
            ]
        )
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()