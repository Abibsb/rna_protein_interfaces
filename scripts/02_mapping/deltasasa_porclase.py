#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# INPUT
# =========================
csv_file = "/media/usuario/Seagate Basic1/Tesis_Maestria/25_ene_26/interface_areas_per_residue_normalized.csv"

# =========================
# READ
# =========================
df = pd.read_csv(csv_file)

required = [
    "polymer_class",
    "interface_class",
    "delta_sasa",
    "buried_fraction_residue"
]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Falta columna requerida: {col}")

# =========================
# CLEAN
# =========================
df["delta_sasa"] = pd.to_numeric(df["delta_sasa"], errors="coerce")
df["buried_fraction_residue"] = pd.to_numeric(df["buried_fraction_residue"], errors="coerce")

df = df.dropna(subset=["polymer_class", "interface_class", "delta_sasa", "buried_fraction_residue"])

# opcional: asegurar rango válido
df = df[(df["buried_fraction_residue"] >= 0) & (df["buried_fraction_residue"] <= 1)]

# nos quedamos con clases relevantes
class_order = ["interface", "strong_interface", "core_hotspot"]
df = df[df["interface_class"].isin(class_order)].copy()

# orden explícito
df["interface_class"] = pd.Categorical(
    df["interface_class"],
    categories=class_order,
    ordered=True
)

polymer_order = ["protein", "rna"]
present_polymer = [x for x in polymer_order if x in set(df["polymer_class"])]
df["polymer_class"] = pd.Categorical(
    df["polymer_class"],
    categories=present_polymer,
    ordered=True
)

# =========================
# SUMMARY TABLES
# =========================
summary_delta = (
    df.groupby(["interface_class", "polymer_class"])["delta_sasa"]
      .agg(["count", "mean", "median", "std", "min", "max"])
      .reset_index()
)

summary_buried = (
    df.groupby(["interface_class", "polymer_class"])["buried_fraction_residue"]
      .agg(["count", "mean", "median", "std", "min", "max"])
      .reset_index()
)

summary_delta.to_csv("summary_delta_sasa_by_class_polymer.csv", index=False)
summary_buried.to_csv("summary_buried_fraction_by_class_polymer.csv", index=False)

print("\nResumen delta_sasa")
print(summary_delta)

print("\nResumen buried_fraction_residue")
print(summary_buried)

# =========================
# PLOT 1: delta_sasa
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, iclass in zip(axes, class_order):
    sub = df[df["interface_class"] == iclass]

    data = [
        sub.loc[sub["polymer_class"] == p, "delta_sasa"].values
        for p in present_polymer
    ]

    ax.boxplot(data, tick_labels=present_polymer, showfliers=False)
    ax.set_title(iclass)
    ax.set_xlabel("polymer_class")
    ax.set_ylabel("delta_sasa")

plt.suptitle("ΔSASA absoluto por clase estructural y polymer_class")
plt.tight_layout()
plt.savefig("delta_sasa_by_class_and_polymer_boxplot.png", dpi=300)
plt.close()

# =========================
# PLOT 2: buried_fraction_residue
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, iclass in zip(axes, class_order):
    sub = df[df["interface_class"] == iclass]

    data = [
        sub.loc[sub["polymer_class"] == p, "buried_fraction_residue"].values
        for p in present_polymer
    ]

    ax.boxplot(data, tick_labels=present_polymer, showfliers=False)
    ax.set_title(iclass)
    ax.set_xlabel("polymer_class")
    ax.set_ylabel("buried_fraction_residue")

plt.suptitle("Fracción enterrada por clase estructural y polymer_class")
plt.tight_layout()
plt.savefig("buried_fraction_by_class_and_polymer_boxplot.png", dpi=300)
plt.close()