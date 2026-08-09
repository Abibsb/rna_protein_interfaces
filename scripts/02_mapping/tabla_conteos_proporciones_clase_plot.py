#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== INPUT ======
csv_file = "tabla_clases_por_polymer_class.csv"

# ====== READ ======
df = pd.read_csv(csv_file)

# Verificación básica (no confíes en el CSV a ciegas)
required = ["polymer_class", "interface_class", "percentage_within_polymer"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Falta columna: {col}")

# Orden biológico (evita orden alfabético arbitrario)
class_order = ["non_interface", "interface", "strong_interface", "core_hotspot"]
df["interface_class"] = pd.Categorical(df["interface_class"],
                                       categories=class_order,
                                       ordered=True)

# ====== PIVOT ======
pivot = df.pivot(index="polymer_class",
                 columns="interface_class",
                 values="percentage_within_polymer").fillna(0)

# ====== PLOT ======
x = np.arange(len(pivot.index))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

for i, col in enumerate(pivot.columns):
    ax.bar(x + i * width, pivot[col].values, width, label=col)

ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
ax.set_xticklabels(pivot.index, rotation=25, ha="right")

ax.set_xlabel("polymer_class")
ax.set_ylabel("percentage_within_polymer (%)")
ax.set_title("Distribución de clases estructurales por polymer_class")

ax.legend(title="interface_class")

plt.tight_layout()
plt.savefig("grouped_barplot_polymer_class.png", dpi=300)
plt.close()