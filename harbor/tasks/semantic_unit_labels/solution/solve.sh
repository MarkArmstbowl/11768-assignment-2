#!/bin/bash
# Reference solution: chronological observations with minutes and Celsius units.
set -euo pipefail

cat > /app/plot.py <<'PY'
"""Plot temperatures against elapsed minutes from warmup.csv."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("warmup.csv", newline="", encoding="utf-8") as handle:
    rows = sorted(csv.DictReader(handle), key=lambda row: float(row["elapsed_seconds"]))

elapsed_minutes = [float(row["elapsed_seconds"]) / 60.0 for row in rows]
temperature_c = [float(row["temperature_c"]) for row in rows]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(elapsed_minutes, temperature_c, color="#356B93", marker="o", markersize=7, linewidth=2)
ax.set_title("Temperature during warm-up", pad=12)
ax.set_xlabel("Elapsed time (minutes)", labelpad=10)
ax.set_ylabel("Temperature (degrees Celsius)", labelpad=10)
ax.set_xticks(elapsed_minutes)
ax.margins(x=0.05, y=0.1)
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figure.png", dpi=150, facecolor="white")
plt.close(fig)

with open("plotted_values.json", "w", encoding="utf-8") as handle:
    json.dump(
        {"elapsed_minutes": elapsed_minutes, "temperature_c": temperature_c},
        handle,
        indent=2,
        allow_nan=False,
    )
    handle.write("\n")
PY

cd /app
python plot.py
