#!/bin/bash
# Reference solution: a shared time scale and one pair of figure-level labels.
set -euo pipefail

cat > /app/plot.py <<'PY'
"""Plot both sensor traces with shared figure-level units from sensors.csv."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("sensors.csv", newline="", encoding="utf-8") as handle:
    rows = sorted(csv.DictReader(handle), key=lambda row: float(row["time_s"]))

time_s = [float(row["time_s"]) for row in rows]
sensor_a_v = [float(row["sensor_a_v"]) for row in rows]
sensor_b_v = [float(row["sensor_b_v"]) for row in rows]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6.8))
axes[0].plot(time_s, sensor_a_v, color="#356B93", marker="o", markersize=6, linewidth=2)
axes[0].set_title("Sensor A", pad=10)
axes[1].plot(time_s, sensor_b_v, color="#B85D43", marker="o", markersize=6, linewidth=2)
axes[1].set_title("Sensor B", pad=10)

for ax in axes:
    ax.set_xticks(time_s)
    ax.margins(x=0.05, y=0.12)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

fig.supxlabel("Elapsed time (s)", x=0.5, y=0.025, fontsize=12)
fig.supylabel("Amplitude (V)", x=0.025, y=0.5, fontsize=12)
# Reserve left and bottom margins for the labels shared by the entire figure.
fig.tight_layout(rect=(0.065, 0.06, 1, 1), h_pad=2)
fig.savefig("figure.png", dpi=150, facecolor="white")
plt.close(fig)

with open("plotted_values.json", "w", encoding="utf-8") as handle:
    json.dump(
        {"time_s": time_s, "sensor_a_v": sensor_a_v, "sensor_b_v": sensor_b_v},
        handle,
        indent=2,
        allow_nan=False,
    )
    handle.write("\n")
PY

cd /app
python plot.py
