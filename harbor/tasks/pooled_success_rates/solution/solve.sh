#!/bin/bash
set -euo pipefail

cat > /app/plot.py <<'PY'
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


months = ["Jan", "Feb", "Mar", "Apr"]
successes = dict.fromkeys(months, 0)
attempts = dict.fromkeys(months, 0)
with open("cohorts.csv", newline="", encoding="utf-8") as source:
    for group in csv.DictReader(source):
        month = group["month"]
        successes[month] += int(group["successes"])
        attempts[month] += int(group["attempts"])
percentages = [100.0 * successes[month] / attempts[month] for month in months]

plt.rcdefaults()
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
ax.plot(months, percentages, marker="o", markersize=8, linewidth=2.4, color="#2879B9")
ax.set_title("Monthly pooled success rate", fontsize=16, pad=16)
ax.set_xlabel("Month", labelpad=10)
ax.set_ylabel("Success rate (%)", labelpad=10)
ax.set_ylim(0, 100)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for month, percentage in zip(months, percentages):
    ax.annotate(
        f"{percentage:.1f}%",
        xy=(month, percentage),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12,
    )
fig.tight_layout()
fig.savefig("figure.png", dpi=160, metadata={"Software": "matplotlib"})
plt.close(fig)

plotted_values = dict(zip(months, percentages))
with open("plotted_values.json", "w", encoding="utf-8") as output:
    json.dump(plotted_values, output, indent=2, allow_nan=False)
    output.write("\n")
PY

cd /app
python plot.py
