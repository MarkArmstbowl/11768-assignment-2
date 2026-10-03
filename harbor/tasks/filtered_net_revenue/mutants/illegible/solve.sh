#!/bin/bash
# MUTANT: correct content, but 2pt text on a 300x220-pixel canvas.
# Expected totals: North 145, South 0, East 95, West 105.
# Expected reward: 1; structural checks do not evaluate rendered legibility.
# This expectation still requires a Harbor run; it is not a measured reward.
set -euo pipefail

cat > /app/plot.py <<'PY'
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

regions = ["North", "South", "East", "West"]
totals = {region: 0.0 for region in regions}
with open("orders.csv", newline="") as source:
    for row in csv.DictReader(source):
        if row["status"] != "completed":
            continue
        totals[row["region"]] += float(row["gross"]) - float(row["refund"])
values = [totals[region] for region in regions]

# BUG: every text element is too small to read in the delivered image.
plt.rcParams.update({"font.size": 2})
fig, ax = plt.subplots(figsize=(3.0, 2.2))
bars = ax.bar(regions, values, color="#4C78A8")
ax.set_title("Completed-order net revenue", fontsize=2, pad=2)
ax.set_xlabel("Region", fontsize=2, labelpad=2)
ax.set_ylabel("Net revenue (USD)", fontsize=2, labelpad=2)
ax.tick_params(labelsize=2, pad=2, length=2)
ax.set_ylim(0, max(max(values) * 1.18, 1))
for bar, value in zip(bars, values):
    ax.annotate(
        f"{value:.0f}",
        xy=(bar.get_x() + bar.get_width() / 2, value),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=2,
    )
# Reserve room for every label: the intended defect is text size.
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.84)
fig.savefig("figure.png", dpi=100)
with open("plotted_values.json", "w") as target:
    json.dump(dict(zip(regions, values)), target, indent=2)
PY

cd /app
python plot.py
