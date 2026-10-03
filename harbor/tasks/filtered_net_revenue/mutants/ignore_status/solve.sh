#!/bin/bash
# MUTANT: omit the status filter and include failed and pending orders.
# Expected totals: North 345, South 225, East 140, West 105.
# Expected reward: 0; the numeric-content checks should reject these totals.
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
        # BUG: every status contributes, rather than only "completed".
        totals[row["region"]] += float(row["gross"]) - float(row["refund"])
values = [totals[region] for region in regions]

plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(7, 4.8))
bars = ax.bar(regions, values, color="#4C78A8")
ax.set_title("Completed-order net revenue", fontsize=15, pad=12)
ax.set_xlabel("Region", fontsize=12, labelpad=8)
ax.set_ylabel("Net revenue (USD)", fontsize=12, labelpad=8)
ax.tick_params(labelsize=11)
ax.set_ylim(0, max(max(values) * 1.18, 1))
for bar, value in zip(bars, values):
    ax.annotate(
        f"{value:.0f}",
        xy=(bar.get_x() + bar.get_width() / 2, value),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
    )
fig.subplots_adjust(left=0.14, right=0.97, bottom=0.17, top=0.85)
fig.savefig("figure.png", dpi=100)
with open("plotted_values.json", "w") as target:
    json.dump(dict(zip(regions, values)), target, indent=2)
PY

cd /app
python plot.py
