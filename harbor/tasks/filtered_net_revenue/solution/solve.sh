#!/bin/bash
set -euo pipefail

cat > /app/plot.py <<'PY'
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


regions = ["North", "South", "East", "West"]
net_by_region = dict.fromkeys(regions, 0.0)
with open("orders.csv", newline="", encoding="utf-8") as source:
    for order in csv.DictReader(source):
        if order["status"] == "completed":
            net_by_region[order["region"]] += (
                float(order["gross"]) - float(order["refund"])
            )
net_revenues = [net_by_region[region] for region in regions]

plt.rcdefaults()
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
bars = ax.bar(regions, net_revenues, width=0.62, color="#2879B9")
ax.set_title("Completed-order net revenue", fontsize=16, pad=16)
ax.set_xlabel("Region", labelpad=10)
ax.set_ylabel("Net revenue (USD)", labelpad=10)
ax.set_ylim(0, max(1.0, max(net_revenues) * 1.18))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, revenue in zip(bars, net_revenues):
    ax.annotate(
        f"{revenue:.0f}",
        xy=(bar.get_x() + bar.get_width() / 2, revenue),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12,
    )
fig.tight_layout()
fig.savefig("figure.png", dpi=160, metadata={"Software": "matplotlib"})
plt.close(fig)

plotted_values = dict(zip(regions, net_revenues))
with open("plotted_values.json", "w", encoding="utf-8") as output:
    json.dump(plotted_values, output, indent=2, allow_nan=False)
    output.write("\n")
PY

cd /app
python plot.py
