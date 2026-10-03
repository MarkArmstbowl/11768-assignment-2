#!/bin/bash
# Reference solution: independent labels and units for both channel panels.
set -euo pipefail

cat > /app/plot.py <<'PY'
"""Plot channel orders and refund percentages from channels.csv."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("channels.csv", newline="", encoding="utf-8") as handle:
    rows_by_channel = {row["channel"]: row for row in csv.DictReader(handle)}

channels = ["Retail", "Wholesale", "Online", "Partners"]
orders = [int(rows_by_channel[channel]["orders"]) for channel in channels]
refund_rate_pct = [float(rows_by_channel[channel]["refund_rate_pct"]) for channel in channels]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
axes[0].bar(channels, orders, color="#356B93", width=0.65)
axes[0].set_title("Orders by channel", pad=12)
axes[0].set_xlabel("Sales channel", labelpad=10)
axes[0].set_ylabel("Orders (count)", labelpad=10)
axes[0].set_ylim(0, max(orders) * 1.15)

axes[1].bar(channels, refund_rate_pct, color="#B85D43", width=0.65)
axes[1].set_title("Refund rate by channel", pad=12)
axes[1].set_xlabel("Sales channel", labelpad=10)
axes[1].set_ylabel("Refund rate (%)", labelpad=10)
axes[1].set_ylim(0, max(refund_rate_pct) * 1.15)

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(w_pad=3)
fig.savefig("figure.png", dpi=150, facecolor="white")
plt.close(fig)

with open("plotted_values.json", "w", encoding="utf-8") as handle:
    json.dump(
        {"channels": channels, "orders": orders, "refund_rate_pct": refund_rate_pct},
        handle,
        indent=2,
        allow_nan=False,
    )
    handle.write("\n")
PY

cd /app
python plot.py
