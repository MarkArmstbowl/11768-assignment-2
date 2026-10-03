# panel_unit_labels

<!-- Copied from tasks/panel_unit_labels/task.json. This is the prompt the three fixed agents
     were given, so do NOT reword it: check-submission holds both your runs
     and this file to the descriptor's wording. Add what your verifier needs
     under the heading below instead. -->

Using matplotlib, read channels.csv. Each row gives a sales channel, its order count, and its refund rate as a percentage; use the numbers directly without aggregation.
Create one figure with two panels arranged in one row and two columns. In both panels, keep the sales channels in this left-to-right order: Retail, Wholesale, Online, Partners.
The left panel must be a vertical bar chart of orders, with the title "Orders by channel", a separate x-axis label "Sales channel", and a separate y-axis label "Orders (count)".
The right panel must be a vertical bar chart of refund_rate_pct, with the title "Refund rate by channel", a separate x-axis label "Sales channel", and a separate y-axis label "Refund rate (%)".
Both panels must have their own x- and y-axis labels; channel tick names or a single figure-level label do not replace these per-panel labels. Bar colors are unrestricted. Make the labels and tick names readable.
Save the complete figure as figure.png in the workspace.

## Required outputs

Create the following three files in `/app`, which is also the working directory.
In `plot.py`, refer to `channels.csv`, `figure.png`, and `plotted_values.json`
by bare filename rather than by absolute path. The verifier re-runs the script
in a copied workspace, so absolute paths would point outside that copy.

Extra exploratory images are allowed. Give every extra image a different stem
from `figure`; do not save another `figure.<anything>` file. The answer is graded
from `figure.png`.

1. **`/app/plot.py`** - the plotting script. Self-contained and re-runnable:
   `python plot.py` from `/app` must reproduce both output files with no arguments and
   no manual steps. It must read the input file rather than hard-code results,
   and it must render the same image every time it runs.
2. **`/app/figure.png`** - the chart, as saved by that script.
3. **`/app/plotted_values.json`** - the values shown in the chart, generated from
   the same variables passed to the plotting functions. It must contain one JSON
   object with exactly three keys: `channels`, `orders`, and `refund_rate_pct`.
   `channels` must be an array of four strings, in this order: `Retail`,
   `Wholesale`, `Online`, `Partners`. `orders` and `refund_rate_pct` must each be
   an array of four JSON numbers, aligned with `channels`, giving the respective
   bar heights directly from the input. Numbers must not be strings, booleans,
   or objects, and must not include unit suffixes. This all-zero example shows
   the schema only; use the actual CSV values in your output:

   ```json
   {"channels": ["Retail", "Wholesale", "Online", "Partners"], "orders": [0, 0, 0, 0], "refund_rate_pct": [0, 0, 0, 0]}
   ```

Do not modify `channels.csv`. Keep the plotting script and its outputs
deterministic: no random styling, timestamps, or other run-dependent content.

Only the libraries already installed are available and there is no network
access; everything you need is in the image.
