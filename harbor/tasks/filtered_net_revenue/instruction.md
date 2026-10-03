# filtered_net_revenue

<!-- Copied from tasks/filtered_net_revenue/task.json. This is the prompt the three fixed agents
     were given, so do NOT reword it: check-submission holds both your runs
     and this file to the descriptor's wording. Add what your verifier needs
     under the heading below instead. -->

Using matplotlib, read orders.csv. Each row is an order; gross and refund are amounts in USD.
Keep only rows whose status is exactly "completed". For each region, compute net revenue by summing gross minus refund over those retained rows.
Plot one vertical bar per region in this left-to-right order: North, South, East, West. Include all four regions even if a region has no retained rows; its net revenue is then zero. Do not plot individual orders or include failed or pending orders.
Use one panel titled "Completed-order net revenue", with a separate x-axis label "Region" and a separate y-axis label "Net revenue (USD)". Start the y-axis at zero.
Annotate every bar with its integer net revenue, including a visible "0" for a zero-height bar. Bar colors are unrestricted. Keep the annotations, labels, and region tick names readable.
Save the figure as figure.png in the workspace.

## Required outputs

Create the following three files in `/app`, which is also the working directory.
In `plot.py`, refer to `orders.csv`, `figure.png`, and `plotted_values.json` by
bare filename rather than by absolute path. The verifier re-runs the script in
a copied workspace, so an absolute path would point outside that copy.

Extra exploratory images are allowed, but give each one a different stem from
`figure`: do not create another image named `figure.<anything>`. The answer is
graded from `figure.png`.

1. **`/app/plot.py`** - a self-contained, re-runnable Python plotting script.
   Running `python plot.py` from `/app` must read `orders.csv` and reproduce both
   `figure.png` and `plotted_values.json` with no arguments or manual steps.
   Keep this program on disk and compute the results from the CSV rather than
   hard-coding totals. Every run must render the same image and sidecar: do not
   use random colors, timestamps, or other changing content.
2. **`/app/figure.png`** - a valid, readable PNG chart saved by `plot.py`.
   Re-running the script must reproduce this exact image.
3. **`/app/plotted_values.json`** - one JSON object with exactly the four keys
   `North`, `South`, `East`, and `West`. Each value must be a JSON number giving
   that region's completed-order net revenue in USD, including numeric zero
   for a region with no completed orders. Do not use strings, booleans, nulls,
   unit suffixes, or extra keys. Generate the object from the same variables
   passed to the plotting function. An all-zero schema example is
   `{"North": 0, "South": 0, "East": 0, "West": 0}`.

Do not modify `orders.csv`.

Only the libraries already installed are available and there is no network
access; everything you need is in the image.
