# pooled_success_rates

<!-- Copied from tasks/pooled_success_rates/task.json. This is the prompt the three fixed agents
     were given, so do NOT reword it: check-submission holds both your runs
     and this file to the descriptor's wording. Add what your verifier needs
     under the heading below instead. -->

Using matplotlib, read cohorts.csv. Each row gives successes and attempts for one group in one month. For each month, compute the overall success rate as 100 times the sum of successes divided by the sum of attempts across that month's groups. Do not take the unweighted arithmetic mean of the groups' individual success percentages, and do not pool different months together.
Plot one line with circular markers for the four monthly percentages in this left-to-right order: Jan, Feb, Mar, Apr. Show one point per month, not one point per group.
Use one panel titled "Monthly pooled success rate", with a separate x-axis label "Month" and a separate y-axis label "Success rate (%)". Set the y-axis limits to 0 and 100.
Annotate each point with its percentage to one decimal place and a percent sign. The line color is unrestricted. Keep the annotations, labels, and month tick names readable.
Save the figure as figure.png in the workspace.

## Required outputs

Create the following three files in `/app`, which is also the working directory.
In `plot.py`, refer to `cohorts.csv`, `figure.png`, and `plotted_values.json` by
bare filename rather than by absolute path. The verifier re-runs the script in
a copied workspace, so an absolute path would point outside that copy.

Extra exploratory images are allowed, but give each one a different stem from
`figure`: do not create another image named `figure.<anything>`. The answer is
graded from `figure.png`.

1. **`/app/plot.py`** - a self-contained, re-runnable Python plotting script.
   Running `python plot.py` from `/app` must read `cohorts.csv` and reproduce both
   `figure.png` and `plotted_values.json` with no arguments or manual steps.
   Keep this program on disk and compute the results from the CSV rather than
   hard-coding percentages. Every run must render the same image and sidecar:
   do not use random colors, timestamps, or other changing content.
2. **`/app/figure.png`** - a valid, readable PNG chart saved by `plot.py`.
   Re-running the script must reproduce this exact image.
3. **`/app/plotted_values.json`** - one JSON object with exactly the four keys
   `Jan`, `Feb`, `Mar`, and `Apr`. Each value must be a JSON number giving that
   month's pooled success percentage on the 0-to-100 scale, computed as
   `100 * sum(successes) / sum(attempts)` within the month. Do not use strings,
   booleans, nulls, percent signs, fractions on the 0-to-1 scale, or extra keys.
   Generate the object from the same variables passed to the plotting function.
   An all-zero schema example is `{"Jan": 0, "Feb": 0, "Mar": 0, "Apr": 0}`.

Artifact and replay checks: `plot.py`, both outputs, and `cohorts.csv` must be
regular files, not symbolic links. `figure.png` must be a valid PNG at least
100 pixels wide and 100 pixels high, larger than 1,000 bytes, and containing
more than four distinct RGB colors. These are basic artifact-validity guards,
not a substitute for the readability requirements above. On a clean replay
from a copied workspace, `python plot.py` must exit successfully within
60 seconds, reproduce `figure.png` byte-for-byte, and reproduce an equal
decoded `plotted_values.json` object. Use finite JSON numbers and unique
object keys.

Do not modify `cohorts.csv`, including during replay.

Only the libraries already installed are available and there is no network
access; everything you need is in the image.
