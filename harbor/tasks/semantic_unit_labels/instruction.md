# semantic_unit_labels

<!-- Copied from tasks/semantic_unit_labels/task.json. This is the prompt the three fixed agents
     were given, so do NOT reword it: check-submission holds both your runs
     and this file to the descriptor's wording. Add what your verifier needs
     under the heading below instead. -->

Using matplotlib, read warmup.csv and plot the six supplied temperature observations in chronological order as one line with circular markers. The input time column is elapsed_seconds; convert it to minutes for the plotted x-values. The temperature_c column is already in degrees Celsius.
Use a single panel with the title "Temperature during warm-up". Give it a separate x-axis label that states elapsed time in minutes and a separate y-axis label that states temperature in degrees Celsius.
The axis-label wording is not required to be verbatim: "Time (min)" and "Elapsed time (minutes)" are both acceptable x-axis labels, and "Temperature (C)" and "Temperature (degrees Celsius)" are both acceptable y-axis labels. Other wording that clearly expresses the same quantity and unit is also acceptable. Numeric tick labels alone do not satisfy the axis-label requirements.
The line color is unrestricted. Make all labels and plotted observations readable.
Save the figure as figure.png in the workspace.

## Required outputs

Create the following three files in `/app`, which is also the working directory.
In `plot.py`, refer to `warmup.csv`, `figure.png`, and `plotted_values.json`
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
   the same variables passed to the plotting function. It must contain one JSON
   object with exactly two keys: `elapsed_minutes` and `temperature_c`. Each
   value must be an array of six JSON numbers. Match observations by array index,
   in chronological order; `elapsed_minutes` contains the plotted x-values after
   converting seconds to minutes, and `temperature_c` contains the plotted
   temperatures in degrees Celsius. Numbers must not be strings, booleans, or
   objects, and must not include unit suffixes. This all-zero example shows the
   schema only; use the actual CSV values in your output:

   ```json
   {"elapsed_minutes": [0, 0, 0, 0, 0, 0], "temperature_c": [0, 0, 0, 0, 0, 0]}
   ```

Artifact and replay checks: `plot.py`, both outputs, and `warmup.csv` must be
regular files, not symbolic links. `figure.png` must be a valid PNG at least
100 pixels wide and 100 pixels high, larger than 1,000 bytes, and containing
more than four distinct RGB colors. These are basic artifact-validity guards,
not a substitute for the readability requirements above. On a clean replay
from a copied workspace, `python plot.py` must exit successfully within
60 seconds, reproduce `figure.png` byte-for-byte, and reproduce an equal
decoded `plotted_values.json` object. Use finite JSON numbers and unique
object keys.

Do not modify `warmup.csv`, including during replay. Keep the plotting script and its outputs
deterministic: no random styling, timestamps, or other run-dependent content.
Equivalent axis-label wording remains acceptable as described above.

Only the libraries already installed are available and there is no network
access; everything you need is in the image.
