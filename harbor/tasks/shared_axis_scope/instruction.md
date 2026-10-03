# shared_axis_scope

<!-- Copied from tasks/shared_axis_scope/task.json. This is the prompt the three fixed agents
     were given, so do NOT reword it: check-submission holds both your runs
     and this file to the descriptor's wording. Add what your verifier needs
     under the heading below instead. -->

Using matplotlib, read sensors.csv. Create one figure containing two line-plot panels arranged in two rows and one column. Use time_s as the x-values in both panels, in ascending order.
The top panel must plot sensor_a_v and have the title "Sensor A". The bottom panel must plot sensor_b_v and have the title "Sensor B". Include all six supplied observations in each panel.
The panels must share the x-axis scale. Use exactly one figure-level x-axis label "Elapsed time (s)" beneath the pair of panels and exactly one figure-level y-axis label "Amplitude (V)" alongside the pair. These shared figure-level labels are the requested axis labels for both panels; do not add separate per-panel x- or y-axis labels. Ordinary numeric tick labels are still allowed.
The line colors are unrestricted. Make the shared labels, panel titles, ticks, and lines readable and keep them inside the saved image.
Save the complete figure as figure.png in the workspace.

## Required outputs

Create the following three files in `/app`, which is also the working directory.
In `plot.py`, refer to `sensors.csv`, `figure.png`, and `plotted_values.json`
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
   object with exactly three keys: `time_s`, `sensor_a_v`, and `sensor_b_v`.
   Each value must be an array of six JSON numbers, with observations aligned by
   array index in ascending time order. `time_s` holds the x-values used in both
   panels; `sensor_a_v` holds the top panel's y-values and `sensor_b_v` the bottom
   panel's y-values, in volts. Numbers must not be strings, booleans, or objects,
   and must not include unit suffixes. This all-zero example shows the schema
   only; use the actual CSV values in your output:

   ```json
   {"time_s": [0, 0, 0, 0, 0, 0], "sensor_a_v": [0, 0, 0, 0, 0, 0], "sensor_b_v": [0, 0, 0, 0, 0, 0]}
   ```

Artifact and replay checks: `plot.py`, both outputs, and `sensors.csv` must be
regular files, not symbolic links. `figure.png` must be a valid PNG at least
100 pixels wide and 100 pixels high, larger than 1,000 bytes, and containing
more than four distinct RGB colors. These are basic artifact-validity guards,
not a substitute for the readability requirements above. On a clean replay
from a copied workspace, `python plot.py` must exit successfully within
60 seconds, reproduce `figure.png` byte-for-byte, and reproduce an equal
decoded `plotted_values.json` object. Use finite JSON numbers and unique
object keys.

Do not modify `sensors.csv`, including during replay. Keep the plotting script and its outputs
deterministic: no random styling, timestamps, or other run-dependent content.

Only the libraries already installed are available and there is no network
access; everything you need is in the image.
