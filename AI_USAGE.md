# AI usage declaration

Author: Yubo Chen (Andrew ID: yuboc)

I used AI assistance in completing this assignment. The tools and their roles
are described below; AI assistance was not limited to proofreading.

## OpenAI Codex / conversational coding assistance

I used an interactive OpenAI coding assistant to explain and translate the
assignment, understand the provided workflow, and discuss evaluation design.
The assistant helped inspect tasks, trajectories, input data, and figures;
diagnose validator errors; propose and implement changes to
`validator/solution.py`; write offline tests; and interpret experiment results.
It also assisted with deployment troubleshooting, context and response-format
handling, and comparing unsuccessful experimental revisions with the selected
final implementation.

The assistant helped design the five authored tasks and their synthetic CSV
inputs, check the intended calculations, and implement the Harbor task
instructions, reference solutions, deterministic verifiers, and two mutants.
It drafted reference annotations and a local review aid for the 15 generated
runs. I reviewed the runs and confirmed the final labels; the annotations were
AI-assisted rather than independently drafted entirely by me. The supplied
seed labels were used as provided and were not rewritten to agree with the
validator.

The assistant substantially assisted in drafting and revising `report.tex`,
preparing its figures, checking the correspondence between reported results
and saved predictions, and compiling the PDF. It also helped audit submission
files and identify local files that should not be submitted. The report includes
failed attempts and remaining classification/evidence errors, rather than
treating AI explanations or passing local tests as proof of correctness.

This development assistant is not called by the submitted `validate(run)`
implementation. No additional development-time model is used as a judge at
validator inference time.

## My own involvement and learning process

I worked through the assignment step by step with the assistant's support. My
role included operating the experiments, asking for explanations, reviewing
results, and deciding how to proceed; the assistant provided substantial
implementation and writing assistance as described above.

- **Running and observing experiments.** I performed environment and Modal
  setup, configured my local endpoint credentials, and ran the assignment's
  evaluation, label-initialization, label-validation, and scoring commands.
  I also ran the Harbor worked-example oracle, no-op, and illegible-mutant
  experiments and the first authored-task oracle check. I shared actual
  terminal output and errors for diagnosis. I authorized the assistant to
  perform subsequent repetitive Harbor verification and offline tests, so I
  do not claim that I personally ran every development check.
- **Working on conceptual understanding.** I asked about the distinction
  between the visualization-generation agents and the validator, how validator
  predictions are compared with ground-truth labels, and what false positives
  and false negatives mean. I also asked about the fixed-model restriction,
  context limits, step limits, long evaluation times, and GPU billing. These
  discussions helped me connect the commands and files to the evaluation
  concepts rather than treating the workflow only as a sequence of commands.
- **Reviewing annotations and ambiguous figures.** I reviewed the 15-run review
  index and explicitly confirmed all 15 proposed labels. In discussing the
  figures, I questioned whether the South zero-value annotation was actually
  unreadable and identified the shared-axis figure's left-side text overlap.
  The assistant helped inspect the images and trajectories and articulate the
  evidence. My confirmation was human review with AI assistance; it was not an
  independent unaided derivation of every annotation or numerical target.
- **Questioning the process and checking requirements.** I asked for each
  part's requirements to be explained before implementation, checked the
  prescribed command order, and requested that commands intended for me be
  left for me to run. I repeatedly requested comparisons between the writeup,
  report, actual outputs, and submission files, including credential checks
  and preservation of earlier experiment results. Those detailed file and
  code audits were performed with the assistant's help.
- **Interpreting limitations and making decisions.** I asked why judgments
  remained incorrect, whether failed Qwen generation runs had to be replaced,
  and whether passing tests implied a correct validator. When the final
  experimental support-review revision still produced incomplete results, I
  agreed to restore the previously fully evaluated version. I then ran the
  final authored evaluation and both final scoring commands. The resulting
  seed and authored macro-MCC values were 0.4462 and 0.5000. Our analysis
  distinguishes completing all runs, predicting the correct error family, and
  giving correct explanatory evidence; none of these alone proves the others.

I am responsible for reviewing the submission and being able to explain its
design and limitations. The involvement above does not change the substantial
AI contribution to the code, tests, tasks, annotations, and report, and I do
not present those artifacts as entirely unaided work.

## Required validator model

I used `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`, served through the provided
OpenAI-compatible Modal deployment, for baseline and modified-validator
judgments. It is the only model called by the submitted validator. Its outputs
were evaluated against the provided seed labels and the human-confirmed
authored labels; model-generated predictions were not substituted for those
reference labels. The OpenAI-compatible client is a transport interface to
this Qwen deployment, not a call to an OpenAI judge model.

## Required visualization-generation agents

I used the three prescribed models through the provided generation workflow:

- `Qwen/Qwen2.5-Coder-3B-Instruct`
- `mistralai/Ministral-3-14B-Instruct-2512`
- `zai-org/GLM-4.7-Flash`

Each agent was run once on each of the five authored tasks. Their saved outputs
and complete trajectories are included under `runs/`, including runs that did
not deliver a figure. They generated the candidate visualizations to be
evaluated; they were not additional judge models. Harbor reference solutions
and mutants are separate, AI-assisted development artifacts, not replacements
for these fixed-agent runs.

The final metrics in the report are computed from the saved prediction files
with the assignment's scoring workflow. I remain responsible for understanding
and explaining the submitted code and evaluation choices.
