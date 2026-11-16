# Reproducing the study

## Three different levels

| Level | Included inputs | What it establishes |
|---|---|---|
| Processed replication | Public prompts, worlds, decoded responses, and intervention records | Recomputed behavioral counts and whole-answer repair outcomes |
| Original native audit | Saved registrations, checksums, and audit receipts; **not** all raw tensors | A record of the original checks, not a locally repeatable tensor audit |
| New model execution | Dated sources plus separately downloaded pinned weights, GPU, and new output roots | A new experiment; numerical equivalence must be checked, not assumed |

The default package and CI cover the first level. Recomputing a table from saved
responses does not reproduce a model forward pass. Native audit receipts report
CPU recomputation of saved component changes and checks of saved native score
vectors; they do not mean the model was rerun on a CPU. The large native score
shards and model weights are not shipped in this Git release. Selected smaller
capture/kernel tensors are retained under `data/original/`; they are insufficient
to replay a full-vocabulary native audit. The default analysis does not load them.
Historical path strings and hashes in records identify original inputs;
they are not instructions to connect to an original machine.

## Offline processed replication

After the [quickstart](../README.md#quickstart-no-gpu-or-model-download):

```bash
python -m names_mechinterp.reproduce --data-root data/processed --output results/reproduced.json
python -m pytest tests -q
```

Run from the repository root. The generated result is a new local file, not a
replacement for a historical observation. The test suite uses only released
small data and synthetic inputs. Its network guard prevents accidental network
calls during tests. No pretrained weights are loaded. Archived experiment tests
are deliberately outside default test discovery: some require PyTorch,
Transformers, tiny randomly initialized models, or original artifact roots.

## Pinned checkpoints and original inference environment

| Checkpoint | Immutable revision |
|---|---|
| [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| [`microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | `2fe192450127e6a83f7441aef6e3ca586c338b77` |

The original recorded environment used Python 3.12.15, NVIDIA A40, CUDA 12.4,
PyTorch `2.6.0+cu124`, Transformers `5.14.1`, tokenizers `0.22.2`, NumPy `2.5.1`,
safetensors `0.8.0`, and huggingface-hub `1.33.0`. These are historical recorded
versions, not a claim that every platform can install the same wheels. The
release-facing standard-library analysis does not depend on them.

Model execution used batch size one, frozen evaluation-mode parameters, native
float32, eager attention, TensorFloat-32 disabled, one CPU thread, and greedy
generation of at most eight new tokens including termination. Every step
recomputed the sequence with caching disabled. Decoding stripped only a final
end-of-sequence token. Full-response case/whitespace/final-punctuation
normalization did not discard explanations or truncated suffixes.

## GPU rerun: explicit, opt-in, and not tested by CI

The original runs succeeded on a 48 GB A40. Use a dedicated Linux/CUDA environment,
download the weights separately under their upstream licenses, and allocate
substantial disk space for weights and native outputs. The behavioral runner
allows up to 16 GB of native output for Qwen and 4 GB for Phi, in addition to
weights, token assets, and later stages. It fails closed on its storage/deadline
checks. A comparable GPU is not a guarantee of bitwise-equivalent results.

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r requirements-gpu.txt

python experiments/error_reference_study_20261114/error_run.py prepare \
  --output runs/behavior \
  --dependency-root experiments/e30_14_work \
  --cache-root model-cache

python experiments/error_reference_study_20261114/error_run.py check \
  --output runs/behavior \
  --dependency-root experiments/e30_14_work
```

Preparation uses public pinned configuration/tokenizer assets. A `run` command
downloads the pinned safetensors weights if absent. The loader sets
`trust_remote_code=False`; no access credentials or machine-provisioning utilities
are supplied. To execute one checkpoint, replace the deadline below with a future
UTC ISO-8601 timestamp no more than six hours away:

```bash
python experiments/error_reference_study_20261114/error_run.py run \
  --output runs/behavior \
  --dependency-root experiments/e30_14_work \
  --model qwen --split development \
  --deadline 'REPLACE_WITH_FUTURE_UTC_ISO8601'

python experiments/error_reference_study_20261114/error_run.py rescore \
  --output runs/behavior \
  --dependency-root experiments/e30_14_work \
  --model qwen
```

Use `--model phi` for the other behavioral checkpoint. Never reuse or overwrite a
failed output namespace; preparation and each model run are single-use. The
runner only authorizes the development split. It does not authorize forwarding
the 64 reserved confirmation worlds.

The downstream causal sequence is preserved under the dated `experiments/`
directories: `causal_run.py` → `routing_run.py` → `validation_run.py` →
`decomposition_run.py`. Each exposes `prepare`, `check`, `run`, and `rescore`;
inspect its `--help` and source before use. These stages require the preceding
stage's full native artifacts, sources, custody, and verified audit receipt—not
just this repository's processed JSON. They cannot be restarted directly from
the public processed release. Fresh reruns may change eligibility or head
selection and must not be represented as a bitwise replay of the original.
No GPU rerun or new paid computation was performed to package this release.
`experiments/e30_14_work/` predates this study and is retained as an execution
helper dependency; its older task fixtures are not additional observations in
the 32-world results.

## Paper build

```bash
python -m pip install -e ".[paper]"
python paper/build_submission.py
python paper/export_latex.py --compile
```

The Word/figure/data build is CPU-only. Editable paper sources and final build
instructions are in [`paper/`](../paper/); generated deliverables are in
[`submission/`](../submission/). The LaTeX PDF build additionally requires a local
TeX distribution and `latexmk`; those executables are not installed by pip or CI.
If `latexmk` is not on PATH, pass `--latexmk /path/to/latexmk` to the exporter.
The editable `paper/manuscript.tex`, `appendix.tex`, `cover_letter.tex`, and
`references.bib` produce `submission/Manuscript.pdf`, `S1_Appendix.pdf`, and
`Cover_letter.pdf`. The main submission PDF intentionally leaves figures in
separate files. Vendored journal styles retain their own license notices.
Document layout and binary hashes can vary with fonts and toolchain versions;
that is separate from numerical reproduction of the study.

## Tested versus not tested

- Default CI: released processed-data checks and synthetic tests, with no model
  download, GPU, cloud credentials, or paid provisioning.
- Historical evidence: the original recorded A40 executions and their saved
  audit receipts, as documented in the paper and provenance records.
- Not certified by default CI: the archived full GPU pipeline, native tensor
  replay without the missing raw artifacts, alternative hardware/precision,
  additional models, or the reserved confirmation worlds.
