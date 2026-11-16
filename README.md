# Equivalent names and factual retrieval

Research code, processed observations, and the draft paper **Equivalent names yield
inconsistent factual answers in two language models**.

The study asks whether two names explicitly assigned to the same object produce
the same factual answer. This repository preserves the positive behavioral finding
and the failed single-component repair, including controls and negative results.
It does not claim a new general binding mechanism or that models lack object
representations. See [study scope](docs/study.md) before interpreting the results.

Release version: **0.1.0**; the intended study snapshot is tag `v0.1.0`.
The [submission files](submission/) are a manuscript draft, not an accepted paper.

## Main observations

Both checkpoints evaluated the same 32 development worlds, with 3,072 questions
per model. Qwen answered 1,657/2,304 factual questions correctly and produced 647
completed wrong-object answers. Phi answered 1,980/2,304 correctly, with 260
wrong-object answers and 64 other factual errors.

In Qwen, a selected attention-head change made the first token correct in 5/8
discovery cases. Frozen testing on six other development worlds repaired only
1/8 complete errors. Reopening those six worlds for a nine-condition diagnosis
found no new complete-answer repair despite substantial prediction-score changes.
An additional 64 reserved worlds received no model forward passes.

## Quickstart: no GPU or model download

Use Python 3.12 or newer. From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m names_mechinterp.reproduce --data-root data/processed --output results/reproduced.json
```

The release-facing package has no runtime dependencies beyond Python's standard
library. Installing the test environment may access PyPI; the tests and processed
reproduction do not need network access, model weights, tokens, cloud accounts,
or paid resources. They read the included responses rather than run a model.

The output recomputes released behavioral and complete-answer intervention
summaries from processed records. It is **not** a replay of native full-vocabulary
score tensors. Those large raw tensors are not included; saved audit receipts
record the original checks but cannot substitute for the missing tensors.
See [reproduction levels and GPU instructions](docs/reproducing.md).

## Repository map

| Path | Purpose |
|---|---|
| [`src/names_mechinterp/`](src/names_mechinterp/) | Small offline reproduction interface |
| [`tests/`](tests/) | Network-free release checks and synthetic unit tests |
| [`data/processed/`](data/processed/) | Prompts, worlds, generated answers, and processed intervention records |
| [`experiments/`](experiments/) | Dated original experiment and audit sources, kept separate from the release interface |
| [`paper/`](paper/) | Editable manuscript, appendix, bibliography, and build sources |
| [`submission/`](submission/) | Manuscript/appendix files, figures, cover letter, and supporting dataset |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Record fields, joins, units, and provenance limits |
| [`docs/reproducing.md`](docs/reproducing.md) | Reproduction commands, model revisions, original runtime, and limitations |

## Paper and figures

```bash
python -m pip install -e ".[paper]"
python paper/build_submission.py
# Optional: requires a TeX distribution and latexmk, separately installed.
python paper/export_latex.py --compile
```

This CPU-only build creates the Word submission artifacts, figures, and processed
supporting-data archive. [Paper build details](docs/reproducing.md#paper-build)
describe the separate LaTeX/PDF toolchain. Neither build runs a research model.

## Citation and licenses

Use [CITATION.cff](CITATION.cff), citing the release or exact commit you used. No
paper DOI or archival repository DOI is asserted by this release. Please retain
the distinction between development results and independent confirmation.

Code is [MIT licensed](LICENSE). Original study data, documentation, figures, and
manuscript are [CC BY 4.0](LICENSE-DATA.md). Model weights and third-party
dependencies retain their own licenses and are not redistributed here. Vendored
paper templates/styles retain the notices in [`paper/vendor/`](paper/vendor/).

Corrections and reproducibility reports are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
