# Equivalent names and factual retrieval

Research code, processed observations, and the draft paper **Equivalent names yield
inconsistent factual answers in two language models**.

The study asks whether two names explicitly assigned to the same object produce
the same factual answer. This repository preserves the naming failure, the earlier
failed repair tests, and a completed independent test of the unchanged repair.
It does not claim a new general binding mechanism or that models lack object
representations. See [study scope](docs/study.md) before interpreting the results.

Working update for planned version **0.2.0**: the 2026-11-16 confirmation results
below are now included. The latest published paper snapshot remains `v0.1.1`
until the next release is published; this working update is not a verified raw-data deposit.
The [submission files](submission/) are a manuscript draft, not an accepted paper.

## Main observations

Both models completed the same **64 previously untouched scenarios**, called
worlds in the code, with 6,144 questions per model. Each scenario describes four
objects and their properties, including two names for the same object.

| Confirmation outcome | Qwen | Phi |
|---|---:|---:|
| Correct factual answers | 3,656 / 4,608 | 3,782 / 4,608 |
| Completed answers about another listed object | 952 | 546 |
| Other factual errors | 0 | 280 |
| Equivalent-name pairs with one correct and one wrong-object answer | 265 / 1,536 | 166 / 1,536 |
| Scenarios containing such a disagreement | 48 / 64 | 38 / 64 |

The unchanged internal repair was tested on every eligible scenario: 30 in total.
It corrected 16 of 40 initially wrong complete answers and kept 50 of 50 tested
correct answers correct. The fixed average-pattern comparison corrected 7 of 40;
the same-name color comparison corrected 6 of 40 and damaged one correct answer.
The prespecified uncertainty ranges do **not** establish a generally useful
repair or its absence. See [study scope](docs/study.md) for the exact measures and limits.

The earlier development sample remains separate: both models evaluated the same
32 scenarios, with 3,072 questions per model. Qwen answered 1,657/2,304 factual
questions correctly; Phi answered 1,980/2,304 correctly.

In Qwen, a selected attention-head change made the first token correct in 5/8
discovery cases. Frozen testing on six other development worlds repaired only
1/8 complete errors. Reopening those six worlds for a nine-condition diagnosis
found no new complete-answer repair despite substantial prediction-score changes.
Those development failures remain failed; the independent test does not revise
their success requirements or turn the original mechanism proposal into a discovery.

## Quickstart: no GPU or model download

Use Python 3.12 or newer. From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m names_mechinterp.reproduce --data-root data/processed --output results/reproduced.json
python tools/reproduce_confirmation.py --data-root data/processed/confirmation --output results/confirmation_reproduced.json
```

The release-facing package has no runtime dependencies beyond Python's standard
library. Installing the test environment may access PyPI; the tests and processed
reproduction do not need network access, model weights, tokens, cloud accounts,
or paid resources. They read the included responses rather than run a model.

The original reproduction command recomputes the earlier development results;
the confirmation helper checks the separate new sample and its answers against
the printed object lists. The confirmation output must be a new file.
Neither command replays native full-vocabulary scores, hidden states, or kernels.
The original 410-shard development gap remains. The new confirmation has saved
native-audit receipts, but a public full-raw deposit is not yet asserted; see
[`data/CONFIRMATION_AVAILABILITY.json`](data/CONFIRMATION_AVAILABILITY.json).
See [reproduction levels and GPU instructions](docs/reproducing.md).

## Repository map

| Path | Purpose |
|---|---|
| [`src/names_mechinterp/`](src/names_mechinterp/) | Small offline reproduction interface |
| [`tests/`](tests/) | Network-free release checks and synthetic unit tests |
| [`data/processed/`](data/processed/) | Prompts, worlds, generated answers, and processed intervention records |
| [`data/processed/confirmation/`](data/processed/confirmation/) | Separate 64-world confirmation inputs, complete answers, statistics, and CPU reconstruction |
| [`tools/reproduce_confirmation.py`](tools/reproduce_confirmation.py) | Standard-library confirmation reconstruction; no model or tensor replay |
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
