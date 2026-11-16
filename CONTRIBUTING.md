# Contributing

Please open an issue for a scientific discrepancy, missing reproducibility input,
or proposed change before making a large modification. Include the release or
commit, command, Python version, and a minimal example. Do not attach credentials,
downloaded model weights, private machine details, or large native tensors.

For code changes, add a small test demonstrating the intended behavior, run the
CPU checks in the [quickstart](README.md#quickstart-no-gpu-or-model-download), and
explain whether any released result changes. Keep new analyses separate from
historical records: do not rewrite a saved observation to make it agree with a
new implementation.

The evaluated worlds, labels, and outcomes are public development data. Any
analysis informed by them must be labeled exploratory. A new confirmation study
needs a frozen design and genuinely untouched examples; reusing the six-world
diagnosis is not independent confirmation. Report complete answers and preserve
world-level clustering rather than treating repeated queries as independent.

The default test and analysis workflow must remain CPU-only and must not download
models or provision paid resources. Keep pretrained-model tests and GPU runs
explicitly opt-in. A passing unit suite is not a replication of the original
native-score audit.

Contributions of code are under [MIT](LICENSE); contributions of original data,
documentation, or manuscript content are under [CC BY 4.0](LICENSE-DATA.md).
Only contribute material you have the right to license. Cite upstream work and
retain its applicable notices.
