# Reproducing the study

## Different reproduction levels

| Level | Included inputs | What it establishes |
|---|---|---|
| Processed replication | Public prompts, worlds, decoded responses, and intervention records | Recomputed behavioral counts and whole-answer repair outcomes |
| Native file integrity | Separately downloaded complete native-data assets and the checkout's metadata | Matching payload bytes and file coverage, not mathematical replay |
| Original mathematical native audit | Native payloads, frozen sources, registrations, and the original environment/path setup | A replay that requires additional setup; saved receipts alone are not a new audit |
| New model execution | Dated sources plus separately downloaded pinned weights, GPU, and new output roots | A new experiment; numerical equivalence must be checked, not assumed |

The default package and CI cover the first level, separately for the original
development data and the completed 64-scenario confirmation. Recomputing a table
from saved responses does not reproduce a model forward pass. Native audit receipts report
CPU recomputation of saved component changes and checks of saved native score
vectors; they do not mean the model was rerun on a CPU. The large native arrays
are distributed separately as release assets, not in the Git checkout. Selected
smaller captures are also retained under `data/original/`. The default analysis
does not load tensor objects. Model checkpoints and learned parameter tensors
are not distributed in either location.
Historical path strings and hashes in records identify original inputs;
they are not instructions to connect to an original machine.
`data/AVAILABILITY.json` is the historical development-checkout inventory (0/410
score shards); `data/CONFIRMATION_AVAILABILITY.json` separately records the 620
confirmation score shards and 30 captures absent from Git. These are not current
public-archive gap reports. The separate archive inventory, checksums, and
publication status are in [`data/NATIVE_RELEASE.json`](../data/NATIVE_RELEASE.json).

## Offline processed replication

After the [quickstart](../README.md#quickstart-no-gpu-or-model-download):

```bash
python -m names_mechinterp.reproduce --data-root data/processed --output results/reproduced.json
python -m pytest tests -q
```

For the completed confirmation, use its separate directory and helper:

```bash
python tools/reproduce_confirmation.py --data-root data/processed/confirmation --output results/confirmation_reproduced.json
```

The helper uses only Python's standard library. It verifies the frozen analysis
source hashes, grades all 6,144 answers per model, rebuilds eligible repair cases
and all four conditions, and recomputes the registered uncertainty ranges.
A separately frozen parser derives facts from the printed question text and
checks the counts independently. Saved `RESULT.json` files, when present, must
agree: counts exactly, floating statistics within `1e-12`. The output path must
not already exist. `CPU_REPRODUCTION.json` in the data folder is a saved result
of this process, not evidence of a new model execution.

This does not recheck native greedy choices, tokenizer decoding, head arithmetic,
or full-score custody. The saved `NATIVE_AUDIT.json` receipts record separate
cloud CPU audits of those original tensors. They cannot be rerun from JSON alone.

## Download and check the native data

The [native-data release](https://github.com/Biodyn-AI/names-mechinterp/releases/tag/native-data-20261116-v1)
contains four development archives, five Qwen confirmation archives, one Phi
confirmation archive, and one repair archive. Together these supply 1,030 score
shards and 39 measurement files: 410 score shards plus 9 captures for development,
and 620 score shards plus 30 captures for confirmation.
The scientific payloads total 16,814,720,490 bytes; the archives download about
15.2 GB. Allow at least 40 GB of free space to keep both archives and extracted
data. These files are optional for processed replication and require no GPU.

Download all 11 `.tar.gz` assets using a browser or the separately installed
GitHub CLI. From the repository root, use a new download directory; do not use
`--clobber`:

```bash
mkdir native-downloads
gh release download native-data-20261116-v1 \
  --repo Biodyn-AI/names-mechinterp --dir native-downloads \
  --pattern 'names-development-part*.tar.gz' \
  --pattern 'names-confirmation-qwen-part*.tar.gz' \
  --pattern 'names-confirmation-phi-part*.tar.gz' \
  --pattern 'names-confirmation-repair-part*.tar.gz'
```

[`data/NATIVE_RELEASE.json`](../data/NATIVE_RELEASE.json) lists the archive
checksums and publication status. Compare each downloaded archive with that
catalog before extracting: use `sha256sum native-downloads/*.tar.gz` on Linux or
`shasum -a 256 native-downloads/*.tar.gz` on macOS. Stop if a filename or checksum
does not match.

The following Python 3.12+ example extracts only the 11 named assets into a
**new** `native/` folder, rejecting links, special
files, escaping names, and duplicate output paths. It never executes archive
contents. If interrupted, retain the partial folder for inspection and use a
different new folder for a retry.

```bash
python - <<'PY'
from pathlib import Path, PurePosixPath
import tarfile

downloads, destination = Path("native-downloads"), Path("native")
names = [f"names-development-part{i:03}.tar.gz" for i in range(1, 5)]
names += [f"names-confirmation-qwen-part{i:03}.tar.gz" for i in range(1, 6)]
names += ["names-confirmation-phi-part001.tar.gz",
          "names-confirmation-repair-part001.tar.gz"]
if any(not (downloads / name).is_file() for name in names):
    raise FileNotFoundError("Download every named archive before extracting")
destination.mkdir()  # Refuse an existing folder; never merge into user data.
seen = set()
for name in names:
    with tarfile.open(downloads / name, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or member.name != path.as_posix()
                    or any(part in (".", "..") for part in path.parts)
                    or not path.parts or path.parts[0] not in ("development", "confirmation")
                    or "\\" in member.name or path.as_posix() in seen):
                raise ValueError("Unsafe or duplicate native archive member")
            seen.add(path.as_posix())
        archive.extractall(destination, members=members, filter="data")
PY
```

The archives preserve two roots. Development files sit below
`native/development/cloud_reference_*_metadata_20261115/<stage>/`; confirmation
files sit below `native/confirmation/{qwen,phi,repair}/`. Check their complete-file
SHA-256 values against the original scientific metadata:

```bash
python -m names_mechinterp.verify --repo-root . \
  --native-root native/development --verify-native --require-native \
  --output native/development_integrity.json
python tools/verify_confirmation_native.py \
  --native-root native/confirmation --output native/confirmation_integrity.json
```

The development report should show 410/410 score shards, nine included tensor
captures, and an empty `missing_non_native_tensor_artifacts` list. Its
`--require-native` exit check covers score-shard completeness, so also inspect
the capture fields. The confirmation helper is strict over all 650 expected
files: 620/620 score shards and 30/30 case captures. It checks every expected file
against the matching metadata index and custody checksum, checks sidecars when
present, rejects escaping paths, and never loads tensor objects. Missing files
produce `complete=false` and a nonzero exit; `--allow-missing` permits an explicitly
incomplete progress report. The optional output path must be new. Indices record
tensor shapes, not exact encoded-file sizes: the helper reports observed sizes
and checks full-file hashes rather than inventing a size guarantee.

## Mathematical replay portability

File integrity is not a mathematical replay. The original cloud native audits
passed, but their frozen registration binds specific cloud source, runtime and
cache paths, and their audit commands seal outputs in single-use namespaces.
The archived token JSON also requires lossless decompression to its original
filenames in a separate working copy.
An arbitrary checkout plus downloaded tensors is therefore **not** promised to
run those exact commands unchanged. The complete payloads support portable byte
checks and new native analyses; reproducing the original mathematical audit requires
an explicitly documented environment/path setup, not editing the frozen evidence
or treating existing audit receipts as fresh checks.

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
The independent confirmation used an NVIDIA RTX A6000 with the same pinned
model and package versions and numerical settings. It is a new evaluation, not
a bitwise replay of the A40 development answers.

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
archived runner only authorizes the development split and still refuses
confirmation inputs. The completed confirmation used the separate, publicly
frozen `experiments/error_reference_confirmation_20261116/` runner; its records
are not permission to reopen the older failed mechanism search.

The downstream causal sequence is preserved under the dated `experiments/`
directories: `causal_run.py` → `routing_run.py` → `validation_run.py` →
`decomposition_run.py`. Each exposes `prepare`, `check`, `run`, and `rescore`;
inspect its `--help` and source before use. These stages require the preceding
stage's full native artifacts, sources, custody, and verified audit receipt—not
just this repository's processed JSON. They cannot be restarted directly from
the public processed release. Fresh reruns may change eligibility or head
selection and must not be represented as a bitwise replay of the original.
The earlier 32-world release was packaged without new inference; the separate
2026-11-16 confirmation was a new bounded cloud experiment. Copying its completed
metadata into this working release does not perform any further model calls.
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

The checklist PDF is a rendering of its Word file, not a separate LaTeX source.
If you change that Word file, regenerate its PDF using LibreOffice or Word before
running `python tools/package_submission.py`. The package command makes the
standalone source archive, complete submission ZIP, and file checksums; it does
not silently regenerate or review documents.

## Tested versus not tested

- Default CI: released processed-data checks and synthetic tests, with no model
  download, GPU, cloud credentials, or paid provisioning.
- Historical evidence: the original recorded A40 executions and their saved
  audit receipts, as documented in the paper and provenance records.
- Release validation also passed 293 archived CPU tests and 4 subtests, with
  pretrained downloads and network access blocked. With the optional research
  dependencies installed, rerun them using
  `python experiments/release_tools/run_archived_tests.py`. This checks the
  archived implementation, not a new inference run on the study models.
- Not certified by default CI: the archived full GPU pipeline, native tensor
  mathematical replay from arbitrary paths, alternative hardware/precision,
  additional models, or a new GPU execution of the confirmation. The new
  confirmation JSON counts and statistics are checked offline, while its saved
  native-audit receipts describe the separately completed cloud checks.
