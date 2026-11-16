# Data dictionary

All JSON is UTF-8. The original top-level processed files refer to the 32-world
development set; the `confirmation/` subtree is a separate 64-world evaluation.
The Git checkout omits the large arrays, which are supplied as separate
[native-data release assets](https://github.com/Biodyn-AI/names-mechinterp/releases/tag/native-data-20261116-v1).
Neither location contains model checkpoints or learned parameter tensors.
Use IDs to join records; array order is not a substitute for an ID. Fields
retained from historical records may contain original path strings. Their
presence does not make those paths public
or available.

## Behavioral inputs and responses

| File under `data/processed/` | Shape and unit |
|---|---|
| `worlds.json` | 32 scenario records; one registry of four objects per entry |
| `prompts.json` | 3,072 objects; one question per entry, shared by both models |
| `qwen_generations.json` | 3,072 objects; one Qwen response per question |
| `phi_generations.json` | 3,072 objects; one Phi response per question |
| `package_scope.json` | Release scope and processed figure coordinates |
| `evidence_digest.json` | Derived cross-stage summaries, per-world values, and original input hashes |

### World fields

| Field | Meaning |
|---|---|
| `world_id` | Stable string such as `development:000` |
| `split` | `development` in the original top-level files; `confirmation` in the separate new subtree |
| `objects` | Mapping A–D to `nameplate`, `color`, `shape`, and `home` strings |
| `object_order` | Order in which the four records are printed |
| `aliases` | Four lexical alias strings in semantic-role order |
| `alias_targets` | Corresponding A–D targets; entries 0 and 1 share a target |
| `alias_order` | Alias-role indices in printed binding-line order |
| `base_object_id` | Object targeted by the equivalent aliases |
| `unaliased_object_id` | The one record with no assigned alias |
| `template` | Wording format, 0 or 1; assigned between worlds |
| `status_word` | Literal answer for the global status probe |

Within a world, values are distinct for each factual field. A response matching
another object's value for that same field therefore identifies a unique listed
wrong object. Individual words recur across worlds; a world is not an entirely
new vocabulary sample.

## Independent confirmation files

All paths in this section are below `data/processed/confirmation/`.

| Path | Contents |
|---|---|
| `ROWS.json` | Complete shared grid of 6,144 questions; every `split` is `confirmation` |
| `WORLDS.json` | 64 scenarios; both models use this same set |
| `PLAN.json` | New publicly frozen plan; old failed success requirements remain failed |
| `qwen/GENERATIONS.json`, `phi/GENERATIONS.json` | 6,144 complete response records each |
| `qwen/RESULT.json`, `phi/RESULT.json` | Recomputed whole-answer counts, scenario summaries and reference intervals |
| `repair/CASES.json` | One deterministic qualifying context from every eligible Qwen scenario: 30 contexts |
| `repair/GENERATIONS.json` | 360 answers: 30 contexts × three properties × four comparisons |
| `repair/RESULT.json` | Exact repair/damage counts, pooled and scenario-balanced rates, uncertainty and limitations |
| `LITERAL_AUDIT.json` | Saved independently parsed text-answer checks; not native tensor replay |
| `CPU_REPRODUCTION.json` | Portable helper's saved reconstruction of the processed inputs |

The original `worlds.json` and `prompts.json` are unchanged development inputs.
The two sets contain 32 and 64 scenarios, not two copies of a common sample.
Questions within a scenario are repeated measurements. The repair contexts come
from the confirmation scenarios, not from another 30 independent scenarios.

Repair records join to their case through `case_id` and to the original question
through `row_id`. Their own `id` distinguishes condition and property. The fixed
conditions are `selfRfield` (unchanged), `primaryDcolor` (correct-name color
profile), `frozenDiscoveryMean` (old fixed average profile), and `Rcolor`
(recipient-name color profile). All preserve the same operation settings.
`initial_wrong_fields` includes every initially incorrect recipient property,
not only the finished wrong-object answer that makes the context eligible.
`protected_correct_fields` identifies the initially correct recipient properties.

The primary repair corrected 16/40 initially wrong facts and damaged 0/50 tested
correct facts. Its scenario-balanced rate differs from the pooled fraction;
neither should replace the other silently. The reported usefulness conclusion
is `uncertain`. Reference intervals and their sampling assumptions are retained
from the frozen plan, not recalibrated after the results.

Original confirmation metadata are under
`data/original/cloud_reference_confirmation_metadata_20261116/`. The two large
`TOKENS-*.json.gz` files decompress to the exact historical bytes; their adjacent
`TOKENS-*.json.sha256` sidecars refer to those decompressed bytes. The deterministic
compression and every imported byte are documented in `IMPORT_RECEIPT.json`.
Other raw-score and case-capture sidecars refer to payloads outside Git, supplied
in the separate native-data assets. `data/CONFIRMATION_AVAILABILITY.json` describes
this sample's checkout inventory; it is distinct from the historical development
inventory in `data/AVAILABILITY.json`. Use `data/NATIVE_RELEASE.json` for current
archive inventory, checksums, and publication status.

### Native payloads for both phases

| Sample | Score shards | Measurement files |
|---|---:|---:|
| Development | 410 | 9 |
| Confirmation | 620 | 30 |
| Total | 1,030 | 39 |

All 1,069 scientific payloads total 16,814,720,490 bytes. Four development archives
also carry 419 original 65-byte checksum sidecars. Seven confirmation archives
hold the two models' score shards and the repair captures; their checksum sidecars
are retained with the original metadata in Git. Archive names and checksums are
recorded in [`data/NATIVE_RELEASE.json`](../data/NATIVE_RELEASE.json).
See [download and integrity commands](reproducing.md#download-and-check-the-native-data).

After extraction, the development root mirrors paths relative to `data/original/`:
`native/development/cloud_reference_*_metadata_20261115/<stage>/`. The confirmation
root uses `native/confirmation/{qwen,phi,repair}/`. Pass these different roots to
the matching verification helpers; do not flatten the directories.

`raw-*.safetensors` stores saved complete-vocabulary output scores. The `.pt` files
store input-dependent internal signals, attention probabilities, applied changes,
and audit measurements. `PREFIXES.pt` contains signals captured before the output
projection, not that projection's learned weights. `OPERATORS.pt` and related
captures describe the applied changes. `KERNELS.pt` is a historical filename for
calculated intervention differences, profiles, and checks—not a model-parameter
matrix. Do not confuse per-input attention "weights" with learned model parameters.
Reading these tensor objects is unnecessary for the standard-library byte checks.

### Prompt fields

| Field | Meaning |
|---|---|
| `id` | Unique question key, used to join either model's response |
| `world_id`, `split`, `template` | Registry and format identifiers |
| `text` | Complete task text before model-specific chat templating |
| `probe_kind` | `fact`, `identity`, or `status` |
| `field` | Requested factual field, or JSON `null` for a nonfactual probe |
| `name_swap` | 0 or 1: consistent exchange of the equivalent alias names |
| `focus_condition` | `none`, `coref_0`, `coref_1`, or `distractor` |
| `query_alias_index` | Alias role queried; factual roles 0 and 1 co-refer |
| `aliases` | Alias strings for this context after the consistent exchange |
| `expected_answer` | Literal canonical answer supplied by the generator |
| `alias_end_span` | Half-open character interval of the explicit alias-end marker |
| `field_start` | Character position at which the field portion begins |

Offsets above refer to `text`, not to token indices or bytes. The 96 questions in
a world comprise two name assignments × four Focus conditions × twelve probes.
Nine probes in a context are factual (three aliases × three fields), two ask for
identifiers, and one asks for the status word.

### Response fields

| Field | Meaning |
|---|---|
| `id` | Matching prompt key |
| `expected_answer` | Retained generator answer, not a model prediction |
| `generated_ids` | Generated token IDs, including a final EOS when present |
| `generated_text` | Decoded answer with only final EOS removed |
| `terminated` | Whether generation ended on configured EOS |
| `truncated` | Whether the token limit was reached without termination |
| `stop_reason` | Recorded stop category |
| `step_endpoint_ids` | IDs of original native-score endpoints, not their tensors |

Correctness requires the entire response to match after case, whitespace, and
final ASCII-punctuation normalization, with `terminated=true` and
`truncated=false`. A prefix such as `Object A` is not the requested bare `A`.
Do not reclassify such output without labeling a separate formatting analysis.
An eligible equivalent-name pair compares roles 0 and 1 for a fixed world,
name assignment, Focus condition, and field: one response is correct and the
other is a completed listed wrong-object answer. There are 24 pairs per world,
not 24 independent worlds.

## Fixed-component diagnosis

Files under `data/processed/decomposition/` preserve the nine-condition diagnosis
on the same six worlds reopened after failed development validation.

- `PLAN.json` specifies 36 natural capture jobs and 162 generation jobs: six
  worlds × three recipient fields × nine conditions. Each generation job has an
  `id`, `case_id`, `world_id`, `row_id`, `field`, `condition`, `baseline_correct`,
  `original_query_position`, and endpoint IDs.
- `INPUTS.json` contains the 36 selected tokenized prompt rows, corresponding
  original responses, six worlds, original color profiles, and provenance. In
  tokenized rows, `input_ids` and `offset_mapping` refer to `rendered_prompt`;
  `original_query_position` is a token index, not `field_start`.
- `stage/GENERATIONS.json` and `stage/RESULT.json` retain whole decoded answers and their stop
  flags. Join planned jobs to generations by the job `id`, not the original
  behavioral `row_id`, since each row has nine interventions.
- `INDEPENDENT_LITERAL_AUDIT.json` records independently parsed whole-answer
  agreement. It is not a native tensor replay.
- `reference-decomposition-20261115-effect-readout.json` stores processed
  first-step canonical-token score changes and aggregate summaries. These are
  derived from original native tensors, supplied in the separate native-data assets.
- `reference-decomposition-20261115-native-audit.json` and custody/source records
  preserve the original audit anchors. A hash alone does not provide a missing
  input or permit the audit to be rerun.

| Condition | Change at fixed layer 28, head 0 |
|---|---|
| `self` | Exact unchanged control |
| `oldDcolorMass` | Original donor-color four-record profile |
| `sameFieldMass` | Same-field donor record profile, recipient within-record weights |
| `sameFieldWithinOnly` | Donor within-record patterns, recipient record profile |
| `sameFieldProfileWithin` | Donor profile and patterns at recipient record total |
| `sameFieldRecordContribution` | Donor-minus-recipient record weighted-value contribution |
| `sameFieldNonrecordResidual` | Difference of actual head outputs minus their record contributions |
| `sameFieldFullHead` | Donor-minus-recipient actual complete head output |
| `normMatchedRandomFullHead` | Seeded random direction matched only to the full-head change norm |

Each change is computed for a particular world, field, and condition, then held
fixed at the original query position throughout greedy generation. The eight
initial errors and ten initially correct facts are the same across conditions.
`repairs` count newly correct complete answers among those eight errors;
`damages` count newly incorrect answers among the ten protected facts. Report
repairing worlds as well as questions: two errors from one world remain clustered.

The processed margin is canonical correct-token logit minus the fixed original
wrong-response first-token logit. It is not an accuracy, a probability, or the
accepted-variant score used in earlier head discovery. First-token correctness
does not imply complete-answer correctness. See [study limits](study.md).

## Provenance and integrity

Release checksums describe the bytes distributed here. Historical SHA-256 anchors
describe original sources and execution inputs and are retained separately. A
copied metadata record can still name an unavailable raw artifact; do not claim
its availability or independent verification merely because the metadata hashes.
The default reproduction checks the included records and reconstructs outcomes.
Downloading the native assets and matching their checksums is a separate byte
integrity check, not a new mathematical audit or model execution. The frozen
native-audit sources retain original cloud paths and environment requirements;
the full arrays do not make those exact commands portable without additional setup.

Original research data are [CC BY 4.0](../LICENSE-DATA.md). Upstream model weights
and dependency licenses are outside that grant.

`data/original/` preserves selected historical metadata, source copies, and
smaller tensor captures with their original names. Its sidecars identify original
payloads even when those payloads live in release assets rather than Git.
`data/AVAILABILITY.json` and `data/CONFIRMATION_AVAILABILITY.json` preserve the
checkout distinction; they must not be read as current public-archive gap reports.
The included tensors are not needed by the standard-library reproduction path;
inspecting them requires a separate PyTorch environment. Do not load untrusted
tensor pickle files from other sources.
