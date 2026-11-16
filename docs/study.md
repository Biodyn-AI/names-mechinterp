# Study scope and interpretation

## Completed independent confirmation

On 2026-11-16, both models answered all 6,144 questions in the same 64 previously
untouched scenarios. These are separate from the 32 development scenarios below.
Each supplies 72 factual questions and 24 equivalent-name pairs. Two models do
not make 128 independent scenarios.

| Outcome | Qwen3-4B-Instruct-2507 | Phi-3.5-mini-instruct |
|---|---:|---:|
| Correct factual answers | 3,656 / 4,608 | 3,782 / 4,608 |
| Completed wrong-object factual answers | 952 / 4,608 | 546 / 4,608 |
| Other factual errors | 0 / 4,608 | 280 / 4,608 |
| Correct-versus-wrong-object equivalent-name pairs | 265 / 1,536 | 166 / 1,536 |
| Scenarios with those paired disagreements | 48 / 64 | 38 / 64 |

The [new plan and source snapshot](https://github.com/Biodyn-AI/names-mechinterp/tree/f1d7836c653a82c0183c8bfed59c2c1153f73e42/experiments/error_reference_confirmation_20261116)
was published before these model answers were opened. It asks a narrower question
than the earlier unsuccessful mechanism proposal. It does not override the old
failed requirements or search for another component.

Every eligible Qwen scenario contributed one context chosen by the fixed ordering:
30 contexts, with 40 initially wrong and 50 initially correct property answers.
The same layer-28, head-0 record-share change corrected 16/40 errors, preserving
50/50 correct answers. Its fixed average-pattern comparison corrected 7/40 and
the same-name color comparison corrected 6/40, damaging one correct answer.
The zero-change comparison reproduced all original answers.

The primary repair's pooled rate is 40%. Giving each eligible scenario equal
weight yields 41.67%, with a conservative reference range of 16.87–66.46%.
That range includes the prespecified 50% benefit threshold: usefulness remains
uncertain, rather than confirmed or excluded. The corresponding advantage over
the fixed average pattern is 23.89 percentage points, with a range of
−25.70 to 73.48 points. Zero observed damage does not demonstrate a damage risk
below 10%: the upper one-sided bound is 10.15% across the 28 scenarios containing
initially correct answers. These are separate intervals, not a joint confidence
claim. They use the sampling approximation stated in the frozen plan; balanced
generation and unique assignments couple the scenarios.

The repair sample is selected from the same 64 scenarios used for the answer
tests, not a second independent dataset or an unconditional rate over all errors.
Complete results and saved audits are in `data/processed/confirmation/` and
`data/original/cloud_reference_confirmation_metadata_20261116/`. Raw payload
access is provided separately through the
[native-data release](https://github.com/Biodyn-AI/names-mechinterp/releases/tag/native-data-20261116-v1)
and its `data/NATIVE_RELEASE.json` archive catalog. The Git checkout itself omits
the large arrays: its confirmation inventory is recorded in
`data/CONFIRMATION_AVAILABILITY.json`, not a claim of a current public-data gap.
Metadata and checksums alone do not provide the tensors; download the assets for
byte-level checks. They cover all 1,030 score shards and 39 measurement files from
both study phases. Matching those bytes is not a fresh mathematical audit or a
model rerun, and the original native audits retain frozen cloud-path requirements.

## Earlier development measurements

Both checkpoints evaluated the same 32 independently generated contextual
registries ("worlds"). Each world contains four objects, three factual fields,
and four aliases. Two aliases name the same object. Consistent exchanges of
those names and four highlighting conditions give repeated questions within
each world: 96 questions, including 72 factual questions, per model.

| Outcome | Qwen3-4B-Instruct-2507 | Phi-3.5-mini-instruct |
|---|---:|---:|
| Correct factual answers | 1,657 / 2,304 | 1,980 / 2,304 |
| Completed wrong-object factual answers | 647 / 2,304 | 260 / 2,304 |
| Other factual errors | 0 / 2,304 | 64 / 2,304 |
| Eligible equivalent-name disagreements | 117 / 768 | 69 / 768 |
| Worlds with eligible disagreements | 23 / 32 | 14 / 32 |

An eligible disagreement has one correct response and one completed response
matching another object's listed value for the requested field. It is narrower
than any string difference. Facts were graded using the entire response, not a
selected first line. Phi often prefixed identifier answers with `Object`; its low
strict bare-identifier score is not evidence that it could not identify objects.

The independent unit is a world, not a question. Both models share those worlds;
they do not provide 64 independent worlds. The original descriptive factual
accuracy intervals resample whole worlds 2,000 times (seed 20261114). They are
not question-level binomial intervals or population-wide guarantees.

## Earlier development interventions

Only Qwen received internal interventions. A deterministic selection found 14
eligible development worlds: eight for exploratory component selection and six
initially unused for intervention selection. Natural attention across 1,152 heads
was ranked on the eight labeled discovery cases. Two candidates were tested and
layer 28, head 0 (zero-based) was selected by a fixed ordering.

A four-record allocation transfer made the first token correct in 5/8 discovery
cases. This was not a completed-answer repair measurement. On the six other
development worlds, it repaired only 1/8 complete factual errors and preserved
10/10 initially correct facts. It failed the fixed repair and steering criteria.

A later nine-condition diagnosis reopened those same six worlds. Six new
targeted changes each repaired 0/8 errors; the original color-derived transfer
reproduced 1/8. All conditions preserved the ten tested correct facts. Complete
donor-head transfer still changed prediction scores substantially, so no repair
does not mean numerical inactivity or lack of causal influence.

## What is not established

- No new general retrieval mechanism or abstract entity representation is claimed.
- No absence of object representations, universal repair failure, or exhaustive
  search over components is established.
- Behavioral replication across two checkpoints is not causal replication across
  model families; only Qwen was intervened on.
- The reopened diagnosis is exploratory, not a fresh validation sample.
- A whole-head donor can carry answer information and is not a mathematical
  upper bound on the restricted changes' performance.
- The one random direction is norm-matched only to the whole-head change.
- The tested ten preserved facts do not establish preservation on unrelated tasks.

The original development plans were internally hash-frozen, not publicly
preregistered. Their 64 reserved scenarios were subsequently evaluated only under
the new, publicly frozen narrower plan described above. The failed development
results are preserved, not relabeled as independent confirmation. The working
paper update and its public release status are tracked separately from the data.
