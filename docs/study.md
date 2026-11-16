# Study scope and interpretation

## What was measured

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

## What the interventions showed

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

An additional 64 reserved worlds were generated and tokenized but received no
model forward passes. They are not part of the reported evaluation. Internal
hash-frozen execution plans improve traceability but are not public preregistration.
See the [manuscript and appendix](../submission/) for the full methods and gates.
