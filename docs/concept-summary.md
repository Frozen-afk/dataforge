# Recurrent depth as an inference-time compute axis

**Latent Loop Lab — one-page concept summary.** DataForge 2026, Pathway Track.

## The design pressure

A Transformer that reasons harder mostly reasons *longer in text*. Chain of
thought buys accuracy by emitting intermediate tokens; each costs a forward
pass, a slot in a key-value cache that grows with the sequence, and
un-parallelisable latency. The trace is legible, a real benefit, but the cost
scales with how much thinking you want.

Recurrent latent computation takes the other route. One fixed-shape state is
updated by the same parameters repeatedly and nothing is emitted until the end,
so the budget is an iteration count `R` chosen at inference time. Memory stops
growing with the amount of thinking, and the thinking stops being readable.

## The claim

> Repeated application of a **shared** state-update rule increases effective
> computational depth without generating intermediate language tokens — within
> the learned dynamics' capacity and stability limits.

*Shared* carries the argument. One block reused every step makes depth a loop
bound at constant parameter count; a distinct block per step fixes depth when
training ends. The artifact measures both: 11,169 parameters at any depth,
against 39,585 with a hard ceiling at the trained depth of 4.

## Mechanism, small enough to see

The task is directed-graph reachability. A learner moves `R` and watches a
noisy-OR recurrence `h(r+1)[v] = max(h(r)[v], 1 − ∏(1 − α·h(r)[u]))` spread
activation one edge per step. The invariant `h(R)[q] > ε ⟺ d(s,q) ≤ R` is
checked against an independent breadth-first search and holds on 10,000 seeded
stratified cases, so `R` has an exact meaning: edges information can travel.

The same shape is then learned. A shared-weight recurrent graph network trains
on distances 1–4 only, with `R` resampled from {1..4} per batch, then runs at
depths to 10 on distances to 10. Five seeds, frozen weights, only `R` varies:

| `R` | learned | 95% CI | distances ≤ 4 | distances ≥ 5 | exact |
|---|---|---|---|---|---|
| 1 | 0.552 | [0.546, 0.558] | 0.621 | 0.506 | 0.550 |
| 4 | 0.700 | [0.700, 0.700] | 1.000 | 0.500 | 0.700 |
| 10 | 0.991 | [0.972, 1.000] | 0.994 | 0.988 | 1.000 |

Distances ≥ 5 sit at chance until `R` exceeds 4 — correct, since the answer is
not yet knowable at the target — then climb to 0.988, solving paths longer than
anything trained on by applying the same weights more times. Distances ≤ 4 peak
at 1.000 near `R = 5` and erode to 0.994 by `R = 10` as the state norm
saturates. Depth is not free.

Making this measure anything required fixing the data. An earlier generator
built negatives as reversed chains, leaving the target with no incoming edges,
so reachability collapsed to a one-hop lookup and accuracy was 1.000
everywhere. Each negative is now cut from a specific positive, matching node
and edge counts exactly; two smaller gaps remain and are reported, not hidden.

## Where BDH and BDH-CQ sit

BDH (Kosowski et al., arXiv:2509.26507) is a post-Transformer architecture of
high-dimensional positive activations, low-rank communication and a recurrent
associative state, with attention reformulated as synaptic memory; reported to
rival GPT-2 at matched parameters from 10M to 1B, with sparse activations and
synapse-level monosemanticity. Its GPU formulation combines ReLU low-rank
transformations with linear attention and is not a Mamba-style state-space
model.

BDH-CQ (Engdahl et al., arXiv:2608.09888) contains **two** recurrences that are
routinely conflated. `S_t = U_θ(S_{t−1}, D_t)` is contextual memory advancing
once per demonstration, driven by new input, with additive special case
`S_t = S_{t−1} + U_θ(D_t)`. `H_{r+1} = F_θ(H_r, S_K)` is a query-time workspace
advancing with no new input and `S_K` held fixed. Only the second is a
reasoning step, and only the second is what this artifact models — the lab has
no contextual recurrence at all. A live toy lets a learner drive each and
produce the distinguishing failure: withhold a demonstration and no latent
budget recovers the missing association.

| | Chain of thought | Recurrent depth | BDH-CQ |
|---|---|---|---|
| Compute knob | tokens emitted | iterations `R` | latent effort level |
| State growth | KV cache grows | fixed shape | fixed-shape recurrent memory |
| Observability | readable trace | none | none |

## Evidence, labelled

BDH-CQ reports LOW 21%, MEDIUM 27%, HIGH 29.5% pass@2 on public ARC-AGI-1, at
22%, 11% and 0% cost reduction relative to HIGH; the headline is 29.5% at a
computed $0.00070 per task for 150M parameters, claimed to break the previously
reported cost–accuracy Pareto frontier.

That is a **benchmark reported by the developers**, not a deployment. The
report describes a black-box audit reproducing 29.5%, but by **co-authors**
from Bielik and New York University — stronger than a self-report, weaker than
an independent reproduction. Cost comparisons set measured hardware time for
BDH-CQ against leaderboard costs for others that may be API prices, so they are
not like-for-like. Dimensions and update rules are proprietary, so the
mechanism is not externally auditable.

The most important open limitation is general: recurrent depth is reported to
degrade past some depth in looped language models (Yang et al.,
arXiv:2605.26733), the same erosion this lab measures at toy scale. How far
latent depth scales before that ceiling binds is unresolved.

## Scope

Both models here are independent reimplementations written for teaching;
neither is BDH or BDH-CQ, and no checkpoint of either was run. Every BDH and
BDH-CQ figure is quoted and labelled *paper-reported*, never shown beside a
measurement of ours. The learned model has ~11,000 parameters on one synthetic
task, and graph reachability demonstrates computational depth, not semantic
reasoning.

**Continue with:** the Dragon Hatchling paper; the BDH-CQ report; Geiping et
al. (arXiv:2502.05171) on recurrent depth in a language model.
