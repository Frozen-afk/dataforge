# Claim sheet

The single falsifiable sentence this artifact teaches, the reasons to believe
it, and the ways it could be shown wrong.

## The claim

> Repeated application of a **shared** state-update rule increases effective
> computational depth without generating intermediate language tokens. In the
> exact graph mechanism, `R` updates transmit reachability information through
> at most `R` edges. In the learned extension, increasing the inference
> iteration budget `R` can improve extrapolation to longer reasoning paths, but
> only within the learned dynamics' capacity and stability limits.

The second sentence is deliberately weaker than "more loops always improve
reasoning". It states a measurable research question rather than a marketing
line, and pages 5 and 7 show where it stops holding.

## How a learner can falsify it

| Test | Where | What would refute the claim |
|---|---|---|
| Target activates before `R = d(s,q)` | Page 1–3 | Any case where `h(R)[q] > ε` while `d(s,q) > R` |
| Recurrence disagrees with BFS | Page 3 | Any invariant `FAIL` in the 10,000-case suite |
| Depth does not help the learned model | Page 5 | A flat accuracy-versus-`R` curve |
| Depth helps without extra computation | Page 5 | Accuracy rising while the state stops changing |
| Unshared weights work just as well | Page 5 | The ablation running past its trained depth |

## Sub-claims and their evidence

| # | Sub-claim | Evidence | Label |
|---|---|---|---|
| 1 | The exact recurrence activates the target exactly when `d(s,q) ≤ R` | 10,000 stratified seeded cases, checked against an independent BFS oracle | Synthetic data |
| 2 | The update rule does not change with `r` | `core/recurrent.py`: one function, no step index | Live computation |
| 3 | The learned model shares parameters across steps | `learned/model.py`: one `RecurrentBlock`; 11,169 parameters at any depth | Live computation |
| 4 | Depth increases learned accuracy | Mean accuracy 0.552 at `R=1` rising to 0.991 at `R=10`, five seeds | Precomputed result |
| 5 | The learned model extrapolates past its training range | Distances 5–10, never trained on, go from chance to 0.988 as `R` grows | Precomputed result |
| 6 | Depth is not unlimited | Accuracy on distances ≤ 4 peaks at `R≈5` and erodes slightly by `R=10`; state norm saturates | Precomputed result |
| 7 | Shared weights are what make depth adjustable | Ablation: 39,585 parameters, hard ceiling at depth 4 | Precomputed result |
| 8 | BDH-CQ contains a query-time latent recurrence | Published equations `H_{r+1} = F_θ(H_r, S_K)` | Paper-reported result |
| 9 | BDH-CQ's contextual recurrence is a different process | Published equations `S_t = U_θ(S_{t-1}, D_t)` | Paper-reported result |

## What is deliberately not claimed

- That graph reachability proves semantic reasoning.
- That more recurrence always improves reasoning.
- That this toy is BDH, BDH-CQ, or predictive of their behaviour.
- That any architecture is universally superior.
- Any wall-clock speed comparison. None was measured on matched hardware.

## Terminology, frozen

| Term | Meaning here | Never used to mean |
|---|---|---|
| `R` | Number of applications of the update rule at inference | Number of layers, or number of output tokens |
| Recurrent depth | Same as `R` | Network depth in the parameter-count sense |
| `h^(r)` | Exact mechanism state, one coordinate per node | Anything learned |
| `z^(r)` | Learned model latent state, no per-coordinate meaning | Anything interpretable |
| `S_t` | BDH-CQ contextual state, advances per demonstration | Query-time reasoning |
| `H_r` | BDH-CQ query-time workspace, advances per latent step | Context acquisition |
| Reasoning step | Only used for query-time latent computation | Absorbing a demonstration |
