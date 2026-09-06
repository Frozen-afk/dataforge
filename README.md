# Latent Loop Lab

**Can a model think longer without saying more?**

An interactive educational laboratory for understanding **recurrent latent
computation as an inference-time compute axis**, built for the DataForge 2026
Pathway Track.

The lab has two computational layers that are never conflated: an exact,
interpretable graph recurrence that acts as a scientific control, and a small
learned shared-weight recurrent GNN that tests whether the same idea survives
when the transition rule is learned from data rather than designed by hand.

---

## 1. The claim

> Repeated application of a **shared** state-update rule increases effective
> computational depth without generating intermediate language tokens. In the
> exact graph mechanism, `R` updates transmit reachability information through
> at most `R` edges. In the learned extension, increasing the inference
> iteration budget `R` can improve extrapolation to longer reasoning paths, but
> only within the learned dynamics' capacity and stability limits.

The learner can reproduce, test, or challenge this by moving `R`, choosing path
lengths outside the training range, and comparing the learned model against both
the exact mechanism and an independent BFS oracle.

Full breakdown, including what is deliberately *not* claimed:
[`Backend/docs/claim-sheet.md`](Backend/docs/claim-sheet.md).

---

## 2. Headline result

Five seeds, frozen checkpoints, identical inputs. Only `R` changes.

| R | learned mean | 95% CI | distances ≤4 | distances ≥5 | exact | mean ‖z‖ |
|---|---|---|---|---|---|---|
| 1 | 0.552 | [0.546, 0.558] | 0.621 | 0.506 | 0.550 | 3.37 |
| 2 | 0.600 | [0.600, 0.600] | 0.750 | 0.500 | 0.600 | 3.89 |
| 3 | 0.650 | [0.650, 0.650] | 0.875 | 0.500 | 0.650 | 4.28 |
| 4 | 0.700 | [0.700, 0.700] | 1.000 | 0.500 | 0.700 | 4.79 |
| 5 | 0.750 | [0.750, 0.750] | 1.000 | 0.583 | 0.750 | 4.95 |
| 6 | 0.800 | [0.800, 0.800] | 1.000 | 0.667 | 0.800 | 5.02 |
| 7 | 0.850 | [0.850, 0.850] | 1.000 | 0.750 | 0.850 | 5.06 |
| 8 | 0.900 | [0.898, 0.901] | 0.999 | 0.833 | 0.900 | 5.09 |
| 9 | 0.948 | [0.944, 0.951] | 0.994 | 0.916 | 0.950 | 5.10 |
| 10 | 0.991 | [0.972, 1.000] | 0.994 | 0.988 | 1.000 | 5.11 |

Training saw distances 1 to 4 only. Read the table three ways:

- **Distances ≥ 5 sit at chance until `R` exceeds 4.** That is not the model
  failing. At those depths the answer is not yet knowable from the target node,
  so 0.5 is the correct score. Past `R = 4` the column climbs to 0.988 — the
  model solves paths longer than anything it trained on, purely by applying the
  same weights more times.
- **The learned column tracks the exact column** to within 0.01 almost
  everywhere. The learned update reproduced the mechanism's timing.
- **Depth is not free.** Distances ≤ 4 peak at 1.000 around `R = 5` and erode to
  0.994 by `R = 10`, while the state norm saturates near 5.1. Extra iterations
  eventually add drift rather than information.

A single-graph version of the same effect, at a distance never trained on:

```
$ python -m learned.infer --checkpoint results/model_seed1.pt --distance 6 --nodes 16 --sweep

  R   learned p      predicts         exact   correct
  1      0.5042     reachable   unreachable       yes
  4      0.0011   unreachable   unreachable        NO
  5      0.0001   unreachable   unreachable        NO
  6      1.0000     reachable     reachable       yes
 10      1.0000     reachable     reachable       yes
```

Same weights, same graph. The answer flips at exactly the depth the path
requires.

---

## 3. Intended learner

A student or engineer who knows basic ML and graphs, but has not read the
latent-reasoning or post-Transformer literature. Prerequisites: basic Python,
what a directed graph is, roughly what a recurrent update is.

After using the artifact a learner can:

1. Explain what recurrent depth `R` means mechanically.
2. Predict the minimum `R` at which a target activates, before moving the slider.
3. Distinguish token generation from latent state updates.
4. Interpret "same weights, more computation" in a shared-weight recurrent GNN.
5. Distinguish insufficient computation from learned-generalisation limits.
6. Distinguish BDH-CQ contextual recurrence `S_t` from query-time recurrence `H_r`.
7. Name at least one limitation or failure case.

---

## 4. The task, and why it is built the way it is

Every instance is a directed graph with a marked source and target. The label is
whether the target is reachable. Positives are generated at an exact distance
`d`; negatives are **hard**.

This matters more than it sounds. An earlier version of this project built
positives as a forward chain `0 → 1 → … → L` and negatives as the same chain
reversed. In a negative the target then had **zero incoming edges**, so
"is the target reachable?" collapsed to "does the target have any incoming
edge?" — a one-hop question. Measured accuracy was 1.000 at every `R` and every
path length, and the inference-depth experiment measured nothing.

The current generator removes that shortcut in two stages:

1. **Matched local statistics.** The target's in-degree is drawn from the same
   distribution for both classes, and edges into the target are assigned only by
   that controlled step.
2. **Matched global statistics.** Each negative is *derived* from a specific
   positive: cut the edges leaving a randomly chosen BFS layer, then add exactly
   the same number of edges back elsewhere. Node count and edge count match
   **exactly**.

Stage 2 was added after stage 1 alone proved insufficient — the model learned to
read a residual edge-density gap and scored *above the exact mechanism* at depths
where the answer was not yet knowable. With counts matched, the only remaining
signal is global connectivity, which the target cannot observe until `R` reaches
the source. That is exactly the property the experiment is meant to measure.

Two smaller asymmetries remain and are reported rather than hidden: negatives
average a slightly lower target in-degree (1.63 against 1.78) and source
out-degree (1.84 against 1.99). Neither is visible from the target before the
recurrence reaches the source.

---

## 5. Architecture

```
Learner interaction (React, 7 guided pages)
        |
Graph + exact noisy-OR recurrence  ->  BFS oracle      [LIVE COMPUTATION]
        |
Shared-weight learned recurrent GNN                    [LIVE or PRECOMPUTED]
        |
Inference-depth experiment + shared/unshared ablation  [PRECOMPUTED RESULT]
        |
BDH-CQ architectural connection                        [PAPER-REPORTED RESULT]
        |
Limitations + evidence discipline
```

Three objects, never conflated:

| Object | What it does | Evidence |
|---|---|---|
| Exact recurrence `h^(r)` | Hand-designed propagation, one coordinate per node | Live computation |
| Learned GNN `z^(r)` | Learned shared-weight updates, opaque coordinates | Live / precomputed |
| BDH-CQ workspace `H_r` | Published system-level example | Paper-reported result |

### Exact mechanism

```
h(0) = e_source
h(r+1)[v] = max( h(r)[v],  1 - ∏_{u:(u,v)∈E} (1 - α·h(r)[u]) )
ŷ_R = [ h(R)[q] > ε ]
```

Invariant, verified on 10,000 stratified seeded cases:
`h(R)[q] > ε  ⟺  d(s,q) ≤ R`.

### Learned model

```
z_v^(0)   = E_θ(x_v)                              x_v = [is_source, is_target]
m_v^(r)   = AGG_{u∈N⁻(v)} M_θ(z_u^r, z_v^r)       AGG defaults to max
z_v^(r+1) = F_θ(z_v^r, m_v^r)                     gated (GRU) update
ŷ_R       = G_θ(z_q^(R))
```

with `θ_0 = θ_1 = … = θ_{R-1}`. 11,169 parameters at any depth. Max aggregation
is the default because reachability is a logical OR over incoming neighbours,
the same role the noisy-OR plays in the exact layer.

`R` is resampled uniformly from `{1..4}` every batch during training, so the same
weights must work at every depth and the model cannot specialise to one fixed
iteration count.

---

## 6. Repository layout

```
Backend/
├── requirements.txt
├── reproduce.py                # one-command full pipeline
├── server.py                   # FastAPI: /exact/* routes
├── learned_api.py              # /learned/* routes
├── core/
│   ├── graph.py                # graph schema + presets (n ≤ 24)
│   ├── bfs.py                  # independent BFS oracle
│   ├── recurrent.py            # exact noisy-OR recurrence (R ≤ 12)
│   └── generator.py            # distance-stratified graphs + hard negatives
├── learned/
│   ├── model.py                # SharedRecurrentGNN + UnsharedRecurrentGNN
│   ├── dataset.py              # balanced datasets, block-diagonal batching
│   ├── train.py                # training + frozen checkpoint + hash
│   ├── evaluate.py             # depth sweep + exact reference + examples
│   ├── export_results.py       # multi-seed aggregation with CIs
│   └── infer.py                # deterministic inference script
├── experiments/generate.py     # stratified case suite
├── tests/test_exact_invariant.py
├── docs/                       # claim-sheet, citation-ledger, source-matrix
└── results/                    # checkpoints + JSON outputs

Frontend/
└── src/
    ├── App.jsx                 # 7-page lesson shell
    ├── lib/api.js
    ├── components/             # GraphView, Chart, Evidence, Common
    └── pages/                  # Hook, Mechanism, Verification,
                                # LearnedBridge, Generalisation, BDHCQ, Evidence
```

---

## 7. Setup

### Backend

```bash
cd Backend
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
uvicorn server:app --reload
```

Runs at `http://127.0.0.1:8000`; interactive docs at `/docs`.

### Frontend

```bash
cd Frontend
npm install
npm run dev
```

Runs at `http://127.0.0.1:5173`. Point it elsewhere with `VITE_API_BASE`.

### Reproduce every number

```bash
cd Backend
python reproduce.py              # tests, 5 seeds, ablation, sweeps, export  (~2 min)
python reproduce.py --quick      # smaller sweep, smoke check
python reproduce.py --tests-only # validation suites only
```

Everything is seeded. The same machine reproduces the same checkpoint hashes.

---

## 8. The seven pages

| Page | Question | Learner action |
|---|---|---|
| 1 Hook | Can a model think longer without saying more? | Move `R` until the target activates |
| 2 Mechanism | What is the latent state doing? | Predict the minimum `R` before moving the slider |
| 3 Verification | Is the effect real or just animation? | Try to break the invariant |
| 4 AI bridge | What if the rule is learned? | Push inference depth past training depth |
| 5 Generalisation | More computation or more memorisation? | Choose a path length outside the training range |
| 6 BDH-CQ | Where does this appear in real AI? | Compare the two recurrences |
| 7 Evidence | What counts as evidence here? | Audit the provenance of every number |

Pages 1 to 3 work with no trained checkpoint. Pages 4 and 5 degrade with an
explicit message rather than inventing numbers.

---

## 9. How to test each feature

### Backend

| Test | Command | Expected |
|---|---|---|
| Unit tests | `python -m tests.test_exact_invariant` | `OK` (10 tests) |
| 10,000-case invariant | `python -m tests.test_exact_invariant --large` | `OK: 10000 cases passed.` + per-stratum table |
| Generator guarantees | `python -m tests.test_exact_invariant --generator` | every case has the distance it claims |
| R too small | `python -m experiments.run_exact --preset line --R 3` | `estimate: false`, `bfsDistance: 4`, `invariantPass: true` |
| R sufficient | `python -m experiments.run_exact --preset line --R 4` | `estimate: true` |
| Health | `curl localhost:8000/health` | `{"status":"ok"}` |
| Generated graph | `POST /exact/generate {"distance":7,"nodes":16,"R":7}` | `estimate: true`, `bfsDistance: 7` |
| Learned status | `GET /learned/status` | checkpoint hash, live-inference flag |
| Depth sweep | `POST /learned/sweep {"distance":6,"nodes":16}` | `flipDepthsAgree: true` |
| Missing model honesty | same call with no checkpoint | 503 with an actionable message |

### Frontend

- **Page 1** opens with the mechanism already running and stops at `R = 4`. Grab
  the slider to take over.
- **Page 2** blocks the reveal until you commit to a prediction. Predict 2 on the
  branching graph and it tells you the answer is 3.
- **Page 3** switch to `disconnected` and push `R` to 12. The target must never
  activate and the invariant must stay `PASS`.
- **Page 4** set path length 6 and sweep `R`. Both models should flip at 6, and
  the badge should read "same depth".
- **Page 5** every plot carries `Precomputed result`. The negative findings
  (erosion at large `R`, chance-level performance below `R = d`) are shown, not
  hidden.
- **Page 6** the two recurrences are separately coloured and separately labelled.
- **Page 7** the ledger auto-detects whether this deployment runs live inference.

### The sixty-second test

Open the app, see the preset already running, move `R`, watch the target
activate at `R = 4`, read the truth panel. If that fails, fix it before anything
else.

---

## 10. Evidence discipline

| Element | Label |
|---|---|
| Exact graph recurrence, BFS oracle | Live computation |
| 10,000-case invariant suite | Synthetic data |
| Learned depth curves, ablation | Precomputed result |
| Single-graph learned sweep (page 4) | Live computation when a checkpoint is deployed |
| BDH-CQ effort scores, ARC-AGI numbers, literature claims | Paper-reported result |

No paper-reported number appears on the same visual footing as a live
measurement. Mapping of every claim to its primary source:
[`Backend/docs/citation-ledger.md`](Backend/docs/citation-ledger.md) and
[`Backend/docs/source-matrix.csv`](Backend/docs/source-matrix.csv).

---

## 11. Failures shown on purpose

| Failure | Lesson |
|---|---|
| `R < d(s,q)` | Insufficient computational depth, not a wrong answer |
| Disconnected graph | More iterations cannot create a missing path |
| Learned model, small `R` | Insufficient inference depth |
| Learned model, long unseen path | More compute does not guarantee extrapolation |
| Very large `R` | Recurrence has stability and capacity limits |
| Unshared weights past depth 4 | Not a worse result — not a possible one |

---

## 12. Caps and approximations, stated rather than hidden

- Exact engine: `n ≤ 24`, `R ≤ 12`, `ε = 1e-12`.
- Learned experiment: distances 1–10, `R ≤ 10`, hidden dim 32, five seeds.
- Training saw distances 1–4 only.
- Negatives match positives on node and edge count exactly; target in-degree
  still differs slightly (1.63 vs 1.78).
- Confidence intervals use a Student *t* critical value, appropriate for `n = 5`.
- Exact-layer coordinates are interpretable **by design**. Production latent
  states are not generally human-readable.
- No wall-clock comparison is made against any real system. None was measured.

---

## 13. Scope boundary

The graph recurrence is an educational mechanism-level model. The learned GNN is
a small experimental bridge, an **independent reimplementation** written for this
lab. Neither is BDH or BDH-CQ. The BDH-CQ equations shown describe the published
conceptual decomposition; exact production dimensions and update rules remain
outside this project. No BDH or BDH-CQ checkpoint was run.

---

## 14. References

1. Hao et al. *Training LLMs to Reason in a Continuous Latent Space.* arXiv:2412.06769, 2024.
2. Geiping et al. *Scaling up Test-Time Compute with Latent Reasoning.* arXiv:2502.05171, 2025.
3. Saunshi et al. *Reasoning with Latent Thoughts: On the Power of Looped Transformers.* ICLR 2025.
4. Zhu et al. *Reasoning by Superposition.* arXiv:2505.12514, 2025.
5. Kosowski et al. *The Dragon Hatchling.* arXiv:2509.26507, 2025.
6. Ben-Kish et al. *Overflow Prevention Enhances Long-Context Recurrent LLMs.* COLM 2025.
7. Wei et al. *Stabilizing Recurrent Dynamics for Test-Time Scalable Reasoning.* arXiv:2605.26733, 2026.
8. Engdahl et al. *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.* arXiv:2608.09888, 2026.

---

## 15. Credits, licenses, provenance

- Code: written by [team names], [license, e.g. MIT].
- Third-party libraries: FastAPI, PyTorch, React, Vite (their respective licenses).
- No external datasets. All graph data is synthetic and seeded.
- Graphics: inline SVG, system fonts only. No external assets.

## 16. AI assistance disclosure

[Describe exactly how AI tools were used. Every equation, experiment design,
result, and claim must be reviewed, run, and verified by the team, and every
component must be traceable and explainable by a team member.]

---

## 17. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: core` | Run commands from the `Backend/` root |
| `uvicorn: command not found` | Activate the venv, `pip install -r requirements.txt` |
| Port 8000 in use | `uvicorn server:app --port 8001` and set `VITE_API_BASE` |
| Frontend banner: backend not responding | Start the backend; check `/health` |
| Pages 4–5 empty | `python reproduce.py` to train and export |
| `/learned/run` returns 503 | Expected with no checkpoint. Train one, or use the precomputed sweep |
