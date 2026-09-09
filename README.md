# Latent Loop Lab

**Can a model think longer without saying more?**

An interactive lab for **recurrent latent computation as an inference-time
compute axis**, built for the DataForge 2026 Pathway Track.

Everything computes in the browser. There is no server, no install, and no
sign-in: open the URL and the mechanism is already running.

- **Artifact:** `PUBLIC_ARTIFACT_URL` — pending deployment (see §7)
- **Submission package:** [`submission/`](submission/) — start at
  [`submission/SUBMISSION.md`](submission/SUBMISSION.md)
- **One-page concept summary:** [`submission/docs/concept-summary.pdf`](submission/docs/concept-summary.pdf)
- **Blog post:** [`submission/docs/blog-post.pdf`](submission/docs/blog-post.pdf) — topic 19,
  demonstration coverage as a predictor of extrapolation success
- **Claim sheet:** [`submission/evidence/claim-sheet.md`](submission/evidence/claim-sheet.md)
- **Citation ledger:** [`submission/evidence/citation-ledger.md`](submission/evidence/citation-ledger.md)
- **AI assistance disclosure:** [`submission/AI-DISCLOSURE.md`](submission/AI-DISCLOSURE.md)

The lab has two computational layers that are never conflated: an exact,
interpretable graph recurrence that acts as a scientific control, and a small
learned shared-weight recurrent network that tests whether the same idea
survives when the transition rule is learned from data rather than designed by
hand.

---

## 1. The claim

> Repeated application of a **shared** state-update rule increases effective
> computational depth without generating intermediate language tokens. In the
> exact graph mechanism, `R` updates transmit reachability information through
> at most `R` edges. In the learned extension, increasing the inference
> iteration budget `R` can improve extrapolation to longer reasoning paths, but
> only within the learned dynamics' capacity and stability limits.

A learner can reproduce, test, or challenge this by moving `R`, choosing path
lengths outside the training range, and comparing the learned model against
both the exact mechanism and an independent breadth-first search.

What is deliberately **not** claimed is in the claim sheet.

---

## 2. Headline result

Five seeds, frozen checkpoints, identical inputs. Only `R` changes.

| R | learned mean | 95% CI | distances ≤4 | distances ≥5 | exact | mean ‖z‖ |
|---|---|---|---|---|---|---|
| 1 | 0.552 | [0.546, 0.558] | 0.621 | 0.506 | 0.550 | 3.36 |
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
  so 0.5 is the correct score. Read it as an aggregate, not as a per-case
  behaviour: the model is not reporting uncertainty below the threshold, it is
  answering confidently and being right about half the time. The sweep below
  shows one such case, at 0.0000 rather than near 0.5. Past `R = 4` the column climbs to 0.988 — the
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
  1      0.5038     reachable   unreachable       yes
  4      0.0000   unreachable   unreachable        NO
  5      0.0000   unreachable   unreachable        NO
  6      0.9995     reachable     reachable       yes
 10      1.0000     reachable     reachable       yes
```

Same weights, same graph. The answer flips at exactly the depth the path
requires.

---

## 3. Intended learner

A student or engineer who knows basic machine learning and graphs, but has not
read the latent-reasoning or post-Transformer literature.

**Prerequisites:** basic Python (only to reproduce; the artifact needs none),
what a directed graph is, roughly what a recurrent update is.

**After using the artifact a learner can:**

1. Say what recurrent depth `R` means mechanically, in one sentence.
2. Predict the minimum `R` at which a target activates, before moving the control.
3. Tell apart generating tokens and updating a latent state.
4. Read "same weights, more computation" off a shared-weight recurrent model.
5. Tell apart insufficient computation and a limit of what was learned.
6. Keep BDH-CQ's contextual state `S_t` distinct from its query-time workspace `H_r`.
7. Name at least one limitation or failure case.

Section 8 of the artifact asks the learner to do exactly these seven and marks
their own answers against the lab's.

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
average a slightly lower target in-degree (1.55 against 1.82) and source
out-degree (2.10 against 2.31) in the shipped case bank. Neither is visible from
the target before the recurrence reaches the source. Section 7 of the artifact
computes these from the bundle actually being served, so the figure on screen
cannot drift away from the data behind it.

---

## 5. Architecture

```
Browser (React, 9 guided sections)
  |
  |-- engine/graph.js    BFS oracle                          [LIVE, in browser]
  |-- engine/exact.js    noisy-OR graph recurrence           [LIVE, in browser]
  |-- engine/gnn.js      learned shared-weight recurrent GNN [LIVE, in browser]
  |-- engine/assoc.js    two-recurrence associative toy      [LIVE, in browser]
  |
  `-- public/data/       built by Backend/export_web.py
        model.json         frozen checkpoint weights
        cases.json         seeded graph instances            [SYNTHETIC DATA]
        experiment.json    multi-seed depth sweep            [PRECOMPUTED]
        examples.json      categorised successes/failures    [PRECOMPUTED]
        manifest.json      hashes, build revision, sizes

Backend (Python) — the reference implementation and the training pipeline.
Not required to view the artifact.
  core/       exact recurrence, BFS, generator
  learned/    model, dataset, training, evaluation, export
  tests/      invariant suite, generator guarantees, browser-vs-PyTorch parity
  server.py   optional FastAPI mirror of the same computation
```

**Why the compute moved into the browser.** The lab used to require a local
FastAPI process. That is fine on a laptop and useless as a public artifact: a
visitor saw a red banner instead of the mechanism, and there was a network round
trip between the depth control and the picture — the one interaction the whole
lesson rests on. Both computational layers now run client-side from the same
frozen weights, and `Backend/tests/test_js_parity.py` checks the JavaScript
against PyTorch on seeded cases, agreeing to about **2 × 10⁻⁷ relative**.

Graph instances are the one thing shipped as data rather than recomputed.
Reproducing `core/generator.py` in JavaScript would mean maintaining two copies
of 500 lines of subtle rejection sampling, and any drift between them would
silently change what the experiment measures. They are labelled *Synthetic data*
in the interface, which is what they always were.

Three objects, never conflated:

| Object | What it does | Evidence label |
|---|---|---|
| Exact recurrence `h^(r)` | Hand-designed propagation, one coordinate per node | Live computation |
| Learned GNN `z^(r)` | Learned shared-weight updates, opaque coordinates | Live computation |
| BDH-CQ workspace `H_r` | Published system-level example | Paper-reported result |

### Exact mechanism

```
h(0) = e_source
h(r+1)[v] = max( h(r)[v],  1 - ∏_{u:(u,v)∈E} (1 - α·h(r)[u]) )
ŷ_R = [ h(R)[q] > ε ]
```

Invariant, verified on 10,000 stratified seeded cases:
`h(R)[q] > ε  ⟺  d(s,q) ≤ R`.

The equivalence is exact in real arithmetic for every `α` in `(0, 1]`. The `ε`
threshold turns it into a decision, and that decision is only well posed while
the smallest activation the recurrence can produce stays above `ε`. A reached
node sits at distance at most `R`, so `α^R` is the lower bound and `α^R > ε` is
the condition. At the default `α = 1` it holds at every `R`; at `α = 1e-4` and
`R = 4` it does not, and the target would underflow to a false negative. Both
engines reject such inputs and name the condition rather than returning a
quietly wrong answer.

### Learned model

```
z_v^(0)   = E_θ(x_v)                              x_v = [is_source, is_target]
m_v^(r)   = AGG_{u∈N⁻(v)} M_θ(z_u^r, z_v^r)       AGG defaults to max
z_v^(r+1) = F_θ(z_v^r, m_v^r)                     gated (GRU) update
ŷ_R       = G_θ(z_q^(R))
```

with `θ_0 = θ_1 = … = θ_{R-1}`. 11,169 parameters at any depth. Max aggregation
is the default because reachability is a logical OR over incoming neighbours,
the same role the noisy-OR plays in the exact layer. That is an argument, not a
measurement, so `sum` and `mean` are trained too — and on the two seeds
available the intervals overlap, so **the ablation does not separate them.** The
artifact says so rather than reading the small gap as a result.

`R` is resampled uniformly from `{1..4}` every batch during training, so the same
weights must work at every depth and the model cannot specialise to one fixed
iteration count.

---

## 6. Repository layout

```
Backend/
├── requirements.txt
├── reproduce.py                # one-command full pipeline
├── export_web.py               # build the browser bundle
├── server.py                   # optional FastAPI mirror: /exact/* routes
├── learned_api.py              # optional: /learned/* routes
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
├── tests/
│   ├── test_exact_invariant.py # 10,000-case invariant + generator guarantees
│   ├── test_js_parity.py       # browser engine vs PyTorch
│   └── js_parity_runner.mjs    # runs the browser engine under Node
└── results/                    # checkpoints + JSON outputs

Frontend/
├── vite.config.js              # BASE_PATH for subpath deployments
├── scripts/smoke.mjs           # renders every page + exercises the engine
└── src/
    ├── App.jsx                 # 9-section lesson shell
    ├── App.css                 # design tokens, light and dark
    ├── engine/                 # graph.js, exact.js, gnn.js, assoc.js
    ├── lib/lab.js              # data loading + compute layer
    ├── components/             # GraphView, Chart, TwoRecurrences, Common
    └── pages/                  # Start, Hook, Mechanism, Verification,
                                # LearnedBridge, Generalisation, BDHCQ,
                                # Evidence, Recap

submission/                     # every judge-facing document, single copy
├── SUBMISSION.md               # package index and the judge's five-minute path
├── AI-DISCLOSURE.md            # AI assistance, ownership, source verification
├── SOURCES-AND-LICENSES.md     # code, data, weights, fonts, libraries
├── LICENSE                     # MIT
├── docs/
│   ├── concept-summary.md      # source text for the one-page summary
│   ├── concept-summary.pdf     # the submitted PDF, one page, 918 words
│   ├── blog-post.md            # source text for the blog post
│   ├── blog-post.pdf           # the submitted PDF, two pages, 802 words
│   └── build-summary-pdf.py    # md -> print-ready HTML -> PDF, both documents
└── evidence/
    ├── claim-sheet.md          # the claim, how to falsify it, what is not claimed
    ├── citation-ledger.md      # every external claim mapped to a primary source
    └── source-matrix.csv       # per-page claim / source / evidence label
```

Both PDFs are generated from their Markdown, never edited directly:

```bash
pip install weasyprint                                      # or use headless Chrome
python submission/docs/build-summary-pdf.py --doc all --pdf # rebuilds both
```

---

## 7. Setup

### Just view the artifact

```bash
./run.sh
```

Starts the lab and the reference API and opens a browser.
See [`RUNNING.md`](RUNNING.md) for the options.

Or by hand, with no backend at all:

```bash
cd Frontend
npm install
npm run dev            # http://127.0.0.1:5173
```

No Python, no backend, no checkpoint download. The data bundle is committed
under `Frontend/public/data`.

### Build for deployment

```bash
cd Frontend
npm run build                        # served from the domain root
BASE_PATH=/your-repo/ npm run build  # served from a subpath, e.g. GitHub Pages
```

`dist/` is a plain static directory. Any static host serves it: GitHub Pages,
Netlify, Vercel, Cloudflare Pages, S3. Nothing needs to run server-side.

`.github/workflows/deploy.yml` does this on every push to `main`: it installs,
runs the smoke suite, builds with `BASE_PATH` set to the repository name, and
publishes to GitHub Pages. Enable it once under Settings → Pages → Source →
GitHub Actions. When the deployment is live, replace `PUBLIC_ARTIFACT_URL` on
line 11 of this file and in row 1 of
[`submission/SUBMISSION.md`](submission/SUBMISSION.md). Those are the only two
places the URL appears.

### Reproduce every number

```bash
cd Backend
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt

python reproduce.py              # tests, 5 seeds, ablations, sweeps, export  (~4 min)
python reproduce.py --quick      # smaller sweep, smoke check
python reproduce.py --tests-only # validation suites only
```

`reproduce.py` finishes by rebuilding `Frontend/public/data` and running the
browser-versus-PyTorch parity check, so a trained result and the artifact cannot
fall out of step.

### Optional: the FastAPI reference server

```bash
cd Backend && uvicorn server:app --reload   # http://127.0.0.1:8000, docs at /docs
```

The frontend no longer uses it. It stays because it is the reference
implementation the browser engine is checked against, and because `/docs` is a
convenient way to poke at the mechanism from a terminal.

---

## 8. The nine sections

| # | Section | Question | Learner action |
|---|---|---|---|
| 0 | Start here | What is this and who is it for? | Watch the wavefront advance before touching anything |
| 1 | The dial | Can a model think longer without saying more? | Move `R` until the target activates |
| 2 | The state | What is the latent state doing? | Predict the minimum `R` before the marker appears |
| 3 | The check | Is the effect real or just animation? | Try to break the invariant, with two one-click setups |
| 4 | Learned rule | What if the rule is learned? | Push inference depth past training depth |
| 5 | The limits | More computation or more memorisation? | Choose a path length outside the training range |
| 6 | BDH-CQ | Where does this appear in a real system? | Drive two recurrences separately and break one |
| 7 | Evidence | What counts as evidence here? | Audit the provenance of every number |
| 8 | Explain it back | Can you say it in your own words? | Answer seven prompts, then check and self-mark |

Sections 0 to 3 need no trained checkpoint. Sections 4 and 5 degrade with an
explicit message rather than inventing numbers.

---

## 9. How to test each feature

### Backend

| Test | Command | Expected |
|---|---|---|
| Unit tests | `python -m tests.test_exact_invariant` | `OK` (10 tests) |
| 10,000-case invariant | `python -m tests.test_exact_invariant --large` | `OK: 10000 cases passed.` + per-stratum table |
| Generator guarantees | `python -m tests.test_exact_invariant --generator` | every case has the distance it claims |
| Browser vs PyTorch | `python -m tests.test_js_parity --verbose` | `OK` (5 tests) + per-field deviations under 1e-6 |
| R too small | `python -m experiments.run_exact --preset line --R 3` | `estimate: false`, `bfsDistance: 4`, `invariantPass: true` |
| R sufficient | `python -m experiments.run_exact --preset line --R 4` | `estimate: true` |
| Bundle export | `python export_web.py` | 132 matched pairs, 0 dropped, checkpoint hash printed |

### Frontend

| Test | Command | Expected |
|---|---|---|
| Every page renders, engine agrees with BFS | `npm run smoke` | `All checks passed.` |
| Production build | `npm run build` | `dist/` with `data/` copied in |

### By hand

- **Section 1** opens with the mechanism already running and stops at `R = 4`.
  Grab the control to take over.
- **Section 2** hides the distance marker until you commit to a prediction.
  Predict 2 on the branching graph and it tells you the answer is 3.
- **Section 3** click "No path exists" and push `R` to 12. The target must never
  activate and the invariant must hold at all 13 depths.
- **Section 4** set path length 6 and sweep `R`. Both models should commit at 6,
  and the badge should read "same depth". Try 9 or 10 and watch it fail.
- **Section 5** every plot carries `Precomputed result`. The negative findings
  are shown, not hidden — including an ablation that came out inconclusive.
- **Section 6** cut the demonstrations to 3 and raise the latent budget. The
  answer does not improve, because the missing association is not a compute
  problem.
- **Section 7** the matched-statistics table is computed from the bundle you are
  actually browsing.

### The sixty-second test

Open the artifact, see the preset already running, move `R`, watch the target
activate at `R = 4`, read the truth panel beside it. If that fails, fix it before
anything else.

---

## 10. Evidence discipline

| Element | Label |
|---|---|
| Exact graph recurrence, BFS oracle, learned forward pass, section 6 toy | Live computation |
| Graph instances, 10,000-case invariant suite, parity suite | Synthetic data |
| Learned depth curves, aggregation and sharing ablations | Precomputed result |
| BDH-CQ effort scores, ARC-AGI-1 numbers, architecture claims | Paper-reported result |

No paper-reported number appears on the same visual footing as a live
measurement. Mapping of every claim to its primary source:
[`submission/evidence/citation-ledger.md`](submission/evidence/citation-ledger.md)
and
[`submission/evidence/source-matrix.csv`](submission/evidence/source-matrix.csv).

Every BDH and BDH-CQ figure quoted here was re-verified against the arXiv
listings on 2026-09-09, section by section. Every quoted figure matches its
primary source exactly. One attribution was narrowed in the process: reference
6 studies the capacity of a fixed-size recurrent memory, not instability with
depth, and the ledger now says so. The results are tabulated in
[`submission/AI-DISCLOSURE.md`](submission/AI-DISCLOSURE.md) §6.

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
| Aggregation ablation | Inconclusive at two seeds, and reported as inconclusive |
| Section 6 toy, demonstrations withheld | A context failure that no latent budget can repair |

---

## 12. Caps and approximations, stated rather than hidden

- Exact engine: `n ≤ 24`, `R ≤ 12`, `ε = 1e-12`, and `α^R > ε` (see §5).
- Learned experiment: distances 1–10, `R ≤ 10`, hidden dim 32, five seeds.
- Training saw distances 1–4 only.
- Negatives match positives on node and edge count exactly; target in-degree
  still differs slightly (1.55 vs 1.82 in the shipped bank).
- Confidence intervals use a Student *t* critical value, appropriate for `n = 5`.
  The aggregation ablation has `n = 2`, and its intervals are correspondingly
  wide.
- The browser runs the checkpoint in float64 and the Python pipeline in float32.
  They agree to about 2 × 10⁻⁷ relative — float32 noise, three orders of
  magnitude finer than anything displayed.
- Exact-layer coordinates are interpretable **by design**. Production latent
  states are not generally human-readable.
- No wall-clock comparison is made against any real system. None was measured.

---

## 13. Scope boundary

The graph recurrence is an educational mechanism-level model. The learned GNN is
a small experimental bridge, an **independent reimplementation** written for this
lab. The associative toy in section 6 is a hand-built illustration with no
trained parameter. None of them is BDH or BDH-CQ. The BDH-CQ equations shown
describe the published conceptual decomposition; that report states its exact
dimensions and update rules are proprietary, so they are outside this project
and outside anyone else's. No BDH or BDH-CQ checkpoint was run.

---

## 14. References

1. Hao et al. *Training Large Language Models to Reason in a Continuous Latent Space.* arXiv:2412.06769, 2024.
2. Geiping et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* arXiv:2502.05171, 2025.
3. Saunshi et al. *Reasoning with Latent Thoughts: On the Power of Looped Transformers.* ICLR 2025.
4. Zhu et al. *Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought.* arXiv:2505.12514, 2025.
5. Kosowski et al. *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* arXiv:2509.26507, 2025.
6. Ben-Kish et al. *Overflow Prevention Enhances Long-Context Recurrent LLMs.* COLM 2025; arXiv:2505.07793.
7. Yang et al. *Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models.* arXiv:2605.26733, 2026.
8. Engdahl et al. *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.* arXiv:2608.09888, 2026.

---

## 15. Credits, licenses, provenance

- **Code:** written by the Latent Loop Lab team for DataForge 2026. Licensed
  under the MIT License; see [`LICENSE`](LICENSE).
  _Team members: Aryan Sisodiya, Abhishek Shahi, Farhan Alam, Ariyan Bhakat._
- **Third-party libraries:** React and React DOM (MIT), Vite and
  `@vitejs/plugin-react` (MIT), `vite-node` (MIT), FastAPI (MIT), Uvicorn (BSD
  3-Clause), Pydantic (MIT), PyTorch (BSD 3-Clause), NumPy (BSD 3-Clause),
  WeasyPrint (BSD 3-Clause, used only to build the summary PDF).
- **Fonts:** Space Grotesk (SIL Open Font License 1.1), Source Serif 4 (SIL OFL
  1.1), IBM Plex Mono (SIL OFL 1.1), served from Google Fonts. Every face has a
  system fallback, so the artifact renders correctly if the font host is
  unreachable.
- **Graphics:** all inline SVG, generated by this project. No external image,
  icon set, or illustration is used.
- **Data:** no external dataset. Every graph is synthetic and seeded, generated
  by `Backend/core/generator.py`.
- **Model weights:** trained from scratch by this project on the synthetic task
  above. No pretrained weights of any kind are used, downloaded, or fine-tuned.
- **Quoted results:** all BDH and BDH-CQ figures are quoted from references 5
  and 8 above under fair-use citation, labelled *Paper-reported result*, and
  never presented as measurements of this project.

---

## 16. AI assistance disclosure

The track rules require this in the README. The full record — tools, per
component origin, source verification, mentorship, and the team's signed
statement — is
[`submission/AI-DISCLOSURE.md`](submission/AI-DISCLOSURE.md). The summary:

**AI assistance was used substantially and throughout**, in four areas: writing
code, debugging it, refining the architecture, and implementing the design.
Prose was drafted the same way — the team decided what each document had to say,
and the AI wrote it down. One tool was used, Claude Code running Claude Opus 5,
between 2026-09-04 and 2026-09-09.

**The team decided what this project is.** Reading the problem statement, the
choice of concept, the one falsifiable sentence in §1, the design of both
computational layers, the reachability task, the experimental design in §5 and
the nine-section lesson structure are the team's work. The code was written to
serve those decisions, and every component was read and accepted before it
shipped.

The blog post and the concept summary were planned and reviewed by the team;
the AI wrote the planned content and the explanations down. Nothing here is
generated content the team has not read.

**No AI produced any number, weight, dataset or graphic.** Every measurement is
output of `python reproduce.py` over frozen checkpoints. Every checkpoint was
trained from scratch on this project's synthetic task. Every graph instance is
seeded output of `Backend/core/generator.py`. Every graphic is inline SVG from
this project's own code. Every external figure is quoted from a cited paper and
labelled *Paper-reported result*.

### Forks and reuse

This repository is not a fork and contains no code copied from another project.
Assets and licences are recorded in
[`submission/SOURCES-AND-LICENSES.md`](submission/SOURCES-AND-LICENSES.md).

Every number in this repository is reproducible with `python reproduce.py`, and
every external claim is mapped to a primary source in the citation ledger. Those
two facts make the disclosure checkable rather than a formality.

---

## 17. Troubleshooting

| Symptom | Fix |
|---|---|
| Artifact shows "The data bundle did not load" | Run `python export_web.py` in `Backend`, then rebuild the frontend |
| Sections 4–5 say no checkpoint | `python reproduce.py`, then `python export_web.py` |
| Blank page after deploying to a subpath | Rebuild with `BASE_PATH=/your-repo/ npm run build` |
| `ModuleNotFoundError: core` | Run Python commands from the `Backend/` root |
| `test_js_parity` skips | Install Node, or run `python export_web.py` first |
| `uvicorn: command not found` | Activate the venv, `pip install -r requirements.txt` |
| Fonts look wrong offline | Expected; the system fallback stack takes over |
