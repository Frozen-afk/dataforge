# Latent Loop Lab — DataForge 2026 Pathway Track submission

**Team:** VibeCoders

**Members:** 
- Aryan Sisodiya
- Abhishek Shahi
- Farhan Alam
- Ariyan Bhakat

**Track topic:** Inference-Time Scaling (approved topic, "Reasoning and
generalisation" group), taught through recurrent latent computation.

---

## The one-sentence claim

> Repeated application of a **shared** state-update rule increases effective
> computational depth without generating intermediate language tokens. In the
> exact graph mechanism, `R` updates transmit reachability information through
> at most `R` edges. In the learned extension, increasing the inference
> iteration budget `R` can improve extrapolation to longer reasoning paths, but
> only within the learned dynamics' capacity and stability limits.

A learner falsifies or confirms it in under a minute by moving one control
(`R`) and reading the ground-truth panel printed beside the model's answer.

---

## Required package contents

| # | Requirement | Where it is | Status |
|---|---|---|---|
| 1 | Public artifact URL, no sign-in | _FILL IN after enabling GitHub Pages_ | **BLOCKED** |
| 2 | Public source repository | https://github.com/Frozen-afk/dataforge | **BLOCKED** — verify public, merge `dev` → `main` |
| 3 | Blog post as PDF | [`docs/blog-post.pdf`](docs/blog-post.pdf) | Done — 2 pages, 802 words |
| 4 | One-page concept summary PDF | [`docs/concept-summary.pdf`](docs/concept-summary.pdf) | Done — 1 page, 918 words |
| 5 | Complete README | [`../README.md`](../README.md) | Partial — §15 team names, §16 team's own AI rows |
| 6 | Setup instructions for local components | [`../RUNNING.md`](../RUNNING.md), README §7 | Done — verified on a clean checkout |
| 7 | ≥3 primary papers, 2022–2026, cited beside claims | [`evidence/citation-ledger.md`](evidence/citation-ledger.md), [`evidence/source-matrix.csv`](evidence/source-matrix.csv) | Done — 8 primary sources; **2 need re-verification** |
| 8 | Source and license record | [`SOURCES-AND-LICENSES.md`](SOURCES-AND-LICENSES.md) | Done |
| 9 | AI assistance, code, data, asset, license disclosure | [`AI-DISCLOSURE.md`](AI-DISCLOSURE.md) | **BLOCKED** — template only, team must complete |

---

## Folder contents

```
submission/
├── SUBMISSION.md              this file — the package index
├── CHECKLIST.md               done / not done, judged against the rubric
├── AI-DISCLOSURE.md           TEMPLATE — the team must complete this
├── SOURCES-AND-LICENSES.md    code, data, weights, fonts, graphics, libraries
├── LICENSE                    MIT
├── docs/
│   ├── concept-summary.pdf    the required one-page summary
│   ├── concept-summary.md     its source text
│   ├── blog-post.pdf          the required blog PDF
│   └── blog-post.md           its source text
├── evidence/
│   ├── claim-sheet.md         the claim, how to falsify it, what is not claimed
│   ├── citation-ledger.md     every external claim mapped to a primary source
│   └── source-matrix.csv      per-page claim / source / evidence label
└── artifact-build/            static build of the artifact, root-relative paths
```

`artifact-build/` is a snapshot for offline review. The deployed artifact is
rebuilt by CI from source on every push to `main`; see the checklist.

---

## How a judge should evaluate this in five minutes

1. Open the artifact URL. Section 1 is already running. Move `R`. The target
   activates at `R = 4`, and the BFS ground truth sits beside the estimate.
2. Section 3, click "No path exists", push `R` to 12. The target must never
   activate, and the invariant must hold at all 13 depths.
3. Section 4, set path length 6 and sweep `R`. Both the exact and the learned
   model commit at 6 — a distance never trained on.
4. Section 5 shows where the effect stops: accuracy on short paths erodes past
   `R ≈ 5`, and one ablation is reported as inconclusive rather than as a result.
5. Section 7 audits the provenance of every number on screen, computed from the
   bundle actually being served.

## Reproducing every number

```bash
cd Backend
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python reproduce.py          # tests, 5 seeds, ablations, sweeps, export (~4 min)
```

Verified on 2026-09-09, Python 3.14.7, torch 2.14.0+cpu, Node 22.23.1:

| Suite | Command | Result |
|---|---|---|
| Unit tests | `python -m tests.test_exact_invariant` | 11 tests, OK |
| 10,000-case invariant | `python -m tests.test_exact_invariant --large` | 10000 passed, 0 failures |
| Generator guarantees | `python -m tests.test_exact_invariant --generator` | 2000 cases, 0 distance mismatches |
| Browser vs PyTorch parity | `python -m tests.test_js_parity` | 5 tests, OK |
| Frontend smoke | `npm run smoke` | All checks passed |
| Production build | `npm run build` | 305 kB JS, 17 kB CSS |

Nine defects were found and fixed during the 2026-09-09 audit; they are listed
in [`CHECKLIST.md`](CHECKLIST.md) §B and summarised in README §16.1. No
checkpoint was retrained, so the shipped hash is still `a58f090f344e3f8b` and
the accuracy table in README §2 is unchanged.

## Scope boundary

The graph recurrence, the learned GNN, and the section 6 associative toy are
independent educational reimplementations written for this lab. **None of them
is BDH or BDH-CQ, and no BDH or BDH-CQ checkpoint was run.** Every BDH and
BDH-CQ figure in the artifact is quoted from the primary sources, labelled
*Paper-reported result*, and never shown on the same visual footing as a live
measurement.
