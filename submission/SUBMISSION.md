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
| 1 | Public artifact URL, no sign-in | `PUBLIC_ARTIFACT_URL` — pending deployment | **Pending** — the one open item |
| 2 | Public source repository | https://github.com/Frozen-afk/dataforge | Done — merge `dev` → `main` before judging |
| 3 | Blog post as PDF | [`docs/blog-post.pdf`](docs/blog-post.pdf) | Done — 2 pages, 802 words |
| 4 | One-page concept summary PDF | [`docs/concept-summary.pdf`](docs/concept-summary.pdf) | Done — 1 page, 918 words, inside the recommended 500–950 |
| 5 | Complete README | [`../README.md`](../README.md) | Done |
| 6 | Setup instructions for local components | [`../RUNNING.md`](../RUNNING.md), README §7 | Done — verified on a clean checkout |
| 7 | ≥3 primary papers, 2022–2026, cited beside claims | [`evidence/citation-ledger.md`](evidence/citation-ledger.md), [`evidence/source-matrix.csv`](evidence/source-matrix.csv) | Done — 8 primary sources, all re-verified against arXiv on 2026-09-09 |
| 8 | Source and license record | [`SOURCES-AND-LICENSES.md`](SOURCES-AND-LICENSES.md) | Done |
| 9 | AI assistance, code, data, asset, license disclosure | [`AI-DISCLOSURE.md`](AI-DISCLOSURE.md) | Done |

The artifact URL is the only outstanding item. When the deployment is live,
replace `PUBLIC_ARTIFACT_URL` in row 1 above and in README line 11. Those are
the only two places it appears.

---

## Folder contents

```
submission/
├── SUBMISSION.md              this file — the package index
├── AI-DISCLOSURE.md           AI assistance, tools, ownership, sources, mentorship
├── SOURCES-AND-LICENSES.md    code, data, weights, fonts, graphics, libraries
├── LICENSE                    MIT
├── docs/
│   ├── concept-summary.pdf    the required one-page summary
│   ├── concept-summary.md     its source text
│   ├── blog-post.pdf          the required blog PDF
│   ├── blog-post.md           its source text
│   └── build-summary-pdf.py   builds both PDFs from the Markdown
└── evidence/
    ├── claim-sheet.md         the claim, how to falsify it, what is not claimed
    ├── citation-ledger.md     every external claim mapped to a primary source
    └── source-matrix.csv      per-page claim / source / evidence label
```

This folder holds every judge-facing document. It is the only copy of each: the
PDFs, their Markdown sources and the evidence files live here and nowhere else
in the repository. The artifact itself is built from `Frontend/` and deployed by
CI on every push to `main`.

Both PDFs are generated from their Markdown, never edited directly:

```bash
pip install weasyprint
python submission/docs/build-summary-pdf.py --doc all --pdf
```

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

Nine defects were found and fixed during the 2026-09-09 audit against the
problem statement; commit `c65eca3` carries all of them. Two changed a
published number: the mean-‖z‖ confidence interval, and `R = 1`'s state norm
moving from 3.37 to 3.36. No checkpoint was retrained, so the shipped hash is
still `a58f090f344e3f8b` and the accuracy table in README §2 is otherwise
unchanged.

## Citation verification

Every BDH and BDH-CQ figure quoted in the artifact was re-checked against the
arXiv listings on 2026-09-09, section by section. Every quoted figure matches
its primary source exactly, including the latent-effort table, the co-author
audit's affiliations, and the MIN-versus-STANDARD comparison being
statistically unresolved. One attribution was narrowed: reference 6 studies the
capacity of a fixed-size recurrent memory, not instability with depth, and the
ledger now cites it only for what it shows. The section-by-section results are
in [`AI-DISCLOSURE.md`](AI-DISCLOSURE.md) §6.

## Scope boundary

The graph recurrence, the learned GNN, and the section 6 associative toy are
independent educational reimplementations written for this lab. **None of them
is BDH or BDH-CQ, and no BDH or BDH-CQ checkpoint was run.** Every BDH and
BDH-CQ figure in the artifact is quoted from the primary sources, labelled
*Paper-reported result*, and never shown on the same visual footing as a live
measurement.
