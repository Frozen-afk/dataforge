# Submission checklist

Audited against `Pathway PS.pdf` on 2026-09-09, then fixed. Every test suite in
this repository was executed after the fixes; results are in the verification
log at the bottom.

Legend: **[x]** done and verified · **[ ]** not done · **[~]** partly done

---

## A. Blocking — only the team can finish these

- [ ] **A1. Publish the artifact and record its URL.** The track requires a
  public URL that opens without sign-in. Nothing is deployed today. The CI
  workflow fires only on push to `main`, and `main` is behind `dev`. Steps:
  merge `dev` into `main`, enable Settings → Pages → Source: GitHub Actions,
  wait for the run, then fill the URL into `README.md` line 11 and
  `SUBMISSION.md` row 1. Expected: `https://frozen-afk.github.io/dataforge/`.
- [ ] **A2. Confirm the repository is public.** `github.com/Frozen-afk/dataforge`
  must open in a logged-out browser. Judges cannot score a private repository.
- [ ] **A3. Complete the AI assistance disclosure.** `README.md` §16 now has a
  filled structure with the audit's own changes recorded, but the team's own
  rows are still blank. The long form is `AI-DISCLOSURE.md`. The rules require
  it, and 15 of 100 points ride on technical ownership.
- [ ] **A4. Add team member names.** `README.md` line 510 still reads "add your
  names here". Also fill the team block at the top of `SUBMISSION.md`.
- [ ] **A5. Re-verify two citations against the live arXiv listings.** Sources 7
  (`arXiv:2605.26733`) and 8 (`arXiv:2608.09888`, the BDH-CQ report) carry 2026
  identifiers, and the BDH-CQ report is the backbone of the required BDH module.
  Every quoted figure — the 21/27/29.5% latent-effort points, $0.00070 per task,
  150M parameters, the 111/400 against 118/400 comparison, the co-author audit —
  must be checked against the actual document, section by section, as the
  citation ledger claims. An invented or misattributed number here is the single
  largest scoring risk in the package: "Incorrect claims receive a major
  penalty."

## B. Defects found and fixed

- [x] **B1. The state-norm chart in section 5 could not render correctly.**
  `learned/export_results.py` clamped every confidence-interval upper bound to
  1.0. Right for accuracies, wrong for `meanStateNorm`, which is an unbounded
  L2 norm. All ten rows shipped with `high = 1.0` against means of 3.37 to 5.11,
  so the upper bound sat below the mean, and `Generalisation.jsx` derived the
  chart ceiling from that field. Since `Chart.jsx` clamps plotted values to the
  ceiling, the curve rendered as a flat line pinned to the top of the plot —
  which read as a plausible result while contradicting the paragraph beneath it.

  Load-bearing: README §2 and claim-sheet sub-claim 6 both cite "the state norm
  saturates near 5.1", read off this chart.

  **Fixed.** `mean_ci` now takes a `bounds` pair, either side of which may be
  `None`; the norm passes `(0, None)`. Verified in a browser: the curve climbs
  from 3.36 and flattens near 5.11, with a tenth of headroom above the widest
  interval so the plateau does not read as clipping.

- [x] **B2. The data bundle recorded a stale build revision.** `manifest.json`
  reported `gitRevision: e3ce01f`, two commits behind. Section 7 shows
  provenance to the learner, so the revision on screen should be the one that
  produced the numbers. **Fixed:** re-exported, now `49cf07f`.

- [x] **B3. The mean state norm was a mean of batch means.** `evaluate.py`
  averaged per-batch means over unequally sized batches, so a node in the short
  final batch counted for more than a node in a full one. **Fixed:** it now sums
  node norms and divides by node count. Moved `R = 1` from 3.37 to 3.36; every
  other depth is unchanged to two decimals. README §2 updated.

- [x] **B4. Checkpoints were loaded with `weights_only=False`.** That unpickles
  arbitrary objects, and it sat in the exact function anyone reproducing the
  results points at a `.pt` file. These checkpoints hold tensors and plain
  scalars only. **Fixed:** `weights_only=True` in `evaluate.py` and
  `export_web.py`, which covers `infer.py` and `learned_api.py` through
  `load_checkpoint`. Verified: the inference sweep in README §2 is unchanged.

- [x] **B5. The browser engine did not check node bounds.**
  `core/recurrent.py` rejected an out-of-range source or target;
  `engine/exact.js` did not, so `h[target]` read as `undefined` and
  `undefined > epsilon` returned a confident "unreachable" for a node that does
  not exist. **Fixed:** the JavaScript now rejects the same inputs.

- [x] **B6. A failed bundle load was cached forever.** `lab.js` stored the
  rejected promise, so one dropped request made the error permanent until a full
  page reload. **Fixed:** only the success is cached.

- [x] **B7. Section 4 flashed an error when the distance changed.** Changing
  path length invalidates the node count before the size effect replaces it —
  distance 1 is offered at 8 nodes, distance 6 is not — and that intermediate
  pair reached the case bank as a real lookup, failed, and replaced the whole
  section with an error banner for a frame. **Fixed:** the sweep waits for a
  valid pair. Verified in a browser: switching from 6 to 1 now moves the size
  control from 16 to 8 nodes with no error.

- [x] **B8. The reference server advertised credentialed wildcard CORS.**
  `allow_origins=["*"]` with `allow_credentials=True` is the combination
  browsers refuse, and it claimed a trust relationship that does not exist —
  the server carries no session or cookie. **Fixed:** credentials off, methods
  narrowed to GET, POST and OPTIONS.

- [x] **B9. One table could overflow on a narrow screen.** The published
  latent-effort table on the BDH-CQ page was the only three-column table not
  wrapped in the codebase's own `scroll-x` container. **Fixed.**

## C. Required package contents

- [x] **C1. Explorable artifact, not a static article.** Nine guided sections,
  both computational layers running client-side.
- [x] **C2. A learner can change a meaningful input and see the consequence.**
  `R`, path length, graph preset, aggregation, demonstration count. Every
  control maps to a real variable.
- [ ] **C3. Public artifact URL.** See A1.
- [ ] **C4. Public source repository.** See A2.
- [x] **C5. Blog post as a PDF.** `docs/blog-post.pdf`, 2 pages, 802 words,
  generated from Markdown by `docs/build-summary-pdf.py`.
- [x] **C6. One-page concept summary as a PDF.** `docs/concept-summary.pdf`,
  1 page, 918 words — inside the recommended 500–950. Contains a design-pressure
  section, the mechanism, a three-way comparison table against chain of thought
  and BDH-CQ, labelled evidence, and a scope statement. No number in it changed
  during the audit, so it did not need rebuilding.
- [~] **C7. Complete README.** Covers the claim, learner and prerequisites,
  seven learning objectives, architecture, the role of every component, evidence
  labels, reproduction, caps, scope, references, credits and troubleshooting.
  Outstanding: the artifact URL (A1), team names (A4), the team's own AI
  disclosure rows (A3).
- [x] **C8. Setup instructions for local components.** `RUNNING.md` plus README
  §7, verified from a clean shell.
- [x] **C9. At least three primary papers, 2022–2026, cited beside claims.**
  Eight primary sources, mapped claim-by-claim in `evidence/citation-ledger.md`
  and `evidence/source-matrix.csv`. Subject to A5.
- [x] **C10. Source and licence record.** `SOURCES-AND-LICENSES.md` covers code,
  libraries, data, weights, graphics, fonts and quoted results.
- [ ] **C11. AI, code, data, asset and licence disclosure.** See A3.

## D. Rubric coverage, 100 points

- [x] **Technical correctness and depth, 25.** All suites pass after the fixes:
  11 unit tests; 10,000-case invariant, 0 failures; 2,000-case generator check,
  0 distance mismatches; 5 parity tests agreeing with PyTorch to about 2e-7
  relative. The `α^R > ε` well-posedness condition is derived and enforced
  rather than assumed. B1 and B3 are corrected. Subject to A5.
- [ ] **Technical ownership and live defence, 15.** Blocked on A3. Rehearse: how
  the hard-negative generator matches node and edge counts exactly; why max
  aggregation mirrors the noisy-OR; why distances ≥ 5 sit at 0.5 below the depth
  threshold and why that is the correct score rather than a failure; where each
  BDH-CQ number lives in the report. Be ready to explain the nine fixes in §B —
  a judge may ask what changed and why.
- [x] **Learning effectiveness, 15.** One falsifiable claim, a stated audience
  with prerequisites, seven learning objectives, a guided nine-section narrative,
  a sixty-second test documented in README §9, and a self-marked explain-it-back
  section.
- [x] **Interactive substrate and honesty, 15.** Real computation, not
  animation. Ground truth sits beside every estimate. Feedback is local, with no
  network round trip. Failures are shown on purpose, including an ablation
  reported as inconclusive.
- [x] **BDH / BDH-CQ integration and evidence discipline, 10.** Section 6 is a
  full section, not a footnote: the contextual recurrence and the query-time
  workspace are driven separately, and the learner breaks one. Evidence strength
  is graded honestly — the report's "independent evaluation" is named as a
  co-author black-box audit. Subject to A5.
- [x] **Craft, robustness, accessibility, provenance, 10.** Build is 305 kB JS
  and 17 kB CSS, fonts have system fallbacks, licences are recorded, and
  `reproduce.py` regenerates every number. B6 through B9 hardened the robustness
  and narrow-screen cases. Link stability still depends on A1.
- [x] **One-page concept summary, 10.** See C6.

## E. Worth doing if time allows

- [ ] **E1. Open the artifact on a real phone.** The responsive CSS is sound on
  inspection: a breakpoint at 860px turns the sidebar into a horizontal
  scroller, single-column grids kick in at 1080px, and every wide table is now
  in a scrolling container. A real device check is still the only way to be
  sure, and mobile usability is named in the craft criterion.
- [ ] **E2. Confirm `git status` is clean at submission time.** The
  `__pycache__` directories are gitignored but present locally.
- [ ] **E3. Keep `Pathway PS.pdf` out of the repository.** It is the organisers'
  document, not the team's work, and it is currently untracked at the root.
- [ ] **E4. Consider submitting to the NeurIPS 2026 Education Track.** The
  problem statement points at its Call for Educational Resources. Check the
  deadline first.

---

## Verification log

Run on 2026-09-09 after every fix in §B. Python 3.14.7, torch 2.14.0+cpu,
Node 22.23.1.

| Suite | Result |
|---|---|
| `python -m tests.test_exact_invariant` | 11 tests, **OK** |
| `python -m tests.test_exact_invariant --large` | **10000 passed, 0 failures** |
| `python -m tests.test_exact_invariant --generator` | 2000 cases, **0 distance mismatches** |
| `python -m tests.test_js_parity` | 5 tests, **OK** |
| `npm run smoke` | **All checks passed** |
| `npm run build` | **304.72 kB JS, 17.06 kB CSS, data bundle copied** |
| README §2 accuracy table vs `experiment.json` | **matches to every published digit** |
| README §2 inference sweep vs `learned.infer` | **reproduced exactly** |
| `meanStateNorm` interval ordering | **low ≤ mean ≤ high at all ten depths** |
| Section 5 state-norm chart, in a browser | **curve climbs and saturates, no clipping** |
| Section 4 distance change, in a browser | **size control follows, no error banner** |
| Browser console across sections | **no errors** |
| `docs/concept-summary.pdf` | **1 page, 918 words** |

No checkpoint was retrained. The shipped checkpoint hash is unchanged at
`a58f090f344e3f8b`, so the model the README describes is the model that ships.
