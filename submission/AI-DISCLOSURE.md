# AI assistance, code, data, asset and licence disclosure

**Team VibeCoders — Latent Loop Lab.** DataForge 2026, Pathway Track.

Members: Aryan Sisodiya, Abhishek Shahi, Farhan Alam, Ariyan Bhakat.

The track rules state: *"AI-assisted coding, writing, design, and research are
allowed, but teams must understand and defend every component. All
AI-generated, reused, or forked work must be disclosed in the README. [...] The
same applies to the blog, be ready to explain every claim, sentence, and
citation."* This document is that disclosure. README §16 points here.

---

## 1. Summary

AI assistance was used substantially and throughout, concentrated in four
areas: **writing code, debugging it, refining the architecture, and
implementing the design.** Prose was drafted the same way — the team decided
what each document had to say, and the AI wrote it down.

What the AI did not do is decide what this project is. The team read the
problem statement together, chose the concept and the one falsifiable sentence
the artifact teaches, designed the model engine and the task it runs on, and
set the core teaching principle that every section must let a learner change a
real variable and see the consequence beside the ground truth. Those decisions
came first and the code was written to serve them.

Every component below was read and accepted by the team before it shipped. The
team can trace any number on screen back to the code that produced it.

---

## 2. Tools used

| Tool | Model | When | Used for |
|---|---|---|---|
| Claude Code | Claude Opus 5 | 2026-09-04 to 2026-09-09 | Writing code, debugging, refining the architecture, implementing the team's design, drafting prose from the team's outlines, auditing the repository against the problem statement |

No other AI tool was used. No AI-generated image, icon, font, audio or video
appears anywhere in the submission; every graphic is inline SVG produced by
this project's own code.

Two commits carry an explicit AI co-authorship trailer in the git history and
are the easiest place to see the working pattern:

| Commit | What it did |
|---|---|
| `e3ce01f` (2026-09-06) | Rebuilt the task generator after the team found the experiment was measuring nothing, and rewrote the training and evaluation pipeline around it |
| `c65eca3` (2026-09-09) | Audited the repository against the problem statement, fixed nine defects, assembled the submission package |

---

## 3. What the team did

| Contribution | Detail |
|---|---|
| Reading the problem statement | The whole team went through the Pathway PS together and chose the topic, the audience, and the format |
| The one-sentence claim | Written by the team. It is deliberately weaker than "more loops always help", because the artifact has to be able to show it failing |
| The model engine | The team designed both computational layers: the noisy-OR graph recurrence as an exact, interpretable control, and the shared-weight recurrent GNN as the learned bridge that tests whether the same effect survives when the rule is learned |
| The core principle | Two layers, never conflated. An exact mechanism whose state a learner can read coordinate by coordinate, and a learned model whose state they cannot — with the ground truth printed beside every estimate |
| The task | Directed-graph reachability, chosen because `R` gets an exact meaning: the number of edges information can travel |
| The lesson design | The nine-section sequence, the question each section answers, and the one action the learner takes in each |
| Experimental design | Train on distances 1–4 only, resample `R` per batch, evaluate to distance 10, five seeds, unshared-weights and aggregation ablations |
| Review of every document | The team read and corrected the blog post, the concept summary and this README before they were built into PDFs |

The most consequential debugging call in the project was the team's. An earlier
version scored 1.000 accuracy at every `R` and every path length. The team
recognised that as a measurement failure rather than a result, traced it to
negatives built as reversed chains, and specified the replacement: derive each
negative from a specific positive so node and edge counts match exactly. README
§4 records both the failure and the fix.

---

## 4. What was AI-assisted, component by component

"AI-assisted" below means the AI wrote or substantially rewrote the code to the
team's design, and the team reviewed it. Nothing here was accepted unread.

| Component | Origin | Team review |
|---|---|---|
| `Backend/core/recurrent.py` — exact recurrence | Equation and the `α^R > ε` well-posedness condition specified by the team; implementation AI-written | Checked against the hand-derived equation and the 10,000-case invariant suite |
| `Backend/core/bfs.py` — independent BFS oracle | AI-written, deliberately kept independent of the recurrence | Read line by line; independence is the whole point of the check |
| `Backend/core/generator.py` — hard-negative generator | Matching strategy specified by the team after the shortcut was found; implementation AI-written | Verified by the generator guarantee suite: every case has the distance it claims |
| `Backend/learned/model.py` — shared-weight GNN | Architecture chosen by the team (one reused block, max aggregation as the differentiable noisy-OR); implementation AI-written | Parameter count checked against the layer shapes by hand: 11,169 shared, 39,585 unshared |
| `Backend/learned/train.py`, `dataset.py`, `evaluate.py`, `export_results.py` | AI-written to the team's experimental design | Reviewed; the team re-ran `reproduce.py` and matched the published numbers |
| `Backend/tests/` — invariant and parity suites | Invariant specified by the team; suites AI-written | Reviewed and run |
| `Backend/server.py`, `learned_api.py` — optional reference API | AI-written | Reviewed; not used by the deployed artifact |
| `Frontend/src/engine/` — browser compute layer | AI-written port of the team's Python reference | Held to the parity suite: browser and PyTorch agree to about 2 × 10⁻⁷ relative |
| `Frontend/src/pages/`, `components/` — the nine sections | Section sequence, learner action and copy decisions by the team; JSX and CSS AI-written | Reviewed section by section against the lesson design |
| `README.md`, `submission/SUBMISSION.md`, `submission/SOURCES-AND-LICENSES.md` | AI-drafted from the repository and the team's notes | Read and corrected by the team |
| `submission/docs/concept-summary.md` | Content, structure and every technical judgement planned by the team; AI wrote the prose | Reviewed sentence by sentence, including the maturity rating and the evidence-strength wording |
| `submission/docs/blog-post.md` | **Team-supervised.** The team chose the topic, the falsifiable claim, the table, the failure case and the "how to prove me wrong" section; the AI wrote the planned content and the explanation down | Reviewed and corrected by the team before the PDF was built |
| `submission/evidence/` — claim sheet, citation ledger, source matrix | AI-drafted | Checked against the papers; see §6 |
| This file | Drafted from the team's account of who did what | The team's own statement of its process |

---

## 5. What no AI produced

- **The model weights.** Every checkpoint under `Backend/results/` was trained
  from scratch by this project on the synthetic task in README §4. No
  pretrained weights were used, downloaded, fine-tuned or distilled. Shipped
  checkpoint: 11,169 parameters, hash `a58f090f344e3f8b`.
- **The data.** Every graph instance is synthetic, seeded and produced by
  `Backend/core/generator.py`. No external, scraped, licensed or human-subject
  dataset is used anywhere.
- **The graphics.** All inline SVG generated by this project's code. No stock
  asset, icon set, screenshot or illustration.
- **The numbers.** Every measurement in the README, the blog post and the
  concept summary is output of `python reproduce.py` over frozen checkpoints,
  or a figure quoted from a cited paper and labelled *Paper-reported result*.
  No number anywhere in this submission was written by a language model from
  memory.

---

## 6. Primary sources, and how they were checked

Claims were checked against the primary documents rather than against a model's
recollection of them. On 2026-09-09 every BDH and BDH-CQ figure quoted in the
artifact was re-verified against the arXiv listings, section by section. That
pass was run with AI assistance: the documents were fetched and the quoted
figures located in them, and the results are below. The team read the sources
while writing the artifact; this pass is a check on that reading, not a
substitute for it, and a team member should be able to open each report at the
section named here during live defence.

| Quoted claim | Verified against | Result |
|---|---|---|
| Latent effort LOW 21% / MEDIUM 27% / HIGH 29.5% pass@2, cost reduction 22% / 11% / 0% | arXiv:2608.09888, table 5, §7 | Matches exactly |
| 29.5% pass@2 on public ARC-AGI-1 at a computed $0.00070 per task, 150M parameters | arXiv:2608.09888, abstract and §5 | Matches exactly |
| The black-box audit reproducing 29.5% was conducted by co-authors from Bielik and New York University, without access to weights | arXiv:2608.09888, §5, "Independent evaluation" | Matches exactly |
| MIN effort scored 111/400 against STANDARD's 118/400 at one third the cost | arXiv:2608.09888, §6.6 | Matches exactly, including that the comparison is statistically unresolved |
| Contextual recurrence `S_t = U_θ(S_{t−1}, D_t)`, additive special case | arXiv:2608.09888, eq. (1), §3.2 | Matches exactly |
| Query-time recurrence `H_{r+1} = F_θ(H_r, S_K)` | arXiv:2608.09888, eqs. (2)–(4), §3.3 | Matches exactly |
| Dimensions, exact update rules and implementation details are proprietary | arXiv:2608.09888, §3.3 | Matches exactly |
| BDH rivals GPT-2 at matched parameters from 10M to 1B; sparse positive activations; monosemanticity; heavy-tailed connectivity | arXiv:2509.26507, abstract | Matches exactly |
| Recurrent depth peaks then collapses with further recurrence in looped language models | arXiv:2605.26733, abstract | Matches exactly |
| A fixed-size recurrent memory has bounded capacity and long context is underused because of it | arXiv:2505.07793 (COLM 2025), abstract | Matches, and the ledger was **narrowed** to say it: the earlier wording also cited this paper for depth instability, which is not what it studies |

The two 2026 identifiers were the ones most worth re-checking, because they are
the newest and the BDH-CQ report is the backbone of the required BDH module.
Both resolve to the papers cited, with the authors and titles as listed in
[`evidence/citation-ledger.md`](evidence/citation-ledger.md).

Every quoted figure sits beside its section reference in the ledger, so a judge
can open the report at that section and point at the number.

---

## 7. Forks and reused work

- **Forked from another project:** No. This repository was written from
  scratch for DataForge 2026.
- **Code copied from another repository:** None. No file in this repository is
  a copy, a vendored dependency or a translated port of another project's code.
- **Reused assets, graphics or datasets:** None. Third-party libraries are
  installed from their public registries and never vendored; they are listed
  with their licences in
  [`SOURCES-AND-LICENSES.md`](SOURCES-AND-LICENSES.md).
- **BDH and BDH-CQ code or weights:** none obtained, none run. The graph
  recurrence, the learned GNN and the section 6 associative toy are independent
  educational reimplementations written for this lab. None of them is BDH or
  BDH-CQ, and the artifact says so on the page where they appear.

---

## 8. Mentorship

No mentor, senior student, alumnus or external researcher advised on, reviewed
or contributed to this submission. The work is the registered team's.

---

## 9. Team statement

We used AI assistance heavily for writing and debugging code, for refining and
implementing the architecture, and for turning our planned content into
finished prose. We did not use it to decide what to build or what to claim. The
concept, the falsifiable sentence, the design of both computational layers, the
experimental design and the teaching structure are ours, and we read and
accepted every line that shipped. Every number in this submission is
reproducible from this repository with `python reproduce.py`, and every
external claim is mapped to a primary source and a section within it, which we
can open on request. Each of us can explain and defend any component listed
above.

**Team VibeCoders** — Aryan Sisodiya, Abhishek Shahi, Farhan Alam, Ariyan
Bhakat. 2026-09-09.
