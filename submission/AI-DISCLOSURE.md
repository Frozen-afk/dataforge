# AI assistance, code, data, asset and licence disclosure

> **STATUS: TEMPLATE. THE TEAM MUST COMPLETE THIS BEFORE SUBMITTING.**
>
> Every `_FILL IN_` below must be replaced with an honest answer. This document
> cannot be completed by an AI tool on the team's behalf, because it is a record
> of what the team itself did and understands.
>
> The track rules state: "AI-assisted coding, writing, design, and research are
> allowed, but teams must understand and defend every component. All
> AI-generated, reused, or forked work must be disclosed in the README." Judges
> award 15 of 100 points for technical ownership and live defence.

---

## 1. Tools used

| Tool | Version or model | Used for |
|---|---|---|
| _FILL IN_ | _FILL IN_ | _FILL IN — e.g. code generation, refactoring, prose drafting, design, literature search, review_ |

Known and already recorded:

| Tool | When | Used for |
|---|---|---|
| Claude Code (Claude Opus 5) | 2026-09-09 | Read the problem statement, audited the repository against it, ran every test suite, assembled `submission/`, and drafted `SUBMISSION.md`, `SOURCES-AND-LICENSES.md`, `CHECKLIST.md` and this template. Also fixed nine defects found in the audit, listed in `CHECKLIST.md` §B and README §16.1, and re-ran the evaluation over the existing frozen checkpoints. Designed no model, wrote no lesson content, and retrained nothing. |

The audit's nine fixes touched `learned/export_results.py`, `learned/evaluate.py`,
`export_web.py`, `server.py`, `engine/exact.js`, `lib/lab.js`,
`pages/LearnedBridge.jsx`, `pages/Generalisation.jsx` and `pages/BDHCQ.jsx`. Two
of them changed a published number: the mean-‖z‖ confidence interval, and
`R = 1`'s state norm moving from 3.37 to 3.36. A team member should read that
diff before live defence — judges may ask what changed and why.

## 2. What was AI-drafted and what was written by hand

State this per component. Be specific: "substantially AI-drafted, reviewed
line by line" and "written by hand" are both acceptable answers; silence is not.

| Component | Origin | Reviewed by |
|---|---|---|
| `Backend/core/recurrent.py` — exact recurrence | _FILL IN_ | _FILL IN_ |
| `Backend/core/generator.py` — hard-negative generator | _FILL IN_ | _FILL IN_ |
| `Backend/core/bfs.py` — BFS oracle | _FILL IN_ | _FILL IN_ |
| `Backend/learned/model.py` — shared-weight GNN | _FILL IN_ | _FILL IN_ |
| `Backend/learned/train.py`, `dataset.py`, `evaluate.py` | _FILL IN_ | _FILL IN_ |
| `Backend/tests/` — invariant and parity suites | _FILL IN_ | _FILL IN_ |
| `Frontend/src/engine/` — browser compute layer | _FILL IN_ | _FILL IN_ |
| `Frontend/src/pages/`, `components/` — interface | _FILL IN_ | _FILL IN_ |
| `README.md` | _FILL IN_ | _FILL IN_ |
| `docs/concept-summary.md` | _FILL IN_ | _FILL IN_ |
| `docs/blog-post.md` | _FILL IN_ | _FILL IN_ |

## 3. How each claim was verified by a person

The rules require that the team verified the work rather than accepting it.

| Item | How a team member verified it | Who |
|---|---|---|
| The exact recurrence equation in README §5 | _FILL IN — e.g. derived by hand, checked against the 10,000-case suite_ | _FILL IN_ |
| The `α^R > ε` well-posedness condition | _FILL IN_ | _FILL IN_ |
| The learned model's 11,169-parameter count | _FILL IN_ | _FILL IN_ |
| The depth-curve numbers in README §2 | _FILL IN_ | _FILL IN_ |
| The hard-negative matching argument in README §4 | _FILL IN_ | _FILL IN_ |
| The browser-versus-PyTorch parity result | _FILL IN_ | _FILL IN_ |

## 4. Primary sources actually read

The rules require that claims be checked against primary sources rather than
memory or secondhand summaries. For each source in
[`evidence/citation-ledger.md`](evidence/citation-ledger.md), record who read it
and how much.

| # | Source | Read by | Read in full, or the cited section only |
|---|---|---|---|
| 1 | Hao et al., arXiv:2412.06769 | _FILL IN_ | _FILL IN_ |
| 2 | Geiping et al., arXiv:2502.05171 | _FILL IN_ | _FILL IN_ |
| 3 | Saunshi et al., ICLR 2025 | _FILL IN_ | _FILL IN_ |
| 4 | Zhu et al., arXiv:2505.12514 | _FILL IN_ | _FILL IN_ |
| 5 | Kosowski et al., *The Dragon Hatchling*, arXiv:2509.26507 | _FILL IN_ | _FILL IN_ |
| 6 | Ben-Kish et al., COLM 2025 / arXiv:2505.07793 | _FILL IN_ | _FILL IN_ |
| 7 | Yang et al., arXiv:2605.26733 | _FILL IN_ | _FILL IN_ |
| 8 | Engdahl et al., *BDH-CQ*, arXiv:2608.09888 | _FILL IN_ | _FILL IN_ |

Every BDH-CQ number in the artifact is attributed to a specific section or table
of source 8. A team member must be able to open the report at that section and
point at the number during live defence.

## 5. Forks and reused work

- **Forked from another project:** No. This repository was written from scratch.
- **Code copied from another repository:** _FILL IN — state "none" if none._
- **Reused assets, graphics, or datasets:** None. See
  [`SOURCES-AND-LICENSES.md`](SOURCES-AND-LICENSES.md).

## 6. Mentorship

The rules require that mentor involvement be disclosed.

- **Mentors consulted:** _FILL IN — names and roles, or "none"._
- **What they advised on:** _FILL IN._
- **What they did not do:** _FILL IN — mentors may advise, review and challenge,
  but the registered team must build, understand and defend the submission._

## 7. Team statement

> _FILL IN — one paragraph, signed by the team, stating that every team member
> can explain and defend every component listed above._

Signed: _FILL IN — names and date_
