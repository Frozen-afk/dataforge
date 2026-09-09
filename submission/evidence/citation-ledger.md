# Citation ledger

Every externally checkable claim in the artifact, with its primary source and
the page where it appears. Secondary summaries are not used as sources.

## Primary sources

| # | Work | Identifier |
|---|---|---|
| 1 | Hao, Sukhbaatar, Su et al. *Training Large Language Models to Reason in a Continuous Latent Space* | arXiv:2412.06769 (2024) |
| 2 | Geiping, McLeish, Jain et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* | arXiv:2502.05171 (2025) |
| 3 | Saunshi, Dikkala, Li, Kumar, Reddi. *Reasoning with Latent Thoughts: On the Power of Looped Transformers* | ICLR 2025 |
| 4 | Zhu, Hao, Hu, Jiao, Russell, Tian. *Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought* | arXiv:2505.12514 (2025) |
| 5 | Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz. *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain* | arXiv:2509.26507 (2025) |
| 6 | Ben-Kish, Zimerman, Mirza et al. *Overflow Prevention Enhances Long-Context Recurrent LLMs* | COLM 2025; arXiv:2505.07793v2 |
| 7 | Yang, Han, Zhang, Wei, Shao, Guo, Li. *Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models* | arXiv:2605.26733 (2026) |
| 8 | Engdahl, Kosowski, Chorowski et al. *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning* | arXiv:2608.09888 (2026) |
| 9 | NeurIPS *2026 Call for Educational Resources* | neurips.cc |

## Claim-to-source mapping

| Page | Claim as stated in the artifact | Source | Label |
|---|---|---|---|
| 2 | Reasoning can happen in a continuous state instead of emitted tokens | 1, 4 | Paper-reported result |
| 5 | Test-time compute can be scaled through recurrent depth | 2 | Paper-reported result |
| 5 | Reusing a block gives effective depth without per-layer parameters | 3 | Paper-reported result |
| 5 | Recurrent depth peaks at some iteration count and then degrades in looped language models | 7, abstract | Paper-reported result |
| 5 | A fixed-size recurrent memory has bounded capacity, and long context is underused because of it | 6, abstract | Paper-reported result |
| 6 | BDH is a brain-inspired post-Transformer family with recurrent associative state | 5 | Paper-reported result |
| 6 | BDH-GPU is a GPU-efficient formulation built from ReLU low-rank transformations with linear attention | 5 | Paper-reported result |
| 6 | BDH-CQ combines in-context learning through evolving recurrent memory with latent iterative computation | 8, abstract and §3 | Paper-reported result |
| 6 | Contextual recurrence `S_t = U_θ(S_{t-1}, D_t)` | 8, eq. (1), §3.2 | Paper-reported result |
| 6 | Query-time recurrence `H_{r+1} = F_θ(H_r, S_K)` | 8, eqs. (2)–(4), §3.3 | Paper-reported result |
| 6 | Latent-effort points: LOW 21%/22%, MEDIUM 27%/11%, HIGH 29.5%/0% | 8, table 5, §7 | Paper-reported result |
| 6 | 29.5% pass@2 on public ARC-AGI-1 at a computed $0.00070 per task, 150M parameters | 8, §5 | Paper-reported result |
| 6 | A black-box audit reproduced 29.5%, conducted by co-authors from Bielik and New York University | 8, §5, "Independent evaluation" | Paper-reported result |
| 6 | MIN effort scored 111/400 against STANDARD's 118/400 at one third the cost | 8, §6.6 | Paper-reported result |
| 6 | BDH-CQ's dimensions, exact update rules and implementation details are proprietary | 8, §3.3 | Paper-reported result |
| 6 | BDH rivals GPT-2 at matched parameters from 10M to 1B; sparse positive activations; monosemanticity; heavy-tailed connectivity | 5, abstract | Paper-reported result |

## Numbers this project measured itself

These have no external citation because they are measurements of this
artifact's own toy models. They are reproducible via `python reproduce.py`.

| Number | Where | Label |
|---|---|---|
| 10,000-case invariant suite passes | Section 3 | Synthetic data |
| Learned accuracy 0.552 at `R=1` to 0.991 at `R=10` | Section 5 | Precomputed result |
| Unseen distances: chance until `R=4`, 0.988 at `R=10` | Section 5 | Precomputed result |
| Shared 11,169 vs unshared 39,585 parameters | Section 5 | Precomputed result |
| Aggregation ablation: max, sum and mean not separated at two seeds | Section 5 | Precomputed result |
| Browser engine agrees with PyTorch to 2e-7 relative | Section 7 | Synthetic data |
| Case-bank matched statistics (nodes and edges exact; in-degree 1.82 vs 1.55) | Section 7 | Synthetic data |
| State norm saturating near 5.1 | Section 5 | Precomputed result |
| Per-graph learned sweep and commit depth | Section 4 | Live computation |

## Evidence-strength notes

These distinctions matter more than the citations themselves, and the artifact
states them on the BDH-CQ page rather than burying them here.

- The 29.5% ARC-AGI-1 figure is a **benchmark result reported by the system's
  developers**. It is not a deployment.
- The report's "independent evaluation" is a black-box audit conducted by
  **co-authors** of the same paper, from Bielik and New York University, under a
  documented protocol without access to weights. That is stronger than a bare
  self-report and weaker than a third-party reproduction, and the artifact says
  so in exactly those terms.
- Cost comparisons place measured hardware time for BDH-CQ against
  leaderboard-reported costs for other systems, which the report notes may be
  hardware estimates or API prices. That is not a like-for-like measurement.
- No commercial partnership is cited anywhere in this artifact as evidence of
  performance.

## Deliberate exclusions

- The MIN-versus-STANDARD comparison is cited in the evidence-strength table on
  the BDH-CQ page but is deliberately **not** merged into the latent-effort
  table, because it varies a different setting.
- No BDH or BDH-CQ checkpoint was run. No internal detail of those systems is
  inferred beyond what the cited papers state.
- No wall-clock or cost comparison is made between this toy and any real system.
