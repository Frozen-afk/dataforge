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
| 7 | Wei et al. *Stabilizing Recurrent Dynamics for Test-Time Scalable Reasoning* | arXiv:2605.26733 (2026) |
| 8 | Engdahl, Kosowski, Chorowski et al. *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning* | arXiv:2608.09888 (2026) |
| 9 | NeurIPS *2026 Call for Educational Resources* | neurips.cc |

## Claim-to-source mapping

| Page | Claim as stated in the artifact | Source | Label |
|---|---|---|---|
| 2 | Reasoning can happen in a continuous state instead of emitted tokens | 1, 4 | Paper-reported result |
| 5 | Test-time compute can be scaled through recurrent depth | 2 | Paper-reported result |
| 5 | Reusing a block gives effective depth without per-layer parameters | 3 | Paper-reported result |
| 5 | Recurrent systems can destabilise as depth grows | 6, 7 | Paper-reported result |
| 6 | BDH is a brain-inspired post-Transformer family with recurrent associative state | 5 | Paper-reported result |
| 6 | BDH-GPU is a GPU-efficient formulation built from ReLU low-rank transformations with linear attention | 5 | Paper-reported result |
| 6 | BDH-CQ combines in-context learning through evolving recurrent memory with latent iterative computation | 8 | Paper-reported result |
| 6 | Contextual recurrence `S_t = U_θ(S_{t-1}, D_t)` | 8 | Paper-reported result |
| 6 | Query-time recurrence `H_{r+1} = F_θ(H_r, S_K)` | 8 | Paper-reported result |
| 6 | Latent-effort points: Low 21%/22%, Medium 27%/11%, High 29.5%/0% | 8 | Paper-reported result |

## Numbers this project measured itself

These have no external citation because they are measurements of this
artifact's own toy models. They are reproducible via `python reproduce.py`.

| Number | Where | Label |
|---|---|---|
| 10,000-case invariant suite passes | Page 3 | Synthetic data |
| Learned accuracy 0.552 at `R=1` to 0.991 at `R=10` | Page 5 | Precomputed result |
| Unseen distances: chance until `R=4`, 0.988 at `R=10` | Page 5 | Precomputed result |
| Shared 11,169 vs unshared 39,585 parameters | Page 5 | Precomputed result |
| State norm saturating near 4.6 | Page 5 | Precomputed result |
| Per-graph learned sweep and flip depth | Page 4 | Live computation |

## Deliberate exclusions

- A MIN-versus-STANDARD comparison exists in the BDH-CQ literature whose
  statistical result is unresolved. It is not merged into the effort table.
- No BDH or BDH-CQ checkpoint was run. No internal detail of those systems is
  inferred beyond what the cited papers state.
- No wall-clock or cost comparison is made between this toy and any real system.
