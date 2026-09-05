# Latent Loop Lab

**Can a model think longer without saying more?**

An interactive educational laboratory for understanding **recurrent latent computation as an inference-time compute axis**, built for the DataForge 2026 Pathway Track (NeurIPS 2026 Education Track style).

---

## 1. The one-sentence claim

> Repeated application of a shared state-update rule can increase effective computational depth without generating intermediate language tokens. In the exact graph mechanism, `R` updates transmit reachability information through at most `R` edges. In the learned extension, increasing the inference iteration budget `R` can improve extrapolation to longer reasoning paths, but only within the learned dynamics' capacity and stability limits.

The interaction lets the learner **reproduce, test, or challenge** this claim by moving the `R` slider, changing graph presets, and comparing the learned model against the exact mechanism and an independent BFS oracle.

---

## 2. Intended learner and prerequisites

**Intended learner:** a student or engineer who knows basic ML (neural networks, gradients, graphs) but has not read the latent-reasoning / post-Transformer literature.

**Prerequisites:**

- Basic Python and command line.
- What a directed graph, source, and target are.
- Rough idea of what a recurrent update is.

**Learning objectives.** After using the artifact, the learner can:

1. Explain what recurrent depth `R` means mechanically.
2. Predict the minimum `R` at which a target activates, before moving the slider.
3. Distinguish token generation from latent state updates.
4. Interpret "same weights, more computation" in a shared-weight recurrent GNN.
5. Distinguish insufficient computation from learned-generalisation limits.
6. Distinguish BDH-CQ contextual recurrence `S_t` from query-time recurrence `H_r`.
7. Name at least one limitation or failure case of recurrent latent computation.

---

## 3. What this artifact is (and is not)

It **is**:

- A controlled graph instrument (exact noisy-OR recurrence + independent BFS oracle).
- A small learned shared-weight recurrent GNN used as an experimental bridge.
- An inference-depth experiment (same weights, different `R`).
- A guided 7-page lesson connecting the mechanism to BDH-CQ.

It **is not**:

- A chatbot, a reproduction of BDH-CQ, or a benchmark leaderboard.
- Proof that graph reachability implies semantic reasoning.
- A claim that more recurrence always improves reasoning.

---

## 4. Architecture overview

```
Learner interaction (React frontend)
        |
Graph + exact recurrence  ->  BFS truth          [LIVE COMPUTATION]
        |
Shared-weight learned recurrent GNN             [PRECOMPUTED RESULT / LIVE if deployed]
        |
Generalisation / inference-depth experiment     [PRECOMPUTED RESULT]
        |
AI architecture interpretation -> BDH-CQ        [PAPER-REPORTED RESULT]
        |
Limitations + evidence discipline
```

Three computational objects are never conflated:

| Object | What it does | Evidence |
|---|---|---|
| Exact graph recurrence `h(r)` | Hand-designed propagation, interpretable coordinates | Live computation |
| Learned recurrent GNN `z(r)` | Learned shared-weight updates on graph tasks | Precomputed result (live if `/learned/run` deployed) |
| BDH-CQ workspace `H_r` | Published system-level example | Paper-reported result |

---

## 5. Repository structure

```
repo/
├── README.md
├── Backend/
│   ├── requirements.txt
│   ├── server.py                  # FastAPI app (exact engine + learned router)
│   ├── learned_api.py             # /learned/* endpoints
│   ├── core/
│   │   ├── graph.py               # Graph schema + presets (n <= 24)
│   │   ├── bfs.py                 # Independent BFS oracle
│   │   └── recurrent.py           # Exact noisy-OR recurrence (R <= 12)
│   ├── experiments/
│   │   ├── generate.py            # Seeded case generator
│   │   └── run_exact.py           # CLI runner
│   ├── tests/
│   │   └── test_exact_invariant.py# Unit tests + 10,000-case suite
│   ├── learned/
│   │   ├── model.py               # SharedRecurrentGNN (theta shared across steps)
│   │   ├── dataset.py             # Path-graph datasets (train <= 4, test 1..10)
│   │   ├── train.py               # Training + frozen checkpoint + hash
│   │   ├── evaluate.py            # Inference-depth sweep R = 1..10, 5 seeds
│   │   └── export_results.py      # Combined JSON export for frontend
│   └── results/                   # Checkpoints + JSON outputs
└── Frontend/
    ├── package.json
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx                # Navigation shell (7 pages)
        ├── App.css
        ├── ExactLab.jsx           # Pages 1-3 (Hook, Mechanism, Verification)
        └── pages/
            ├── LearnedBridge.jsx  # Page 4 (AI bridge)
            ├── Generalisation.jsx # Page 5 (Generalisation)
            ├── BDHCQ.jsx          # Page 6 (BDH-CQ)
            └── Evidence.jsx       # Page 7 (Evidence & limitations)
```

---

## 6. Setup and run

### 6.1 Backend

```bash
cd Backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --reload
```

`requirements.txt` should contain:

```
fastapi
uvicorn[standard]
pydantic
torch
numpy
```

Backend runs at `http://127.0.0.1:8000`. Interactive API docs at `/docs`.

### 6.2 Frontend

```bash
cd Frontend
npm install
npm run dev
```

Frontend runs at `http://127.0.0.1:5173`.

The frontend expects the backend at `http://127.0.0.1:8000` (constant `API_BASE` in each page file; change it there if you deploy elsewhere).

### 6.3 Optional: train the learned model

Pages 1-3 work without this. Pages 4-5 use it (with graceful fallback messages if missing).

```bash
cd Backend
source .venv/bin/activate

# Train 5 seeds (paths of length <= 4, R <= 4)
python -m learned.train --seed 1 --epochs 80
python -m learned.train --seed 2 --epochs 80
python -m learned.train --seed 3 --epochs 80
python -m learned.train --seed 4 --epochs 80
python -m learned.train --seed 5 --epochs 80

# Inference-depth sweep (R = 1..10, seen + unseen path lengths)
python -m learned.evaluate --seed 1 --checkpoint results/model_seed1.pt
python -m learned.evaluate --seed 2 --checkpoint results/model_seed2.pt
python -m learned.evaluate --seed 3 --checkpoint results/model_seed3.pt
python -m learned.evaluate --seed 4 --checkpoint results/model_seed4.pt
python -m learned.evaluate --seed 5 --checkpoint results/model_seed5.pt

# Combine for the frontend
python -m learned.export_results --seeds 1 2 3 4 5
```

Outputs: `results/learned_experiment.json`, `results/learned_examples.json`, checkpoints `results/model_seed*.pt`.

---

## 7. How to test each feature

### 7.1 Backend: exact engine

| Test | Command | Expected |
|---|---|---|
| Unit tests | `python -m tests.test_exact_invariant` | `OK` (3 tests) |
| 10,000-case invariant | `python -m tests.test_exact_invariant --large --cases 10000` | `OK: 10000 cases passed.` |
| R too small | `python -m experiments.run_exact --preset line --R 3` | `estimate: false`, `bfsDistance: 4`, `invariantPass: true` |
| R sufficient | `python -m experiments.run_exact --preset line --R 4` | `estimate: true`, `invariantPass: true` |
| Disconnected | `python -m experiments.run_exact --preset disconnected --R 12` | `estimate: false`, `reachable: false`, `invariantPass: true` |
| Health | `curl http://127.0.0.1:8000/health` | `{"status":"ok"}` |
| Preset API | `curl http://127.0.0.1:8000/exact/preset/line/4` | JSON with trajectory, estimate, bfsDistance |
| Custom graph API | `POST /exact/run` with `{"graph":{"n":5,"edges":[[0,1],[1,2],[2,3],[3,4]]},"source":0,"target":4,"R":4}` | `estimate: true` |

The invariant being tested: `h(R)[q] > eps  iff  d(s, q) <= R`.

### 7.2 Backend: learned model

| Test | Command / check | Expected |
|---|---|---|
| Live learned inference | `POST /learned/run` `{"pathLength":3,"reachable":true,"R":3}` | 200 JSON with prediction/confidence (if checkpoint exists) |
| Missing model honesty | Same call before training | 503 with clear message |
| Experiment JSON | `GET /learned/experiment` | 5 seeds, accuracy per R=1..10 |
| Examples JSON | `GET /learned/examples` | success + failure examples |

### 7.3 Page 1 — Hook

1. Open the app; the four-hop line preset is already running (no blank canvas).
2. Move the `R` slider 0 → 4.
3. **Expected:** active nodes spread one edge per step; target activates exactly at `R = 4`.
4. **Aha check:** the update rule never changes; only the number of applications changes.

### 7.4 Page 2 — Mechanism

1. Watch the vector panel and timeline `h(0) ... h(R)`.
2. Before moving the slider, predict the minimum `R` for the current preset.
3. **Expected:** the prediction matches the BFS shortest-path distance shown beside it.

### 7.5 Page 3 — Verification

1. Check the truth panel: recurrent estimate, BFS oracle result, shortest-path distance, PASS/FAIL invariant.
2. Switch preset to **Disconnected graph**.
3. **Expected:** target never activates, distance shows `∞`, invariant stays `PASS`.
4. **Expected:** BFS is an independent implementation (see `core/bfs.py`), not derived from the recurrence.

### 7.6 Page 4 — AI Bridge

1. Default: path length 6 (outside training), `R = 4`.
2. Move `R` 1 → 10 with the same frozen weights.
3. **Expected:** badge "Same weights reused at every iteration" visible; learned prediction/confidence changes with `R`.
4. Uncheck "Target reachable".
5. **Expected:** failure explanation states more iterations cannot create a missing path.
6. Set path length 3 (inside training) and compare with length 7.
7. **Expected:** key sentence visible: "The model has not gained new parameters. It has only been allowed to compute longer."

### 7.7 Page 5 — Generalisation

1. Chart 1 shows accuracy vs `R`: faint per-seed lines, bold mean, exact mechanism dashed at 1.0.
2. Select path length 7 (unseen) from the dropdown.
3. **Expected:** curve reflects only length-7 results; training depths shaded.
4. Move the fixed-`R` slider on chart 2 (path length vs accuracy).
5. **Expected:** green "training range" shading at lengths 1-4; exact expected step line at `L <= R`.
6. Check seed chips and the success/failure example cards.
7. **Expected:** every plot carries the `Precomputed result` badge. A negative result (no extrapolation) is displayed honestly, not hidden.

### 7.8 Page 6 — BDH-CQ

1. Click the green row (context acquisition). **Expected:** `S_t = U(S_{t-1}, D_t)` card; labeled "recurrence over demonstrations/context".
2. Click the blue row (query-time reasoning). **Expected:** `H_0, H_{r+1}, y_hat` card; labeled "recurrence over query-time latent computation".
3. **Expected:** the learner rule "do not call both processes reasoning steps" is visible.
4. **Expected:** analogy sentence verbatim: "Latent Loop Lab h(r) is an abstract educational analogue of BDH-CQ H_r. It is not the whole BDH-CQ system."
5. **Expected:** effort table (Low 21%/22%, Medium 27%/11%, High 29.5%/0%) carries `Paper-reported result` and a provenance note.

### 7.9 Page 7 — Evidence and limitations

1. All four badges render with meanings and usage.
2. The ledger row for learned GNN auto-detects live vs precomputed deployment.
3. **Expected:** all seven limitations and both never-claim statements are visible.

### 7.10 The sixty-second test

A new user should, within one minute: open the app, see the preset running, move `R`, watch the target activate at `R = 4`, and read the truth panel. If this fails, fix before adding anything else.

---

## 8. Evidence discipline

| Element | Badge |
|---|---|
| Exact graph recurrence | Live computation |
| 10,000-case sweep | Synthetic data |
| Learned GNN results | Precomputed result (Live computation only if `/learned/run` is deployed) |
| BDH-CQ effort scores | Paper-reported result |
| ARC-AGI-1 numbers | Paper-reported result |
| Literature claims | Paper-reported result with citation |

No paper-reported number is shown on the same visual footing as a live measurement without a provenance label.

---

## 9. Failure cases shown on purpose

| Failure | Lesson |
|---|---|
| `R < d(s,q)` | Insufficient computational depth |
| Disconnected graph | More iterations cannot create a missing path |
| Learned model, small `R` | Insufficient inference depth |
| Learned model, long unseen path | More compute does not guarantee extrapolation |
| Very large `R` | Recurrence is a compute axis with stability/capacity limits |

---

## 10. Reproduction commands (one-command summary)

```bash
# Backend tests
python -m tests.test_exact_invariant --large --cases 10000

# Exact CLI example
python -m experiments.run_exact --preset line --R 4

# Learned pipeline
python -m learned.train --seed 1 --epochs 80
python -m learned.evaluate --seed 1 --checkpoint results/model_seed1.pt
python -m learned.export_results --seeds 1 2 3 4 5

# Serve
uvicorn server:app --reload        # Backend
npm run dev                        # Frontend
```

---

## 11. Caps and approximations (stated, not hidden)

- Exact engine: `n <= 24`, `R <= 12`, `epsilon = 1e-12`.
- Learned experiment: paths `<= 10`, `R <= 10`, hidden dim 32, 5 seeds.
- Exact layer interpretable by design; production latent states are not generally human-readable.
- Toy models are not BDH-CQ.

---

## 12. References

1. S. Hao et al. Training Large Language Models to Reason in a Continuous Latent Space. arXiv:2412.06769, 2024.
2. J. Geiping et al. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv:2502.05171, 2025.
3. N. Saunshi et al. Reasoning with Latent Thoughts: On the Power of Looped Transformers. ICLR 2025.
4. A. Kosowski et al. The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain. arXiv:2509.26507, 2025.
5. B. Engdahl et al. BDH-CQ: In-Context Learning with Recurrent Latent Reasoning. arXiv:2608.09888, 2026.

---

## 13. Credits, licenses, provenance

- Code: written by [team names], [license, e.g. MIT].
- Third-party libraries: FastAPI, PyTorch, React, Vite ([their licenses]).
- No external datasets; all data synthetic and seeded.
- Graphics/fonts: [list or "system fonts only"].
- [List any forked/reused components and what you added.]

## 14. AI assistance disclosure

[Describe exactly how AI tools were used: e.g. "AI assistants helped draft boilerplate UI code and this README; all equations, experiment design, results, and final wording were reviewed, run, and verified by the team. Every component can be traced and explained by a team member."]

---

## 15. Troubleshooting

| Symptom | Fix |
|---|---|
| `uvicorn: command not found` | Activate venv; `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'core'` | Run commands from the `Backend/` root |
| Port 8000 in use | `uvicorn server:app --reload --port 8001` and update `API_BASE` |
| Frontend: "Could not connect to backend" | Start backend; check `http://127.0.0.1:8000/health` |
| Page 5 error banner | Run the learned pipeline (Section 6.3) and restart backend |
| Page 4 shows precomputed note | Train `model_seed1.pt` or accept the honest fallback |
