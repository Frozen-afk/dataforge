import React, { useState } from "react";

const LINEAGE = [
  {
    name: "BDH",
    desc: "The underlying Dragon Hatchling architectural family: high-dimensional positive activations, low-rank communication, and recurrent associative state.",
    cite: "[5]",
  },
  {
    name: "BDH-GPU",
    desc: "A GPU-efficient formulation of the underlying dynamics.",
    cite: "[5]",
  },
  {
    name: "BDH-CQ",
    desc: "Combines inference-time in-context learning through evolving recurrent memory with iterative computation in a latent workspace.",
    cite: "[8]",
  },
];

const POSITIONING = [
  {
    system: "Token CoT",
    obj: "Generated token sequence",
    role: "Baseline: computation expressed as tokens.",
    cite: "—",
  },
  {
    system: "Coconut",
    obj: "Continuous hidden state",
    role: "Reasoning without verbalising every intermediate step.",
    cite: "[1]",
  },
  {
    system: "Recurrent-depth models",
    obj: "Shared latent block",
    role: "Test-time compute through repeated application.",
    cite: "[2]",
  },
  {
    system: "Looped Transformer",
    obj: "Reused block",
    role: "Effective depth without distinct parameters per layer.",
    cite: "[3]",
  },
  {
    system: "BDH-CQ",
    obj: "S_t and H_r",
    role: "Real system combining contextual recurrence and latent query computation.",
    cite: "[8]",
  },
  {
    system: "Latent Loop Lab",
    obj: "h(r) and z(r)",
    role: "Transparent mechanism plus learned experimental bridge.",
    cite: "this artifact",
  },
];

const REFERENCES = [
  { id: 1, text: "S. Hao, S. Sukhbaatar, D. Su, et al. Training Large Language Models to Reason in a Continuous Latent Space. arXiv:2412.06769, 2024.", url: "https://arxiv.org/abs/2412.06769" },
  { id: 2, text: "J. Geiping, S. McLeish, N. Jain, et al. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv:2502.05171, 2025.", url: "https://arxiv.org/abs/2502.05171" },
  { id: 3, text: "N. Saunshi, N. Dikkala, Z. Li, S. Kumar, S. J. Reddi. Reasoning with Latent Thoughts: On the Power of Looped Transformers. ICLR, 2025.", url: "https://openreview.net/forum?id=din0lGfZFd" },
  { id: 5, text: "A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz. The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain. arXiv:2509.26507, 2025.", url: "https://arxiv.org/abs/2509.26507" },
  { id: 8, text: "B. Engdahl, A. Kosowski, J. Chorowski, et al. BDH-CQ: In-Context Learning with Recurrent Latent Reasoning. arXiv:2608.09888, 2026.", url: "https://arxiv.org/abs/2608.09888" },
];

function Arrow() {
  return <span className="flow-arrow">→</span>;
}

export default function BDHCQ() {
  const [focus, setFocus] = useState(null); // "context" | "query" | null

  return (
    <div className="page">
      <header className="page-header">
        <div className="tag">Page 6 — BDH-CQ</div>
        <h1>Where does this appear in real AI?</h1>
        <p className="muted">
          This page connects the lab mechanism to BDH-CQ as a conceptual correspondence.
          It does not claim equivalence and does not infer unpublished internals.
        </p>
      </header>

      {/* Lineage */}
      <section className="panel">
        <h3>Conceptual lineage</h3>
        <div className="lineage">
          {LINEAGE.map((item, i) => (
            <React.Fragment key={item.name}>
              <div className="lineage-card">
                <strong>{item.name}</strong>
                <p className="muted">{item.desc}</p>
                <span className="cite">{item.cite}</span>
              </div>
              {i < LINEAGE.length - 1 && <Arrow />}
            </React.Fragment>
          ))}
        </div>
      </section>

      {/* Pipeline decomposition */}
      <section className="panel">
        <div className="panel-head">
          <h3>Two different processes, one architecture</h3>
          <span className="badge other">Click a row to inspect it</span>
        </div>

        <div
          className={
            "flow-row green " +
            (focus === "context" ? "focused" : focus === "query" ? "dimmed" : "")
          }
          onClick={() => setFocus(focus === "context" ? null : "context")}
        >
          <span className="flow-title">Context acquisition</span>
          <div className="flow-nodes">
            <span className="flow-node">D1</span>
            <Arrow />
            <span className="flow-node">D2</span>
            <Arrow />
            <span className="flow-node">…</span>
            <Arrow />
            <span className="flow-node">DK</span>
            <Arrow />
            <span className="flow-node strong-green">S_K</span>
          </div>
        </div>

        <div
          className={
            "flow-row blue " +
            (focus === "query" ? "focused" : focus === "context" ? "dimmed" : "")
          }
          onClick={() => setFocus(focus === "query" ? null : "query")}
        >
          <span className="flow-title">Query-time latent reasoning</span>
          <div className="flow-nodes">
            <span className="flow-node">x*</span>
            <Arrow />
            <span className="flow-node">H0</span>
            <Arrow />
            <span className="flow-node">H1</span>
            <Arrow />
            <span className="flow-node">…</span>
            <Arrow />
            <span className="flow-node">HR</span>
            <Arrow />
            <span className="flow-node strong-blue">ŷ</span>
          </div>
        </div>

        {focus === "context" && (
          <div className="focus-card green-border">
            <h4>Contextual recurrence</h4>
            <pre className="equation">{"S_t = U(S_{t-1}, D_t)"}</pre>
            <p>
              This state accumulates task-specific associations from demonstrations.
              It is recurrence <strong>over demonstrations/context</strong>. It is not
              query-time reasoning.
            </p>
          </div>
        )}

        {focus === "query" && (
          <div className="focus-card blue-border">
            <h4>Query-time recurrence</h4>
            <pre className="equation">
              {"H_0 = E(x*, S_K)\nH_{r+1} = F(H_r, S_K)\ny_hat = G(H_R)"}
            </pre>
            <p>
              This state carries computation for the current query. It is recurrence{" "}
              <strong>over query-time latent computation</strong> — the process that
              corresponds conceptually to the lab&apos;s h(r) and z(r).
            </p>
          </div>
        )}

        <p className="key-sentence">
          Learner rule: do not call both processes “reasoning steps.” The first is
          recurrence over demonstrations; the second is recurrence over query-time
          latent computation.
        </p>
      </section>

      {/* Analogy mapping */}
      <section className="panel">
        <h3>The analogy, stated precisely</h3>

        <div className="bridge-grid three-col">
          <div className="analogy-card">
            <strong>h(r)</strong>
            <p className="muted">Exact graph recurrence. Interpretable coordinates, BFS truth. This lab.</p>
            <span className="badge live">Live computation</span>
          </div>
          <div className="analogy-card">
            <strong>z(r)</strong>
            <p className="muted">Learned shared-weight recurrent GNN. This lab.</p>
            <span className="badge precomputed">Precomputed / live</span>
          </div>
          <div className="analogy-card">
            <strong>H_r</strong>
            <p className="muted">BDH-CQ query-time latent workspace. Published system-level example [8].</p>
            <span className="badge paper">Paper-reported</span>
          </div>
        </div>

        <p className="key-sentence">
          Latent Loop Lab h(r) is an abstract educational analogue of BDH-CQ H_r. It is
          not the whole BDH-CQ system.
        </p>

        <ul className="muted limit-list">
          <li>The lab has no demonstration memory S_t; BDH-CQ has one.</li>
          <li>Production dimensions and update rules of BDH-CQ are outside this project.</li>
          <li>The correspondence is conceptual, not an implementation claim.</li>
        </ul>
      </section>

      {/* Paper-reported evidence */}
      <section className="panel">
        <div className="panel-head">
          <h3>Reported latent-effort operating points</h3>
          <span className="badge paper">Paper-reported result</span>
        </div>

        <table className="evidence-table">
          <thead>
            <tr>
              <th>Latent effort</th>
              <th>ARC-AGI-1 pass@2</th>
              <th>Reported cost reduction</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Low</td>
              <td>21%</td>
              <td>22%</td>
            </tr>
            <tr>
              <td>Medium</td>
              <td>27%</td>
              <td>11%</td>
            </tr>
            <tr>
              <td>High</td>
              <td>29.5%</td>
              <td>0%</td>
            </tr>
          </tbody>
        </table>

        <p className="muted">
          Source: BDH-CQ technical report [8]. These are paper-reported results, not
          measurements of Latent Loop Lab. They are shown on separate visual footing
          from live measurements, as required.
        </p>
      </section>

      {/* Positioning */}
      <section className="panel">
        <h3>Where the idea appears in current research</h3>

        <table className="evidence-table">
          <thead>
            <tr>
              <th>System</th>
              <th>Recurrent object</th>
              <th>Role in the lesson</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {POSITIONING.map((row) => (
              <tr key={row.system}>
                <td>{row.system}</td>
                <td>{row.obj}</td>
                <td>{row.role}</td>
                <td>{row.cite}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* References */}
      <section className="panel">
        <h3>Primary sources used on this page</h3>
        <ul className="ref-list">
          {REFERENCES.map((r) => (
            <li key={r.id}>
              <span className="cite">[{r.id}]</span>{" "}
              <a href={r.url} target="_blank" rel="noreferrer">
                {r.text}
              </a>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
