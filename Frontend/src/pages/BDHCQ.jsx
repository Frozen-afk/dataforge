import React, { useState } from "react";
import { EvidenceBadge, EvidenceNote } from "../components/Evidence.jsx";
import { Equation, PageHeader, Panel, Takeaway } from "../components/Common.jsx";

// Every figure on this page is quoted from published sources. None of it is
// measured by Latent Loop Lab, and the badges say so on each block.
const EFFORT_TABLE = [
  { effort: "Low", arc: "21%", cost: "22%" },
  { effort: "Medium", arc: "27%", cost: "11%" },
  { effort: "High", arc: "29.5%", cost: "0%" },
];

const LINEAGE = [
  {
    name: "BDH",
    full: "Dragon Hatchling",
    description:
      "The underlying brain-inspired post-Transformer family: high-dimensional positive activations, low-rank communication, and a recurrent associative state in which attention is reformulated as synaptic memory.",
  },
  {
    name: "BDH-GPU",
    full: "GPU-efficient formulation",
    description:
      "A GPU-friendly formulation of the same dynamics, built from ReLU low-rank transformations with linear attention. It is not a state-space model in the Mamba sense.",
  },
  {
    name: "BDH-CQ",
    full: "In-context learning with recurrent latent reasoning",
    description:
      "Combines inference-time in-context learning through an evolving recurrent memory with iterative computation in a latent workspace. This is the system this page connects to.",
  },
];

export default function BDHCQ({ onNext }) {
  const [selected, setSelected] = useState("context");

  return (
    <div className="page">
      <PageHeader
        step="6"
        question="Where does this appear in real AI?"
        goal="Connect the mechanism to a published system without claiming the two are the same thing."
      />

      <div className="scope-warning">
        <strong>Read this before the diagrams.</strong>
        <p>
          Everything on this page describes the published conceptual
          decomposition of BDH-CQ. Exact production dimensions and update rules
          are outside this project. The toy model in pages 1 to 5 is an
          independent reimplementation written for teaching, and is not a BDH or
          BDH-CQ checkpoint.
        </p>
      </div>

      <Panel title="Architectural lineage">
        <div className="lineage">
          {LINEAGE.map((item, index) => (
            <React.Fragment key={item.name}>
              <div className="lineage-node">
                <h4>{item.name}</h4>
                <span className="lineage-full">{item.full}</span>
                <p>{item.description}</p>
              </div>
              {index < LINEAGE.length - 1 && <div className="lineage-arrow">→</div>}
            </React.Fragment>
          ))}
        </div>
        <EvidenceNote evidence={{
          evidenceType: "Paper-reported result",
          source: "Kosowski et al., The Dragon Hatchling (arXiv:2509.26507); Engdahl et al., BDH-CQ (arXiv:2608.09888)",
        }} />
      </Panel>

      <Panel
        title="Two different recurrences, often confused"
        subtitle="Click a row. These are not the same process, and calling both of them 'reasoning steps' is the mistake this page exists to prevent."
      >
        <div className="recurrence-picker">
          <button
            className={`recurrence-row context ${selected === "context" ? "active" : ""}`}
            onClick={() => setSelected("context")}
          >
            <span className="rr-label">Contextual recurrence</span>
            <code>S_t = U_θ(S_t₋₁, D_t)</code>
            <span className="rr-note">recurrence over demonstrations / context</span>
          </button>

          <button
            className={`recurrence-row query ${selected === "query" ? "active" : ""}`}
            onClick={() => setSelected("query")}
          >
            <span className="rr-label">Query-time recurrence</span>
            <code>H_r₊₁ = F_θ(H_r, S_K)</code>
            <span className="rr-note">recurrence over query-time latent computation</span>
          </button>
        </div>

        {selected === "context" ? (
          <div className="recurrence-detail context">
            <h4>Recurrence over demonstrations</h4>
            <Equation label="one demonstration absorbed per step">
              S_t = U_θ(S_t₋₁, D_t)
            </Equation>
            <div className="flow">
              <span className="flow-item">D₁</span><span className="flow-arrow">→</span>
              <span className="flow-item">D₂</span><span className="flow-arrow">→</span>
              <span className="flow-item">⋯</span><span className="flow-arrow">→</span>
              <span className="flow-item">D_K</span><span className="flow-arrow">⇒</span>
              <span className="flow-item state">S_K</span>
            </div>
            <p>
              This state accumulates task-specific associations from the
              demonstrations. It advances once per demonstration, and what drives
              it is <strong>new input</strong>. When the demonstrations run out,
              this recurrence stops.
            </p>
            <p className="contrast">
              Nothing in pages 1 to 5 corresponds to this. The lab has no
              demonstrations and no in-context learning.
            </p>
          </div>
        ) : (
          <div className="recurrence-detail query">
            <h4>Recurrence over query-time latent computation</h4>
            <Equation label="workspace initialised from the query and the context state">
              H₀ = E_θ(x*, S_K)
            </Equation>
            <Equation label="refined repeatedly, with the context state held fixed">
              H_r₊₁ = F_θ(H_r, S_K)
            </Equation>
            <Equation label="decoded once, at the end">
              ŷ = G_θ(H_R)
            </Equation>
            <div className="flow">
              <span className="flow-item state">H₀</span><span className="flow-arrow">→</span>
              <span className="flow-item state">H₁</span><span className="flow-arrow">→</span>
              <span className="flow-item">⋯</span><span className="flow-arrow">→</span>
              <span className="flow-item state">H_R</span><span className="flow-arrow">⇒</span>
              <span className="flow-item">ŷ</span>
            </div>
            <p>
              This state carries computation for the <strong>current query</strong>.
              It advances without consuming any new input, and R is a budget that
              can be chosen at inference time. This is the recurrence the lab is
              an analogue of.
            </p>
            <p className="contrast">
              Note that S_K appears in every step but never changes during this
              loop. Context acquisition has finished; only latent refinement is
              running.
            </p>
          </div>
        )}

        <div className="learner-rule">
          <strong>The rule to remember:</strong> the first is recurrence over
          demonstrations and context. The second is recurrence over query-time
          latent computation. Do not call both of them &ldquo;reasoning
          steps&rdquo;.
        </div>
      </Panel>

      <Panel title="What maps onto what">
        <div className="mapping">
          <div className="mapping-side">
            <span className="mapping-head">Latent Loop Lab</span>
            <code>h⁽⁰⁾ → h⁽¹⁾ → ⋯ → h⁽ᴿ⁾ → ŷ</code>
            <span className="mapping-note">exact mechanism, pages 1–3</span>
            <code>z⁽⁰⁾ → z⁽¹⁾ → ⋯ → z⁽ᴿ⁾ → ŷ</code>
            <span className="mapping-note">learned extension, pages 4–5</span>
          </div>
          <div className="mapping-arrow">is an abstract analogue of</div>
          <div className="mapping-side">
            <span className="mapping-head">BDH-CQ</span>
            <code>H₀ → H₁ → ⋯ → H_R → ŷ</code>
            <span className="mapping-note">query-time latent workspace</span>
          </div>
        </div>

        <div className="analogy-statement">
          Latent Loop Lab h⁽ʳ⁾ is an abstract educational analogue of BDH-CQ H_r.
          It is <strong>not</strong> the whole BDH-CQ system.
        </div>

        <div className="mapping-limits">
          <h4>Where the analogy stops</h4>
          <ul>
            <li>
              The lab has no contextual recurrence S_t at all, so half of BDH-CQ
              has no counterpart here.
            </li>
            <li>
              The lab&rsquo;s coordinates are interpretable by construction. A
              production latent workspace is not generally human-readable.
            </li>
            <li>
              The lab solves graph reachability. Sharing a recurrence shape with a
              language model does not make it one.
            </li>
            <li>
              Scale differs by many orders of magnitude, and behaviour at this
              scale does not license claims about behaviour at that one.
            </li>
          </ul>
        </div>
      </Panel>

      <Panel
        title="Published latent-effort operating points"
        subtitle="Reported for BDH-CQ. These are quoted, not measured here."
      >
        <table className="data-table">
          <thead>
            <tr>
              <th>Latent effort</th>
              <th>ARC-AGI-1 pass@2</th>
              <th>Reported cost reduction</th>
            </tr>
          </thead>
          <tbody>
            {EFFORT_TABLE.map((row) => (
              <tr key={row.effort}>
                <td>{row.effort}</td>
                <td>{row.arc}</td>
                <td>{row.cost}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="provenance">
          <EvidenceBadge type="Paper-reported result" />
          <p>
            Source: the BDH-CQ technical report. These numbers are not
            measurements of Latent Loop Lab, are not reproduced here, and are not
            on the same footing as anything on pages 1 to 5. They are shown
            because they illustrate the same variable this lab lets you move:
            more latent effort against accuracy and cost.
          </p>
          <p className="caveat">
            A separate MIN-versus-STANDARD comparison exists in that literature
            whose statistical result is unresolved. It is deliberately not merged
            into this table.
          </p>
        </div>
      </Panel>

      <Takeaway>
        Query-time latent recurrence is a real component of a real system, and it
        is the component this lab models. Contextual recurrence is a separate
        mechanism that this lab does not model at all.
      </Takeaway>

      <div className="page-nav">
        <button className="primary" onClick={onNext}>
          Next: evidence and limitations
        </button>
      </div>
    </div>
  );
}
