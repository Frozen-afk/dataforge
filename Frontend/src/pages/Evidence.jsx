import React, { useEffect, useState } from "react";
import { EVIDENCE, EvidenceBadge } from "../components/Evidence.jsx";
import { PageHeader, Panel, Pill, Takeaway } from "../components/Common.jsx";
import { api } from "../lib/api.js";

const LEDGER = [
  { element: "Exact graph recurrence", badge: "Live computation",
    detail: "Computed per request by core/recurrent.py on the graph on screen." },
  { element: "BFS ground truth", badge: "Live computation",
    detail: "Independent implementation in core/bfs.py, never derived from the recurrence." },
  { element: "10,000-case invariant suite", badge: "Synthetic data",
    detail: "Seeded generator stratified by node count, density and distance." },
  { element: "Learned GNN depth curves", badge: "Precomputed result", dynamic: true,
    detail: "Multi-seed sweep over frozen checkpoints, exported to JSON." },
  { element: "Single-graph learned sweep (page 4)", badge: "Live computation", dynamic: true,
    detail: "Forward passes run per request when a checkpoint is deployed." },
  { element: "BDH-CQ latent-effort scores", badge: "Paper-reported result",
    detail: "Quoted from the BDH-CQ technical report. Not reproduced here." },
  { element: "ARC-AGI-1 numbers", badge: "Paper-reported result",
    detail: "Quoted. Not measured by this project." },
  { element: "Architecture and lineage claims", badge: "Paper-reported result",
    detail: "Sourced to the Dragon Hatchling paper and the BDH-CQ report." },
];

const FAILURES = [
  { failure: "R < d(s, q)", sees: "Target remains inactive.",
    lesson: "Insufficient computational depth. The answer was never wrong, only unavailable." },
  { failure: "Disconnected graph", sees: "Target never activates at any R.",
    lesson: "More iterations cannot create a missing path." },
  { failure: "Learned model, R too small", sees: "Incorrect prediction, low confidence.",
    lesson: "Insufficient inference depth in the learned model too." },
  { failure: "Learned model, long unseen path", sees: "Degraded accuracy past the training range.",
    lesson: "More compute does not guarantee algorithmic extrapolation." },
  { failure: "Large R", sees: "Accuracy on already-solved cases erodes; state norm saturates.",
    lesson: "Recurrence is a compute axis with stability and capacity limits." },
];

const LIMITATIONS = [
  "The lab is a toy. It is not BDH-CQ, does not contain BDH-CQ, and its behaviour licenses no claim about BDH-CQ's behaviour.",
  "Latent coordinates here are interpretable by design. Production latent states are not generally human-readable.",
  "More recurrence consumes compute. Depth is not free, even when it is not visible as generated tokens.",
  "Excessive recurrence may saturate or destabilise learned dynamics. This is measured on page 5, not assumed.",
  "Graph reachability demonstrates computational depth, not general intelligence or semantic reasoning.",
  "The learned model has roughly 11,000 parameters and solves one synthetic task. Nothing here scales automatically.",
  "Broader independent evidence is required before claiming any architecture is universally superior.",
];

const NEVER_CLAIMED = [
  "That graph reachability proves semantic reasoning.",
  "That more recurrence always improves reasoning.",
];

export default function Evidence() {
  const [status, setStatus] = useState(null);

  // The ledger reports what this deployment actually does, so a row cannot
  // claim "live computation" on a server with no checkpoint.
  useEffect(() => {
    api.learnedStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  function badgeFor(row) {
    if (!row.dynamic || !status) return row.badge;
    if (row.element.startsWith("Single-graph")) {
      return status.liveInferenceAvailable
        ? "Live computation"
        : "Precomputed result";
    }
    return row.badge;
  }

  return (
    <div className="page">
      <PageHeader
        step="7"
        question="What is actually evidence here, and what is not?"
        goal="Make the provenance of every number on this site checkable, and state the limits plainly."
      />

      <Panel title="The four labels" subtitle="Every number in this lab carries exactly one of these.">
        <div className="badge-grid">
          {Object.entries(EVIDENCE).map(([type, info]) => (
            <div className="badge-card" key={type}>
              <EvidenceBadge type={type} />
              <p>{info.meaning}</p>
            </div>
          ))}
        </div>
        <p className="callout">
          No paper-reported number appears anywhere in this lab on the same
          visual footing as a live measurement. The distinction is part of the
          project, not bureaucratic overhead.
        </p>
      </Panel>

      <Panel title="Evidence ledger">
        <table className="data-table wide">
          <thead>
            <tr><th>Element</th><th>Label</th><th>What produces it</th></tr>
          </thead>
          <tbody>
            {LEDGER.map((row) => (
              <tr key={row.element}>
                <td>{row.element}</td>
                <td><EvidenceBadge type={badgeFor(row)} /></td>
                <td className="detail-cell">{row.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {status && (
          <p className="muted">
            This deployment:{" "}
            {status.liveInferenceAvailable
              ? `a checkpoint is loaded (${status.meta?.checkpointHash}), so page 4 runs live inference.`
              : "no checkpoint is loaded, so learned numbers come from the exported sweep only."}
          </p>
        )}
      </Panel>

      <Panel title="Failures shown on purpose"
             subtitle="A demo that only shows wins would be an advertisement, not an explanation.">
        <table className="data-table wide">
          <thead>
            <tr><th>Situation</th><th>What you see</th><th>What it means</th></tr>
          </thead>
          <tbody>
            {FAILURES.map((row) => (
              <tr key={row.failure}>
                <td><code>{row.failure}</code></td>
                <td>{row.sees}</td>
                <td className="detail-cell">{row.lesson}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <div className="split">
        <Panel title="Limitations">
          <ul className="limit-list">
            {LIMITATIONS.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </Panel>

        <Panel title="What this lab never claims" className="never-panel">
          <ul className="never-list">
            {NEVER_CLAIMED.map((item) => (
              <li key={item}><span className="never-mark">×</span> {item}</li>
            ))}
          </ul>

          <div className="toy-notice">
            <Pill tone="no">toy ≠ BDH-CQ</Pill>
            <p>
              The models in pages 1 to 5 were written for this lab. They are an
              independent reimplementation of a general idea, not any official
              model, and are labelled as such everywhere they appear.
            </p>
          </div>
        </Panel>
      </div>

      <Panel title="Caps and approximations, stated rather than hidden">
        <ul className="tick-list">
          <li>Exact engine: n ≤ 24 nodes, R ≤ 12, ε = 10⁻¹².</li>
          <li>Learned experiment: distances 1 to 10, R ≤ 10, hidden dimension 32, five seeds.</li>
          <li>Training saw distances 1 to 4 only; 5 to 10 are never trained on.</li>
          <li>Negatives are matched to positives on node and edge count, but target in-degree still differs slightly (about 1.63 against 1.78).</li>
          <li>Confidence intervals use a Student t critical value, appropriate for five seeds.</li>
        </ul>
      </Panel>

      <Panel title="Reproduce everything">
        <pre className="repro">{`cd Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python reproduce.py            # tests, training, sweeps, export
python -m learned.infer --checkpoint results/model_seed1.pt --distance 6 --sweep`}</pre>
        <p className="muted">
          Every figure on pages 1 to 5 comes from that pipeline. Checkpoint
          hashes are printed during training and shown on pages 4 and 5, so a
          displayed number can be traced to the exact weights that produced it.
        </p>
      </Panel>

      <Takeaway>
        The central claim is narrow and testable: repeated application of a
        shared update rule increases effective computational depth without
        generating intermediate tokens, within limits this lab measures rather
        than assumes.
      </Takeaway>
    </div>
  );
}
