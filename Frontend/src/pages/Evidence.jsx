import React, { useEffect, useState } from "react";
import { EVIDENCE, EvidenceBadge } from "../components/Evidence.jsx";
import {
  NextButton, PageHeader, Panel, Pill, Stat, Takeaway,
} from "../components/Common.jsx";
import { labStatus, labManifest, bankMatching } from "../lib/lab.js";

const LEDGER = [
  { element: "Exact graph recurrence", badge: "Live computation",
    detail: "Applied per depth in your browser by src/engine/exact.js, a port of Backend/core/recurrent.py." },
  { element: "Breadth-first ground truth", badge: "Live computation",
    detail: "src/engine/graph.js, an implementation that never reads the recurrence and is never read by it." },
  { element: "Graph instances", badge: "Synthetic data",
    detail: "Built by the seeded generator in Backend/core/generator.py, verified against BFS, and shipped as cases.json." },
  { element: "10,000-case invariant suite", badge: "Synthetic data",
    detail: "Seeded cases stratified by node count, density and distance. Run with python -m tests.test_exact_invariant --large." },
  { element: "Learned model, single graph (section 4)", badge: "Live computation",
    detail: "Forward pass of the frozen checkpoint in src/engine/gnn.js, from the exported weights." },
  { element: "Browser-versus-PyTorch agreement", badge: "Synthetic data",
    detail: "python -m tests.test_js_parity compares the two implementations on seeded cases at 1e-6 relative tolerance." },
  { element: "Learned depth curves and ablation (section 5)", badge: "Precomputed result",
    detail: "Multi-seed sweep over five frozen checkpoints, exported to JSON. Not recomputed as you browse." },
  { element: "Associative-memory toy (section 6)", badge: "Live computation",
    detail: "A hand-built illustration in src/engine/assoc.js. No trained parameter, and not BDH-CQ." },
  { element: "BDH-CQ latent-effort scores and ARC-AGI-1 numbers", badge: "Paper-reported result",
    detail: "Quoted from arXiv:2608.09888. Not reproduced here." },
  { element: "Architecture and lineage claims", badge: "Paper-reported result",
    detail: "Sourced to arXiv:2509.26507 and arXiv:2608.09888, cited beside the claims they support." },
];

const FAILURES = [
  { when: "R < d(s, q)", sees: "The target stays dark.",
    means: "Insufficient computational depth. The answer was never wrong, only unavailable." },
  { when: "No path exists", sees: "The target never activates, at any R.",
    means: "More iterations cannot create a missing edge. Depth buys reach, not connectivity." },
  { when: "Learned model, R too small", sees: "Wrong prediction, confidence near half.",
    means: "The same shortage of depth, in a model that cannot tell you so directly." },
  { when: "Learned model, long unseen path", sees: "Accuracy degrades past the training range.",
    means: "More compute does not guarantee algorithmic extrapolation." },
  { when: "Large R", sees: "Accuracy on already-solved cases erodes; state norm saturates.",
    means: "Recurrence is a compute axis with stability and capacity limits." },
  { when: "Section 6 toy, K cut short", sees: "The state goes to zero and reads as nothing.",
    means: "A missing association is a context failure. No latent budget substitutes for it." },
];

const LIMITATIONS = [
  "The lab is a toy. It is not BDH-CQ, does not contain BDH-CQ, and its behaviour licenses no claim about BDH-CQ's behaviour.",
  "Latent coordinates in the exact layer are interpretable by design. Production latent states are not generally human-readable.",
  "More recurrence consumes compute. Depth is not free, even when it is invisible because no tokens are emitted.",
  "Excessive recurrence saturates the learned dynamics. That is measured in section 5, not assumed.",
  "Graph reachability demonstrates computational depth, not semantic reasoning or general intelligence.",
  "The learned model has about 11,000 parameters and solves one synthetic task. Nothing here scales automatically.",
  "No wall-clock or cost comparison is made against any real system, because none was measured on matched hardware.",
];

const NEVER_CLAIMED = [
  "That graph reachability proves semantic reasoning.",
  "That more recurrence always improves reasoning.",
  "That any architecture is universally superior.",
  "That this lab reproduces, tests, or verifies any published BDH or BDH-CQ result.",
];

export default function Evidence({ onNext }) {
  const [status, setStatus] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [matching, setMatching] = useState(null);

  // The ledger reports what this build actually does, so no row can claim live
  // computation in a deployment that has no checkpoint.
  useEffect(() => {
    labStatus().then(setStatus).catch(() => {});
    labManifest().then(setManifest).catch(() => {});
    bankMatching().then(setMatching).catch(() => {});
  }, []);

  function badgeFor(row) {
    if (!status) return row.badge;
    if (row.element.startsWith("Learned model, single graph")) {
      return status.liveInferenceAvailable ? "Live computation" : "Precomputed result";
    }
    return row.badge;
  }

  return (
    <div className="page">
      <PageHeader
        step="Section 7 of 8"
        question="What counts as evidence here, and what does not?"
        goal="Make the provenance of every number on this site checkable, and state the limits plainly."
      />

      <Panel title="The four labels" subtitle="Every number in this lab carries exactly one.">
        <div className="grid-auto">
          {Object.entries(EVIDENCE).map(([type, info]) => (
            <div key={type}>
              <EvidenceBadge type={type} />
              <p className="muted" style={{ marginTop: 8 }}>{info.meaning}</p>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="The ledger">
        <div className="scroll-x">
          <table>
            <thead>
              <tr><th>What</th><th>Label</th><th>What produces it</th></tr>
            </thead>
            <tbody>
              {LEDGER.map((row) => (
                <tr key={row.element}>
                  <td>{row.element}</td>
                  <td><EvidenceBadge type={badgeFor(row)} /></td>
                  <td className="muted">{row.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {status && (
          <div className="readout" style={{ marginTop: 18 }}>
            <Stat
              label="Where computation happens"
              value="this browser"
              note="no server is involved"
            />
            <Stat
              label="Checkpoint loaded"
              value={status.meta?.checkpointHash?.slice(0, 8) || "none"}
              note={status.liveInferenceAvailable
                ? "section 4 runs live inference"
                : "learned numbers come from the exported sweep only"}
            />
            {manifest?.generatedAt && (
              <Stat
                label="Bundle built"
                value={manifest.generatedAt.slice(0, 10)}
                note={manifest.gitRevision ? `revision ${manifest.gitRevision}` : undefined}
              />
            )}
          </div>
        )}
      </Panel>

      <Panel
        title="What the negative examples are matched on"
        subtitle="If the unreachable graphs were easier than the reachable ones, the whole depth measurement would be meaningless. These are the actual statistics of the bundle you are browsing."
      >
        {matching ? (
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Statistic</th>
                  <th className="num">reachable</th>
                  <th className="num">unreachable</th>
                  <th>matched?</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>mean node count</td>
                  <td className="num">{matching.positive.nodes}</td>
                  <td className="num">{matching.negative.nodes}</td>
                  <td><Pill tone="yes">exactly, by construction</Pill></td>
                </tr>
                <tr>
                  <td>mean edge count</td>
                  <td className="num">{matching.positive.edges}</td>
                  <td className="num">{matching.negative.edges}</td>
                  <td><Pill tone="yes">exactly, by construction</Pill></td>
                </tr>
                <tr>
                  <td>mean target in-degree</td>
                  <td className="num">{matching.positive.targetInDegree}</td>
                  <td className="num">{matching.negative.targetInDegree}</td>
                  <td><Pill tone="no">not matched, gap reported</Pill></td>
                </tr>
                <tr>
                  <td>mean source out-degree</td>
                  <td className="num">{matching.positive.sourceOutDegree}</td>
                  <td className="num">{matching.negative.sourceOutDegree}</td>
                  <td><Pill tone="no">not matched, gap reported</Pill></td>
                </tr>
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">Loading the bundle's own statistics.</p>
        )}

        <p style={{ marginTop: 16 }}>
          Each unreachable graph is cut from a specific reachable one: the edges
          leaving a randomly chosen breadth-first layer are removed, and exactly
          the same number of edges is added back elsewhere. So node and edge
          counts match exactly. Two smaller gaps remain, shown above rather than
          hidden. Neither is visible from the target node until the recurrence
          reaches the source, which is why they do not substitute for
          propagation.
        </p>
        <p className="muted">
          An earlier version of this task did have a shortcut: negatives were
          the reversed chain, so the target had no incoming edges at all and the
          question collapsed to a one-hop lookup. Accuracy was 1.000 at every
          depth and the experiment measured nothing. That is why this table
          exists.
        </p>
      </Panel>

      <Panel
        title="Failures shown on purpose"
        subtitle="A demo that only shows wins is an advertisement, not an explanation."
      >
        <div className="scroll-x">
          <table>
            <thead>
              <tr><th>Situation</th><th>What you see</th><th>What it means</th></tr>
            </thead>
            <tbody>
              {FAILURES.map((row) => (
                <tr key={row.when}>
                  <td><code>{row.when}</code></td>
                  <td>{row.sees}</td>
                  <td className="muted">{row.means}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="split">
        <Panel title="Limitations">
          <ul className="plain">
            {LIMITATIONS.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </Panel>

        <Panel title="What this lab never claims">
          <ul className="crosses">
            {NEVER_CLAIMED.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </Panel>
      </div>

      <Panel title="Caps and approximations, stated rather than hidden">
        <ul className="plain">
          <li>Exact engine: at most 24 nodes, depth at most 12, threshold ε = 10⁻¹².</li>
          <li>Learned experiment: distances 1 to 10, depth at most 10, hidden dimension 32, five seeds.</li>
          <li>Training saw distances 1 to 4 only. Distances 5 to 10 are never trained on.</li>
          <li>Confidence intervals use a Student t critical value, appropriate for five seeds.</li>
          <li>
            The browser runs the checkpoint in double precision and the Python
            pipeline runs it in single. They agree to about 2 × 10⁻⁷ relative,
            which is float32 noise and three orders of magnitude finer than
            anything shown on screen.
          </li>
          <li>
            Graph instances are shipped as data rather than generated in the
            browser, so the sandbox draws from the same distribution the
            experiment measured.
          </li>
        </ul>
      </Panel>

      <Panel title="Reproduce all of it">
        <pre className="repro">{`cd Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python reproduce.py                    # tests, five seeds, ablation, sweeps
python export_web.py                   # rebuild the browser bundle
python -m tests.test_js_parity         # browser engine against PyTorch
python -m tests.test_exact_invariant --large`}</pre>
        <p className="muted">
          Every figure in sections 1 to 5 comes from that pipeline. Checkpoint
          hashes are printed during training and shown in sections 4 and 5, so a
          number on screen can be traced to the exact weights that produced it.
        </p>
      </Panel>

      <Takeaway>
        The central claim is narrow and testable: repeated application of a
        shared update rule increases effective computational depth without
        generating intermediate tokens, within limits this lab measures rather
        than assumes.
      </Takeaway>

      <NextButton onNext={onNext}>Last: can you say it in your own words?</NextButton>
    </div>
  );
}
