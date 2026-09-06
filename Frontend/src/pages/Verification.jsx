import React, { useEffect, useMemo, useState } from "react";
import GraphView from "../components/GraphView.jsx";
import { EvidenceNote, EvidenceBadge } from "../components/Evidence.jsx";
import {
  DepthAxis, ErrorBanner, Loading, NextButton, PageHeader, Panel, Pill, Stat,
  Takeaway,
} from "../components/Common.jsx";
import { exactPreset, MAX_R } from "../lib/lab.js";

const LABELS = {
  line: "Four-hop line",
  branch: "Branching graph",
  diamond: "Diamond graph",
  cycle: "Directed cycle",
  disconnected: "Disconnected graph",
};

// Two ways a learner might try to break the invariant, offered as one click
// each. A "try to break it" instruction that leaves the reader to construct
// the adversarial case usually gets skipped.
const CHALLENGES = [
  {
    id: "disconnected",
    title: "No path exists",
    preset: "disconnected",
    R: MAX_R,
    expect: "The target must stay dark at every depth, all the way to 12.",
    why: "More iterations cannot create an edge that is not there. Depth buys reach, not connectivity.",
  },
  {
    id: "cycle",
    title: "A loop in the graph",
    preset: "cycle",
    R: 3,
    expect: "At R = 3 the target is still dark: going round the cycle does not shorten the path.",
    why: "The recurrence spreads along edges, so a cycle re-reaches nodes it already had rather than skipping ahead.",
  },
];

export default function Verification({ onNext }) {
  const [preset, setPreset] = useState("line");
  const [R, setR] = useState(3);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [showTable, setShowTable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setError("");
    exactPreset(preset)
      .then((result) => !cancelled && setData(result))
      .catch((err) => !cancelled && setError(err.message));
    return () => { cancelled = true; };
  }, [preset]);

  // Every depth is checked independently against the oracle, not derived from
  // the depth on screen, so a learner can see that the invariant holds across
  // the whole trajectory rather than at one convenient point.
  const allPass = useMemo(
    () => data?.perDepth?.every((entry) => entry.invariantPass) ?? false,
    [data]
  );

  if (error) return <div className="page"><ErrorBanner message={error} /></div>;
  if (!data) return <div className="page"><Loading label="Checking the invariant" /></div>;

  const here = data.perDepth[R];
  const distance = data.bfsDistance;

  const explanation =
    distance === null
      ? "Breadth-first search reports no path at all. No amount of recurrence can create an edge that does not exist, so the target stays inactive at every R."
      : R < distance
      ? `The shortest path is ${distance} edges long and R is only ${R}. The state has not had enough updates to cross the remaining ${distance - R}.`
      : `R is at least the shortest-path distance of ${distance}, so activation has had time to arrive.`;

  function runChallenge(challenge) {
    setPreset(challenge.preset);
    setR(challenge.R);
  }

  return (
    <div className="page">
      <PageHeader
        step="Section 3 of 8"
        question="Is the effect real, or just animation?"
        goal="Check the recurrence against ground truth produced by a completely separate algorithm."
      />

      <div className="controls">
        <label>
          Graph
          <select
            value={preset}
            onChange={(event) => { setPreset(event.target.value); setR(0); }}
          >
            {Object.entries(LABELS).map(([name, label]) => (
              <option key={name} value={name}>{label}</option>
            ))}
          </select>
        </label>
        <div className="grow">
          <DepthAxis
            id="verify-depth"
            value={R}
            min={0}
            max={MAX_R}
            onChange={setR}
            threshold={distance}
          />
        </div>
      </div>

      <div className="duo">
        <div>
          <span className="head">What the recurrence says</span>
          <span className="verdict">
            {here.estimate ? "target active" : "target inactive"}
          </span>
          <span className="note">
            h({R})[q] = {here.targetActivation.toFixed(3)}, from the noisy-OR update
          </span>
        </div>
        <div>
          <span className="head">What breadth-first search says</span>
          <span className="verdict">
            {here.expected ? "reachable within R" : "not reachable within R"}
          </span>
          <span className="note">
            d(s, q) = {distance === null ? "∞" : distance}, from an implementation
            that never reads the recurrence
          </span>
        </div>
      </div>

      <div className="readout" style={{ marginTop: 0, borderTop: 0 }}>
        <Stat
          label="Invariant at this depth"
          value={here.invariantPass ? "holds" : "broken"}
          note="h(R)[q] > ε ⟺ d(s,q) ≤ R"
        />
        <Stat
          label="Across all 13 depths"
          value={allPass ? "13 / 13 hold" : "a depth disagrees"}
          note="rechecked on every graph change"
        />
        <div>
          <span className="small-label">Status</span>
          <Pill tone={allPass && here.invariantPass ? "yes" : "alarm"}>
            {allPass && here.invariantPass ? "agreement" : "disagreement"}
          </Pill>
        </div>
      </div>

      <p style={{ marginTop: 18 }}>{explanation}</p>

      <div className="split">
        <GraphView
          graph={data.graph}
          source={data.source}
          target={data.target}
          state={data.trajectory[R]}
          epsilon={data.epsilon}
        />

        <div className="stack">
          <Panel title="Try to break it">
            <p className="muted">
              The interesting cases are the boundaries. Each button sets up one
              and tells you what should happen before it happens.
            </p>
            {CHALLENGES.map((challenge) => (
              <div key={challenge.id} style={{ marginBottom: 16 }}>
                <button className="btn ghost small" onClick={() => runChallenge(challenge)}>
                  {challenge.title}
                </button>
                <p className="muted" style={{ marginTop: 7 }}>
                  <strong>Expect:</strong> {challenge.expect} {challenge.why}
                </p>
              </div>
            ))}
            <button className="btn ghost small" onClick={() => setShowTable((v) => !v)}>
              {showTable ? "Hide" : "Show"} every depth at once
            </button>
          </Panel>

          <Panel title="Why the oracle counts as independent">
            <p>
              The estimate comes from repeatedly applying the noisy-OR update to
              a vector of activations. The ground truth comes from breadth-first
              search over an adjacency list, in a module that never reads the
              recurrence, and the recurrence never reads it back.
            </p>
            <p>
              Two unrelated procedures agreeing on every case is evidence. One
              procedure agreeing with itself would not be.
            </p>
            <ul className="ticks">
              <li>10,000 seeded cases, stratified by node count, edge density and distance</li>
              <li>Distances 1 to 10, plus pairs with no path at all</li>
              <li>Depths swept across each case's own boundary, where a failure would show first</li>
              <li>
                Every case passes. Reproduce with{" "}
                <code>python -m tests.test_exact_invariant --large</code>
              </li>
              <li>
                The browser engine is checked against the Python one to 2 × 10⁻⁷
                relative, by <code>python -m tests.test_js_parity</code>
              </li>
            </ul>
            <div className="evidence-note">
              <EvidenceBadge type="Synthetic data" />
              <span>Seeded generator; validates the stated invariant</span>
            </div>
          </Panel>
        </div>
      </div>

      {showTable && (
        <Panel title="Full trajectory" subtitle="Every depth, with its own invariant check.">
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th className="num">r</th>
                  <th className="num">active nodes</th>
                  <th className="num">h(r)[q]</th>
                  <th>recurrence says</th>
                  <th>BFS says</th>
                  <th>invariant</th>
                </tr>
              </thead>
              <tbody>
                {data.perDepth.map((entry) => (
                  <tr key={entry.R} className={entry.R === R ? "current" : ""}>
                    <td className="num">{entry.R}</td>
                    <td className="num">{entry.activeNodeCount}</td>
                    <td className="num">{entry.targetActivation.toFixed(3)}</td>
                    <td>{entry.estimate ? "active" : "inactive"}</td>
                    <td>{entry.expected ? "reachable" : "not reachable"}</td>
                    <td>{entry.invariantPass ? "holds" : "BROKEN"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <EvidenceNote evidence={data.evidence} />

      <Takeaway>
        The mechanism is live computation checked against an independent oracle,
        not a scripted animation. That is what makes it usable as a control for
        the experiment in the next section.
      </Takeaway>

      <NextButton onNext={onNext}>Next: replace the rule with a learned one</NextButton>
    </div>
  );
}
