import React, { useEffect, useState } from "react";
import GraphView from "../components/GraphView.jsx";
import { EvidenceNote } from "../components/Evidence.jsx";
import {
  ErrorBanner, Loading, PageHeader, Panel, Pill, Slider, Takeaway,
} from "../components/Common.jsx";
import { api } from "../lib/api.js";

const PRESETS = ["line", "branch", "diamond", "cycle", "disconnected"];

export default function Verification({ onNext }) {
  const [preset, setPreset] = useState("line");
  const [R, setR] = useState(3);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [showTrajectory, setShowTrajectory] = useState(false);

  // Unlike pages 1 and 2, every R here is a fresh backend call. The point of
  // this page is that the invariant is checked by the server for the exact
  // depth on screen, not derived in the browser from a cached trajectory.
  useEffect(() => {
    let cancelled = false;
    setError("");

    api
      .exactPreset(preset, R)
      .then((result) => !cancelled && setData(result))
      .catch((err) => !cancelled && setError(err.message));

    return () => { cancelled = true; };
  }, [preset, R]);

  if (error) return <div className="page"><ErrorBanner message={error} /></div>;
  if (!data) return <div className="page"><Loading label="Checking the invariant..." /></div>;

  const distance = data.bfsDistance;
  const explanation =
    distance === null
      ? "BFS reports no path at all. No amount of recurrence can create an edge that does not exist, so the target stays inactive at every R."
      : R < distance
      ? `The shortest path is ${distance} edges long but R is only ${R}. The state has not had enough updates to cross the remaining ${distance - R}.`
      : `R is at least the shortest-path distance of ${distance}, so activation has had time to arrive.`;

  return (
    <div className="page">
      <PageHeader
        step="3"
        question="Is the effect real, or just animation?"
        goal="Check the recurrence against ground truth computed by a completely separate algorithm."
      />

      <div className="control-card row">
        <label>
          Graph
          <select value={preset} onChange={(e) => { setPreset(e.target.value); setR(0); }}>
            {PRESETS.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </label>
        <Slider id="verify-depth" label="Recurrent depth R" value={R} min={0} max={12} onChange={setR} />
      </div>

      <Panel className="truth-panel" title="Truth beside estimate">
        <div className="truth-grid">
          <div>
            <span className="small-label">Recurrent estimate</span>
            <Pill tone={data.estimate ? "yes" : "no"}>
              {data.estimate ? "target active" : "target inactive"}
            </Pill>
            <span className="sub">h(R)[q] = {data.targetActivation.toFixed(3)}</span>
          </div>

          <div>
            <span className="small-label">BFS oracle</span>
            <Pill tone={data.expected ? "yes" : "no"}>
              {data.expected ? "reachable within R" : "not reachable within R"}
            </Pill>
            <span className="sub">independent implementation</span>
          </div>

          <div>
            <span className="small-label">Shortest-path distance</span>
            <span className="big-value">{distance === null ? "∞" : distance}</span>
            <span className="sub">d(s, q)</span>
          </div>

          <div>
            <span className="small-label">Invariant</span>
            <Pill tone={data.invariantPass ? "yes" : "no"}>
              {data.invariantPass ? "PASS" : "FAIL"}
            </Pill>
            <span className="sub">h(R)[q] &gt; ε ⟺ d(s,q) ≤ R</span>
          </div>
        </div>

        <p className="explanation">{explanation}</p>
        <EvidenceNote evidence={data.evidence} />
      </Panel>

      <div className="split">
        <GraphView
          graph={data.graph}
          source={data.source}
          target={data.target}
          state={data.trajectory[R]}
          epsilon={data.epsilon}
        />

        <div className="stack">
          <Panel title="Why the oracle is independent">
            <p>
              The estimate comes from repeatedly applying the noisy-OR update to
              a vector of activations. The ground truth comes from breadth-first
              search over an adjacency list, in a separate module
              (<code>core/bfs.py</code>) that never reads the recurrence, and the
              recurrence never reads it back.
            </p>
            <p>
              Two unrelated procedures agreeing on every case is evidence. One
              procedure agreeing with itself would not be.
            </p>
            <ul className="tick-list">
              <li>10,000 seeded cases, stratified by node count, edge density and distance</li>
              <li>Distances 1 to 10 plus unreachable pairs</li>
              <li>Depths swept across each case&rsquo;s own boundary, where failure would show first</li>
              <li>Every case passes: run <code>python -m tests.test_exact_invariant --large</code></li>
            </ul>
            <EvidenceNote
              evidence={{
                evidenceType: "Synthetic data",
                source: "Seeded generator; validates the stated invariant",
              }}
            />
          </Panel>

          <Panel title="Try to break it">
            <p className="muted">
              The interesting cases are the boundaries. Set the graph to{" "}
              <strong>disconnected</strong> and push R to 12: the target must
              never activate. Set it to <strong>cycle</strong> and check that
              going around the loop does not shorten the distance.
            </p>
            <button className="secondary" onClick={() => setShowTrajectory((v) => !v)}>
              {showTrajectory ? "Hide" : "Show"} full trajectory
            </button>

            {showTrajectory && (
              <div className="trajectory-table">
                <table>
                  <thead>
                    <tr><th>r</th><th>active nodes</th><th>h(r)[q]</th><th>target active</th></tr>
                  </thead>
                  <tbody>
                    {data.trajectory.map((state, index) => (
                      <tr key={index} className={index === R ? "current" : ""}>
                        <td>{index}</td>
                        <td>{state.filter((v) => v > data.epsilon).length}</td>
                        <td>{state[data.target].toFixed(3)}</td>
                        <td>{state[data.target] > data.epsilon ? "yes" : "no"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        </div>
      </div>

      <Takeaway>
        The mechanism is live computation checked against an independent oracle,
        not a scripted animation. That makes it a usable control for the
        experiment on the next page.
      </Takeaway>

      <div className="page-nav">
        <button className="primary" onClick={onNext}>
          Next: replace the rule with a learned one
        </button>
      </div>
    </div>
  );
}
