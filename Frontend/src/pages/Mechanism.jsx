import React, { useEffect, useMemo, useState } from "react";
import GraphView from "../components/GraphView.jsx";
import { EvidenceNote } from "../components/Evidence.jsx";
import {
  DepthAxis, Equation, ErrorBanner, Loading, NextButton, PageHeader, Panel,
  Stat, Takeaway, Verdict,
} from "../components/Common.jsx";
import { exactPreset, presetList, MAX_R } from "../lib/lab.js";

const LABELS = {
  line: "Four-hop line",
  line_short: "Two-hop line",
  branch: "Branching graph",
  diamond: "Diamond graph",
  cycle: "Directed cycle",
  disconnected: "Disconnected graph",
};

export default function Mechanism({ onNext }) {
  const [presets, setPresets] = useState([]);
  const [preset, setPreset] = useState("branch");
  const [data, setData] = useState(null);
  const [R, setR] = useState(0);
  const [error, setError] = useState("");

  // The learner commits to a prediction before the answer is revealed. Without
  // that, the page can only ever confirm what the control already showed, and
  // a confirmation is not a test.
  const [guess, setGuess] = useState("");
  const [submitted, setSubmitted] = useState(null);

  useEffect(() => {
    presetList().then(setPresets).catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    setError("");
    setSubmitted(null);
    setGuess("");

    exactPreset(preset)
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setR(0);
      })
      .catch((err) => !cancelled && setError(err.message));

    return () => { cancelled = true; };
  }, [preset]);

  const state = data?.trajectory?.[R];
  const distance = data?.bfsDistance ?? null;

  const activeCount = useMemo(
    () => (state ? state.filter((v) => v > (data?.epsilon ?? 0)).length : 0),
    [state, data]
  );

  if (error) return <div className="page"><ErrorBanner message={error} /></div>;
  if (!data) return <div className="page"><Loading /></div>;

  const correct =
    submitted !== null &&
    ((distance === null && submitted === "never") || Number(submitted) === distance);

  return (
    <div className="page">
      <PageHeader
        step="Section 2 of 8"
        question="What is the latent state doing?"
        goal="Make the recurrence visible: a real vector, one coordinate per node, updated by a rule that never changes."
      />

      <div className="controls">
        <label>
          Graph
          <select value={preset} onChange={(event) => setPreset(event.target.value)}>
            {(presets.length ? presets.map((p) => p.name) : Object.keys(LABELS)).map(
              (name) => (
                <option key={name} value={name}>{LABELS[name] || name}</option>
              )
            )}
          </select>
        </label>
        <div className="grow">
          <DepthAxis
            id="mechanism-depth"
            value={R}
            min={0}
            max={MAX_R}
            onChange={setR}
            threshold={submitted !== null ? distance : null}
            hint={
              submitted === null
                ? "The distance marker appears once you have committed to a prediction."
                : undefined
            }
          />
        </div>
      </div>

      {submitted === null ? (
        <Panel
          title="Before you move anything: predict"
          subtitle="At what depth R does the target first become active? Count the columns in the picture below, then commit."
        >
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            {[0, 1, 2, 3, 4, 5, 6].map((value) => (
              <button
                key={value}
                className="chip"
                aria-pressed={guess === String(value)}
                onClick={() => setGuess(String(value))}
              >
                {value}
              </button>
            ))}
            <button
              className="chip"
              aria-pressed={guess === "never"}
              onClick={() => setGuess("never")}
            >
              never
            </button>
            <button className="btn" disabled={!guess} onClick={() => setSubmitted(guess)}>
              Lock it in
            </button>
          </div>
        </Panel>
      ) : (
        <Verdict
          tone={correct ? "good" : "bad"}
          title={`You predicted ${submitted === "never" ? "never" : `R = ${submitted}`}. ${correct ? "Correct." : "Not quite."}`}
        >
          The shortest path from source to target is{" "}
          {distance === null ? "infinite: no path exists" : `${distance} edges`}.
          The target activates exactly when R reaches that number, because each
          update moves activation along exactly one edge.
        </Verdict>
      )}

      <div className="split">
        <GraphView
          graph={data.graph}
          source={data.source}
          target={data.target}
          state={state}
          epsilon={data.epsilon}
          highlightPath={submitted !== null}
          caption={`h(${R}). A node is non-zero exactly when it is reachable from the source within ${R} hops.`}
        />

        <div className="stack">
          <Panel title="The update rule, locked">
            <Equation caption="one-hot at the source">h(0) = e_s</Equation>
            <Equation caption="applied identically at every depth">
              {`h(r+1)[v] = max( h(r)[v],  1 − ∏ (1 − α·h(r)[u]) )`}
            </Equation>
            <p className="muted">
              The product runs over every node u with an edge u → v. This is a
              noisy-OR: a node becomes active when activation arrives from any
              incoming neighbour. With α = 1 it is exact reachability spreading
              one edge per update.
            </p>
            <p className="muted">
              Nothing in that equation depends on r. There is no step index, no
              per-layer weight, and no schedule.
            </p>
          </Panel>

          <Panel title={`State vector h(${R})`}>
            <div className="vector">
              {state.map((value, node) => (
                <div
                  className={`vector-row ${node === data.target ? "is-target" : ""}`}
                  key={node}
                >
                  <span className="key">
                    h[{node}]
                    {node === data.source && <span className="tag-mini src">s</span>}
                    {node === data.target && <span className="tag-mini tgt">q</span>}
                  </span>
                  <span className="track">
                    <span className="fill" style={{ width: `${Math.min(100, value * 100)}%` }} />
                  </span>
                  <span className="val">{value.toFixed(2)}</span>
                </div>
              ))}
            </div>

            <div className="readout plain" style={{ marginTop: 16 }}>
              <Stat label="Active nodes" value={activeCount} />
              <Stat
                label="Shortest path"
                value={distance === null ? "∞" : distance}
                note="d(s, q)"
              />
              <Stat
                label="State width"
                value={data.graph.n}
                note="fixed at every depth"
              />
            </div>
          </Panel>
        </div>
      </div>

      <Panel
        title="Every state the recurrence passed through"
        subtitle="Click one to inspect it. The boundary between dark and lit is the shortest-path distance."
      >
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {data.trajectory.map((step, r) => {
            const lit = step[data.target] > data.epsilon;
            return (
              <button
                key={r}
                className="chip"
                aria-pressed={r === R}
                onClick={() => setR(r)}
                title={`${step.filter((v) => v > data.epsilon).length} active nodes`}
                style={lit ? { borderColor: "var(--charge)", color: "var(--charge)" } : undefined}
              >
                h({r})
              </button>
            );
          })}
        </div>
      </Panel>

      <div className="keyline">
        The state has a fixed width. Depth 12 uses exactly as much memory as
        depth 1. That is the property a growing key-value cache does not have,
        and it is why recurrent depth is a different resource from context
        length.
      </div>

      <EvidenceNote evidence={data.evidence} />

      <Takeaway>
        Recurrent depth R has a precise mechanical meaning. It is the number of
        edges information can travel, so the target cannot activate before R
        reaches the shortest-path distance.
      </Takeaway>

      <NextButton onNext={onNext}>Next: is this real, or just animation?</NextButton>
    </div>
  );
}
