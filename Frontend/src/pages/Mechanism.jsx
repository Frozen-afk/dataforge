import React, { useEffect, useMemo, useState } from "react";
import GraphView from "../components/GraphView.jsx";
import { EvidenceNote } from "../components/Evidence.jsx";
import {
  Equation, ErrorBanner, Loading, PageHeader, Panel, Pill, Slider, Takeaway,
} from "../components/Common.jsx";
import { api } from "../lib/api.js";

const MAX_R = 12;

const PRESET_LABELS = {
  line: "Four-hop line",
  line_short: "Two-hop line",
  branch: "Branching graph",
  diamond: "Diamond graph",
  cycle: "Directed cycle",
  disconnected: "Disconnected graph",
};

export default function Mechanism({ onNext }) {
  const [preset, setPreset] = useState("branch");
  const [data, setData] = useState(null);
  const [R, setR] = useState(3);
  const [error, setError] = useState("");

  // The learner commits to a prediction before the answer is revealed. Without
  // this, the page would only ever confirm what the slider already showed.
  const [guess, setGuess] = useState("");
  const [submitted, setSubmitted] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setError("");
    setSubmitted(null);
    setGuess("");

    api
      .exactPreset(preset, MAX_R)
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setR(0);
      })
      .catch((err) => !cancelled && setError(err.message));

    return () => {
      cancelled = true;
    };
  }, [preset]);

  const state = data?.trajectory?.[R];
  const distance = data?.bfsDistance ?? null;

  const activeCount = useMemo(
    () => (state ? state.filter((v) => v > (data?.epsilon ?? 0)).length : 0),
    [state, data]
  );
  const activationMass = useMemo(
    () => (state ? state.reduce((a, b) => a + b, 0) : 0),
    [state]
  );

  if (error) return <div className="page"><ErrorBanner message={error} /></div>;
  if (!data) return <div className="page"><Loading /></div>;

  const guessCorrect =
    submitted !== null &&
    ((distance === null && submitted === "never") ||
      Number(submitted) === distance);

  return (
    <div className="page">
      <PageHeader
        step="2"
        question="What is the latent state doing?"
        goal="Make the recurrence mathematically visible: a real vector, one coordinate per node, updated by a rule that never changes."
      />

      <div className="control-card row">
        <label>
          Graph
          <select value={preset} onChange={(e) => setPreset(e.target.value)}>
            {Object.entries(PRESET_LABELS).map(([key, label]) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </select>
        </label>

        <Slider
          id="mechanism-depth"
          label="Recurrent depth R"
          value={R}
          min={0}
          max={MAX_R}
          onChange={setR}
        />
      </div>

      {submitted === null && (
        <Panel
          className="predict"
          title="Before you move the slider: predict"
          subtitle="At what depth R will the target first become active? Commit to an answer, then check it."
        >
          <div className="predict-row">
            {[0, 1, 2, 3, 4, 5, 6].map((value) => (
              <button
                key={value}
                className={guess === String(value) ? "chip active" : "chip"}
                onClick={() => setGuess(String(value))}
              >
                {value}
              </button>
            ))}
            <button
              className={guess === "never" ? "chip active" : "chip"}
              onClick={() => setGuess("never")}
            >
              never
            </button>
            <button
              className="primary"
              disabled={!guess}
              onClick={() => setSubmitted(guess)}
            >
              Lock in
            </button>
          </div>
        </Panel>
      )}

      {submitted !== null && (
        <div className={`predict-result ${guessCorrect ? "right" : "wrong"}`}>
          <strong>
            You predicted {submitted === "never" ? "never" : `R = ${submitted}`}.{" "}
            {guessCorrect ? "Correct." : "Not quite."}
          </strong>
          <p>
            The shortest path from source to target is{" "}
            {distance === null ? "infinite: no path exists" : distance}. The
            target activates exactly when R reaches that number, because each
            update moves activation along exactly one edge.
          </p>
        </div>
      )}

      <div className="split">
        <GraphView
          graph={data.graph}
          source={data.source}
          target={data.target}
          state={state}
          epsilon={data.epsilon}
          caption={`h(${R}): nodes with non-zero activation are those reachable from the source within ${R} hops.`}
        />

        <div className="stack">
          <Panel title="The update rule, locked">
            <Equation label="initial state: one-hot at the source">
              h(0) = e_source
            </Equation>
            <Equation label="applied identically at every depth">
              {`h(r+1)[v] = max( h(r)[v],  1 - ∏ (1 - α · h(r)[u]) )`}
            </Equation>
            <p className="muted">
              The product runs over every node u with an edge u &rarr; v. This is
              a noisy-OR: a node becomes active when activation arrives from any
              incoming neighbour. With &alpha; = 1 it is exact reachability
              spreading one edge per update.
            </p>
            <p className="muted">
              Nothing in this equation depends on r. That is what makes R a pure
              compute dial rather than a different model.
            </p>
          </Panel>

          <Panel title="State vector h(r)">
            <div className="vector-list">
              {state.map((value, index) => (
                <div className="vector-row" key={index}>
                  <div className="vector-label">
                    h[{index}]
                    {index === data.source && <span className="tag-mini src">s</span>}
                    {index === data.target && <span className="tag-mini tgt">q</span>}
                  </div>
                  <div className="bar-bg">
                    <div className="bar-fill" style={{ width: `${Math.min(100, value * 100)}%` }} />
                  </div>
                  <div className="vector-value">{value.toFixed(2)}</div>
                </div>
              ))}
            </div>
            <div className="readout tight">
              <div>
                <span className="small-label">Active nodes</span>
                <span className="big-value">{activeCount}</span>
              </div>
              <div>
                <span className="small-label">Activation mass</span>
                <span className="big-value">{activationMass.toFixed(2)}</span>
              </div>
              <div>
                <span className="small-label">Shortest path</span>
                <span className="big-value">{distance === null ? "∞" : distance}</span>
              </div>
            </div>
          </Panel>
        </div>
      </div>

      <Panel title="Iteration timeline" subtitle="Every state the recurrence passed through. Click one to inspect it.">
        <div className="timeline">
          {data.trajectory.map((step, index) => {
            const active = step[data.target] > data.epsilon;
            return (
              <button
                key={index}
                className={`time ${index === R ? "active-time" : ""} ${active ? "reached" : ""}`}
                onClick={() => setR(index)}
                title={`${step.filter((v) => v > data.epsilon).length} active nodes`}
              >
                h({index})
              </button>
            );
          })}
        </div>
        <p className="muted">
          Steps shaded green are those where the target is active. The boundary
          between grey and green is the shortest-path distance.
        </p>
      </Panel>

      <div className="why-ai">
        <h3>Why this is a question about AI, not just about graphs</h3>
        <p>
          A transformer that reasons in text has to write each intermediate step
          into its output before it can use it. The state above is the opposite
          case: the intermediate work lives in a fixed-size vector that gets
          rewritten in place, and nothing is emitted until the end. Recurrent
          latent reasoning is that idea applied to real models, where the
          coordinates are learned rather than standing for graph nodes.
        </p>
        <p>
          This page uses a hand-designed rule so the coordinates mean something
          you can check. Page 4 replaces that rule with a learned one and asks
          whether the same story survives.
        </p>
      </div>

      <EvidenceNote evidence={data.evidence} />

      <Takeaway>
        Recurrent depth R has a precise mechanical meaning: it is the number of
        edges information can travel. The target cannot activate before R reaches
        the shortest-path distance.
      </Takeaway>

      <div className="page-nav">
        <button className="primary" onClick={onNext}>
          Next: is this effect real, or just animation?
        </button>
      </div>
    </div>
  );
}
