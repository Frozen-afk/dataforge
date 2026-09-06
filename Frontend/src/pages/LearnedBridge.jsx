import React, { useEffect, useState } from "react";
import GraphView from "../components/GraphView.jsx";
import { EvidenceBadge, EvidenceNote } from "../components/Evidence.jsx";
import {
  DepthAxis, ErrorBanner, Loading, NextButton, PageHeader, Panel, Pill, Stat,
  Takeaway, Verdict,
} from "../components/Common.jsx";
import { learnedSweep, nodeCountsFor } from "../lib/lab.js";

const MAX_R = 10;

function explain({ result, distance, R, trainingMax, reachable }) {
  if (!result || result.unavailable) return null;

  if (!reachable) {
    return result.correct
      ? "Correct. The target genuinely is unreachable, and no number of updates can manufacture a path that does not exist."
      : "Wrong: the model claims a path exists when none does. More iterations cannot fix a false positive, because there is nothing to propagate.";
  }

  if (result.correct) {
    return distance > trainingMax
      ? `Correct at distance ${distance}, which is longer than anything in training. The same update block, applied more times, generalised past its training range.`
      : `Correct, and this distance was inside the training range.`;
  }

  if (R < distance) {
    return `R = ${R} is smaller than the ${distance}-edge path. The target node's state cannot yet have been influenced by the source, so this is insufficient computation rather than a broken model. Increase R.`;
  }

  return distance > trainingMax
    ? `R is large enough in principle, but distance ${distance} is outside the training range. More compute does not guarantee algorithmic extrapolation, and this is a learned-generalisation limit rather than a shortage of depth.`
    : "Wrong even though depth and distance were both inside the training regime. A small model on a hard instance simply gets some cases wrong.";
}

export default function LearnedBridge({ onNext }) {
  const [distance, setDistance] = useState(6);
  const [reachable, setReachable] = useState(true);
  const [nodes, setNodes] = useState(16);
  const [variant, setVariant] = useState(0);
  const [R, setR] = useState(3);

  const [sizes, setSizes] = useState([16]);
  const [sweep, setSweep] = useState(null);
  const [error, setError] = useState("");

  // Keep the size control honest: it only ever offers sizes the bank actually
  // holds for the chosen distance, so no selection can produce an empty state.
  useEffect(() => {
    let cancelled = false;
    nodeCountsFor(distance).then((available) => {
      if (cancelled || !available.length) return;
      setSizes(available);
      setNodes((current) => (available.includes(current) ? current : available[0]));
    });
    return () => { cancelled = true; };
  }, [distance]);

  // One call returns the whole depth sweep for one fixed graph, so moving R
  // cannot change the input as well as the depth.
  useEffect(() => {
    let cancelled = false;
    setError("");
    learnedSweep({ distance, reachable, nodes, variant, maxR: MAX_R })
      .then((result) => !cancelled && setSweep(result))
      .catch((err) => !cancelled && setError(err.message));
    return () => { cancelled = true; };
  }, [distance, reachable, nodes, variant]);

  if (error) {
    return (
      <div className="page">
        <PageHeader
          step="Section 4 of 8"
          question="What if the rule is learned instead of designed?"
          goal="Swap the hand-designed update for a trained one and ask whether the same story holds."
        />
        <ErrorBanner message={error}>
          <p className="muted">
            Sections 1 to 3 do not depend on the learned model.
          </p>
        </ErrorBanner>
      </div>
    );
  }

  if (!sweep) return <div className="page"><Loading label="Running the learned model" /></div>;

  const meta = sweep.meta || {};
  const trainingMax = meta.trainingMaxPathLength ?? 4;
  const trainingMaxR = meta.trainingMaxR ?? 4;
  const result = sweep.results.find((entry) => entry.R === R);
  const actualDistance = sweep.case.bfsDistance;
  const message = explain({ result, distance: actualDistance, R, trainingMax, reachable });
  const maxNorm = Math.max(1, ...(result?.targetStateNormPerStep || [1]));

  return (
    <div className="page">
      <PageHeader
        step="Section 4 of 8"
        question="What if the rule is learned instead of designed?"
        goal="Hold the weights fixed and vary only how many times the learned update is applied."
      />

      <div className="grid-2">
        <div>
          <h4>Sections 1 to 3</h4>
          <p className="mono" style={{ fontSize: 14 }}>
            h⁽⁰⁾ → h⁽¹⁾ → h⁽²⁾ → ⋯ → h⁽ᴿ⁾ → ŷ
          </p>
          <p className="muted">
            Hand-designed rule. One coordinate per node, so every number means
            something you can check.
          </p>
        </div>
        <div>
          <h4>This section</h4>
          <p className="mono" style={{ fontSize: 14, color: "var(--latent)" }}>
            z⁽⁰⁾ → z⁽¹⁾ → z⁽²⁾ → ⋯ → z⁽ᴿ⁾ → ŷ
          </p>
          <p className="muted">
            Same shape, trained rather than designed. The coordinates now mean
            nothing in particular, which is the honest situation.
          </p>
        </div>
      </div>

      <div className="controls">
        <label>
          Path length
          <select value={distance} onChange={(e) => setDistance(Number(e.target.value))}>
            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((d) => (
              <option key={d} value={d}>
                {d} {d <= trainingMax ? "— seen in training" : "— never trained on"}
              </option>
            ))}
          </select>
        </label>

        <label>
          Graph size
          <select value={nodes} onChange={(e) => setNodes(Number(e.target.value))}>
            {sizes.map((n) => <option key={n} value={n}>{n} nodes</option>)}
          </select>
        </label>

        <label className="inline">
          <input
            type="checkbox"
            checked={reachable}
            onChange={(e) => setReachable(e.target.checked)}
          />
          Target is reachable
        </label>

        <button className="btn ghost small" onClick={() => setVariant((v) => v + 1)}>
          Another graph
        </button>
      </div>

      <div className="box sunk" style={{ marginBottom: 22 }}>
        <div className="readout plain">
          <Stat
            label="Parameters"
            value={meta.numParameters?.toLocaleString() ?? "—"}
            note="the same count at every depth"
          />
          <Stat label="Trained on distances" value={`≤ ${trainingMax}`} />
          <Stat label="Trained at depths" value={`1 – ${trainingMaxR}`} />
          <Stat
            label="Checkpoint"
            value={meta.checkpointHash?.slice(0, 8) ?? "—"}
            note="frozen, seed 1"
          />
        </div>
        <p className="muted" style={{ marginTop: 12 }}>
          One update block, reused at every step: θ₀ = θ₁ = ⋯ = θ_R₋₁. Raising R
          adds applications of the same weights and nothing else.
        </p>
      </div>

      {!reachable && (
        <p className="footnote">
          This unreachable graph was cut from the reachable one beside it, so
          the two have the same node count and the same edge count. The answer
          cannot be read off the target's neighbourhood.
        </p>
      )}

      <div className="box">
        <DepthAxis
          id="bridge-depth"
          label="Inference depth R"
          value={R}
          min={1}
          max={MAX_R}
          onChange={setR}
          threshold={actualDistance}
          thresholdLabel="shortest path d(s,q)"
          hint={
            R > trainingMaxR
              ? `Past the maximum depth used in training (${trainingMaxR}). No new parameters exist here — only more applications of the same ones.`
              : "Inside the depth range used during training."
          }
        />
      </div>

      <div className="split" style={{ marginTop: 24 }}>
        <div className="stack">
          <Panel title="The exact mechanism, for reference">
            <GraphView
              graph={sweep.case.graph}
              source={sweep.case.source}
              target={sweep.case.target}
              state={result?.exact?.trajectory?.[R]}
              showValues={false}
              highlightPath={false}
            />
            <div className="readout plain" style={{ marginTop: 14 }}>
              <div>
                <span className="small-label">Says</span>
                <Pill tone={result?.exact?.estimate ? "yes" : "no"}>
                  {result?.exact?.estimate ? "reachable" : "not reachable"}
                </Pill>
              </div>
              <Stat label="Nodes reached" value={result?.exact?.activeNodeCount ?? "—"} />
              <Stat
                label="h(R)[q]"
                value={(result?.exact?.targetActivation ?? 0).toFixed(2)}
              />
            </div>
            <div className="evidence-note">
              <EvidenceBadge type="Live computation" />
              <span>Noisy-OR recurrence on the identical graph</span>
            </div>
          </Panel>
        </div>

        <div className="stack">
          <Panel title="The learned recurrent network">
            <span className="small-label">
              Target state magnitude ‖z⁽ʳ⁾_q‖ after each step
            </span>
            <div className="bars">
              {(result?.targetStateNormPerStep || []).map((norm, step) => (
                <div
                  key={step}
                  style={{ height: `${Math.max(3, (norm / maxNorm) * 100)}%` }}
                  title={`step ${step}: ${norm.toFixed(2)}`}
                />
              ))}
            </div>
            <div className="bars-axis" aria-hidden="true">
              {(result?.targetStateNormPerStep || []).map((_, step) => (
                <span key={step}>{step}</span>
              ))}
            </div>
            <span className="hint">
              The learned state has no per-coordinate meaning, so its magnitude
              is shown instead. That is what is honestly observable.
            </span>

            <span className="small-label" style={{ marginTop: 20 }}>
              P(reachable), read out after each step
            </span>
            <div className="bars prob">
              {(result?.probabilityPerStep || []).map((p, step) => (
                <div
                  key={step}
                  className={p > 0.5 ? "over" : ""}
                  style={{ height: `${Math.max(2, p * 100)}%` }}
                  title={`after ${step} update${step === 1 ? "" : "s"}: ${p.toFixed(3)}`}
                />
              ))}
            </div>
            <div className="bars-axis" aria-hidden="true">
              {(result?.probabilityPerStep || []).map((_, step) => (
                <span key={step}>{step}</span>
              ))}
            </div>
            <span className="hint">Bars above half the height cross the decision threshold.</span>

            <div className="readout plain" style={{ marginTop: 20 }}>
              <div>
                <span className="small-label">Predicts</span>
                <Pill tone={result?.prediction === 1 ? "yes" : "no"}>
                  {result?.prediction === 1 ? "reachable" : "not reachable"}
                </Pill>
              </div>
              <Stat
                label="Confidence"
                value={`${((result?.confidence ?? 0) * 100).toFixed(1)}%`}
              />
              <div>
                <span className="small-label">Correct</span>
                <Pill tone={result?.correct ? "yes" : "alarm"}>
                  {result?.correct ? "yes" : "no"}
                </Pill>
              </div>
            </div>
            <EvidenceNote evidence={sweep.evidence} />
          </Panel>
        </div>
      </div>

      {message && (
        <Verdict tone={result?.correct ? "good" : "bad"} title={result?.correct ? "Correct" : "Wrong"}>
          {message}
        </Verdict>
      )}

      <Panel
        title="Where each model commits"
        subtitle="The depth from which the answer becomes “reachable” and stays there."
      >
        <div className="readout">
          <Stat
            label="Exact mechanism commits at"
            value={sweep.exactFlipDepth ?? "never"}
            note="equal to the shortest path, by construction"
          />
          <Stat
            label="Learned model commits at"
            value={sweep.learnedFlipDepth ?? "never"}
            note="measured from the sweep above"
          />
          <div>
            <span className="small-label">Agreement</span>
            <Pill tone={sweep.flipDepthsAgree ? "yes" : "alarm"}>
              {sweep.flipDepthsAgree ? "same depth" : "different"}
            </Pill>
          </div>
        </div>
        <p className="muted" style={{ marginTop: 14 }}>
          When these agree on a path longer than the training range, the learned
          update has reproduced the mechanism's timing without ever having seen
          a path that long. When they disagree, the gap is the lesson. Try
          distance 9 or 10 and watch the agreement start to fail.
        </p>
      </Panel>

      <div className="keyline">
        The model has not gained parameters. It has only been allowed to compute
        for longer.
      </div>

      <Takeaway>
        A learned update block reused across steps behaves like the exact
        mechanism: depth buys reach. But it is an approximation, and the next
        section measures where that approximation holds and where it breaks.
      </Takeaway>

      <NextButton onNext={onNext}>Next: more computation, or more memorisation?</NextButton>
    </div>
  );
}
