import React, { useEffect, useState } from "react";
import GraphView from "../components/GraphView.jsx";
import { EvidenceBadge, EvidenceNote } from "../components/Evidence.jsx";
import {
  Equation, ErrorBanner, Loading, PageHeader, Panel, Pill, Slider, Takeaway,
} from "../components/Common.jsx";
import { api } from "../lib/api.js";

const MAX_R = 10;

function explainOutcome({ result, distance, R, trainingMax, reachable }) {
  if (!result || result.unavailable) return null;

  const correct = result.correct;

  if (!reachable) {
    return correct
      ? "Correct. The target genuinely is unreachable, and no number of updates can manufacture a path that does not exist."
      : "Wrong: the model claims a path exists when none does. This is a false positive, and more iterations cannot fix it — there is nothing to propagate.";
  }

  if (correct) {
    return distance > trainingMax
      ? `Correct at distance ${distance}, which is longer than anything in training (≤ ${trainingMax}). The same update block, applied more times, generalised past its training range.`
      : `Correct, and this distance was inside the training range (≤ ${trainingMax}).`;
  }

  if (R < distance) {
    return `R = ${R} is smaller than the ${distance}-edge path. The target node's state cannot yet have been influenced by the source, so this is insufficient computation, not a broken model. Increase R.`;
  }

  return distance > trainingMax
    ? `R is large enough in principle, but distance ${distance} is outside the training range (≤ ${trainingMax}). More compute does not guarantee algorithmic extrapolation — this is a learned-generalisation limit.`
    : "Wrong even though depth and distance were both inside the training regime. A small model on a hard instance simply gets some cases wrong.";
}

export default function LearnedBridge({ onNext, status }) {
  const [distance, setDistance] = useState(6);
  const [reachable, setReachable] = useState(true);
  const [nodes, setNodes] = useState(16);
  const [caseSeed, setCaseSeed] = useState(0);
  const [R, setR] = useState(3);

  const [sweep, setSweep] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // One request returns the whole depth sweep for one fixed graph, so moving
  // the R slider cannot accidentally change the input as well as the depth.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");

    api
      .learnedSweep({ distance, reachable, nodes, caseSeed, maxR: MAX_R })
      .then((result) => !cancelled && setSweep(result))
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false));

    return () => { cancelled = true; };
  }, [distance, reachable, nodes, caseSeed]);

  if (error) {
    return (
      <div className="page">
        <PageHeader step="4" question="Now replace the rule with a learned one"
                    goal="Swap the hand-designed update for a trained one and ask whether the same story holds." />
        <ErrorBanner message={error}>
          <p>
            Pages 1 to 3 do not depend on the learned model. To enable this page,
            train a checkpoint: <code>python reproduce.py</code> from the Backend
            directory.
          </p>
        </ErrorBanner>
      </div>
    );
  }

  if (!sweep) return <div className="page"><Loading label="Running the learned model..." /></div>;

  const meta = sweep.meta || {};
  const trainingMax = meta.trainingMaxPathLength ?? 4;
  const result = sweep.results.find((r) => r.R === R);
  const actualDistance = sweep.case.bfsDistance;
  const exactState = result?.exact?.trajectory?.[R];

  const explanation = explainOutcome({
    result, distance: actualDistance, R, trainingMax, reachable,
  });

  return (
    <div className="page">
      <PageHeader
        step="4"
        question="Now replace the rule with a learned one"
        goal="Hold the weights fixed and vary only how many times the learned update is applied."
      />

      <div className="bridge-chain">
        <div className="chain exact">
          <span className="chain-label">Mechanism layer, pages 1–3</span>
          <code>h⁽⁰⁾ → h⁽¹⁾ → h⁽²⁾ → ⋯ → h⁽ᴿ⁾ → ŷ</code>
          <span className="chain-note">hand-designed rule, interpretable coordinates</span>
        </div>
        <div className="chain-arrow">same shape, learned instead of designed</div>
        <div className="chain learned">
          <span className="chain-label">AI layer, this page</span>
          <code>z⁽⁰⁾ → z⁽¹⁾ → z⁽²⁾ → ⋯ → z⁽ᴿ⁾ → ŷ</code>
          <span className="chain-note">trained update block, opaque coordinates</span>
        </div>
      </div>

      <div className="control-card row wrap">
        <label>
          Path length
          <select value={distance} onChange={(e) => setDistance(Number(e.target.value))}>
            {[1,2,3,4,5,6,7,8,9,10].map((d) => (
              <option key={d} value={d}>
                {d} {d <= trainingMax ? "(seen in training)" : "(never trained on)"}
              </option>
            ))}
          </select>
        </label>

        <label className="checkbox">
          <input type="checkbox" checked={reachable}
                 onChange={(e) => setReachable(e.target.checked)} />
          Target is reachable
        </label>

        <label>
          Graph size
          <select value={nodes} onChange={(e) => setNodes(Number(e.target.value))}>
            {[12, 16, 20, 24].map((n) => <option key={n} value={n}>{n} nodes</option>)}
          </select>
        </label>

        <button className="secondary" onClick={() => setCaseSeed((s) => s + 1)}>
          New graph
        </button>
      </div>

      <div className="param-badge">
        <div>
          <strong>Same weights reused at every iteration.</strong>
          <span>
            {meta.numParameters?.toLocaleString()} parameters, one update block,
            θ₀ = θ₁ = ⋯ = θ_R₋₁. Checkpoint {meta.checkpointHash}, frozen.
          </span>
        </div>
        <div className="training-depth">
          <span className="small-label">Trained on distances</span>
          <span className="big-value">≤ {trainingMax}</span>
        </div>
      </div>

      <Slider
        id="bridge-depth"
        label="Inference depth R"
        value={R}
        min={1}
        max={MAX_R}
        onChange={setR}
        hint={
          R > (meta.trainingMaxR ?? 4)
            ? `Beyond the maximum depth used in training (${meta.trainingMaxR ?? 4}). The model has no new parameters here — only more applications of the same ones.`
            : "Inside the depth range used during training."
        }
      />

      {loading && <Loading label="Recomputing..." />}

      <div className="split">
        <Panel title="Exact mechanism" className="side exact-side">
          <GraphView
            graph={sweep.case.graph}
            source={sweep.case.source}
            target={sweep.case.target}
            state={exactState}
            height={340}
            showValues={false}
          />
          <div className="readout tight">
            <div>
              <span className="small-label">Says</span>
              <Pill tone={result?.exact?.estimate ? "yes" : "no"}>
                {result?.exact?.estimate ? "reachable" : "not reachable"}
              </Pill>
            </div>
            <div>
              <span className="small-label">Active nodes</span>
              <span className="big-value">{result?.exact?.activeNodeCount ?? "–"}</span>
            </div>
            <div>
              <span className="small-label">h(R)[q]</span>
              <span className="big-value">
                {(result?.exact?.targetActivation ?? 0).toFixed(2)}
              </span>
            </div>
          </div>
          <EvidenceBadge type="Live computation" />
        </Panel>

        <Panel title="Learned recurrent GNN" className="side learned-side">
          <div className="latent-strip">
            <span className="small-label">
              Target state magnitude ‖z⁽ʳ⁾_q‖ at each step
            </span>
            <div className="latent-bars">
              {(result?.targetStateNormPerStep || []).map((norm, index) => {
                const max = Math.max(...(result.targetStateNormPerStep || [1]));
                return (
                  <div className="latent-bar" key={index} title={`step ${index}: ${norm.toFixed(2)}`}>
                    <div
                      className="latent-fill"
                      style={{ height: `${Math.max(3, (norm / max) * 100)}%` }}
                    />
                    <span>{index}</span>
                  </div>
                );
              })}
            </div>
            <span className="hint">
              The learned state has no per-coordinate meaning. Its magnitude is
              shown because that is what is honestly observable.
            </span>
          </div>

          <div className="prob-track">
            <span className="small-label">
              P(reachable) read out after each step
            </span>
            <div className="prob-bars">
              {(result?.probabilityPerStep || []).map((p, index) => (
                <div
                  key={index}
                  className={`prob-bar ${p > 0.5 ? "yes" : "no"}`}
                  style={{ height: `${Math.max(2, p * 100)}%` }}
                  title={`after ${index} update${index === 1 ? "" : "s"}: ${p.toFixed(3)}`}
                />
              ))}
            </div>
            <span className="hint">50% line is the decision threshold.</span>
          </div>

          <div className="readout tight">
            <div>
              <span className="small-label">Predicts</span>
              <Pill tone={result?.prediction === 1 ? "yes" : "no"}>
                {result?.prediction === 1 ? "reachable" : "not reachable"}
              </Pill>
            </div>
            <div>
              <span className="small-label">Confidence</span>
              <span className="big-value">
                {((result?.confidence ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="small-label">Correct</span>
              <Pill tone={result?.correct ? "yes" : "no"}>
                {result?.correct ? "yes" : "no"}
              </Pill>
            </div>
          </div>
          <EvidenceNote evidence={sweep.evidence} />
        </Panel>
      </div>

      {explanation && (
        <div className={`outcome ${result?.correct ? "good" : "bad"}`}>
          <strong>{result?.correct ? "Correct" : "Wrong"}</strong>
          <p>{explanation}</p>
        </div>
      )}

      <Panel title="Where each model flips" subtitle="The depth at which the answer changes to 'reachable' and stays there.">
        <div className="flip-grid">
          <div>
            <span className="small-label">Exact mechanism flips at</span>
            <span className="big-value">
              {sweep.exactFlipDepth ?? "never"}
            </span>
          </div>
          <div>
            <span className="small-label">Learned model flips at</span>
            <span className="big-value">
              {sweep.learnedFlipDepth ?? "never"}
            </span>
          </div>
          <div>
            <span className="small-label">Agreement</span>
            <Pill tone={sweep.flipDepthsAgree ? "yes" : "no"}>
              {sweep.flipDepthsAgree ? "same depth" : "different"}
            </Pill>
          </div>
        </div>
        <p className="muted">
          When these agree on a path longer than the training range, the learned
          update has reproduced the mechanism&rsquo;s timing without ever having
          seen a path that long. When they disagree, the gap is the lesson.
        </p>
      </Panel>

      <div className="key-sentence">
        The model has not gained new parameters. It has only been allowed to
        compute longer.
      </div>

      <Takeaway>
        A learned update block reused across steps behaves like the exact
        mechanism: depth buys reach. But it is an approximation, and the next
        page measures where that approximation holds and where it breaks.
      </Takeaway>

      <div className="page-nav">
        <button className="primary" onClick={onNext}>
          Next: more computation, or more memorisation?
        </button>
      </div>
    </div>
  );
}
