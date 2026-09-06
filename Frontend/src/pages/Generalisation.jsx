import React, { useEffect, useMemo, useState } from "react";
import Chart, { Legend } from "../components/Chart.jsx";
import { EvidenceNote } from "../components/Evidence.jsx";
import {
  ErrorBanner, Loading, PageHeader, Panel, Pill, Slider, Takeaway,
} from "../components/Common.jsx";
import { api } from "../lib/api.js";

const CATEGORY_LABELS = {
  "correct": "Correct, inside training range",
  "correct-extrapolation": "Correct on an unseen longer path",
  "insufficient-depth": "Wrong: R smaller than the path",
  "extrapolation-failure": "Wrong: path longer than training",
  "in-distribution-error": "Wrong inside the training range",
  "false-positive": "Wrong: claimed a path that does not exist",
};

export default function Generalisation({ onNext }) {
  const [experiment, setExperiment] = useState(null);
  const [examples, setExamples] = useState(null);
  const [error, setError] = useState("");
  const [fixedR, setFixedR] = useState(6);

  useEffect(() => {
    let cancelled = false;

    Promise.all([api.learnedExperiment(), api.learnedExamples()])
      .then(([exp, ex]) => {
        if (cancelled) return;
        setExperiment(exp);
        setExamples(ex);
      })
      .catch((err) => !cancelled && setError(err.message));

    return () => { cancelled = true; };
  }, []);

  const charts = useMemo(() => {
    if (!experiment) return null;

    const curve = experiment.depthCurve || [];
    const exact = experiment.exactReference || [];

    return {
      depth: {
        learned: curve.map((e) => ({ x: e.R, y: e.accuracy.mean })),
        learnedBand: curve.map((e) => ({
          x: e.R, low: e.accuracy.low, high: e.accuracy.high,
        })),
        seen: curve.map((e) => ({ x: e.R, y: e.accuracySeenDistances.mean })),
        unseen: curve.map((e) => ({ x: e.R, y: e.accuracyUnseenDistances.mean })),
        exact: exact.map((e) => ({ x: e.R, y: e.accuracy.mean })),
        norm: curve.map((e) => ({ x: e.R, y: e.meanStateNorm.mean })),
        maxNorm: Math.max(1, ...curve.map((e) => e.meanStateNorm.high)),
      },
      curve,
      exact,
    };
  }, [experiment]);

  if (error) {
    return (
      <div className="page">
        <PageHeader step="5" question="More computation, or more memorisation?"
                    goal="Separate useful extra computation from overfitting and unstable iteration." />
        <ErrorBanner message={error}>
          <p>Run <code>python reproduce.py</code> in the Backend directory to generate these results.</p>
        </ErrorBanner>
      </div>
    );
  }

  if (!experiment || !charts) return <div className="page"><Loading /></div>;

  const trainingMax = experiment.trainingMaxPathLength ?? 4;
  const atFixedR = charts.curve.find((e) => e.R === fixedR);
  const exactAtFixedR = charts.exact.find((e) => e.R === fixedR);

  const perDistance = atFixedR
    ? Object.entries(atFixedR.perDistance)
        .filter(([key]) => /^\d+$/.test(key))
        .map(([key, value]) => ({ x: Number(key), y: value.mean }))
        .sort((a, b) => a.x - b.x)
    : [];

  const exactPerDistance = exactAtFixedR
    ? Object.entries(exactAtFixedR.perDistance)
        .filter(([key]) => /^\d+$/.test(key))
        .map(([key, value]) => ({ x: Number(key), y: value.mean }))
        .sort((a, b) => a.x - b.x)
    : [];

  const grouped = {};
  for (const example of examples?.examples || []) {
    (grouped[example.category] ||= []).push(example);
  }

  const ablation = experiment.ablation || {};

  return (
    <div className="page">
      <PageHeader
        step="5"
        question="More computation, or more memorisation?"
        goal="Measure where extra depth genuinely helps, and where it stops helping."
      />

      <div className="stat-row">
        <div className="stat">
          <span className="small-label">Seeds</span>
          <span className="big-value">{experiment.numSeeds}</span>
        </div>
        <div className="stat">
          <span className="small-label">Trained on distances</span>
          <span className="big-value">≤ {trainingMax}</span>
        </div>
        <div className="stat">
          <span className="small-label">Tested to distance</span>
          <span className="big-value">{Math.max(...(experiment.testDistances || [10]))}</span>
        </div>
        <div className="stat">
          <span className="small-label">Parameters</span>
          <span className="big-value">
            {experiment.config?.numParameters?.toLocaleString()}
          </span>
        </div>
      </div>

      <Panel
        title="Accuracy versus inference depth"
        subtitle="Same frozen weights at every point. Only R changes. Bands are 95% confidence intervals across seeds."
      >
        <Chart
          xMin={1}
          xMax={10}
          xLabel="inference depth R"
          yLabel="accuracy"
          chanceLine={0.5}
          shade={{ from: 1, to: experiment.trainingMaxR ?? 4, label: "depths used in training" }}
          series={[
            { points: charts.depth.exact, color: "#0f172a", dash: "5 4", width: 2 },
            { points: charts.depth.learned, band: charts.depth.learnedBand, color: "#2563eb", width: 2.6, showDots: true },
            { points: charts.depth.seen, color: "#16a34a", width: 1.8 },
            { points: charts.depth.unseen, color: "#ea580c", width: 1.8 },
          ]}
        />
        <Legend items={[
          { label: "Exact mechanism (reference)", color: "#0f172a", dash: true },
          { label: "Learned, all distances", color: "#2563eb" },
          { label: "Learned, distances ≤ 4 (seen)", color: "#16a34a" },
          { label: "Learned, distances ≥ 5 (never trained on)", color: "#ea580c" },
        ]} />

        <div className="reading">
          <h4>How to read this</h4>
          <ul>
            <li>
              The orange curve sits flat at chance while R is below 5. That is
              not the model failing — at those depths the answer is not yet
              knowable from the target node, so chance is the correct score.
            </li>
            <li>
              Past R = 4 the orange curve climbs. The model is solving paths
              longer than anything it trained on, purely by iterating the same
              block more times.
            </li>
            <li>
              The green curve peaks and then dips slightly at large R. Extra
              depth is not free: it eventually costs accuracy on cases the model
              had already solved.
            </li>
          </ul>
        </div>

        <EvidenceNote evidence={experiment.evidence} />
      </Panel>

      <Panel
        title="Accuracy versus path length at one fixed depth"
        subtitle="This separates 'not enough computation' from 'cannot generalise'."
      >
        <Slider id="gen-fixed-r" label="Fixed inference depth R" value={fixedR}
                min={1} max={10} onChange={setFixedR} />
        <Chart
          xMin={1}
          xMax={10}
          xLabel="source-to-target distance"
          yLabel="accuracy"
          chanceLine={0.5}
          shade={{ from: 1, to: trainingMax, label: "training range" }}
          series={[
            { points: exactPerDistance, color: "#0f172a", dash: "5 4" },
            { points: perDistance, color: "#2563eb", width: 2.6, showDots: true },
          ]}
        />
        <Legend items={[
          { label: "Exact mechanism", color: "#0f172a", dash: true },
          { label: "Learned model", color: "#2563eb" },
        ]} />
        <p className="muted">
          At a fixed R, both curves fall off once the distance exceeds R. The
          exact mechanism drops to 0.5 exactly at distance R + 1, because it
          scores every unreachable pair correctly and every too-far reachable
          pair wrong. If the learned curve tracks it, the failure is a shortage
          of computation. If the learned curve falls off earlier, that gap is a
          limit of what was learned.
        </p>
      </Panel>

      <Panel
        title="Does the state stay stable as depth grows?"
        subtitle="Recurrence is a compute axis with limits, and this is where they show."
      >
        <Chart
          xMin={1}
          xMax={10}
          yMax={charts.depth.maxNorm}
          xLabel="inference depth R"
          yLabel="mean ‖z‖"
          yFormat={(v) => v.toFixed(1)}
          series={[{ points: charts.depth.norm, color: "#7c3aed", width: 2.4, showDots: true }]}
        />
        <p className="muted">
          The latent state grows with depth and then saturates rather than
          diverging, which is what the gated update is there to ensure. Saturation
          is also why accuracy on already-solved cases erodes at large R: once the
          state stops changing meaningfully, additional iterations add drift
          rather than information.
        </p>
      </Panel>

      {ablation.unsharedWeights && (
        <Panel
          title="Ablation: why this needs shared weights"
          subtitle="Same architecture and same data, but a separate update block per step."
        >
          <div className="ablation-grid">
            <div>
              <h4>Shared weights</h4>
              <div className="ab-stat">
                <span className="small-label">Parameters</span>
                <span className="big-value">
                  {ablation.sharedWeights?.numParameters?.toLocaleString()}
                </span>
              </div>
              <div className="ab-stat">
                <span className="small-label">Maximum inference depth</span>
                <span className="big-value">unbounded</span>
              </div>
              <p className="muted">
                One block, applied as many times as you like. Depth is a runtime
                choice.
              </p>
            </div>
            <div>
              <h4>Unshared weights</h4>
              <div className="ab-stat">
                <span className="small-label">Parameters</span>
                <span className="big-value">
                  {ablation.unsharedWeights?.numParameters?.toLocaleString()}
                </span>
              </div>
              <div className="ab-stat">
                <span className="small-label">Maximum inference depth</span>
                <span className="big-value">
                  {ablation.unsharedWeights?.maxInferenceDepth}
                </span>
              </div>
              <p className="muted">
                Depth is fixed when training ends. Running deeper is not a worse
                result — it is not a possible one, because those layers were never
                created.
              </p>
            </div>
          </div>
          <p className="callout">
            This is the difference between recurrence and simply stacking more
            layers. Only the shared-weight model turns inference depth into a
            dial you can move after training.
          </p>
        </Panel>
      )}

      <Panel title="Successes and failures, side by side"
             subtitle="Individual cases from the frozen checkpoints, grouped by what actually happened.">
        <div className="example-grid">
          {Object.entries(CATEGORY_LABELS).map(([key, label]) => {
            const items = grouped[key] || [];
            if (!items.length) return null;
            const example = items[0];
            const good = key.startsWith("correct");

            return (
              <div className={`example-card ${good ? "good" : "bad"}`} key={key}>
                <div className="example-head">
                  <Pill tone={good ? "yes" : "no"}>{good ? "success" : "failure"}</Pill>
                  <span className="example-count">{items.length} case{items.length === 1 ? "" : "s"}</span>
                </div>
                <h4>{label}</h4>
                <dl>
                  <div><dt>distance</dt><dd>{example.distance ?? "unreachable"}</dd></div>
                  <div><dt>depth R</dt><dd>{example.R}</dd></div>
                  <div><dt>truth</dt><dd>{example.label === 1 ? "reachable" : "unreachable"}</dd></div>
                  <div><dt>predicted</dt><dd>{example.prediction === 1 ? "reachable" : "unreachable"}</dd></div>
                  <div><dt>confidence</dt><dd>{(example.confidence * 100).toFixed(1)}%</dd></div>
                  <div><dt>graph</dt><dd>{example.numNodes} nodes, {example.numEdges} edges</dd></div>
                </dl>
              </div>
            );
          })}
        </div>
        <EvidenceNote evidence={examples?.evidence} />
      </Panel>

      <Panel title="Seed-to-seed variability">
        <div className="seed-chips">
          {(experiment.seeds || []).map((seed) => (
            <span className="seed-chip" key={seed}>
              seed {seed}
              <code>{experiment.checkpointHashes?.[String(seed)]?.slice(0, 8)}</code>
            </span>
          ))}
        </div>
        <p className="muted">
          Every curve above is the mean over these seeds, with 95% confidence
          intervals. The intervals are narrow, which means the depth effect is a
          property of the setup rather than of one lucky initialisation.
        </p>
      </Panel>

      <Takeaway>
        Extra depth genuinely extends what this model can solve, including path
        lengths it never trained on. It is not unlimited: gains flatten, the state
        saturates, and accuracy on easy cases erodes slightly at large R.
      </Takeaway>

      <div className="page-nav">
        <button className="primary" onClick={onNext}>
          Next: where does this appear in real AI?
        </button>
      </div>
    </div>
  );
}
