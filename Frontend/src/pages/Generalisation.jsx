import React, { useEffect, useMemo, useState } from "react";
import Chart, { Legend, SERIES } from "../components/Chart.jsx";
import { EvidenceNote } from "../components/Evidence.jsx";
import {
  ErrorBanner, Loading, NextButton, PageHeader, Panel, Pill, Slider, Stat,
  Takeaway,
} from "../components/Common.jsx";
import { learnedExperiment, learnedExamples } from "../lib/lab.js";

const CATEGORY_LABELS = {
  "correct": "Correct, inside the training range",
  "correct-extrapolation": "Correct on a path longer than training",
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
    Promise.all([learnedExperiment(), learnedExamples()])
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
      curve,
      exact,
      learned: curve.map((e) => ({ x: e.R, y: e.accuracy.mean })),
      band: curve.map((e) => ({ x: e.R, low: e.accuracy.low, high: e.accuracy.high })),
      seen: curve.map((e) => ({ x: e.R, y: e.accuracySeenDistances.mean })),
      unseen: curve.map((e) => ({ x: e.R, y: e.accuracyUnseenDistances.mean })),
      exactLine: exact.map((e) => ({ x: e.R, y: e.accuracy.mean })),
      norm: curve.map((e) => ({ x: e.R, y: e.meanStateNorm.mean })),
      // A tenth of headroom above the widest interval. The point of this plot
      // is that the norm climbs and then flattens, and a curve drawn flush
      // against the ceiling reads as clipped rather than as saturating.
      maxNorm: 1.1 * Math.max(1, ...curve.map((e) => e.meanStateNorm.high)),
    };
  }, [experiment]);

  if (error) {
    return (
      <div className="page">
        <PageHeader
          step="Section 5 of 8"
          question="More computation, or more memorisation?"
          goal="Separate useful extra computation from overfitting and unstable iteration."
        />
        <ErrorBanner message={error} />
      </div>
    );
  }

  if (!experiment || !charts) return <div className="page"><Loading /></div>;

  const trainingMax = experiment.trainingMaxPathLength ?? 4;
  const atFixedR = charts.curve.find((e) => e.R === fixedR);
  const exactAtFixedR = charts.exact.find((e) => e.R === fixedR);

  const toPoints = (entry) =>
    entry
      ? Object.entries(entry.perDistance)
          .filter(([key]) => /^\d+$/.test(key))
          .map(([key, value]) => ({ x: Number(key), y: value.mean }))
          .sort((a, b) => a.x - b.x)
      : [];

  const grouped = {};
  for (const example of examples?.examples || []) {
    (grouped[example.category] ||= []).push(example);
  }

  const ablation = experiment.ablation || {};
  const last = charts.curve[charts.curve.length - 1];
  const peakSeen = charts.curve.reduce(
    (best, e) => (e.accuracySeenDistances.mean > best.accuracySeenDistances.mean ? e : best),
    charts.curve[0]
  );

  return (
    <div className="page">
      <PageHeader
        step="Section 5 of 8"
        question="More computation, or more memorisation?"
        goal="Measure where extra depth genuinely helps, and where it stops helping."
      />

      <div className="stat-row">
        <Stat label="Seeds" value={experiment.numSeeds} note="independent training runs" />
        <Stat label="Trained on distances" value={`≤ ${trainingMax}`} />
        <Stat
          label="Tested to distance"
          value={Math.max(...(experiment.testDistances || [10]))}
        />
        <Stat
          label="Parameters"
          value={experiment.config?.numParameters?.toLocaleString()}
          note="unchanged at every depth"
        />
      </div>

      <Panel
        title="Accuracy against inference depth"
        subtitle="Same frozen weights at every point; only R changes. Bands are 95% confidence intervals across seeds. Hover to read values off the curves."
      >
        <Chart
          xMin={1}
          xMax={10}
          xLabel="inference depth R"
          yLabel="accuracy"
          chanceLine={0.5}
          shade={{ from: 1, to: experiment.trainingMaxR ?? 4, label: "depths used in training" }}
          series={[
            { points: charts.learned, band: charts.band, color: SERIES.learned,
              width: 2.6, showDots: true, label: "learned, all distances" },
            { points: charts.seen, color: SERIES.seen, width: 1.8, label: "distances ≤ 4" },
            { points: charts.unseen, color: SERIES.unseen, width: 1.8, label: "distances ≥ 5" },
            { points: charts.exactLine, color: SERIES.exact, dash: "5 4", width: 1.6,
              label: "exact mechanism" },
          ]}
        />
        <Legend items={[
          { label: "Exact mechanism (reference)", color: SERIES.exact, dash: true },
          { label: "Learned, all distances", color: SERIES.learned },
          { label: "Learned, distances ≤ 4 (seen in training)", color: SERIES.seen },
          { label: "Learned, distances ≥ 5 (never trained on)", color: SERIES.unseen },
        ]} />

        <h4>Three things to read off it</h4>
        <ul className="plain">
          <li>
            The orange curve, distances of 5 and more, sits flat at chance while
            R is below 5. That is not the model failing. At those depths the
            answer is not yet knowable from the target node, so chance is the
            correct score. Read it as an average, not as a description of any
            one prediction: the model is not reporting uncertainty at those
            depths, it is answering confidently and being right about half the
            time. Section 4's per-graph sweep shows a single confidently wrong
            case.
          </li>
          <li>
            Past R = 4 the orange curve climbs to{" "}
            {last?.accuracyUnseenDistances.mean.toFixed(3)}. The model is
            solving paths longer than anything it trained on, purely by
            applying the same weights more times.
          </li>
          <li>
            The green curve, distances of 4 and less, peaks at{" "}
            {peakSeen?.accuracySeenDistances.mean.toFixed(3)} around R ={" "}
            {peakSeen?.R} and slips to{" "}
            {last?.accuracySeenDistances.mean.toFixed(3)} by R = 10. Extra depth
            is not free: it eventually costs accuracy on cases already solved.
          </li>
        </ul>

        <EvidenceNote evidence={experiment.evidence} />
      </Panel>

      <Panel
        title="Accuracy against path length, at one fixed depth"
        subtitle="This is what separates “not enough computation” from “cannot generalise”."
      >
        <div className="controls">
          <Slider
            id="gen-fixed-r"
            label="Fixed inference depth R"
            value={fixedR}
            min={1}
            max={10}
            onChange={setFixedR}
          />
        </div>
        <Chart
          xMin={1}
          xMax={10}
          xLabel="source-to-target distance"
          yLabel="accuracy"
          chanceLine={0.5}
          shade={{ from: 1, to: trainingMax, label: "training range" }}
          series={[
            { points: toPoints(atFixedR), color: SERIES.learned, width: 2.6,
              showDots: true, label: "learned model" },
            { points: toPoints(exactAtFixedR), color: SERIES.exact, dash: "5 4",
              width: 1.6, label: "exact mechanism" },
          ]}
        />
        <Legend items={[
          { label: "Exact mechanism", color: SERIES.exact, dash: true },
          { label: "Learned model", color: SERIES.learned },
        ]} />
        <p>
          At a fixed R, both curves fall off once the distance exceeds R. The
          exact mechanism drops to chance exactly at distance R + 1, because it
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
          yMax={charts.maxNorm}
          xLabel="inference depth R"
          yLabel="mean ‖z‖"
          yFormat={(v) => v.toFixed(1)}
          hoverFormat={(v) => v.toFixed(2)}
          series={[{ points: charts.norm, color: SERIES.latent, width: 2.4,
                     showDots: true, label: "mean state norm" }]}
        />
        <p>
          The latent state grows with depth and then saturates rather than
          diverging, which is what the gated update exists to ensure. Two
          observations sit side by side here: the norm plateaus above R ≈ 7,
          and accuracy on already-solved cases erodes over the same range. A
          plausible reading is that once the state stops changing meaningfully,
          further iterations add drift rather than information — but this
          experiment measures the two curves, it does not establish that the
          first causes the second. Treat the mechanism as a hypothesis. Testing
          it would mean intervening on the norm, for example by rescaling the
          state between steps, and checking whether the erosion moves with it.
          That experiment is not in this lab.
        </p>
        <p className="muted">
          Published work on looped language models reports the same failure mode
          at a much larger scale, where accuracy peaks at some depth and then
          degrades. This lab shows a small version of it rather than claiming to
          reproduce that result.
        </p>
      </Panel>

      {ablation.unsharedWeights && (
        <Panel
          title="Why this needs shared weights"
          subtitle="Same architecture and same data, but a separate update block per step."
        >
          <div className="duo">
            <div>
              <span className="head">Shared weights</span>
              <span className="verdict">
                {ablation.sharedWeights?.numParameters?.toLocaleString()} parameters
              </span>
              <span className="note">Maximum inference depth: unbounded</span>
              <p className="muted" style={{ marginTop: 12 }}>
                One block, applied as many times as you like. Depth becomes a
                runtime choice that nobody had to decide during training.
              </p>
            </div>
            <div>
              <span className="head">Unshared weights</span>
              <span className="verdict">
                {ablation.unsharedWeights?.numParameters?.toLocaleString()} parameters
              </span>
              <span className="note">
                Maximum inference depth: {ablation.unsharedWeights?.maxInferenceDepth}
              </span>
              <p className="muted" style={{ marginTop: 12 }}>
                Depth is fixed when training ends. Running deeper is not a worse
                result. It is not a possible one, because those layers were
                never created.
              </p>
            </div>
          </div>
          <div className="keyline">
            This is the difference between recurrence and simply stacking more
            layers. Only the shared-weight model turns inference depth into a
            dial you can move after training.
          </div>
        </Panel>
      )}

      {experiment.aggregationAblation && (
        <AggregationAblation ablation={experiment.aggregationAblation} />
      )}

      <Panel
        title="Successes and failures, side by side"
        subtitle="Individual cases from the frozen checkpoints, grouped by what actually happened."
      >
        <div className="grid-auto">
          {Object.entries(CATEGORY_LABELS).map(([key, label]) => {
            const items = grouped[key] || [];
            if (!items.length) return null;
            const example = items[0];
            const good = key.startsWith("correct");

            return (
              <div className="box" key={key}
                   style={{ borderLeftWidth: 3,
                            borderLeftColor: good ? "var(--affirm)" : "var(--deny)" }}>
                <div style={{ display: "flex", justifyContent: "space-between",
                              alignItems: "baseline", marginBottom: 8 }}>
                  <Pill tone={good ? "yes" : "alarm"}>{good ? "success" : "failure"}</Pill>
                  <span className="muted">{items.length} case{items.length === 1 ? "" : "s"}</span>
                </div>
                <h4 style={{ margin: "0 0 10px" }}>{label}</h4>
                <table>
                  <tbody>
                    <tr><td>distance</td><td className="num">{example.distance ?? "none"}</td></tr>
                    <tr><td>depth R</td><td className="num">{example.R}</td></tr>
                    <tr><td>truth</td><td className="num">{example.label === 1 ? "reachable" : "not"}</td></tr>
                    <tr><td>predicted</td><td className="num">{example.prediction === 1 ? "reachable" : "not"}</td></tr>
                    <tr><td>confidence</td><td className="num">{(example.confidence * 100).toFixed(1)}%</td></tr>
                    <tr><td>graph</td><td className="num">{example.numNodes}n / {example.numEdges}e</td></tr>
                  </tbody>
                </table>
              </div>
            );
          })}
        </div>
        <EvidenceNote evidence={examples?.evidence} />
      </Panel>

      <Panel title="Seed-to-seed variability">
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {(experiment.seeds || []).map((seed) => (
            <span className="badge synthetic" key={seed}>
              seed {seed} · {experiment.checkpointHashes?.[String(seed)]?.slice(0, 8)}
            </span>
          ))}
        </div>
        <p className="muted" style={{ marginTop: 12 }}>
          Every curve above is the mean over these seeds, with 95% confidence
          intervals from a Student t critical value. The intervals are narrow,
          which means the depth effect is a property of the setup rather than of
          one lucky initialisation.
        </p>
      </Panel>

      <Takeaway>
        Extra depth genuinely extends what this model can solve, including path
        lengths it never trained on. It is not unlimited: gains flatten, the
        state saturates, and accuracy on easy cases erodes slightly at large R.
      </Takeaway>

      <NextButton onNext={onNext}>Next: where does this appear in a real system?</NextButton>
    </div>
  );
}

// The aggregator is chosen by an argument about the task -- reachability is a
// logical OR over incoming neighbours, and max is its differentiable analogue.
// An argument is not evidence, so the alternatives were trained too. The result
// is a negative one, and this component says so rather than dressing a
// three-thousandth of a point up as a finding.
function AggregationAblation({ ablation }) {
  const [depth, setDepth] = useState(10);

  const rows = ablation.variants.map((variant) => {
    const entry = variant.depthCurve.find((e) => e.R === depth);
    return {
      name: variant.aggregation,
      mean: entry?.accuracy.mean,
      low: entry?.accuracy.low,
      high: entry?.accuracy.high,
    };
  });

  const best = rows.reduce((a, b) => (b.mean > a.mean ? b : a), rows[0]);
  // Any interval containing the leader's mean is a variant this experiment
  // cannot rule out.
  const indistinguishable = rows.filter(
    (row) => row.low <= best.mean && best.mean <= row.high
  );
  const separated = indistinguishable.length === 1;

  return (
    <Panel
      title="Does the choice of aggregator matter?"
      subtitle="Same architecture, same data, same seeds. Only the way a node combines messages from its incoming neighbours changes."
    >
      <p>
        Max is the default because reachability is a logical OR over incoming
        neighbours, and max is its differentiable analogue — the same role the
        noisy-OR plays in the exact mechanism. That is an argument about the
        task, not a measurement, so the alternatives were trained as well.
      </p>

      <div className="controls">
        <Slider
          id="agg-depth"
          label="Read the comparison at depth R"
          value={depth}
          min={1}
          max={10}
          onChange={setDepth}
        />
      </div>

      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>Aggregator</th>
              <th className="num">accuracy at R = {depth}</th>
              <th className="num">95% interval</th>
              <th>reading</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name} className={row.name === "max" ? "current" : ""}>
                <td>
                  {row.name}
                  {row.name === ablation.default && " — the default"}
                </td>
                <td className="num">{row.mean?.toFixed(3)}</td>
                <td className="num">
                  [{row.low?.toFixed(3)}, {row.high?.toFixed(3)}]
                </td>
                <td className="muted">
                  {row.low <= best.mean && best.mean <= row.high
                    ? "cannot be separated from the best"
                    : "separated from the best"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className={`verdict-bar ${separated ? "good" : "wait"}`}>
        <strong>
          {separated
            ? `At R = ${depth}, ${best.name} leads on a crude interval screen.`
            : `At R = ${depth}, this experiment does not separate them.`}
        </strong>
        <p>
          {separated ? (
            <>
              No other interval covers the leader's mean. That is a crude
              screen, not a significance test: with{" "}
              {ablation.comparedOnSeeds?.length ?? 2} seeds the intervals are
              wide and this comparison is unpaired, so read the ordering as
              suggestive and worth more seeds, not as established.
            </>
          ) : (
            <>
              Every interval contains the best mean, on{" "}
              {ablation.comparedOnSeeds?.length ?? 2} seeds. Max stays the
              default because of the argument about the task, not because this
              measurement backs it. Reporting the small gap as a win would be
              reading noise as a result.
            </>
          )}
        </p>
      </div>

      <p className="muted">
        Compared on seeds {ablation.comparedOnSeeds?.join(", ")}. The headline
        curves above use five seeds, so the max column here is recomputed on
        just these two — otherwise the seed count would sit inside the
        difference this table is meant to be about.
      </p>
    </Panel>
  );
}
