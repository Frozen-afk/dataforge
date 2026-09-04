import React, { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

function buildPathGraph(pathLength, reachable) {
  const n = pathLength + 1;
  const edges = [];

  if (reachable) {
    for (let i = 0; i < pathLength; i++) edges.push([i, i + 1]);
  } else {
    for (let i = 1; i <= pathLength; i++) edges.push([i, i - 1]);
  }

  return { n, edges };
}

function EvidenceBadge({ type }) {
  const cls =
    type === "Live computation"
      ? "badge live"
      : type === "Precomputed result"
      ? "badge precomputed"
      : "badge other";

  return <span className={cls}>{type}</span>;
}

function NodeBars({ values, labels }) {
  const max = Math.max(1e-8, ...values);

  return (
    <div className="vector-list">
      {values.map((v, i) => (
        <div className="vector-row" key={i}>
          <div className="vector-label">{labels ? labels[i] : `n${i}`}</div>
          <div className="bar-bg">
            <div
              className="bar-fill"
              style={{ width: `${Math.min(100, (v / max) * 100)}%` }}
            />
          </div>
          <div className="vector-value">{v.toFixed(2)}</div>
        </div>
      ))}
    </div>
  );
}

function explainLearnedResult(d, pathLength, R, reachable) {
  if (!d) return null;

  if (d.correct) {
    if (pathLength > 4) {
      return `Correct on a path longer than training (length ${pathLength} > 4). The shared update block is being reused beyond its training depth.`;
    }
    return "Correct on this case.";
  }

  if (!reachable) {
    return "The model produced a prediction although no path exists. More iterations cannot create a missing path — this is a failure case.";
  }

  if (R < pathLength) {
    return `R = ${R} is smaller than the path length ${pathLength}. The learned model has insufficient inference depth, exactly like the exact mechanism.`;
  }

  if (pathLength > 4) {
    return `Path length ${pathLength} is outside the training regime (<= 4). More compute does not guarantee algorithmic extrapolation.`;
  }

  return "Learned dynamics limit: recurrence provides a computational mechanism, not guaranteed generalisation.";
}

export default function LearnedBridge() {
  const [pathLength, setPathLength] = useState(6);
  const [reachable, setReachable] = useState(true);
  const [R, setR] = useState(4);

  const [exact, setExact] = useState(null);
  const [learned, setLearned] = useState(null);
  const [learnedEvidence, setLearnedEvidence] = useState("Live computation");
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);

      const graph = buildPathGraph(pathLength, reachable);

      // Exact side (always live)
      try {
        const res = await fetch(`${API_BASE}/exact/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            graph,
            source: 0,
            target: pathLength,
            R,
          }),
        });
        if (res.ok && !cancelled) setExact(await res.json());
      } catch (e) {
        console.error(e);
      }

      // Learned side: try live inference first
      try {
        const res = await fetch(`${API_BASE}/learned/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pathLength, reachable, R }),
        });

        if (res.ok) {
          const data = await res.json();
          if (!cancelled) {
            setLearned(data);
            setLearnedEvidence(data.evidence?.evidenceType || "Live computation");
            setNote("");
          }
        } else {
          // Fallback to precomputed examples
          const exRes = await fetch(`${API_BASE}/learned/examples`);
          if (exRes.ok) {
            const exJson = await exRes.json();
            const wantedLabel = reachable ? 1 : 0;

            const match =
              (exJson.examples || []).find(
                (e) => e.pathLength === pathLength && e.label === wantedLabel && e.R === R
              ) ||
              (exJson.examples || []).find(
                (e) => e.pathLength === pathLength && e.label === wantedLabel
              );

            if (match && !cancelled) {
              setLearned(match);
              setLearnedEvidence("Precomputed result");
              setNote(
                `Live model unavailable. Showing precomputed example at R = ${match.R}.`
              );
            } else if (!cancelled) {
              setLearned(null);
              setNote("No learned result for this setting yet. Run the learned pipeline.");
            }
          }
        }
      } catch (e) {
        console.error(e);
        if (!cancelled) setLearned(null);
      }

      if (!cancelled) setLoading(false);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [pathLength, reachable, R]);

  const exactState = exact?.trajectory?.[Math.min(R, (exact?.trajectory?.length || 1) - 1)];

  return (
    <div className="page">
      <header className="page-header">
        <div className="tag">Page 4 — AI bridge</div>
        <h1>Now replace the rule with a learned one</h1>
        <p className="key-sentence">
          The model has not gained new parameters. It has only been allowed to compute longer.
        </p>
      </header>

      <div className="badge-row">
        <span className="badge shared">Same weights reused at every iteration</span>
        <span className="badge other">Training: paths ≤ 4 · R ≤ 4</span>
        <span className="badge other">Inference now: R = {R} · path length = {pathLength}</span>
      </div>

      <div className="control-card">
        <div className="control-row three">
          <label>
            Path length: <strong>{pathLength}</strong>
            <input
              type="range"
              min="1"
              max="10"
              value={pathLength}
              onChange={(e) => setPathLength(Number(e.target.value))}
            />
          </label>

          <label>
            Inference depth R: <strong>{R}</strong>
            <input
              type="range"
              min="1"
              max="10"
              value={R}
              onChange={(e) => setR(Number(e.target.value))}
            />
          </label>

          <label className="check">
            <input
              type="checkbox"
              checked={reachable}
              onChange={(e) => setReachable(e.target.checked)}
            />
            Target reachable (forward path)
          </label>
        </div>

        {pathLength > 4 && (
          <p className="muted">
            You are now outside the training regime (length &gt; 4). Increase R and watch
            whether extra computation helps.
          </p>
        )}
      </div>

      {loading && <div className="loading">Running exact + learned models...</div>}
      {note && <div className="loading">{note}</div>}

      <div className="bridge-grid">
        <section className="panel">
          <div className="panel-head">
            <h3>Exact mechanism h(r)</h3>
            <EvidenceBadge type="Live computation" />
          </div>

          <p className="muted">
            Hand-designed noisy-OR rule. Every coordinate has a clear meaning.
          </p>

          {exactState && <NodeBars values={exactState} labels={exactState.map((_, i) => `h[${i}]`)} />}

          {exact && (
            <div className="truth-grid small">
              <div>
                <div className="small-label">Estimate</div>
                <div className={exact.estimate ? "pill yes" : "pill no"}>
                  {exact.estimate ? "Target active" : "Target inactive"}
                </div>
              </div>
              <div>
                <div className="small-label">BFS distance</div>
                <div className="big-value">
                  {exact.bfsDistance === null ? "∞" : exact.bfsDistance}
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="panel">
          <div className="panel-head">
            <h3>Learned GNN z(r)</h3>
            <EvidenceBadge type={learnedEvidence} />
          </div>

          <p className="muted">
            Shared-weight recurrent GNN. Bars show per-node latent state norms after R steps.
          </p>

          {learned?.nodeStateNorms && (
            <NodeBars
              values={learned.nodeStateNorms}
              labels={learned.nodeStateNorms.map((_, i) => `z[${i}]`)}
            />
          )}

          {learned?.trajectoryNorms && !learned.nodeStateNorms && (
            <NodeBars values={learned.trajectoryNorms} labels={learned.trajectoryNorms.map((_, i) => `step ${i}`)} />
          )}

          {learned?.targetNormSeries && (
            <>
              <div className="small-label" style={{ marginTop: 12 }}>
                Target-node state norm per step
              </div>
              <div className="timeline">
                {learned.targetNormSeries.map((v, i) => (
                  <span className={i === R ? "time active-time" : "time"} key={i}>
                    z({i}) {v.toFixed(2)}
                  </span>
                ))}
              </div>
            </>
          )}

          {learned && (
            <div className="truth-grid small">
              <div>
                <div className="small-label">Prediction</div>
                <div className={learned.prediction === 1 ? "pill yes" : "pill no"}>
                  {learned.prediction === 1 ? "Reachable" : "Not reachable"}
                </div>
              </div>
              <div>
                <div className="small-label">Confidence</div>
                <div className="big-value">{(learned.confidence ?? 0).toFixed(2)}</div>
              </div>
            </div>
          )}

          <p className="explanation">
            {explainLearnedResult(learned, pathLength, R, reachable)}
          </p>
        </section>
      </div>
    </div>
  );
}
