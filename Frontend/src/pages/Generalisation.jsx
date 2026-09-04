import React, { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

function LineChart({ series, xMax, xLabel, yLabel, trainingShade }) {
  const W = 640;
  const H = 320;
  const L = 48;
  const B = 44;
  const T = 18;
  const R = 14;

  const xToPx = (x) => L + (x / xMax) * (W - L - R);
  const yToPx = (y) => H - B - Math.max(0, Math.min(1, y)) * (H - B - T);

  const yTicks = [0, 0.25, 0.5, 0.75, 1.0];
  const xTicks = Array.from({ length: xMax + 1 }, (_, i) => i);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      {yTicks.map((t) => (
        <g key={t}>
          <line x1={L} y1={yToPx(t)} x2={W - R} y2={yToPx(t)} stroke="#e2e8f0" />
          <text x={L - 8} y={yToPx(t) + 4} textAnchor="end" fontSize="11" fill="#64748b">
            {t.toFixed(2)}
          </text>
        </g>
      ))}

      {xTicks.map((t) => (
        <text key={t} x={xToPx(t)} y={H - B + 18} textAnchor="middle" fontSize="11" fill="#64748b">
          {t}
        </text>
      ))}

      {trainingShade && (
        <g>
          <rect
            x={xToPx(trainingShade.from)}
            y={T}
            width={Math.max(0, xToPx(trainingShade.to) - xToPx(trainingShade.from))}
            height={H - B - T}
            fill="#22c55e"
            opacity="0.08"
          />
          <text
            x={(xToPx(trainingShade.from) + xToPx(trainingShade.to)) / 2}
            y={T + 12}
            textAnchor="middle"
            fontSize="10"
            fill="#16a34a"
          >
            {trainingShade.label}
          </text>
        </g>
      )}

      {series.map((s, i) =>
        s.points.length ? (
          <polyline
            key={i}
            fill="none"
            stroke={s.color}
            strokeWidth={s.width || 2}
            strokeDasharray={s.dash || "none"}
            opacity={s.opacity ?? 1}
            points={s.points.map((p) => `${xToPx(p.x)},${yToPx(p.y)}`).join(" ")}
          />
        ) : null
      )}

      <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" fontSize="12" fill="#334155" fontWeight="700">
        {xLabel}
      </text>
      <text
        x={14}
        y={(T + H - B) / 2}
        fontSize="12"
        fill="#334155"
        fontWeight="700"
        transform={`rotate(-90 14 ${(T + H - B) / 2})`}
        textAnchor="middle"
      >
        {yLabel}
      </text>
    </svg>
  );
}

function Legend({ series }) {
  return (
    <div className="legend">
      {series.map((s, i) => (
        <span key={i} className="legend-item">
          <span className="legend-swatch" style={{ background: s.color }} />
          {s.name}
        </span>
      ))}
    </div>
  );
}

export default function Generalisation() {
  const [experiment, setExperiment] = useState(null);
  const [examples, setExamples] = useState([]);
  const [error, setError] = useState("");

  const [pathFilter, setPathFilter] = useState(0); // 0 = pooled over all lengths
  const [fixedR, setFixedR] = useState(6);

  useEffect(() => {
    async function load() {
      try {
        const a = await fetch(`${API_BASE}/learned/experiment`);
        const b = await fetch(`${API_BASE}/learned/examples`);

        if (!a.ok || !b.ok) throw new Error("missing");

        setExperiment(await a.json());
        setExamples((await b.json()).examples || []);
      } catch (e) {
        setError(
          "Learned results not found. Run: python -m learned.train, learned.evaluate, learned.export_results, then restart the backend."
        );
      }
    }
    load();
  }, []);

  const seeds = experiment?.experiments ?? [];
  const Rs = useMemo(() => Array.from({ length: 10 }, (_, i) => i + 1), []);
  const Ls = useMemo(() => Array.from({ length: 10 }, (_, i) => i + 1), []);

  function accAt(exp, rIdx, len) {
    const res = exp.inference_depth_results?.[rIdx];
    if (!res) return null;
    if (len == null) return res.accuracy;
    const pl = res.per_path_length?.[String(len)];
    return pl ? pl.accuracy : null;
  }

  // Plot 1: accuracy vs R
  const perSeedSeries = seeds.map((exp) => ({
    name: `Seed ${exp.seed}`,
    color: "#94a3b8",
    width: 1.5,
    opacity: 0.55,
    points: Rs.map((R) => ({ x: R, y: accAt(exp, R - 1, pathFilter || null) })).filter(
      (p) => p.y != null
    ),
  }));

  const meanSeries = {
    name: "Learned mean",
    color: "#2563eb",
    width: 3,
    points: Rs.map((R) => {
      const vals = seeds.map((e) => accAt(e, R - 1, pathFilter || null)).filter((v) => v != null);
      return { x: R, y: vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null };
    }).filter((p) => p.y != null),
  };

  const exactSeries = {
    name: "Exact mechanism (live)",
    color: "#16a34a",
    width: 2,
    dash: "6 4",
    points: [
      { x: 1, y: 1 },
      { x: 10, y: 1 },
    ],
  };

  // Plot 2: path length vs accuracy at fixed R
  const meanByLength = {
    name: `Learned mean @ R=${fixedR}`,
    color: "#2563eb",
    width: 3,
    points: Ls.map((L) => {
      const vals = seeds.map((e) => accAt(e, fixedR - 1, L)).filter((v) => v != null);
      return { x: L, y: vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null };
    }).filter((p) => p.y != null),
  };

  const exactStepPoints = [];
  for (const L of Ls) {
    const y = L <= fixedR ? 1 : 0;
    if (exactStepPoints.length && exactStepPoints[exactStepPoints.length - 1].y !== y) {
      exactStepPoints.push({ x: L, y: exactStepPoints[exactStepPoints.length - 1].y });
    }
    exactStepPoints.push({ x: L, y });
  }

  const exactStepSeries = {
    name: "Exact expected",
    color: "#16a34a",
    width: 2,
    dash: "6 4",
    points: exactStepPoints,
  };

  const seedChips = seeds.map((exp) => {
    const acc = accAt(exp, fixedR - 1, pathFilter || null);
    return { seed: exp.seed, acc, hash: exp.checkpointHash };
  });

  const successes = examples.filter((e) => e.correct && e.pathLength > 4).slice(0, 4);
  const failures = examples.filter((e) => !e.correct).slice(0, 4);

  return (
    <div className="page">
      <header className="page-header">
        <div className="tag">Page 5 — Generalisation</div>
        <h1>More computation or more memorisation?</h1>
        <p className="muted">
          The model was trained on paths of length ≤ 4. Everything at length 5–10 tests
          whether extra latent computation generalises.
        </p>
      </header>

      {error && <div className="error">{error}</div>}

      <div className="control-card">
        <div className="control-row three">
          <label>
            Accuracy-vs-R view
            <select value={pathFilter} onChange={(e) => setPathFilter(Number(e.target.value))}>
              <option value={0}>All path lengths (pooled)</option>
              {Ls.map((L) => (
                <option value={L} key={L}>
                  Path length {L} {L <= 4 ? "(trained)" : "(unseen)"}
                </option>
              ))}
            </select>
          </label>

          <label>
            Fixed inference depth for length plot: <strong>R = {fixedR}</strong>
            <input
              type="range"
              min="1"
              max="10"
              value={fixedR}
              onChange={(e) => setFixedR(Number(e.target.value))}
            />
          </label>

          <div className="badge-row">
            <span className="badge precomputed">Precomputed result</span>
            <span className="badge other">{seeds.length} seeds</span>
          </div>
        </div>
      </div>

      <div className="bridge-grid">
        <section className="panel">
          <div className="panel-head">
            <h3>Inference depth R vs accuracy</h3>
            <span className="badge precomputed">Precomputed result</span>
          </div>

          <LineChart
            series={[exactSeries, ...perSeedSeries, meanSeries]}
            xMax={10}
            xLabel="Inference depth R"
            yLabel="Accuracy"
            trainingShade={{ from: 0, to: 4, label: "training depths" }}
          />
          <Legend series={[exactSeries, meanSeries, { name: "Individual seeds", color: "#94a3b8" }]} />

          <div className="timeline" style={{ marginTop: 10 }}>
            {seedChips.map((c) => (
              <span className="time" key={c.seed}>
                seed {c.seed}: {c.acc != null ? c.acc.toFixed(2) : "–"}
              </span>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h3>Path length vs accuracy at R = {fixedR}</h3>
            <span className="badge precomputed">Precomputed result</span>
          </div>

          <LineChart
            series={[exactStepSeries, meanByLength]}
            xMax={10}
            xLabel="Path length"
            yLabel="Accuracy"
            trainingShade={{ from: 0, to: 4, label: "training range" }}
          />
          <Legend series={[exactStepSeries, meanByLength]} />

          <p className="muted">
            If the learned curve collapses after length 4 while the exact line stays at 1,
            the failure is a learned-generalisation limit, not a lack of computation.
          </p>
        </section>
      </div>

      <div className="bridge-grid">
        <section className="panel">
          <h3>Successful extrapolation</h3>
          <p className="muted">Correct answers on paths longer than training.</p>

          {successes.length === 0 && <p className="muted">None found in this run. That is an honest result.</p>}

          {successes.map((e) => (
            <div className="example-card" key={e.exampleId}>
              <div className="example-head">
                <strong>len {e.pathLength} · R {e.R}</strong>
                <span className="badge precomputed">Precomputed result</span>
              </div>
              <div className="muted">
                truth {e.label} · pred {e.prediction} · conf {Number(e.confidence).toFixed(2)}
              </div>
            </div>
          ))}
        </section>

        <section className="panel">
          <h3>Failure cases</h3>
          <p className="muted">Where extra computation does not help.</p>

          {failures.length === 0 && <p className="muted">No failures recorded in this run.</p>}

          {failures.map((e) => (
            <div className="example-card" key={e.exampleId}>
              <div className="example-head">
                <strong>len {e.pathLength} · R {e.R}</strong>
                <span className="badge precomputed">Precomputed result</span>
              </div>
              <div className="muted">
                truth {e.label} · pred {e.prediction} · conf {Number(e.confidence).toFixed(2)}
              </div>
            </div>
          ))}
        </section>
      </div>
    </div>
  );
}
