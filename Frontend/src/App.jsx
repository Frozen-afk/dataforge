import React, { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const PRESET_LABELS = {
  line: "Four-hop line",
  line_short: "Two-hop line",
  branch: "Branching graph",
  diamond: "Diamond graph",
  cycle: "Directed cycle",
  disconnected: "Disconnected graph"
};

function getNodePositions(n, width, height) {
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.34;

  const positions = {};

  for (let i = 0; i < n; i++) {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / n;
    positions[i] = {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle)
    };
  }

  return positions;
}

function activationColor(value, isSource, isTarget) {
  if (isSource && isTarget) {
    return "#a855f7";
  }

  if (isSource) {
    return "#22c55e";
  }

  if (isTarget && value > 0) {
    return "#f97316";
  }

  if (isTarget) {
    return "#fb923c";
  }

  if (value <= 0) {
    return "#e5e7eb";
  }

  const intensity = Math.min(1, value);
  const blue = Math.floor(180 - intensity * 80);
  return `rgb(59, 130, ${blue})`;
}

function GraphView({ graph, source, target, state }) {
  const width = 620;
  const height = 430;
  const positions = useMemo(
    () => getNodePositions(graph.n, width, height),
    [graph.n]
  );

  return (
    <div className="graph-card">
      <svg width="100%" viewBox={`0 0 ${width} ${height}`}>
        <defs>
          <marker
            id="arrow"
            markerWidth="10"
            markerHeight="10"
            refX="8"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L0,6 L9,3 z" fill="#64748b" />
          </marker>
        </defs>

        {graph.edges.map(([u, v], idx) => {
          const from = positions[u];
          const to = positions[v];

          const dx = to.x - from.x;
          const dy = to.y - from.y;
          const length = Math.sqrt(dx * dx + dy * dy);

          const nodeRadius = 22;
          const sx = from.x + (dx / length) * nodeRadius;
          const sy = from.y + (dy / length) * nodeRadius;
          const tx = to.x - (dx / length) * nodeRadius;
          const ty = to.y - (dy / length) * nodeRadius;

          return (
            <line
              key={idx}
              x1={sx}
              y1={sy}
              x2={tx}
              y2={ty}
              stroke="#64748b"
              strokeWidth="2"
              markerEnd="url(#arrow)"
            />
          );
        })}

        {Array.from({ length: graph.n }).map((_, node) => {
          const pos = positions[node];
          const value = state?.[node] ?? 0;

          const isSource = node === source;
          const isTarget = node === target;

          return (
            <g key={node}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r="25"
                fill={activationColor(value, isSource, isTarget)}
                stroke={value > 0 ? "#0f172a" : "#94a3b8"}
                strokeWidth={value > 0 ? "3" : "2"}
              />

              <text
                x={pos.x}
                y={pos.y + 5}
                textAnchor="middle"
                fontSize="15"
                fontWeight="700"
                fill="#0f172a"
              >
                {node}
              </text>

              {isSource && (
                <text
                  x={pos.x}
                  y={pos.y - 36}
                  textAnchor="middle"
                  fontSize="12"
                  fontWeight="700"
                  fill="#16a34a"
                >
                  SOURCE
                </text>
              )}

              {isTarget && (
                <text
                  x={pos.x}
                  y={pos.y + 45}
                  textAnchor="middle"
                  fontSize="12"
                  fontWeight="700"
                  fill="#ea580c"
                >
                  TARGET
                </text>
              )}

              <text
                x={pos.x}
                y={pos.y + 66}
                textAnchor="middle"
                fontSize="11"
                fill="#475569"
              >
                h={value.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function VectorPanel({ state }) {
  if (!state) return null;

  return (
    <div className="panel">
      <h3>Current latent state vector</h3>
      <p className="muted">
        Each coordinate corresponds to one graph node. A value near 1 means that
        node has been activated by the recurrent propagation.
      </p>

      <div className="vector-list">
        {state.map((value, idx) => (
          <div className="vector-row" key={idx}>
            <div className="vector-label">h[{idx}]</div>
            <div className="bar-bg">
              <div
                className="bar-fill"
                style={{ width: `${Math.min(100, value * 100)}%` }}
              />
            </div>
            <div className="vector-value">{value.toFixed(2)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Timeline({ trajectory, currentR, setCurrentR }) {
  if (!trajectory) return null;

  return (
    <div className="panel">
      <h3>Iteration timeline</h3>
      <p className="muted">
        Click a step to inspect the hidden state after that many recurrent
        updates.
      </p>

      <div className="timeline">
        {trajectory.map((_, idx) => (
          <button
            key={idx}
            className={idx === currentR ? "time active-time" : "time"}
            onClick={() => setCurrentR(idx)}
          >
            h({idx})
          </button>
        ))}
      </div>
    </div>
  );
}

function TruthPanel({ data }) {
  if (!data) return null;

  let explanation = "";

  if (data.bfsDistance === null) {
    explanation =
      "BFS says the target is unreachable. No amount of recurrence can create a missing path.";
  } else if (data.R < data.bfsDistance) {
    explanation = `The shortest path has length ${data.bfsDistance}, but R is only ${data.R}. The recurrent state has not had enough steps to reach the target.`;
  } else {
    explanation = `R is at least the shortest path distance ${data.bfsDistance}, so the target should be active.`;
  }

  return (
    <div className="panel truth-panel">
      <h3>Truth beside estimate</h3>

      <div className="truth-grid">
        <div>
          <div className="small-label">Recurrent estimate</div>
          <div className={data.estimate ? "pill yes" : "pill no"}>
            {data.estimate ? "Target active" : "Target inactive"}
          </div>
        </div>

        <div>
          <div className="small-label">BFS oracle truth</div>
          <div className={data.expected ? "pill yes" : "pill no"}>
            {data.expected ? "Reachable within R" : "Not reachable within R"}
          </div>
        </div>

        <div>
          <div className="small-label">Shortest path distance</div>
          <div className="big-value">
            {data.bfsDistance === null ? "∞" : data.bfsDistance}
          </div>
        </div>

        <div>
          <div className="small-label">Invariant</div>
          <div className={data.invariantPass ? "pill yes" : "pill no"}>
            {data.invariantPass ? "PASS" : "FAIL"}
          </div>
        </div>
      </div>

      <p className="explanation">{explanation}</p>

      <div className="evidence-badge">
        {data.evidence?.evidenceType || "Live computation"}
      </div>
    </div>
  );
}

function EquationBox() {
  return (
    <div className="panel equation-panel">
      <h3>What the update rule is doing</h3>

      <p>
        The state starts as a one-hot vector at the source node:
      </p>

      <pre>h(0) = e_source</pre>

      <p>
        At each recurrent step, every node checks whether activation has arrived
        from its incoming neighbours:
      </p>

      <pre>
{`h(r+1)[v] = max(
  h(r)[v],
  1 - product over incoming u of (1 - alpha * h(r)[u])
)`}
      </pre>

      <p>
        With <strong>alpha = 1</strong>, this behaves like reachability spreading
        one edge per recurrent update.
      </p>
    </div>
  );
}

function App() {
  const [preset, setPreset] = useState("line");
  const [R, setR] = useState(0);
  const [data, setData] = useState(null);
  const [viewStep, setViewStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function fetchExact(selectedPreset, selectedR) {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API_BASE}/exact/preset/${selectedPreset}/${selectedR}`
      );

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const result = await response.json();
      setData(result);
      setViewStep(selectedR);
    } catch (err) {
      setError(
        "Could not connect to backend. Make sure FastAPI is running on http://127.0.0.1:8000"
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchExact(preset, R);
  }, [preset, R]);

  const currentState = data?.trajectory?.[viewStep];

  return (
    <div className="app">
      <header className="hero">
        <div>
          <div className="tag">Latent Loop Lab</div>
          <h1>Can a model think longer without saying more?</h1>
          <p>
            Move the recurrent-depth slider. The update rule stays fixed. Only
            the number of latent state updates changes.
          </p>
        </div>
      </header>

      <main className="layout">
        <section className="left">
          <div className="control-card">
            <div className="control-row">
              <label>
                Graph preset
                <select
                  value={preset}
                  onChange={(e) => {
                    setPreset(e.target.value);
                    setR(0);
                    setViewStep(0);
                  }}
                >
                  {Object.entries(PRESET_LABELS).map(([key, label]) => (
                    <option value={key} key={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Recurrent depth R: <strong>{R}</strong>
                <input
                  type="range"
                  min="0"
                  max="12"
                  value={R}
                  onChange={(e) => setR(Number(e.target.value))}
                />
              </label>
            </div>

            <p className="muted">
              Current preset:{" "}
              <strong>{data?.description || PRESET_LABELS[preset]}</strong>
            </p>
          </div>

          {loading && <div className="loading">Running recurrence...</div>}
          {error && <div className="error">{error}</div>}

          {data && (
            <GraphView
              graph={data.graph}
              source={data.source}
              target={data.target}
              state={currentState}
            />
          )}

          {data && (
            <Timeline
              trajectory={data.trajectory}
              currentR={viewStep}
              setCurrentR={setViewStep}
            />
          )}
        </section>

        <section className="right">
          {data && <TruthPanel data={data} />}
          {data && <VectorPanel state={currentState} />}
          <EquationBox />
        </section>
      </main>

      <footer className="footer">
        <p>
          This is the exact graph mechanism layer. It is live computation, not a
          scripted animation. BFS is computed independently as the oracle.
        </p>
      </footer>
    </div>
  );
}

export default App;
