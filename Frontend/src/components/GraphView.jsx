import React, { useMemo } from "react";

// Layout is deterministic: the same graph always draws the same way, so moving
// the depth slider changes colour and nothing else. A layout that jittered
// between renders would make the propagation impossible to follow.
function layout(n, width, height, edges) {
  const positions = {};

  // A pure ring hides the path structure once graphs get dense, so nodes are
  // placed on a ring but ordered by BFS layer from node 0 where possible.
  const order = bfsOrder(n, edges);

  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.36;

  order.forEach((node, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / n;
    positions[node] = {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    };
  });

  return positions;
}

function bfsOrder(n, edges) {
  const outgoing = Array.from({ length: n }, () => []);
  edges.forEach(([u, v]) => outgoing[u]?.push(v));

  const seen = new Set([0]);
  const order = [0];
  const queue = [0];

  while (queue.length) {
    const u = queue.shift();
    for (const v of outgoing[u] || []) {
      if (!seen.has(v)) {
        seen.add(v);
        order.push(v);
        queue.push(v);
      }
    }
  }

  for (let node = 0; node < n; node++) {
    if (!seen.has(node)) order.push(node);
  }

  return order;
}

function nodeFill(value, isSource, isTarget, epsilon) {
  if (isSource) return "#16a34a";
  if (value > epsilon) {
    // Warmer as activation approaches 1.
    const intensity = Math.min(1, value);
    return isTarget
      ? "#ea580c"
      : `rgb(${Math.round(96 + intensity * 40)}, ${Math.round(
          150 - intensity * 40
        )}, ${Math.round(235 - intensity * 40)})`;
  }
  return isTarget ? "#fed7aa" : "#e2e8f0";
}

export default function GraphView({
  graph,
  source,
  target,
  state,
  epsilon = 1e-12,
  height = 420,
  showValues = true,
  caption,
}) {
  const width = 640;
  const edges = graph?.edges || [];
  const n = graph?.n || 0;

  const positions = useMemo(
    () => layout(n, width, height, edges),
    [n, height, JSON.stringify(edges)]
  );

  if (!graph) return null;

  const radius = n > 16 ? 15 : n > 10 ? 19 : 23;

  return (
    <figure className="graph-card">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Directed graph with ${n} nodes. Source is node ${source}, target is node ${target}.`}
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="9"
            markerHeight="9"
            refX="8"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
          </marker>
          <marker
            id="arrowhead-active"
            markerWidth="9"
            markerHeight="9"
            refX="8"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L0,6 L9,3 z" fill="#2563eb" />
          </marker>
        </defs>

        {edges.map(([u, v], index) => {
          const from = positions[u];
          const to = positions[v];
          if (!from || !to) return null;

          const dx = to.x - from.x;
          const dy = to.y - from.y;
          const length = Math.hypot(dx, dy) || 1;

          // An edge is "carrying" activation when its sender is already active.
          const carrying = (state?.[u] ?? 0) > epsilon;

          return (
            <line
              key={index}
              x1={from.x + (dx / length) * radius}
              y1={from.y + (dy / length) * radius}
              x2={to.x - (dx / length) * radius}
              y2={to.y - (dy / length) * radius}
              stroke={carrying ? "#2563eb" : "#cbd5e1"}
              strokeWidth={carrying ? 2.2 : 1.2}
              opacity={carrying ? 0.9 : 0.5}
              markerEnd={
                carrying ? "url(#arrowhead-active)" : "url(#arrowhead)"
              }
            />
          );
        })}

        {Array.from({ length: n }).map((_, node) => {
          const position = positions[node];
          if (!position) return null;

          const value = state?.[node] ?? 0;
          const isSource = node === source;
          const isTarget = node === target;
          const active = value > epsilon;

          return (
            <g key={node}>
              <circle
                cx={position.x}
                cy={position.y}
                r={radius}
                fill={nodeFill(value, isSource, isTarget, epsilon)}
                stroke={active ? "#0f172a" : "#94a3b8"}
                strokeWidth={active ? 2.5 : 1.2}
              />
              <text
                x={position.x}
                y={position.y + 4}
                textAnchor="middle"
                fontSize={radius > 18 ? 13 : 11}
                fontWeight="700"
                fill={active || isSource ? "#ffffff" : "#475569"}
              >
                {node}
              </text>

              {isSource && (
                <text
                  x={position.x}
                  y={position.y - radius - 8}
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight="700"
                  fill="#16a34a"
                >
                  SOURCE
                </text>
              )}
              {isTarget && (
                <text
                  x={position.x}
                  y={position.y - radius - 8}
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight="700"
                  fill="#ea580c"
                >
                  TARGET
                </text>
              )}
              {showValues && n <= 14 && (
                <text
                  x={position.x}
                  y={position.y + radius + 14}
                  textAnchor="middle"
                  fontSize="10"
                  fill="#64748b"
                >
                  {value.toFixed(2)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}
