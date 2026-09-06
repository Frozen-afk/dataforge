import React, { useMemo } from "react";
import { bfsLayers, shortestPath, outgoing } from "../engine/graph.js";

// The graph is drawn in columns by breadth-first layer from the source, so
// horizontal position *is* distance. That one decision does most of the
// teaching: the depth ruler under the picture shares this axis, so "R passed
// the tick" and "the wavefront reached the target" are the same event seen
// twice, and a learner can count columns to predict the answer before moving
// anything.
//
// A ring layout, which this replaced, hid exactly the variable the lesson is
// about. Nodes the source cannot reach have no layer at all and are parked in a
// final column marked with an infinity sign, which makes a disconnected graph
// legible at a glance instead of looking like a graph that merely needs more
// depth.
//
// Layout is deterministic. The same graph always draws the same way, so moving
// the depth slider changes colour and nothing else.

const NODE_R = 15;
const COL_MIN = 74;
const ROW = 46;
const PAD_X = 30;
const PAD_TOP = 34;
const PAD_BOTTOM = 26;

function buildLayout(graph, source, target) {
  const { n, edges } = graph;
  const layers = bfsLayers(n, edges, source);

  const reachableDepths = layers.filter((value) => value !== null);
  const maxLayer = reachableDepths.length ? Math.max(...reachableDepths) : 0;

  // One column per layer, plus a parking column for anything unreachable.
  const columns = [];
  for (let depth = 0; depth <= maxLayer; depth++) {
    columns.push({ depth, nodes: [] });
  }
  const stranded = [];

  for (let v = 0; v < n; v++) {
    if (layers[v] === null) stranded.push(v);
    else columns[layers[v]].nodes.push(v);
  }
  if (stranded.length) columns.push({ depth: null, nodes: stranded });

  // Keep the target at the vertical centre of its own column so the eye has a
  // fixed place to watch.
  for (const column of columns) {
    column.nodes.sort((a, b) => {
      if (a === target) return -1;
      if (b === target) return 1;
      return a - b;
    });
  }

  const tallest = Math.max(1, ...columns.map((column) => column.nodes.length));
  const colWidth = Math.max(COL_MIN, 640 / Math.max(1, columns.length));
  const width = PAD_X * 2 + colWidth * columns.length;
  const height = PAD_TOP + PAD_BOTTOM + Math.max(2, tallest) * ROW;

  const positions = {};
  columns.forEach((column, index) => {
    const x = PAD_X + colWidth * index + colWidth / 2;
    const span = column.nodes.length;
    const top = PAD_TOP + (height - PAD_TOP - PAD_BOTTOM - span * ROW) / 2 + ROW / 2;
    column.nodes.forEach((node, row) => {
      positions[node] = { x, y: top + row * ROW };
    });
    column.x = x;
    column.halfWidth = colWidth / 2;
  });

  return { positions, columns, width, height, layers, maxLayer };
}

/** Edges that skip more than one column need a visible arc, or they overlap. */
function edgePath(from, to, sameColumn) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy) || 1;

  const startX = from.x + (dx / length) * NODE_R;
  const startY = from.y + (dy / length) * NODE_R;
  const endX = to.x - (dx / length) * (NODE_R + 6);
  const endY = to.y - (dy / length) * (NODE_R + 6);

  if (sameColumn || Math.abs(dx) < 1) {
    const bulge = 26 * (dy >= 0 ? 1 : -1);
    return `M ${startX} ${startY} C ${startX + bulge} ${startY}, ${endX + bulge} ${endY}, ${endX} ${endY}`;
  }

  const midX = (startX + endX) / 2;
  const lift = Math.min(30, Math.abs(dx) * 0.18);
  return `M ${startX} ${startY} C ${midX} ${startY - lift}, ${midX} ${endY - lift}, ${endX} ${endY}`;
}

export default function GraphView({
  graph,
  source,
  target,
  state,
  epsilon = 1e-12,
  caption,
  showValues = true,
  highlightPath = true,
  showLayerStrip = true,
}) {
  const layout = useMemo(
    () => (graph ? buildLayout(graph, source, target) : null),
    [graph?.n, JSON.stringify(graph?.edges), source, target]
  );

  const path = useMemo(
    () =>
      graph && highlightPath
        ? shortestPath(graph.n, graph.edges, source, target)
        : null,
    [graph?.n, JSON.stringify(graph?.edges), source, target, highlightPath]
  );

  if (!graph || !layout) return null;

  const { positions, columns, width, height, layers } = layout;
  const pathEdges = new Set();
  if (path) {
    for (let i = 0; i < path.length - 1; i++) pathEdges.add(`${path[i]}>${path[i + 1]}`);
  }

  const isActive = (node) => (state?.[node] ?? 0) > epsilon;
  const frontier = state
    ? Math.max(-1, ...layers.map((depth, node) => (isActive(node) ? depth ?? -1 : -1)))
    : -1;

  return (
    <figure className="plate">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={
          `Directed graph, ${graph.n} nodes, drawn in columns by distance from ` +
          `the source. Node ${source} is the source, node ${target} is the ` +
          `target. ${state ? `${state.filter((v) => v > epsilon).length} nodes are currently active.` : ""}`
        }
      >
        <defs>
          <marker id="tip" markerWidth="7" markerHeight="7" refX="6" refY="2.4" orient="auto">
            <path d="M0,0 L0,4.8 L7,2.4 z" fill="var(--rule)" />
          </marker>
          <marker id="tip-live" markerWidth="7" markerHeight="7" refX="6" refY="2.4" orient="auto">
            <path d="M0,0 L0,4.8 L7,2.4 z" fill="var(--signal)" />
          </marker>
          <marker id="tip-path" markerWidth="7" markerHeight="7" refX="6" refY="2.4" orient="auto">
            <path d="M0,0 L0,4.8 L7,2.4 z" fill="var(--charge)" />
          </marker>
        </defs>

        {/* Column guides. These are the same ticks as the depth ruler. */}
        {columns.map((column, index) => (
          <g key={`col-${index}`}>
            {column.depth !== null && state && column.depth <= frontier && (
              <rect
                x={column.x - column.halfWidth}
                y={PAD_TOP - 12}
                width={column.halfWidth * 2}
                height={height - PAD_TOP - PAD_BOTTOM + 18}
                fill="var(--signal)"
                opacity="0.06"
              />
            )}
            <text
              x={column.x}
              y={17}
              textAnchor="middle"
              fontSize="10.5"
              fontFamily="var(--mono)"
              fill={
                column.depth !== null && column.depth <= frontier
                  ? "var(--signal)"
                  : "var(--ink-3)"
              }
            >
              {column.depth === null ? "unreachable" : `layer ${column.depth}`}
            </text>
          </g>
        ))}

        {graph.edges.map(([u, v], index) => {
          const from = positions[u];
          const to = positions[v];
          if (!from || !to) return null;

          const onPath = pathEdges.has(`${u}>${v}`);
          const carrying = isActive(u);

          return (
            <path
              key={index}
              d={edgePath(from, to, Math.abs(to.x - from.x) < 1)}
              fill="none"
              stroke={
                onPath ? "var(--charge)" : carrying ? "var(--signal)" : "var(--rule)"
              }
              strokeWidth={onPath ? 2 : carrying ? 1.7 : 1}
              opacity={onPath ? 0.95 : carrying ? 0.85 : 0.5}
              markerEnd={
                onPath ? "url(#tip-path)" : carrying ? "url(#tip-live)" : "url(#tip)"
              }
            />
          );
        })}

        {Array.from({ length: graph.n }).map((_, node) => {
          const position = positions[node];
          if (!position) return null;

          const value = state?.[node] ?? 0;
          const active = value > epsilon;
          const isSource = node === source;
          const isTarget = node === target;

          // Role is carried by shape, activation by fill. Adding a third and a
          // fourth colour for source and target would compete with the one
          // colour that has to mean "activation arrived".
          const fill = active
            ? isTarget
              ? "var(--charge)"
              : "var(--signal)"
            : "var(--surface)";
          const stroke = isTarget
            ? "var(--charge)"
            : isSource
            ? "var(--ink)"
            : active
            ? "var(--signal)"
            : "var(--rule)";

          return (
            <g key={node}>
              {isTarget && (
                <circle
                  cx={position.x}
                  cy={position.y}
                  r={NODE_R + 4}
                  fill="none"
                  stroke={stroke}
                  strokeWidth={active ? 1.6 : 1}
                  opacity={active ? 1 : 0.55}
                />
              )}
              {isSource ? (
                <rect
                  x={position.x - NODE_R}
                  y={position.y - NODE_R}
                  width={NODE_R * 2}
                  height={NODE_R * 2}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth="2"
                />
              ) : (
                <circle
                  cx={position.x}
                  cy={position.y}
                  r={NODE_R}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={active ? 2 : 1.2}
                />
              )}
              <text
                x={position.x}
                y={position.y + 4}
                textAnchor="middle"
                fontSize="11.5"
                fontFamily="var(--mono)"
                fontWeight="500"
                fill={active ? "var(--surface)" : "var(--ink-2)"}
              >
                {node}
              </text>

              {(isSource || isTarget) && (
                <text
                  x={position.x}
                  y={position.y + NODE_R + (isTarget ? 18 : 15)}
                  textAnchor="middle"
                  fontSize="9.5"
                  fontFamily="var(--display)"
                  fontWeight="600"
                  fill={isTarget ? "var(--charge)" : "var(--ink-2)"}
                >
                  {isSource ? "source s" : "target q"}
                </text>
              )}

              {showValues && graph.n <= 14 && !isSource && !isTarget && (
                <text
                  x={position.x}
                  y={position.y + NODE_R + 13}
                  textAnchor="middle"
                  fontSize="9"
                  fontFamily="var(--mono)"
                  fill="var(--ink-3)"
                >
                  {value.toFixed(2)}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {showLayerStrip && state && (
        <div className="layer-strip" aria-hidden="true">
          {columns.map((column, index) => (
            <span
              key={index}
              className={column.depth !== null && column.depth <= frontier ? "lit" : ""}
            >
              {column.depth === null ? "∞" : column.depth}
            </span>
          ))}
        </div>
      )}

      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}

/** Nodes reachable within r hops, used by pages that quote the frontier. */
export function frontierSize(graph, source, r) {
  const layers = bfsLayers(graph.n, graph.edges, source);
  return layers.filter((depth) => depth !== null && depth <= r).length;
}

export { outgoing };
