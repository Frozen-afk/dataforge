import React, { useState } from "react";

// One line chart, used for every plot in the lab.
//
// Deliberately plain: these figures exist to compare two curves against a
// shaded regime and a chance line, not to decorate. Colours come from CSS
// variables so the plot inverts correctly in dark mode, and the series colour
// carries meaning rather than variety -- ink is the exact reference, signal is
// the learned model, and the two never swap.
//
// Hovering snaps to the nearest depth and reads out every series at once, so a
// value can be read off rather than estimated from pixels.

export default function Chart({
  series,
  xMin = 0,
  xMax,
  yMax = 1,
  yMin = 0,
  xLabel,
  yLabel,
  shade,
  chanceLine,
  height = 300,
  yFormat = (v) => v.toFixed(2),
  hoverFormat = (v) => v.toFixed(3),
}) {
  const [hover, setHover] = useState(null);

  const width = 660;
  const left = 52;
  const right = 16;
  const top = 22;
  const bottom = 46;

  const xToPx = (x) =>
    left + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (width - left - right);
  const yToPx = (y) =>
    height - bottom -
    ((Math.max(yMin, Math.min(yMax, y)) - yMin) / Math.max(1e-9, yMax - yMin)) *
      (height - bottom - top);

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));
  const xTicks = [];
  for (let x = Math.ceil(xMin); x <= xMax; x++) xTicks.push(x);

  function onMove(event) {
    const box = event.currentTarget.getBoundingClientRect();
    const relative = ((event.clientX - box.left) / box.width) * width;
    const value = xMin + ((relative - left) / (width - left - right)) * (xMax - xMin);
    const snapped = Math.round(value);
    setHover(snapped >= xMin && snapped <= xMax ? snapped : null);
  }

  const readouts = hover === null
    ? []
    : series
        .map((s) => {
          const point = s.points?.find((p) => p.x === hover);
          return point && s.label ? { label: s.label, color: s.color, y: point.y } : null;
        })
        .filter(Boolean);

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="chart"
        role="img"
        aria-label={`${yLabel} against ${xLabel}`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {shade && (
          <g>
            <rect
              x={xToPx(shade.from)}
              y={top}
              width={Math.max(0, xToPx(shade.to) - xToPx(shade.from))}
              height={height - bottom - top}
              fill="var(--ink-3)"
              opacity="0.09"
            />
            <line
              x1={xToPx(shade.to)} y1={top}
              x2={xToPx(shade.to)} y2={height - bottom}
              stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="3 3"
            />
            <text
              x={(xToPx(shade.from) + xToPx(shade.to)) / 2}
              y={top + 12}
              textAnchor="middle"
              fontSize="10.5"
              fontFamily="var(--display)"
              fill="var(--ink-3)"
            >
              {shade.label}
            </text>
          </g>
        )}

        {yTicks.map((t) => (
          <g key={t}>
            <line x1={left} y1={yToPx(t)} x2={width - right} y2={yToPx(t)}
                  stroke="var(--rule-soft)" />
            <text x={left - 8} y={yToPx(t) + 4} textAnchor="end" fontSize="10"
                  fontFamily="var(--mono)" fill="var(--ink-3)">
              {yFormat(t)}
            </text>
          </g>
        ))}

        {chanceLine !== undefined && (
          <g>
            <line x1={left} y1={yToPx(chanceLine)} x2={width - right} y2={yToPx(chanceLine)}
                  stroke="var(--ink-3)" strokeWidth="1.2" strokeDasharray="2 4" />
            <text x={width - right - 4} y={yToPx(chanceLine) - 6} textAnchor="end"
                  fontSize="10" fontFamily="var(--display)" fill="var(--ink-3)">
              chance
            </text>
          </g>
        )}

        {hover !== null && (
          <line x1={xToPx(hover)} y1={top} x2={xToPx(hover)} y2={height - bottom}
                stroke="var(--charge)" strokeWidth="1" opacity="0.7" />
        )}

        {xTicks.map((t) => (
          <text key={t} x={xToPx(t)} y={height - bottom + 16} textAnchor="middle"
                fontSize="10" fontFamily="var(--mono)"
                fill={hover === t ? "var(--charge)" : "var(--ink-3)"}>
            {t}
          </text>
        ))}

        {/* Bands first, so the mean lines sit on top of their own interval. */}
        {series.map((s, i) =>
          s.band && s.band.length > 1 ? (
            <polygon
              key={`band-${i}`}
              fill={s.color}
              opacity="0.16"
              points={[
                ...s.band.map((p) => `${xToPx(p.x)},${yToPx(p.high)}`),
                ...[...s.band].reverse().map((p) => `${xToPx(p.x)},${yToPx(p.low)}`),
              ].join(" ")}
            />
          ) : null
        )}

        {series.map((s, i) =>
          s.points?.length ? (
            <polyline
              key={i}
              fill="none"
              stroke={s.color}
              strokeWidth={s.width || 2}
              strokeDasharray={s.dash || "none"}
              strokeLinejoin="round"
              points={s.points.map((p) => `${xToPx(p.x)},${yToPx(p.y)}`).join(" ")}
            />
          ) : null
        )}

        {series.map((s, i) =>
          s.showDots && s.points?.length
            ? s.points.map((p, j) => (
                <circle
                  key={`${i}-${j}`}
                  cx={xToPx(p.x)}
                  cy={yToPx(p.y)}
                  r={hover === p.x ? 4.5 : 2.6}
                  fill={s.color}
                />
              ))
            : null
        )}

        <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom}
              stroke="var(--rule)" />
        <line x1={left} y1={top} x2={left} y2={height - bottom} stroke="var(--rule)" />

        <text x={(left + width - right) / 2} y={height - 6} textAnchor="middle"
              fontSize="11.5" fontFamily="var(--display)" fontWeight="600"
              fill="var(--ink-2)">
          {xLabel}
        </text>
        <text x={12} y={(top + height - bottom) / 2} fontSize="11.5"
              fontFamily="var(--display)" fontWeight="600" fill="var(--ink-2)"
              transform={`rotate(-90 12 ${(top + height - bottom) / 2})`}
              textAnchor="middle">
          {yLabel}
        </text>
      </svg>

      {readouts.length > 0 && (
        <div className="legend" aria-live="polite">
          <span className="legend-item"><strong>{xLabel} = {hover}</strong></span>
          {readouts.map((item) => (
            <span className="legend-item" key={item.label}>
              <span className="legend-swatch" style={{ borderTopColor: item.color }} />
              {item.label} <span className="mono">{hoverFormat(item.y)}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((item) => (
        <span className="legend-item" key={item.label}>
          <span
            className="legend-swatch"
            style={{
              borderTopColor: item.color,
              borderTopStyle: item.dash ? "dashed" : "solid",
            }}
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}

// Series colours, named so a page states what a curve means rather than
// picking a hex. The exact mechanism is always ink; the learned model is
// always signal; latent-state quantities are always violet.
export const SERIES = {
  exact: "var(--ink)",
  learned: "var(--signal)",
  seen: "var(--affirm)",
  unseen: "var(--charge)",
  latent: "var(--latent)",
};
