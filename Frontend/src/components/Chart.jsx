import React from "react";

// A small line chart used for every plot in the lab. Kept deliberately plain:
// the point of these figures is to compare two curves and a shaded regime, not
// to decorate.
export default function Chart({
  series,
  xMax,
  xMin = 0,
  yMax = 1,
  xLabel,
  yLabel,
  shade,
  height = 300,
  yFormat = (v) => v.toFixed(2),
  chanceLine,
}) {
  const width = 640;
  const left = 54;
  const bottom = 46;
  const top = 20;
  const right = 18;

  const xToPx = (x) =>
    left + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (width - left - right);
  const yToPx = (y) =>
    height - bottom - (Math.max(0, Math.min(yMax, y)) / yMax) * (height - bottom - top);

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * yMax);
  const xTicks = [];
  for (let x = xMin; x <= xMax; x++) xTicks.push(x);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart" role="img"
         aria-label={`${yLabel} versus ${xLabel}`}>
      {shade && (
        <g>
          <rect
            x={xToPx(shade.from)}
            y={top}
            width={Math.max(0, xToPx(shade.to) - xToPx(shade.from))}
            height={height - bottom - top}
            fill="#16a34a"
            opacity="0.09"
          />
          <text
            x={(xToPx(shade.from) + xToPx(shade.to)) / 2}
            y={top + 13}
            textAnchor="middle"
            fontSize="10"
            fill="#15803d"
            fontWeight="600"
          >
            {shade.label}
          </text>
        </g>
      )}

      {yTicks.map((t) => (
        <g key={t}>
          <line x1={left} y1={yToPx(t)} x2={width - right} y2={yToPx(t)}
                stroke="#e2e8f0" />
          <text x={left - 8} y={yToPx(t) + 4} textAnchor="end" fontSize="10"
                fill="#64748b">
            {yFormat(t)}
          </text>
        </g>
      ))}

      {chanceLine !== undefined && (
        <g>
          <line
            x1={left} y1={yToPx(chanceLine)} x2={width - right} y2={yToPx(chanceLine)}
            stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="2 3"
          />
          <text x={width - right - 4} y={yToPx(chanceLine) - 5} textAnchor="end"
                fontSize="10" fill="#64748b">
            chance
          </text>
        </g>
      )}

      {xTicks.map((t) => (
        <text key={t} x={xToPx(t)} y={height - bottom + 16} textAnchor="middle"
              fontSize="10" fill="#64748b">
          {t}
        </text>
      ))}

      {/* Confidence bands are drawn first so the mean lines sit on top. */}
      {series.map((s, i) =>
        s.band && s.band.length > 1 ? (
          <polygon
            key={`band-${i}`}
            fill={s.color}
            opacity="0.14"
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
            strokeWidth={s.width || 2.2}
            strokeDasharray={s.dash || "none"}
            opacity={s.opacity ?? 1}
            points={s.points.map((p) => `${xToPx(p.x)},${yToPx(p.y)}`).join(" ")}
          />
        ) : null
      )}

      {series.map((s, i) =>
        s.showDots && s.points?.length
          ? s.points.map((p, j) => (
              <circle key={`${i}-${j}`} cx={xToPx(p.x)} cy={yToPx(p.y)} r="3"
                      fill={s.color} />
            ))
          : null
      )}

      <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom}
            stroke="#94a3b8" />
      <line x1={left} y1={top} x2={left} y2={height - bottom} stroke="#94a3b8" />

      <text x={(left + width - right) / 2} y={height - 6} textAnchor="middle"
            fontSize="11" fill="#334155" fontWeight="700">
        {xLabel}
      </text>
      <text x={13} y={(top + height - bottom) / 2} fontSize="11" fill="#334155"
            fontWeight="700"
            transform={`rotate(-90 13 ${(top + height - bottom) / 2})`}
            textAnchor="middle">
        {yLabel}
      </text>
    </svg>
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
              background: item.dash ? "transparent" : item.color,
              borderTop: item.dash ? `2px dashed ${item.color}` : "none",
            }}
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}
