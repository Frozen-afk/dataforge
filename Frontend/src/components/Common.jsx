import React from "react";

export function PageHeader({ step, question, goal, children }) {
  return (
    <header className="page-header">
      <span className="step">{step}</span>
      <h1>{question}</h1>
      {goal && <p className="goal">{goal}</p>}
      {children}
    </header>
  );
}

export function Panel({ title, subtitle, children, className = "" }) {
  return (
    <section className={`panel ${className}`}>
      {title && <h3>{title}</h3>}
      {subtitle && <p className="sub">{subtitle}</p>}
      {children}
    </section>
  );
}

export function Box({ children, className = "" }) {
  return <div className={`box ${className}`}>{children}</div>;
}

/**
 * The depth ruler.
 *
 * Not a generic slider. Every integer depth gets a tick, the shortest-path
 * distance gets a heavier tick in the one loud colour on the page, and ticks
 * the current depth has already passed take the signal colour. So the control
 * shows the answer's location before you reach it, which is what makes the
 * prediction exercise on page 2 possible: the learner can see where the
 * threshold is and still has to work out why it sits there.
 */
export function DepthAxis({
  id,
  label = "Recurrent depth R",
  value,
  min = 0,
  max = 12,
  onChange,
  threshold = null,
  thresholdLabel = "shortest path d(s,q)",
  hint,
  disabled = false,
}) {
  const span = Math.max(1, max - min);
  const ticks = [];
  for (let t = min; t <= max; t++) ticks.push(t);

  return (
    <div className="axis">
      <div className="axis-head">
        <label className="label" htmlFor={id}>{label}</label>
        <span className="value" aria-hidden="true">{value}</span>
      </div>

      <div className="axis-track">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={1}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(Number(event.target.value))}
          aria-valuetext={
            threshold !== null
              ? `${value} of ${max}. The target needs ${threshold}.`
              : `${value} of ${max}`
          }
        />

        <div className="axis-ticks" aria-hidden="true">
          {ticks.map((t) => {
            const isThreshold = threshold !== null && t === threshold;
            const classes = [
              "axis-tick",
              isThreshold ? "threshold" : "",
              !isThreshold && t <= value && t > min ? "passed" : "",
            ].filter(Boolean).join(" ");

            // The threshold keeps its number and gains a heavier mark. An
            // earlier version replaced the number with a caption, which
            // collided with its neighbours on a ten-tick axis and hid the one
            // value the learner is trying to read.
            return (
              <span
                key={t}
                className={classes}
                style={{ left: `${((t - min) / span) * 100}%` }}
              >
                {t}
              </span>
            );
          })}
        </div>
      </div>

      {threshold !== null && (
        <span className="axis-key">
          <i /> {thresholdLabel} = {threshold}
        </span>
      )}
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

/** A plain slider for controls that are not the depth axis. */
export function Slider({ id, label, value, min, max, onChange, hint }) {
  return (
    <label className="grow" htmlFor={id}>
      {label} <strong className="slider-value">{value}</strong>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

export function Pill({ tone = "no", children }) {
  return <span className={`pill ${tone}`}>{children}</span>;
}

export function Equation({ children, caption }) {
  return (
    <div className="equation">
      <code>{children}</code>
      {caption && <span className="caption">{caption}</span>}
    </div>
  );
}

/** Every page closes with the one thing the learner can now say. */
export function Takeaway({ children }) {
  return (
    <aside className="takeaway">
      <strong>What this page establishes</strong>
      <p>{children}</p>
    </aside>
  );
}

export function Verdict({ tone = "wait", title, children }) {
  return (
    <div className={`verdict-bar ${tone}`} role="status">
      {title && <strong>{title}</strong>}
      <p>{children}</p>
    </div>
  );
}

export function ErrorBanner({ message, children }) {
  if (!message) return null;
  return (
    <div className="error" role="alert">
      <strong>This section could not load.</strong>
      <p>{message}</p>
      {children}
    </div>
  );
}

export function Loading({ label = "Computing" }) {
  return <div className="loading" role="status">{label}</div>;
}

export function NextButton({ onNext, children }) {
  return (
    <div className="page-nav">
      <button className="btn" onClick={onNext}>{children}</button>
    </div>
  );
}

/** A labelled number. Measurements live here rather than inside prose. */
export function Stat({ label, value, note }) {
  return (
    <div>
      <span className="small-label">{label}</span>
      <span className="big-value">{value}</span>
      {note && <span className="sub-value">{note}</span>}
    </div>
  );
}
