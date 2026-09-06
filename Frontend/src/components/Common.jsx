import React from "react";

export function Panel({ title, children, className = "", subtitle }) {
  return (
    <section className={`panel ${className}`}>
      {title && <h3>{title}</h3>}
      {subtitle && <p className="muted">{subtitle}</p>}
      {children}
    </section>
  );
}

export function PageHeader({ step, question, goal, children }) {
  return (
    <header className="page-header">
      <div className="step-chip">Page {step}</div>
      <h1>{question}</h1>
      <p className="goal">{goal}</p>
      {children}
    </header>
  );
}

// Every page ends with the one thing the learner should be able to say now.
export function Takeaway({ children }) {
  return (
    <aside className="takeaway">
      <strong>What this page establishes</strong>
      <p>{children}</p>
    </aside>
  );
}

export function Slider({ label, value, min, max, onChange, hint, id }) {
  return (
    <div className="slider-block">
      <label htmlFor={id}>
        {label} <strong className="slider-value">{value}</strong>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

export function ErrorBanner({ message, children }) {
  if (!message) return null;
  return (
    <div className="error" role="alert">
      <strong>Could not load this section.</strong>
      <p>{message}</p>
      {children}
    </div>
  );
}

export function Loading({ label = "Computing..." }) {
  return <div className="loading">{label}</div>;
}

export function Pill({ tone, children }) {
  return <span className={`pill ${tone}`}>{children}</span>;
}

export function Equation({ children, label }) {
  return (
    <div className="equation">
      <code>{children}</code>
      {label && <span className="equation-label">{label}</span>}
    </div>
  );
}
