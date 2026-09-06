import React from "react";

// The four evidence labels the project is required to keep distinct. Every
// number shown anywhere in the lab carries exactly one of these.
export const EVIDENCE = {
  "Live computation": {
    className: "badge live",
    meaning:
      "Computed in your browser session by the backend, right now, from the " +
      "graph on screen.",
  },
  "Synthetic data": {
    className: "badge synthetic",
    meaning:
      "Generated from a seeded random process rather than measured from the " +
      "world. Reproducible from the seed.",
  },
  "Precomputed result": {
    className: "badge precomputed",
    meaning:
      "Measured earlier by this project from a frozen checkpoint, then shipped " +
      "as a data file. Not recomputed as you browse.",
  },
  "Paper-reported result": {
    className: "badge paper",
    meaning:
      "Taken from a published source. Not measured by this project, and not " +
      "independently verified here.",
  },
};

export function EvidenceBadge({ type, title }) {
  const entry = EVIDENCE[type];

  if (!entry) {
    return <span className="badge other">{type || "Unlabelled"}</span>;
  }

  return (
    <span className={entry.className} title={title || entry.meaning}>
      {type}
    </span>
  );
}

// Wraps any figure so the provenance travels with the number rather than
// living in a caption somewhere else on the page.
export function EvidenceNote({ evidence, extra }) {
  if (!evidence) return null;

  return (
    <div className="evidence-note">
      <EvidenceBadge type={evidence.evidenceType} />
      <span className="evidence-source">{evidence.source}</span>
      {extra && <span className="evidence-extra">{extra}</span>}
    </div>
  );
}
