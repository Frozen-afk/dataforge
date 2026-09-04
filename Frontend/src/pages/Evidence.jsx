import React, { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const BADGES = [
  {
    name: "Live computation",
    cls: "live",
    meaning: "Computed in your browser or by the backend right now.",
    where: "Exact graph recurrence (Pages 1–4); learned model only if live inference is deployed.",
  },
  {
    name: "Synthetic data",
    cls: "synthetic",
    meaning: "Generated test suites validating a stated invariant.",
    where: "The 10,000-case exact invariant sweep.",
  },
  {
    name: "Precomputed result",
    cls: "precomputed",
    meaning: "Expensive model runs shipped as data, frozen and hashed.",
    where: "Learned GNN accuracy curves, seed tables, success/failure examples.",
  },
  {
    name: "Paper-reported result",
    cls: "paper",
    meaning: "Numbers taken from a published source, not measured here.",
    where: "BDH-CQ effort table, ARC-AGI-1 numbers, literature claims.",
  },
];

const LIMITATIONS = [
  "toy ≠ BDH-CQ",
  "latent coordinates are interpretable here by design",
  "production latent states are not generally human-readable",
  "more recurrence consumes compute",
  "excessive recurrence may saturate or destabilise learned systems",
  "graph reachability demonstrates computational depth, not general intelligence",
  "broader independent evidence is required before claiming universal superiority",
];

const NEVER_CLAIM = [
  "We do not claim that graph reachability proves semantic reasoning.",
  "We do not claim that more recurrence always improves reasoning.",
];

export default function Evidence() {
  const [learnedLive, setLearnedLive] = useState(false);
  const [probed, setProbed] = useState(false);

  useEffect(() => {
    async function probe() {
      try {
        const res = await fetch(`${API_BASE}/learned/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pathLength: 3, reachable: true, R: 3 }),
        });
        setLearnedLive(res.ok);
      } catch {
        setLearnedLive(false);
      }
      setProbed(true);
    }
    probe();
  }, []);

  const ledger = [
    {
      element: "Exact graph recurrence",
      interpretation: "Positive-support reachability within R hops.",
      badge: "Live computation",
      cls: "live",
    },
    {
      element: "10,000-case sweep",
      interpretation: "Validates the stated invariant h(R)[q] > eps iff d(s,q) <= R.",
      badge: "Synthetic data",
      cls: "synthetic",
    },
    {
      element: "Learned GNN results",
      interpretation:
        (probed
          ? learnedLive
            ? "Live inference deployed on this server."
            : "Live inference not deployed; results shipped as JSON."
          : "Checking deployment...") ,
      badge: probed ? (learnedLive ? "Live computation" : "Precomputed result") : "...",
      cls: probed ? (learnedLive ? "live" : "precomputed") : "other",
    },
    {
      element: "BDH-CQ effort scores",
      interpretation: "Reported operating points from the BDH-CQ report.",
      badge: "Paper-reported result",
      cls: "paper",
    },
    {
      element: "ARC-AGI-1 results",
      interpretation: "Reported benchmark numbers, not measured here.",
      badge: "Paper-reported result",
      cls: "paper",
    },
    {
      element: "Literature / theoretical claims",
      interpretation: "Claims about Coconut, recurrent depth, looped Transformers.",
      badge: "Paper-reported result",
      cls: "paper",
    },
  ];

  return (
    <div className="page">
      <header className="page-header">
        <div className="tag">Page 7 — Evidence and limitations</div>
        <h1>What is live, what is shipped, what is cited</h1>
        <p className="muted">
          Every numerical result in this artifact carries one of the four badges below.
          The distinction is part of the project, not bureaucratic overhead.
        </p>
      </header>

      <section className="badge-grid">
        {BADGES.map((b) => (
          <div className="panel badge-card" key={b.name}>
            <span className={`badge ${b.cls}`}>{b.name}</span>
            <p className="muted">{b.meaning}</p>
            <p className="small-label">Used for</p>
            <p className="muted">{b.where}</p>
          </div>
        ))}
      </section>

      <section className="panel">
        <h3>Evidence ledger</h3>
        <table className="evidence-table">
          <thead>
            <tr>
              <th>Element</th>
              <th>Permitted interpretation</th>
              <th>Badge</th>
            </tr>
          </thead>
          <tbody>
            {ledger.map((row) => (
              <tr key={row.element}>
                <td>{row.element}</td>
                <td>{row.interpretation}</td>
                <td>
                  <span className={`badge ${row.cls}`}>{row.badge}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h3>Limitations</h3>
        <ul className="limit-list">
          {LIMITATIONS.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      </section>

      <section className="panel">
        <h3>What this artifact never claims</h3>
        <ul className="limit-list">
          {NEVER_CLAIM.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
