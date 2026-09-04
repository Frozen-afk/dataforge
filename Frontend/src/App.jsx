import React, { useState } from "react";
import ExactLab from "./ExactLab.jsx";
import LearnedBridge from "./pages/LearnedBridge.jsx";
import Generalisation from "./pages/Generalisation.jsx";

const NAV = [
  { id: "exact", label: "1–3 · Exact Lab" },
  { id: "bridge", label: "4 · AI Bridge" },
  { id: "gen", label: "5 · Generalisation" },
  { id: "bdh", label: "6 · BDH-CQ" },
  { id: "evidence", label: "7 · Evidence" },
];

function Placeholder({ title, items }) {
  return (
    <div className="page">
      <div className="panel placeholder">
        <h2>{title}</h2>
        <p className="muted">Not built yet. Required components:</p>
        <ul className="muted">
          {items.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("exact");

  return (
    <div className="app">
      <nav className="topnav">
        <span className="brand">Latent Loop Lab</span>
        {NAV.map((n) => (
          <button
            key={n.id}
            className={page === n.id ? "navbtn active" : "navbtn"}
            onClick={() => setPage(n.id)}
          >
            {n.label}
          </button>
        ))}
      </nav>

      {page === "exact" && <ExactLab />}
      {page === "bridge" && <LearnedBridge />}
      {page === "gen" && <Generalisation />}
      {page === "bdh" && (
        <Placeholder
          title="Page 6 — BDH-CQ: Where does this appear in real AI?"
          items={[
            "Show S_t = U(S_{t-1}, D_t) as contextual recurrence over demonstrations",
            "Show H_{r+1} = F(H_r, S_K) as query-time latent reasoning",
            "Map Latent Loop Lab h(r) to BDH-CQ H_r as an analogy, not identity",
            "Paper-reported BDH-CQ effort table with evidence badges",
          ]}
        />
      )}
      {page === "evidence" && (
        <Placeholder
          title="Page 7 — Evidence and limitations"
          items={[
            "Four evidence badges on every number",
            "toy != BDH-CQ",
            "More recurrence consumes compute and may saturate",
            "Graph reachability shows computational depth, not general intelligence",
          ]}
        />
      )}
    </div>
  );
}
