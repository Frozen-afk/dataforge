import React, { useState } from "react";
import ExactLab from "./ExactLab.jsx";
import LearnedBridge from "./pages/LearnedBridge.jsx";
import Generalisation from "./pages/Generalisation.jsx";
import BDHCQ from "./pages/BDHCQ.jsx";
import Evidence from "./pages/Evidence.jsx";

const NAV = [
  { id: "exact", label: "1–3 · Exact Lab" },
  { id: "bridge", label: "4 · AI Bridge" },
  { id: "gen", label: "5 · Generalisation" },
  { id: "bdh", label: "6 · BDH-CQ" },
  { id: "evidence", label: "7 · Evidence" },
];

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
      {page === "bdh" && <BDHCQ />}
      {page === "evidence" && <Evidence />}
    </div>
  );
}
