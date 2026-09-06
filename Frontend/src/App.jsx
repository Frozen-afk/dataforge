import React, { useEffect, useState } from "react";
import Hook from "./pages/Hook.jsx";
import Mechanism from "./pages/Mechanism.jsx";
import Verification from "./pages/Verification.jsx";
import LearnedBridge from "./pages/LearnedBridge.jsx";
import Generalisation from "./pages/Generalisation.jsx";
import BDHCQ from "./pages/BDHCQ.jsx";
import Evidence from "./pages/Evidence.jsx";
import { api } from "./lib/api.js";

// The lesson is a sequence, not a dashboard. Each page answers one question and
// hands off to the next, which is why the nav is numbered and ordered.
const PAGES = [
  { id: "hook", label: "Hook", short: "1", component: Hook,
    question: "Can a model think longer without saying more?" },
  { id: "mechanism", label: "Mechanism", short: "2", component: Mechanism,
    question: "What is the latent state doing?" },
  { id: "verification", label: "Verification", short: "3", component: Verification,
    question: "Is the effect real or just animation?" },
  { id: "bridge", label: "AI bridge", short: "4", component: LearnedBridge,
    question: "What if the rule is learned?" },
  { id: "generalisation", label: "Generalisation", short: "5", component: Generalisation,
    question: "More computation or more memorisation?" },
  { id: "bdhcq", label: "BDH-CQ", short: "6", component: BDHCQ,
    question: "Where does this appear in real AI?" },
  { id: "evidence", label: "Evidence", short: "7", component: Evidence,
    question: "What counts as evidence here?" },
];

export default function App() {
  const [index, setIndex] = useState(0);
  const [status, setStatus] = useState(null);
  const [backendUp, setBackendUp] = useState(null);

  useEffect(() => {
    api.health().then(() => setBackendUp(true)).catch(() => setBackendUp(false));
    api.learnedStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  // Arrow keys move through the lesson, so the whole thing is usable without a
  // mouse.
  useEffect(() => {
    function onKey(event) {
      if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
      if (event.key === "ArrowRight") setIndex((i) => Math.min(PAGES.length - 1, i + 1));
      if (event.key === "ArrowLeft") setIndex((i) => Math.max(0, i - 1));
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const Current = PAGES[index].component;

  function goNext() {
    setIndex((i) => Math.min(PAGES.length - 1, i + 1));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function goTo(i) {
    setIndex(i);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div className="app">
      <nav className="topnav" aria-label="Lesson pages">
        <span className="brand">
          Latent Loop Lab
          <span className="brand-sub">recurrent latent computation, taken apart</span>
        </span>

        <div className="nav-pages">
          {PAGES.map((page, i) => (
            <button
              key={page.id}
              className={`navbtn ${i === index ? "active" : ""} ${i < index ? "visited" : ""}`}
              onClick={() => goTo(i)}
              title={page.question}
              aria-current={i === index ? "page" : undefined}
            >
              <span className="navnum">{page.short}</span>
              <span className="navlabel">{page.label}</span>
            </button>
          ))}
        </div>
      </nav>

      {backendUp === false && (
        <div className="global-banner error" role="alert">
          The backend is not responding. Start it with{" "}
          <code>uvicorn server:app --reload</code> from the <code>Backend</code>{" "}
          directory, then reload this page.
        </div>
      )}

      {backendUp && status && !status.liveInferenceAvailable && index >= 3 && (
        <div className="global-banner warn">
          No trained checkpoint found, so pages 4 and 5 have nothing to show. Run{" "}
          <code>python reproduce.py</code> in the <code>Backend</code> directory.
          Pages 1 to 3 work regardless.
        </div>
      )}

      <main>
        <Current onNext={goNext} status={status} />
      </main>

      <footer className="footer">
        <div>
          <strong>The claim being tested:</strong> repeated application of a
          shared state-update rule increases effective computational depth
          without generating intermediate language tokens — within the learned
          dynamics&rsquo; capacity and stability limits.
        </div>
        <div className="footer-meta">
          Toy models written for this lab. Not BDH, not BDH-CQ, not a benchmark
          leaderboard. Use arrow keys to move between pages.
        </div>
      </footer>
    </div>
  );
}
