import React, { useCallback, useEffect, useMemo, useState } from "react";
import Start from "./pages/Start.jsx";
import Hook from "./pages/Hook.jsx";
import Mechanism from "./pages/Mechanism.jsx";
import Verification from "./pages/Verification.jsx";
import LearnedBridge from "./pages/LearnedBridge.jsx";
import Generalisation from "./pages/Generalisation.jsx";
import BDHCQ from "./pages/BDHCQ.jsx";
import Evidence from "./pages/Evidence.jsx";
import Recap from "./pages/Recap.jsx";
import { labStatus } from "./lib/lab.js";

// The lesson is a sequence, not a dashboard: each page answers one question and
// hands the next one over. The rail is numbered because the order is the
// argument, and the measure at the bottom is how far through that argument the
// learner is.
const PAGES = [
  { id: "start", label: "Start here", component: Start,
    question: "What is this and who is it for?" },
  { id: "hook", label: "The dial", component: Hook,
    question: "Can a model think longer without saying more?" },
  { id: "mechanism", label: "The state", component: Mechanism,
    question: "What is the latent state doing?" },
  { id: "verification", label: "The check", component: Verification,
    question: "Is the effect real, or just animation?" },
  { id: "bridge", label: "Learned rule", component: LearnedBridge,
    question: "What if the rule is learned instead of designed?" },
  { id: "generalisation", label: "The limits", component: Generalisation,
    question: "More computation, or more memorisation?" },
  { id: "bdhcq", label: "BDH-CQ", component: BDHCQ,
    question: "Where does this appear in a real system?" },
  { id: "evidence", label: "Evidence", component: Evidence,
    question: "What counts as evidence here?" },
  { id: "recap", label: "Explain it back", component: Recap,
    question: "Can you say it in your own words?" },
];

function pageFromHash() {
  const id = window.location.hash.replace(/^#\/?/, "");
  const index = PAGES.findIndex((page) => page.id === id);
  return index === -1 ? 0 : index;
}

function initialTheme() {
  try {
    return localStorage.getItem("lll-theme") || "system";
  } catch {
    return "system";
  }
}

export default function App() {
  const [index, setIndex] = useState(pageFromHash);
  const [furthest, setFurthest] = useState(pageFromHash);
  const [status, setStatus] = useState(null);
  const [loadError, setLoadError] = useState("");
  const [theme, setTheme] = useState(initialTheme);

  useEffect(() => {
    labStatus().then(setStatus).catch((error) => setLoadError(error.message));
  }, []);

  // The URL carries the page, so a learner can link to the exact step they
  // want someone else to look at.
  useEffect(() => {
    const onHash = () => setIndex(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    setFurthest((f) => Math.max(f, index));
    if (window.location.hash.replace(/^#\/?/, "") !== PAGES[index].id) {
      window.history.replaceState(null, "", `#/${PAGES[index].id}`);
    }
  }, [index]);

  useEffect(() => {
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("lll-theme", theme);
    } catch {
      // A browser that blocks storage still gets the theme for this visit.
    }
  }, [theme]);

  const goTo = useCallback((next) => {
    setIndex(Math.max(0, Math.min(PAGES.length - 1, next)));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const goNext = useCallback(() => goTo(index + 1), [goTo, index]);

  // Arrow keys move through the lesson, so it is usable without a mouse.
  useEffect(() => {
    function onKey(event) {
      const tag = event.target.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "ArrowRight") goTo(index + 1);
      if (event.key === "ArrowLeft") goTo(index - 1);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goTo, index]);

  const Current = PAGES[index].component;
  const progress = useMemo(
    () => Math.round(((furthest + 1) / PAGES.length) * 100),
    [furthest]
  );

  const nextTheme = { system: "light", light: "dark", dark: "system" }[theme];
  const themeLabel = { system: "Match system", light: "Light", dark: "Dark" }[theme];

  return (
    <div className="app">
      <nav className="rail" aria-label="Lesson">
        <a className="wordmark" href="#/start" onClick={() => goTo(0)}>
          Latent Loop Lab
          <span>Recurrent latent computation, taken apart and measured.</span>
        </a>

        <div className="rail-nav">
          {PAGES.map((page, i) => (
            <button
              key={page.id}
              className={`rail-item ${i < furthest ? "seen" : ""}`}
              onClick={() => goTo(i)}
              title={page.question}
              aria-current={i === index ? "page" : undefined}
            >
              <span className="n">{i}</span>
              <span>{page.label}</span>
            </button>
          ))}
        </div>

        <div className="rail-foot">
          <div>
            <div className="measure" role="progressbar" aria-valuenow={progress}
                 aria-valuemin={0} aria-valuemax={100}
                 aria-label="Progress through the lesson">
              <i style={{ width: `${progress}%` }} />
            </div>
            <span className="measure-label">
              {furthest + 1} of {PAGES.length} sections opened
            </span>
          </div>
          <button
            className="theme-toggle"
            onClick={() => setTheme(nextTheme)}
            aria-label={`Colour theme: ${themeLabel}. Switch to ${nextTheme}.`}
          >
            {themeLabel}
          </button>
        </div>
      </nav>

      <main>
        {loadError && (
          <div className="error" role="alert">
            <strong>The data bundle did not load.</strong>
            <p>{loadError}</p>
            <p className="muted">
              Build it from the Backend directory with{" "}
              <code>python export_web.py</code>. Everything the lab computes
              runs in this browser, so no server needs to be running.
            </p>
          </div>
        )}

        {status && !status.liveInferenceAvailable && index >= 4 && index <= 5 && (
          <div className="notice" role="status">
            <strong>No trained checkpoint in this build.</strong>
            <p>
              This section has nothing to show. Run{" "}
              <code>python reproduce.py</code> and then{" "}
              <code>python export_web.py</code> in the Backend directory.
              Sections 1 to 3 do not need a checkpoint.
            </p>
          </div>
        )}

        <Current onNext={goNext} goTo={goTo} status={status} pages={PAGES} />
      </main>
    </div>
  );
}

export { PAGES };
