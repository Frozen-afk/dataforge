import React, { useEffect, useRef, useState } from "react";
import GraphView from "../components/GraphView.jsx";
import { EvidenceNote } from "../components/Evidence.jsx";
import { ErrorBanner, Loading, PageHeader, Pill, Slider, Takeaway } from "../components/Common.jsx";
import { api } from "../lib/api.js";

const MAX_R = 12;

export default function Hook({ onNext }) {
  const [data, setData] = useState(null);
  const [R, setR] = useState(0);
  const [error, setError] = useState("");
  const [autoplaying, setAutoplaying] = useState(true);
  const timer = useRef(null);

  // The whole trajectory h(0) .. h(12) is fetched once. Scrubbing R then reads
  // the state the backend already returned, so the slider responds instantly
  // instead of waiting on a request per step.
  useEffect(() => {
    let cancelled = false;

    api
      .exactPreset("line", MAX_R)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // Open with the mechanism already in motion rather than a blank canvas.
  useEffect(() => {
    if (!data || !autoplaying) return;

    timer.current = setInterval(() => {
      setR((current) => {
        if (current >= 4) {
          setAutoplaying(false);
          return 4;
        }
        return current + 1;
      });
    }, 700);

    return () => clearInterval(timer.current);
  }, [data, autoplaying]);

  function takeOver(value) {
    setAutoplaying(false);
    clearInterval(timer.current);
    setR(value);
  }

  if (error) {
    return (
      <div className="page">
        <ErrorBanner message={error} />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="page">
        <Loading label="Starting the mechanism..." />
      </div>
    );
  }

  const state = data.trajectory[R];
  const targetActive = state[data.target] > data.epsilon;
  const distance = data.bfsDistance;

  return (
    <div className="page">
      <PageHeader
        step="1"
        question="Can a model think longer without saying more?"
        goal="Separate two ways of spending computation: producing more output, or updating an internal state more times."
      />

      <div className="hero-claim">
        <p>
          Below is a four-hop graph. The rule that updates the state is fixed and
          never changes. The <strong>only</strong> thing you control is how many
          times that rule is applied.
        </p>
      </div>

      <div className="split">
        <div>
          <GraphView
            graph={data.graph}
            source={data.source}
            target={data.target}
            state={state}
            epsilon={data.epsilon}
            caption={`Latent state after ${R} update${R === 1 ? "" : "s"}. Blue edges are carrying activation this step.`}
          />
        </div>

        <div className="stack">
          <div className="control-card">
            <Slider
              id="hook-depth"
              label="Recurrent depth R"
              value={R}
              min={0}
              max={MAX_R}
              onChange={takeOver}
              hint={
                autoplaying
                  ? "Running... grab the slider to take over."
                  : "Drag it. Nothing else on this page changes."
              }
            />

            <div className="readout">
              <div>
                <span className="small-label">Target node</span>
                <Pill tone={targetActive ? "yes" : "no"}>
                  {targetActive ? "active" : "not active"}
                </Pill>
              </div>
              <div>
                <span className="small-label">Updates applied</span>
                <span className="big-value">{R}</span>
              </div>
              <div>
                <span className="small-label">Tokens produced</span>
                <span className="big-value">0</span>
              </div>
            </div>
          </div>

          <div className={`aha ${targetActive ? "reached" : ""}`}>
            {targetActive ? (
              <>
                <strong>That is the whole idea.</strong>
                <p>
                  The target activated at R = {distance}. No text was generated,
                  no parameters changed, and the update rule is the same one it
                  was at R = 0. The system did more work by repeating one step,
                  not by saying more.
                </p>
              </>
            ) : (
              <>
                <strong>Keep going.</strong>
                <p>
                  Activation moves one edge per update. The target sits{" "}
                  {distance} edges from the source, so it cannot light up before
                  R reaches {distance}. You are {Math.max(0, distance - R)} update
                  {distance - R === 1 ? "" : "s"} short.
                </p>
              </>
            )}
          </div>

          <div className="one-sentence">
            The update rule is unchanged; only the number of times we apply it
            changes.
          </div>

          <EvidenceNote evidence={data.evidence} />
        </div>
      </div>

      <Takeaway>
        Extra computation can happen inside a state update rather than in
        generated output. The next page opens up that state so you can see
        exactly what it holds.
      </Takeaway>

      <div className="page-nav">
        <button className="primary" onClick={onNext}>
          Next: what is the latent state doing?
        </button>
      </div>
    </div>
  );
}
