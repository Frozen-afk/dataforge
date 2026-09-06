import React, { useEffect, useRef, useState } from "react";
import GraphView from "../components/GraphView.jsx";
import { EvidenceNote } from "../components/Evidence.jsx";
import {
  DepthAxis, ErrorBanner, Loading, NextButton, PageHeader, Panel, Stat,
  Takeaway, Verdict,
} from "../components/Common.jsx";
import { exactPreset, MAX_R } from "../lib/lab.js";

export default function Hook({ onNext }) {
  const [data, setData] = useState(null);
  const [R, setR] = useState(0);
  const [error, setError] = useState("");
  const [autoplaying, setAutoplaying] = useState(true);
  const timer = useRef(null);

  // The whole trajectory h(0) .. h(12) is computed once. Scrubbing R then reads
  // a state that already exists, so the control redraws in the same frame the
  // learner moves it. This is the interaction the lesson rests on; it is not
  // allowed to wait on anything.
  useEffect(() => {
    let cancelled = false;
    exactPreset("line")
      .then((result) => !cancelled && setData(result))
      .catch((err) => !cancelled && setError(err.message));
    return () => { cancelled = true; };
  }, []);

  // Open with the mechanism in motion rather than a blank canvas and a run
  // button. It stops on the answer and hands over.
  useEffect(() => {
    if (!data || !autoplaying) return;

    timer.current = setInterval(() => {
      setR((current) => {
        if (current >= data.bfsDistance) {
          setAutoplaying(false);
          return data.bfsDistance;
        }
        return current + 1;
      });
    }, 760);

    return () => clearInterval(timer.current);
  }, [data, autoplaying]);

  function takeOver(value) {
    setAutoplaying(false);
    clearInterval(timer.current);
    setR(value);
  }

  if (error) return <div className="page"><ErrorBanner message={error} /></div>;
  if (!data) return <div className="page"><Loading label="Starting the mechanism" /></div>;

  const state = data.trajectory[R];
  const distance = data.bfsDistance;
  const arrived = state[data.target] > data.epsilon;
  const short = Math.max(0, distance - R);

  return (
    <div className="page">
      <PageHeader
        step="Section 1 of 8"
        question="Can a model think longer without saying more?"
        goal="Separate two ways of spending computation: producing more output, or updating an internal state more times."
      />

      <p>
        Below is a four-hop graph. The rule that updates the state is fixed and
        never changes between steps. The only thing you control is how many
        times that rule gets applied.
      </p>

      <div className="split">
        <div>
          <GraphView
            graph={data.graph}
            source={data.source}
            target={data.target}
            state={state}
            epsilon={data.epsilon}
            caption={`State after ${R} update${R === 1 ? "" : "s"}. Teal edges are carrying activation this step; the orange path is the shortest route to the target.`}
          />
        </div>

        <div className="stack">
          <div className="box">
            <DepthAxis
              id="hook-depth"
              value={R}
              min={0}
              max={MAX_R}
              onChange={takeOver}
              threshold={distance}
              hint={
                autoplaying
                  ? "Running. Grab the control to take over."
                  : "Drag it. Nothing else on this page changes."
              }
            />
          </div>

          <div className="readout">
            <Stat label="Updates applied" value={R} />
            <Stat label="Tokens produced" value={0} />
            <Stat
              label="Nodes reached"
              value={state.filter((v) => v > data.epsilon).length}
              note={`of ${data.graph.n}`}
            />
          </div>

          {arrived ? (
            <Verdict tone="good" title="That is the whole idea.">
              The target activated at R = {distance}. No text was generated, no
              parameter changed, and the update rule is the same one it was at
              R = 0. The system did more work by repeating one step, not by
              saying more.
            </Verdict>
          ) : (
            <Verdict tone="wait" title="Keep going.">
              Activation moves one column per update. The target sits {distance}{" "}
              edges from the source, so it cannot light up before R reaches{" "}
              {distance}. You are {short} update{short === 1 ? "" : "s"} short.
            </Verdict>
          )}

          <EvidenceNote evidence={data.evidence} />
        </div>
      </div>

      <div className="keyline">
        The update rule is unchanged. Only the number of times it is applied
        changes. That is what makes R a compute dial rather than a different
        model.
      </div>

      <Panel
        title="Why this is not just a graph question"
        subtitle="The same distinction is what people mean by reasoning in latent space."
      >
        <div className="grid-2">
          <div>
            <h4>Reasoning in tokens</h4>
            <p>
              A model that reasons in text has to write each intermediate step
              into its own output before it can use it. The work is visible, and
              it costs one forward pass per token emitted. You can read it, and
              anyone downstream has to pay for it.
            </p>
          </div>
          <div>
            <h4>Reasoning in state</h4>
            <p>
              The mechanism above is the other case. Intermediate work lives in
              a vector that is rewritten in place, nothing is emitted until the
              end, and the budget is set by how many times you apply the rule.
              You cannot read it, which is a real cost.
            </p>
          </div>
        </div>
      </Panel>

      <Takeaway>
        Extra computation can happen inside a state update rather than in
        generated output. The next section opens that state up so you can see
        exactly what it holds.
      </Takeaway>

      <NextButton onNext={onNext}>Next: what is the latent state doing?</NextButton>
    </div>
  );
}
