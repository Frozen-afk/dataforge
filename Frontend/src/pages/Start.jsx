import React, { useEffect, useRef, useState } from "react";
import GraphView from "../components/GraphView.jsx";
import { EVIDENCE, EvidenceBadge } from "../components/Evidence.jsx";
import { Panel, Stat } from "../components/Common.jsx";
import { exactPreset } from "../lib/lab.js";

// The first thing a visitor sees is the mechanism already running, not a
// description of it. The wavefront advances on load and stops on the answer,
// which is the whole lesson in about eight seconds; everything below is there
// for the reader who then wants to know who this is for and what it will cost
// them to follow.

const OBJECTIVES = [
  "Say what recurrent depth R means mechanically, in one sentence.",
  "Predict the depth at which a target activates before moving the control.",
  "Tell apart generating tokens and updating a latent state.",
  "Read “same weights, more computation” off a shared-weight recurrent model.",
  "Tell apart insufficient computation and a limit of what was learned.",
  "Keep BDH-CQ's contextual state S_t distinct from its query-time workspace H_r.",
  "Name a limitation or failure case without being prompted.",
];

const PREREQUISITES = [
  "What a directed graph is, and what a path through one is.",
  "Roughly what a recurrent update means: a state, a rule, applied again.",
  "Nothing else. No familiarity with post-Transformer work is assumed.",
];

export default function Start({ onNext, goTo }) {
  const [data, setData] = useState(null);
  const [R, setR] = useState(0);
  const [running, setRunning] = useState(true);
  const timer = useRef(null);

  useEffect(() => {
    let cancelled = false;
    exactPreset("line").then((result) => !cancelled && setData(result)).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!data || !running) return;

    timer.current = setInterval(() => {
      setR((current) => {
        if (current >= data.bfsDistance) {
          setRunning(false);
          return data.bfsDistance;
        }
        return current + 1;
      });
    }, 780);

    return () => clearInterval(timer.current);
  }, [data, running]);

  const state = data?.trajectory?.[R];
  const distance = data?.bfsDistance;
  const arrived = data && R >= distance;

  return (
    <div className="page">
      <header className="page-header">
        <span className="step">Start here</span>
        <h1>A model can compute longer without saying more.</h1>
        <p className="goal">
          That sentence is the whole claim, and it is meant to be falsifiable.
          This lab hands you the one variable it depends on and lets you try to
          break it.
        </p>
      </header>

      {data && (
        <>
          <GraphView
            graph={data.graph}
            source={data.source}
            target={data.target}
            state={state}
            epsilon={data.epsilon}
            caption={
              arrived
                ? `The target activated after ${distance} updates. No text was generated and no parameter changed.`
                : `Activation after ${R} update${R === 1 ? "" : "s"}. It moves one column per update.`
            }
          />

          <div className="readout" style={{ marginTop: 18 }}>
            <Stat label="Updates applied" value={R} />
            <Stat label="Tokens produced" value={0} note="none, at any depth" />
            <Stat
              label="Target"
              value={arrived ? "active" : "waiting"}
              note={arrived ? `arrived at R = ${distance}` : `needs R = ${distance}`}
            />
            <Stat label="Parameters changed" value={0} note="the rule is fixed" />
          </div>

          <p className="footnote" style={{ marginTop: 12 }}>
            <EvidenceBadge type="Live computation" /> This is running now, in
            your browser, and is checked against a breadth-first search that
            never reads it. Nothing on this screen is a scripted animation.
          </p>
        </>
      )}

      <div className="claim-slab">
        Repeated application of a <em>shared</em> state-update rule increases
        effective computational depth without generating intermediate language
        tokens — within the learned dynamics' capacity and stability limits.
      </div>

      <Panel title="Who this is for">
        <p>
          A student or engineer who knows some machine learning and some graph
          theory, and has not read the latent-reasoning or post-Transformer
          literature. If you have heard that models can “think longer” and want
          to know what that means mechanically, this is aimed at you.
        </p>
        <h4>What you need first</h4>
        <ul className="plain">
          {PREREQUISITES.map((item) => <li key={item}>{item}</li>)}
        </ul>
      </Panel>

      <Panel title="What you will be able to do afterwards">
        <ul className="ticks">
          {OBJECTIVES.map((item) => <li key={item}>{item}</li>)}
        </ul>
        <p className="muted">
          The last section asks you to do exactly these, and marks your own
          answers against the lab's.
        </p>
      </Panel>

      <Panel
        title="The sixty-second version"
        subtitle="If you only have a minute, do this and stop."
      >
        <ol>
          <li>
            Open the next section. The mechanism is already running.
          </li>
          <li>
            Drag the depth control below 4. The target goes dark.
          </li>
          <li>
            Drag it back to 4. The target lights, and the panel beside it shows
            that a separate algorithm agrees the shortest path is 4 edges long.
          </li>
        </ol>
        <p>
          You have now watched extra computation happen inside a state update
          rather than in generated output, and checked the result against
          ground truth. The remaining sections ask whether that survives when
          the update rule is learned rather than designed.
        </p>
        <button className="btn" onClick={onNext}>Start the sixty seconds</button>
      </Panel>

      <Panel
        title="How to read the labels"
        subtitle="Every number in this lab carries exactly one of these, and they are not interchangeable."
      >
        <div className="grid-auto">
          {Object.entries(EVIDENCE).map(([type, info]) => (
            <div key={type}>
              <EvidenceBadge type={type} />
              <p className="muted" style={{ marginTop: 8 }}>{info.meaning}</p>
            </div>
          ))}
        </div>
        <p className="keyline">
          No result quoted from a paper appears anywhere on the same visual
          footing as something measured here. Keeping those apart is part of
          the project, not paperwork around it.
        </p>
      </Panel>

      <Panel title="What is a toy here, and what is not">
        <p>
          The graph mechanism and the small learned model were both written for
          this lab. They are an independent reimplementation of a general idea.
          Neither is BDH, neither is BDH-CQ, and no checkpoint of either was
          run. Section 6 connects the idea to the published BDH-CQ equations and
          says plainly where the analogy stops.
        </p>
        <p>
          <button className="btn ghost small" onClick={() => goTo(7)}>
            Jump to the evidence ledger
          </button>
        </p>
      </Panel>

      <div className="page-nav">
        <button className="btn" onClick={onNext}>
          Begin: can a model think longer without saying more?
        </button>
      </div>
    </div>
  );
}
