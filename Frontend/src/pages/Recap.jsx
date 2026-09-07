import React, { useState } from "react";
import { PageHeader, Panel, Pill, Stat, Takeaway } from "../components/Common.jsx";

// The lesson's last step is the learner producing the explanation rather than
// receiving it. Each prompt is answered from memory first and only then
// checked, because a reveal you read before thinking teaches nothing. Self
// marks are kept in component state and never leave the page.

const CHECKS = [
  {
    id: "meaning",
    objective: "Say what recurrent depth means mechanically",
    prompt: "Someone asks what R actually is. Answer in one sentence, without using the word “deeper”.",
    answer:
      "R is the number of times one fixed update rule is applied to the state before anything is read out. In the graph mechanism that is the number of edges information can travel, so R is a distance budget, not a network size.",
    trap: "R is not the number of layers and not the number of output tokens. The parameter count is identical at every R.",
    section: 2,
  },
  {
    id: "predict",
    objective: "Predict the threshold before moving anything",
    prompt: "A graph has a shortest source-to-target path of 7 edges. At R = 5, is the model wrong?",
    answer:
      "Wrong is the wrong word for it. At R = 5 no path of 7 edges has been traversed, so the source cannot yet have influenced the target's state and the answer is not present in the state being read. What the model does with that absence is a separate question: it is not obliged to report uncertainty, and this one does not. The shipped seed-1 checkpoint, on a distance-7 graph at R = 5, returns 0.000014 for reachable — confidently wrong, not undecided. It recovers to 0.999 at R = 7, the first depth at which the path fits.",
    trap: "Accuracy near 0.5 on a balanced set of unreachable-at-this-depth cases is an average over many confident answers that happen to split evenly. It is not evidence that any single prediction is calibrated. Section 5's flat orange curve is the aggregate; the per-graph sweep in section 4 is where you see what one prediction actually looks like.",
    section: 4,
  },
  {
    id: "tokens",
    objective: "Tell apart generating tokens and updating a state",
    prompt: "What does this lab's mechanism do that a model writing out its reasoning does not?",
    answer:
      "It keeps intermediate work in a fixed-width state that is rewritten in place and emits nothing until the end. The memory cost does not grow with the amount of computation, and there is no readable trace of the intermediate steps. Both of those are consequences of the same choice.",
    trap: "The absence of a trace is a real cost, not only a saving. You give up the ability to inspect the reasoning.",
    section: 2,
  },
  {
    id: "shared",
    objective: "Read “same weights, more computation” off the architecture",
    prompt: "Why does the depth control only work because the weights are shared?",
    answer:
      "With one update block reused at every step, running deeper is a loop bound. With a separate block per step, depth is fixed when training ends: running at depth 5 is not a worse result, it is not a possible one, because those parameters were never created.",
    trap: "This is what separates recurrence from simply stacking more layers, and it is why the ablation in section 5 has a hard ceiling rather than a lower score.",
    section: 5,
  },
  {
    id: "limit",
    objective: "Tell apart insufficient computation and a limit of learning",
    prompt: "The learned model gets a distance-9 case wrong at R = 10. Which kind of failure is that, and how would you tell?",
    answer:
      "R is large enough for the information to arrive, so it is a limit of what was learned, not a shortage of depth. You tell them apart by comparing against the exact mechanism on the same graph: the exact layer is correct whenever R is at least the distance, so any gap below that line belongs to the learned dynamics.",
    trap: "Distance 9 is outside the training range of 4. More compute does not guarantee algorithmic extrapolation.",
    section: 5,
  },
  {
    id: "recurrences",
    objective: "Keep BDH-CQ's two recurrences apart",
    prompt: "BDH-CQ has two recurrences. What drives each, and which one is a reasoning step?",
    answer:
      "The contextual state S_t advances once per demonstration and is driven by new input; it stops when the demonstrations run out. The workspace H_r advances with no new input, with S_K held fixed, and its budget is chosen at inference time. Only the second is a reasoning step.",
    trap: "A missing association in S cannot be repaired by raising R. The section 6 toy lets you produce that failure deliberately.",
    section: 6,
  },
  {
    id: "limits",
    objective: "Name a limitation without being prompted",
    prompt: "State one thing this lab shows that argues against “more recurrence is better”.",
    answer:
      "Accuracy on cases the model had already solved peaks around R = 5 and erodes by R = 10, while the mean state norm saturates. Once the state stops changing meaningfully, more iterations add drift rather than information. Published work on looped language models reports the same shape at far larger scale.",
    trap: "The lab measures this rather than assuming it, and it is shown on the same chart as the wins.",
    section: 5,
  },
];

const CLAIM_PARTS = [
  { text: "Repeated application of a", tone: null },
  { text: "shared", tone: "why" },
  { text: "state-update rule increases", tone: null },
  { text: "effective computational depth", tone: "what" },
  { text: "without generating intermediate language tokens —", tone: null },
  { text: "within capacity and stability limits", tone: "caveat" },
];

const PART_NOTES = {
  why: "Shared, because unshared weights make depth a training-time decision. Section 5's ablation.",
  what: "Effective depth, not parameters. The count is identical at every R. Sections 1 and 4.",
  caveat: "The caveat is measured, not hedged. Accuracy erodes and the state saturates. Section 5.",
};

export default function Recap({ goTo }) {
  const [open, setOpen] = useState({});
  const [marks, setMarks] = useState({});
  const [highlight, setHighlight] = useState(null);

  const answered = Object.keys(marks).length;
  const got = Object.values(marks).filter(Boolean).length;

  return (
    <div className="page">
      <PageHeader
        step="Section 8 of 8"
        question="Can you say it in your own words?"
        goal="Answer each prompt from memory first, then check. The gap between the two is the part worth rereading."
      />

      <Panel
        title="The claim, part by part"
        subtitle="Hover or tap a highlighted phrase to see which section earns it."
      >
        <div className="claim-slab">
          {CLAIM_PARTS.map((part, index) =>
            part.tone ? (
              <em
                key={index}
                tabIndex={0}
                role="button"
                onMouseEnter={() => setHighlight(part.tone)}
                onMouseLeave={() => setHighlight(null)}
                onFocus={() => setHighlight(part.tone)}
                onBlur={() => setHighlight(null)}
                onClick={() => setHighlight(part.tone)}
                style={{ cursor: "help" }}
              >
                {part.text}
              </em>
            ) : (
              <React.Fragment key={index}>{part.text} </React.Fragment>
            )
          )}
          {CLAIM_PARTS.map((part, index) =>
            part.tone ? <React.Fragment key={`sp-${index}`}> </React.Fragment> : null
          )}
        </div>
        <p className="muted" style={{ minHeight: "3em" }}>
          {highlight ? PART_NOTES[highlight] : "Every clause in that sentence is doing work. Pick one."}
        </p>
      </Panel>

      <div className="stat-row">
        <Stat label="Prompts answered" value={`${answered} of ${CHECKS.length}`} />
        <Stat label="Marked correct" value={got} note="your own marking, kept on this page" />
        <div>
          <span className="small-label">Status</span>
          <Pill tone={answered === CHECKS.length ? "yes" : "no"}>
            {answered === CHECKS.length ? "all seven attempted" : "keep going"}
          </Pill>
        </div>
      </div>

      {CHECKS.map((check, index) => {
        const isOpen = open[check.id];
        const mark = marks[check.id];

        return (
          <section className="panel" key={check.id}>
            <span className="small-label">
              Objective {index + 1}: {check.objective}
            </span>
            <h3 style={{ marginTop: 6 }}>{check.prompt}</h3>

            {!isOpen ? (
              <button
                className="btn ghost"
                onClick={() => setOpen((o) => ({ ...o, [check.id]: true }))}
              >
                Answer it out loud, then reveal
              </button>
            ) : (
              <>
                <div className="box" style={{ borderLeft: "3px solid var(--signal)" }}>
                  <p style={{ margin: 0 }}>{check.answer}</p>
                </div>
                <p className="muted" style={{ marginTop: 12 }}>
                  <strong>Easy to get wrong:</strong> {check.trap}
                </p>

                <div style={{ display: "flex", gap: 8, alignItems: "center",
                              flexWrap: "wrap", marginTop: 14 }}>
                  <span className="small-label" style={{ margin: 0 }}>Did you have it?</span>
                  <button
                    className="chip"
                    aria-pressed={mark === true}
                    onClick={() => setMarks((m) => ({ ...m, [check.id]: true }))}
                  >
                    yes
                  </button>
                  <button
                    className="chip"
                    aria-pressed={mark === false}
                    onClick={() => setMarks((m) => ({ ...m, [check.id]: false }))}
                  >
                    not quite
                  </button>
                  {mark === false && (
                    <button className="btn ghost small" onClick={() => goTo(check.section)}>
                      Reread section {check.section}
                    </button>
                  )}
                </div>
              </>
            )}
          </section>
        );
      })}

      <Panel title="If you take one thing away">
        <p>
          Depth and output length are different resources. A system can spend
          more of the first without spending any of the second, and this lab
          lets you watch that happen and check it against ground truth. What it
          cannot tell you is whether that trade is worth making at scale, in a
          real model, on a real task. Section 6 points at where that question is
          actually being answered, and how strong the evidence there currently
          is.
        </p>
      </Panel>

      <Panel title="Where to go next">
        <ul className="plain">
          <li>
            <a href="https://arxiv.org/abs/2509.26507" target="_blank" rel="noreferrer">
              The Dragon Hatchling
            </a>{" "}
            for the architecture the last section connects to, including its
            synaptic-memory formulation of attention.
          </li>
          <li>
            <a href="https://arxiv.org/abs/2608.09888" target="_blank" rel="noreferrer">
              The BDH-CQ report
            </a>{" "}
            for the two-recurrence decomposition and the ARC-AGI-1 evaluation
            quoted in section 6.
          </li>
          <li>
            <a href="https://arxiv.org/abs/2502.05171" target="_blank" rel="noreferrer">
              Scaling up Test-Time Compute with Latent Reasoning
            </a>{" "}
            for recurrent depth as a compute axis in a language model.
          </li>
          <li>
            <a href="https://arxiv.org/abs/2605.26733" target="_blank" rel="noreferrer">
              Stabilizing Recurrent Dynamics for Test-Time Scalable Latent
              Reasoning in Looped Language Models
            </a>{" "}
            for the erosion-at-large-depth effect measured in section 5, at a
            scale where it matters.
          </li>
        </ul>
      </Panel>

      <Takeaway>
        If you can answer all seven prompts without opening them, you can
        explain recurrent latent computation to someone else, including what it
        does not buy.
      </Takeaway>

      <div className="page-nav">
        <button className="btn ghost" onClick={() => goTo(0)}>
          Back to the start
        </button>
      </div>
    </div>
  );
}
