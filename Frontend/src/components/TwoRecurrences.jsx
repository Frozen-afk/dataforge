import React, { useMemo, useState } from "react";
import { EvidenceBadge } from "./Evidence.jsx";
import { DepthAxis, Pill, Stat } from "./Common.jsx";
import {
  SYMBOLS, N, makeRule, allDemonstrations, accumulateContext, latentReasoning,
  groundTruth,
} from "../engine/assoc.js";

// Two controls, two variables, two equations. The learner moves K and watches
// S fill in; then moves R and watches H iterate with S held fixed. The failure
// the page most wants to produce is easy to reach: cut K short and no latent
// budget can recover the missing association.

export default function TwoRecurrences() {
  const [task, setTask] = useState(1);
  const [K, setK] = useState(6);
  const [R, setR] = useState(3);
  const [start, setStart] = useState(0);

  const rule = useMemo(() => makeRule(task), [task]);
  const demonstrations = useMemo(() => allDemonstrations(rule), [rule]);
  const shown = demonstrations.slice(0, K);
  const contextStates = useMemo(() => accumulateContext(shown), [shown]);
  const S = contextStates[contextStates.length - 1];

  const run = useMemo(() => latentReasoning(S, start, R), [S, start, R]);
  const truth = groundTruth(rule, start, R);
  const answer = run.answer;
  const correct = answer.symbol === truth;

  // Where the chain leaves the demonstrated set. This is what turns "wrong"
  // into a diagnosis rather than a verdict.
  const missingAt = useMemo(() => {
    const written = new Set(shown.map((d) => d.from));
    let node = start;
    for (let r = 0; r < R; r++) {
      if (!written.has(node)) return { step: r, symbol: node };
      node = rule.successor[node];
    }
    return null;
  }, [shown, start, R, rule]);

  return (
    <div>
      <div className="controls">
        <label>
          Task
          <select value={task} onChange={(e) => setTask(Number(e.target.value))}>
            {[1, 2, 3, 4].map((t) => (
              <option key={t} value={t}>Hidden rule {t}</option>
            ))}
          </select>
        </label>
        <label>
          Query starts at
          <select value={start} onChange={(e) => setStart(Number(e.target.value))}>
            {SYMBOLS.map((symbol, index) => (
              <option key={symbol} value={index}>{symbol}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="split even">
        {/* ---------------------------------------------- contextual ---- */}
        <div>
          <h4 style={{ marginTop: 0 }}>Contextual recurrence</h4>
          <div className="equation">
            <code>S_t = U(S_t₋₁, D_t)</code>
            <span className="caption">
              one demonstration absorbed per step, driven by new input
            </span>
          </div>

          <div className="box">
            <DepthAxis
              id="assoc-k"
              label="Demonstrations shown, K"
              value={K}
              min={0}
              max={N}
              onChange={setK}
              hint="This recurrence stops when the demonstrations run out. Nothing can advance it further."
            />
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "14px 0" }}>
            {demonstrations.map((demo, index) => (
              <span
                key={index}
                className={index < K ? "badge live" : "badge other"}
                style={{ fontFamily: "var(--mono)" }}
              >
                {SYMBOLS[demo.from]} → {SYMBOLS[demo.to]}
              </span>
            ))}
          </div>

          <span className="small-label">State S after {K} demonstration{K === 1 ? "" : "s"}</span>
          <Matrix S={S} />
          <p className="muted">
            Each demonstration writes one association into the state: an outer
            product added in place. Nothing is stored per token and nothing
            grows with the number of demonstrations except the values already
            in this fixed-size table.
          </p>
        </div>

        {/* --------------------------------------------- query-time ----- */}
        <div>
          <h4 style={{ marginTop: 0 }}>Query-time recurrence</h4>
          <div className="equation" style={{ borderLeftColor: "var(--latent)" }}>
            <code>H₀ = E(x*, S_K){"\n"}H_r₊₁ = F(H_r, S_K){"\n"}ŷ = G(H_R)</code>
            <span className="caption">
              refined repeatedly with S_K held fixed, driven by budget rather
              than by input
            </span>
          </div>

          <div className="box">
            <DepthAxis
              id="assoc-r"
              label="Latent steps, R"
              value={R}
              min={0}
              max={8}
              onChange={setR}
              hint="No new input is consumed here. You are choosing a compute budget after the context is closed."
            />
          </div>

          <span className="small-label" style={{ marginTop: 14 }}>
            H_r at each step
          </span>
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th className="num">r</th>
                  {SYMBOLS.map((symbol) => (
                    <th className="num" key={symbol}>{symbol}</th>
                  ))}
                  <th>reads as</th>
                </tr>
              </thead>
              <tbody>
                {run.trajectory.map((h, r) => (
                  <tr key={r} className={r === R ? "current" : ""}>
                    <td className="num">{r}</td>
                    {h.map((value, index) => (
                      <td className="num" key={index}
                          style={{ color: value > 0.5 ? "var(--latent)" : "var(--ink-3)" }}>
                        {value === 0 ? "·" : value.toFixed(2)}
                      </td>
                    ))}
                    <td>
                      {run.readouts[r].symbol === null
                        ? "nothing"
                        : SYMBOLS[run.readouts[r].symbol]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="duo" style={{ marginTop: 24 }}>
        <div>
          <span className="head">What the toy answers</span>
          <span className="verdict">
            {answer.symbol === null ? "no answer" : SYMBOLS[answer.symbol]}
          </span>
          <span className="note">
            {answer.symbol === null
              ? "the state went to zero: the association it needed was never written"
              : `confidence ${(answer.confidence * 100).toFixed(0)}%, after ${R} latent step${R === 1 ? "" : "s"}`}
          </span>
        </div>
        <div>
          <span className="head">The rule's actual answer</span>
          <span className="verdict">{SYMBOLS[truth]}</span>
          <span className="note">
            the hidden successor applied {R} time{R === 1 ? "" : "s"} to {SYMBOLS[start]},
            computed without consulting S
          </span>
        </div>
      </div>

      <div className={`verdict-bar ${correct ? "good" : "bad"}`}>
        <strong>{correct ? "Agreement" : "Disagreement"}</strong>
        <p>
          {correct ? (
            <>
              The toy reached the right symbol. It never saw this query, and
              nothing was learned during the loop: the answer came from
              iterating over associations written earlier.
            </>
          ) : missingAt ? (
            <>
              The chain needs {SYMBOLS[missingAt.symbol]} → ? at latent step{" "}
              {missingAt.step + 1}, and that demonstration was never shown. This
              is a context failure, not a compute failure. Raising R cannot fix
              it; raising K can. Try it.
            </>
          ) : (
            <>
              The toy and the rule disagree even though every needed
              association was written. Check the readout table above to see
              where the state stopped concentrating.
            </>
          )}
        </p>
      </div>

      <div className="readout" style={{ marginTop: 20 }}>
        <Stat label="Associations written" value={K} note="advances only with input" />
        <Stat label="Latent steps taken" value={R} note="advances with no input" />
        <div>
          <span className="small-label">Raising R when K is short</span>
          <Pill tone={missingAt ? "alarm" : "yes"}>
            {missingAt ? "cannot help" : "nothing missing"}
          </Pill>
        </div>
      </div>

      <p className="footnote" style={{ marginTop: 16 }}>
        <EvidenceBadge type="Live computation" /> This toy runs in your browser
        and is a hand-built illustration of a structural distinction. It is not
        BDH, not BDH-CQ, and contains no trained parameter. The equations it
        mirrors are quoted from the BDH-CQ report below.
      </p>
    </div>
  );
}

function Matrix({ S }) {
  return (
    <div className="scroll-x">
      <table>
        <thead>
          <tr>
            <th />
            {SYMBOLS.map((symbol) => (
              <th className="num" key={symbol}>{symbol}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {S.map((row, i) => (
            <tr key={i}>
              <th className="num">{SYMBOLS[i]}</th>
              {row.map((value, j) => (
                <td
                  className="num"
                  key={j}
                  style={{
                    color: value ? "var(--signal)" : "var(--ink-3)",
                    fontWeight: value ? 600 : 400,
                  }}
                >
                  {value || "·"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
