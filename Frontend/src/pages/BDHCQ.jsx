import React from "react";
import TwoRecurrences from "../components/TwoRecurrences.jsx";
import { EvidenceBadge, EvidenceNote } from "../components/Evidence.jsx";
import {
  Equation, NextButton, PageHeader, Panel, Takeaway,
} from "../components/Common.jsx";

// Every figure on this page is quoted from a published source. None of it is
// measured by Latent Loop Lab, and the badge beside each block says so.
// Table 5 of the BDH-CQ report, reproduced exactly as published.
const EFFORT = [
  { effort: "LOW", arc: "21%", cost: "22%" },
  { effort: "MEDIUM", arc: "27%", cost: "11%" },
  { effort: "HIGH", arc: "29.5%", cost: "0%" },
];

const LINEAGE = [
  {
    name: "BDH",
    full: "Dragon Hatchling",
    body:
      "The architecture family: high-dimensional positive activations, low-rank communication, and a recurrent associative state in which attention is reformulated as synaptic memory. Reported to rival GPT-2 at matched parameter counts from 10M to 1B.",
  },
  {
    name: "BDH-GPU",
    full: "GPU-oriented formulation",
    body:
      "The same dynamics expressed for hardware: BDH layers combining ReLU low-rank transformations with linear attention in a large neuron space. Do not read this as a state-space model in the Mamba sense.",
  },
  {
    name: "BDH-CQ",
    full: "In-context learning with recurrent latent reasoning",
    body:
      "The reasoning system this page connects to. Demonstrations update a recurrent memory at inference time; a query is then answered by iterating in a latent workspace, without verbalising intermediate steps.",
  },
];

export default function BDHCQ({ onNext }) {
  return (
    <div className="page">
      <PageHeader
        step="Section 6 of 8"
        question="Where does this appear in a real system?"
        goal="Connect the mechanism to a published system without claiming the two are the same thing."
      />

      <div className="notice">
        <strong>Read this before the diagrams.</strong>
        <p>
          Everything on this page describes the published conceptual
          decomposition of BDH-CQ. Its dimensions, exact update rules and
          implementation details are stated as proprietary in the report itself,
          and are outside this project. The models in sections 1 to 5 are an
          independent reimplementation written for teaching, not a BDH or BDH-CQ
          checkpoint.
        </p>
      </div>

      <Panel title="Where BDH-CQ comes from">
        <div className="grid-3">
          {LINEAGE.map((item) => (
            <div key={item.name}>
              <h4 style={{ marginTop: 0 }}>{item.name}</h4>
              <p className="muted" style={{ marginTop: -4, marginBottom: 8 }}>{item.full}</p>
              <p>{item.body}</p>
            </div>
          ))}
        </div>
        <EvidenceNote evidence={{
          evidenceType: "Paper-reported result",
          source:
            "Kosowski et al., The Dragon Hatchling, arXiv:2509.26507; " +
            "Engdahl et al., BDH-CQ, arXiv:2608.09888, sections 3.1 to 3.3",
        }} />
      </Panel>

      <Panel
        title="Two recurrences, routinely confused"
        subtitle="These are not the same process, and calling both of them “reasoning steps” is the mistake this page exists to prevent. Below is a toy where you can drive each one separately and watch them behave differently."
      >
        <TwoRecurrences />
      </Panel>

      <Panel title="The equations the toy mirrors">
        <div className="grid-2">
          <div>
            <h4 style={{ marginTop: 0 }}>Recurrence over demonstrations</h4>
            <Equation caption="equation (1) of the BDH-CQ report">
              S_t = U_θ(S_t₋₁, D_t)
            </Equation>
            <p>
              D_t is the content of the t-th demonstration and θ stays fixed.
              Information available to later inputs depends on associations
              accumulated from earlier ones. The report relates this to
              attention, fast-weight memory and linear-attention views of
              contextual association, naming the additive special case
              S_t = S_t₋₁ + U_θ(D_t) — which is exactly what the toy above does.
            </p>
            <p className="muted">
              Nothing in sections 1 to 5 corresponds to this. The graph lab has
              no demonstrations and does no in-context learning at all.
            </p>
          </div>

          <div>
            <h4 style={{ marginTop: 0 }}>Recurrence over query-time computation</h4>
            <Equation caption="equations (2) to (4), with S_K held fixed throughout">
              {`H₀   = E_θ(x*, S_K)\nH_r₊₁ = F_θ(H_r, S_K)\nŷ    = G_θ(H_R)`}
            </Equation>
            <p>
              This workspace carries computation for the current query. It
              advances without consuming new input, and R is a budget that can
              be chosen at inference time. S_K appears at every step and never
              changes during the loop: context acquisition has finished, and
              only latent refinement is running.
            </p>
            <p className="muted">
              This is the recurrence sections 1 to 5 are an analogue of.
            </p>
          </div>
        </div>

        <div className="keyline">
          The first is recurrence over demonstrations and context. The second is
          recurrence over query-time latent computation. Only the second is a
          reasoning step.
        </div>

        <EvidenceNote evidence={{
          evidenceType: "Paper-reported result",
          source: "Engdahl et al., BDH-CQ, arXiv:2608.09888, sections 3.2 and 3.3",
        }} />
      </Panel>

      <Panel title="What maps onto what">
        <div className="duo">
          <div>
            <span className="head">Latent Loop Lab</span>
            <p className="mono" style={{ fontSize: 14, margin: "4px 0" }}>
              h⁽⁰⁾ → h⁽¹⁾ → ⋯ → h⁽ᴿ⁾ → ŷ
            </p>
            <span className="note">exact mechanism, sections 1 to 3</span>
            <p className="mono" style={{ fontSize: 14, margin: "12px 0 4px", color: "var(--latent)" }}>
              z⁽⁰⁾ → z⁽¹⁾ → ⋯ → z⁽ᴿ⁾ → ŷ
            </p>
            <span className="note">learned extension, sections 4 and 5</span>
          </div>
          <div>
            <span className="head">BDH-CQ</span>
            <p className="mono" style={{ fontSize: 14, margin: "4px 0" }}>
              H₀ → H₁ → ⋯ → H_R → ŷ
            </p>
            <span className="note">query-time latent workspace</span>
            <p style={{ marginTop: 14 }} className="muted">
              An abstract analogue, one direction only. Sharing a recurrence
              shape does not make two systems the same system.
            </p>
          </div>
        </div>

        <h4>Where the analogy stops</h4>
        <ul className="crosses">
          <li>
            The graph lab has no contextual recurrence S_t at all, so half of
            BDH-CQ has no counterpart in sections 1 to 5.
          </li>
          <li>
            The lab's exact coordinates are interpretable by construction. A
            production latent workspace is not generally human-readable.
          </li>
          <li>
            The lab solves graph reachability. BDH-CQ solves ARC-style visual
            transformation tasks in a 150M-parameter system.
          </li>
          <li>
            Scale differs by four orders of magnitude, and behaviour at this
            scale licenses no claim about behaviour at that one.
          </li>
        </ul>
      </Panel>

      <Panel
        title="Published latent-effort operating points"
        subtitle="Reported for BDH-CQ on the public ARC-AGI-1 evaluation set. Quoted, not measured here."
      >
        <table>
          <thead>
            <tr>
              <th>Reasoning effort</th>
              <th className="num">ARC-AGI-1 pass@2</th>
              <th className="num">Cost reduction</th>
            </tr>
          </thead>
          <tbody>
            {EFFORT.map((row) => (
              <tr key={row.effort}>
                <td>{row.effort}</td>
                <td className="num">{row.arc}</td>
                <td className="num">{row.cost}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <p style={{ marginTop: 16 }}>
          Cost reduction is measured against the HIGH setting, so HIGH is 0% by
          definition. More latent effort buys accuracy and costs compute, which
          is the same trade the depth control in this lab makes. That is why the
          table is here: not as a result of this project, but as the real-system
          version of the variable you have been moving.
        </p>

        <div className="evidence-note">
          <EvidenceBadge type="Paper-reported result" />
          <span>
            Engdahl et al., BDH-CQ, arXiv:2608.09888, table 5. Not reproduced
            here and not on the same footing as anything in sections 1 to 5.
          </span>
        </div>
      </Panel>

      <Panel
        title="How strong is the evidence, exactly"
        subtitle="The categories the task's own rules insist on keeping apart."
      >
        <div className="scroll-x">
          <table>
            <thead>
              <tr>
                <th>Reported</th>
                <th>What kind of evidence that is</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>29.5% pass@2 on public ARC-AGI-1 at a computed $0.00070 per task</td>
                <td>
                  A benchmark result reported by the system's developers. Not a
                  deployment, and not an outside reproduction.
                </td>
              </tr>
              <tr>
                <td>A black-box audit reproducing that 29.5% score</td>
                <td>
                  Described in the report as conducted by co-authors from Bielik
                  and New York University under a documented protocol without
                  access to weights. Stronger than a bare self-report, weaker
                  than a fully independent third-party reproduction.
                </td>
              </tr>
              <tr>
                <td>Cost comparisons against other leaderboard systems</td>
                <td>
                  Computed from measured hardware time for BDH-CQ, against costs
                  the leaderboard reports for others, which may be hardware
                  estimates or API prices. Not a like-for-like measurement.
                </td>
              </tr>
              <tr>
                <td>The MIN versus STANDARD effort comparison</td>
                <td>
                  Reported as 111/400 against 118/400 pass@2, a 1.75 point gap
                  at one third the cost. Deliberately kept out of the effort
                  table above rather than merged into it.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ marginTop: 14 }}>
          None of these are measurements of this lab. They are listed so the
          strength of each claim is visible beside the claim itself.
        </p>
      </Panel>

      <Takeaway>
        Query-time latent recurrence is a real component of a real system, and it
        is the component this lab models. Contextual recurrence is a separate
        mechanism, driven by input rather than by budget, that no amount of
        latent compute can substitute for.
      </Takeaway>

      <NextButton onNext={onNext}>Next: what counts as evidence here?</NextButton>
    </div>
  );
}
