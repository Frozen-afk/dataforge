// The exact noisy-OR graph recurrence, ported from Backend/core/recurrent.py.
//
//     h(0)      = e_source
//     h(r+1)[v] = max( h(r)[v],  1 - PROD_{u -> v} (1 - alpha * h(r)[u]) )
//     y_hat_R   = [ h(R)[q] > epsilon ]
//
// Nothing in the update depends on r. That is the property that makes R a pure
// compute dial rather than a different model, and it is the reason the same
// three lines can be read in Python and in JavaScript without divergence.
//
// The invariant this is meant to satisfy, checked against the BFS oracle on
// every call:
//
//     h(R)[q] > epsilon   <=>   d(s, q) <= R

import { bfsDistance, incoming } from "./graph.js";

export const MAX_R = 12;
export const EPSILON = 1e-12;

/** One exact update applied to every node at once. */
export function oneStep(h, incomingAdjacency, alpha) {
  const next = h.slice();

  for (let v = 0; v < incomingAdjacency.length; v++) {
    let product = 1;
    for (const u of incomingAdjacency[v]) product *= 1 - alpha * h[u];

    const candidate = 1 - product;
    if (candidate > next[v]) next[v] = candidate;
  }

  return next;
}

/**
 * Run the recurrence for R steps and check it against BFS.
 *
 * Returns the whole trajectory h(0) .. h(R), so a page can scrub the depth
 * slider through states that were all computed in one pass rather than
 * recomputing per frame.
 */
export function runExact({
  graph,
  source,
  target,
  R,
  alpha = 1,
  epsilon = EPSILON,
}) {
  if (R < 0 || R > MAX_R) {
    throw new Error(`Recurrent depth R must be between 0 and ${MAX_R}.`);
  }
  if (!(alpha > 0 && alpha <= 1)) {
    throw new Error("Alpha must satisfy 0 < alpha <= 1.");
  }

  const incomingAdjacency = incoming(graph.n, graph.edges);

  let h = new Array(graph.n).fill(0);
  h[source] = 1;

  const trajectory = [h.slice()];
  for (let r = 0; r < R; r++) {
    h = oneStep(h, incomingAdjacency, alpha);
    trajectory.push(h.slice());
  }

  const estimate = h[target] > epsilon;

  // Ground truth from a procedure that never reads the recurrence.
  const distance = bfsDistance(graph.n, graph.edges, source, target);
  const reachable = distance !== null;
  const expected = reachable && distance <= R;

  return {
    graph,
    source,
    target,
    alpha,
    epsilon,
    R,
    trajectory,
    estimate,
    targetActivation: h[target],
    bfsDistance: distance,
    reachable,
    expected,
    invariantPass: estimate === expected,
    activeNodeCount: h.filter((value) => value > epsilon).length,
    activationMass: h.reduce((a, b) => a + b, 0),
    evidence: {
      evidenceType: "Live computation",
      source:
        "Exact noisy-OR graph recurrence, run in your browser, checked " +
        "against an independent breadth-first search",
      experimentId: "exact-graph-recurrence",
    },
  };
}

/**
 * The full trajectory once, at MAX_R, so scrubbing R costs nothing.
 * The invariant is then re-derived per depth from the cached states.
 */
export function runExactTrajectory({ graph, source, target, alpha = 1, maxR = MAX_R }) {
  const full = runExact({ graph, source, target, R: maxR, alpha });

  const perDepth = full.trajectory.map((state, R) => {
    const estimate = state[target] > full.epsilon;
    const expected = full.reachable && full.bfsDistance <= R;
    return {
      R,
      estimate,
      expected,
      invariantPass: estimate === expected,
      targetActivation: state[target],
      activeNodeCount: state.filter((value) => value > full.epsilon).length,
      activationMass: state.reduce((a, b) => a + b, 0),
    };
  });

  return { ...full, perDepth };
}
