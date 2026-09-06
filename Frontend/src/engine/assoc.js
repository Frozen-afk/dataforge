// A minimal associative-memory toy, used on the BDH-CQ page.
//
// WHAT THIS IS
// A ten-line system with the same two-recurrence *shape* BDH-CQ describes, so
// a learner can watch the two behave instead of reading that they differ:
//
//     contextual    S_t     = U(S_{t-1}, D_t)      advances per demonstration
//     query-time    H_{r+1} = F(H_r, S_K)          advances per latent step
//
// WHAT THIS IS NOT
// It is not BDH, not BDH-CQ, and not a reimplementation of either. There is no
// training, no learned parameter, and nothing here was taken from a Pathway
// checkpoint. It is a hand-built illustration of a structural distinction, and
// the page says so beside it.
//
// The update U is a Hebbian outer-product write, which is the special case the
// BDH-CQ report names when relating its contextual memory to linear attention:
// state that accumulates additively per demonstration. The update F is
// associative recall, applied repeatedly with S held fixed.
//
// Two properties make the distinction visible rather than asserted:
//
//   * S stops advancing when demonstrations run out. No latent budget can
//     substitute for an association that was never written.
//   * H advances with no new input at all. Its budget is chosen at query time,
//     after the context is closed.

export const SYMBOLS = ["A", "B", "C", "D", "E", "F"];
export const N = SYMBOLS.length;

/** A deterministic successor function: one cycle over all six symbols. */
export function makeRule(seed) {
  const order = [...Array(N).keys()];

  // Fisher-Yates with a small deterministic generator, so a task id always
  // gives the same rule and the page is reproducible.
  let state = (seed * 2654435761) >>> 0;
  const next = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };

  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }

  // successor[order[i]] = order[i+1], wrapping. A single cycle means every
  // symbol has exactly one successor and one predecessor.
  const successor = new Array(N);
  for (let i = 0; i < N; i++) successor[order[i]] = order[(i + 1) % N];

  return { successor, order };
}

/** Every demonstration this task could show, in a fixed order. */
export function allDemonstrations(rule) {
  return rule.order.map((from) => ({ from, to: rule.successor[from] }));
}

/**
 * S_t = S_{t-1} + e_to (x) e_from^T
 *
 * Returns every intermediate state, so the page can step through the
 * demonstrations and show what each one wrote.
 */
export function accumulateContext(demonstrations) {
  let S = Array.from({ length: N }, () => new Array(N).fill(0));
  const states = [S.map((row) => row.slice())];

  for (const { from, to } of demonstrations) {
    S = S.map((row) => row.slice());
    S[to][from] += 1;
    states.push(S.map((row) => row.slice()));
  }

  return states;
}

function matVec(S, h) {
  const out = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    let sum = 0;
    for (let j = 0; j < N; j++) sum += S[i][j] * h[j];
    out[i] = sum;
  }
  return out;
}

/**
 * H_0 = e_{x*};  H_{r+1} = normalise(S_K H_r)
 *
 * The context state S_K appears at every step and never changes during the
 * loop. Context acquisition has finished; only latent refinement is running.
 */
export function latentReasoning(S, start, R) {
  let h = new Array(N).fill(0);
  h[start] = 1;

  const trajectory = [h.slice()];
  const readouts = [decode(h)];

  for (let r = 0; r < R; r++) {
    const raw = matVec(S, h);
    const mass = raw.reduce((a, b) => a + b, 0);

    // A zero state means the association needed at this step was never
    // written. The state stays zero rather than being renormalised into a
    // confident guess, so the interface can report "no answer" honestly.
    h = mass > 0 ? raw.map((v) => v / mass) : raw;

    trajectory.push(h.slice());
    readouts.push(decode(h));
  }

  return { trajectory, readouts, answer: readouts[readouts.length - 1] };
}

function decode(h) {
  const mass = h.reduce((a, b) => a + b, 0);
  if (mass <= 1e-12) return { symbol: null, confidence: 0 };

  let best = 0;
  for (let i = 1; i < N; i++) if (h[i] > h[best]) best = i;
  return { symbol: best, confidence: h[best] };
}

/** Ground truth: apply the real rule R times. Never consults S. */
export function groundTruth(rule, start, R) {
  let node = start;
  for (let r = 0; r < R; r++) node = rule.successor[node];
  return node;
}
