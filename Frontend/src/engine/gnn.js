// The learned shared-weight recurrent GNN, running in the browser.
//
// This is a forward-pass port of Backend/learned/model.py over the weights
// exported by Backend/export_web.py. It is the same frozen checkpoint, read
// from JSON instead of a .pt file:
//
//     z_v^(0)   = E(x_v)                                 x_v = [is_source, is_target]
//     m_v^(r)   = AGG_{u in N-(v)} M(z_u^r, z_v^r)        AGG = max
//     z_v^(r+1) = F(z_v^r, m_v^r)                         gated (GRU) update
//     y_hat_R   = G(z_q^(R))
//
// with the same parameters at every step, so R buys computation and never
// parameters. Because the block is one object applied in a loop, running past
// the training depth is a loop bound here, not a new set of weights -- which
// is the claim the lesson is testing, visible directly in the code.
//
// Every operation below mirrors a specific PyTorch operation, and
// Backend/tests/test_js_parity.py runs this file under Node against PyTorch on
// seeded cases and fails if any output differs by more than 1e-6. If you edit
// this file, run that test.

// ---------------------------------------------------------------- primitives

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const relu = (x) => (x > 0 ? x : 0);

/** nn.Linear: y = x @ W^T + b, with W stored as [outFeatures, inFeatures]. */
function linear(x, weight, bias) {
  const out = new Float64Array(weight.length);

  for (let i = 0; i < weight.length; i++) {
    const row = weight[i];
    let sum = bias ? bias[i] : 0;
    for (let j = 0; j < row.length; j++) sum += row[j] * x[j];
    out[i] = sum;
  }

  return out;
}

/** Rows of a matrix laid out as an array of Float64Array node states. */
function linearRows(rows, weight, bias) {
  return rows.map((row) => linear(row, weight, bias));
}

function applyReLU(vector) {
  const out = new Float64Array(vector.length);
  for (let i = 0; i < vector.length; i++) out[i] = relu(vector[i]);
  return out;
}

function concat(a, b) {
  const out = new Float64Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

export function l2Norm(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) sum += vector[i] * vector[i];
  return Math.sqrt(sum);
}

/**
 * torch.nn.GRUCell.
 *
 * PyTorch packs the three gates into one matrix in the order (r, z, n), and
 * applies the reset gate to the *hidden* contribution only:
 *
 *     r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
 *     z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
 *     n = tanh   (W_in x + b_in + r * (W_hn h + b_hn))
 *     h' = (1 - z) * n + z * h
 *
 * Getting the r-inside-the-parenthesis detail wrong is the classic way a
 * hand-written GRU silently disagrees with the framework, which is what the
 * parity test exists to catch.
 */
function gruCell(x, h, weights) {
  const gi = linear(x, weights.weight_ih, weights.bias_ih);
  const gh = linear(h, weights.weight_hh, weights.bias_hh);

  const hidden = h.length;
  const next = new Float64Array(hidden);

  for (let i = 0; i < hidden; i++) {
    const r = sigmoid(gi[i] + gh[i]);
    const z = sigmoid(gi[hidden + i] + gh[hidden + i]);
    const n = Math.tanh(gi[2 * hidden + i] + r * gh[2 * hidden + i]);
    next[i] = (1 - z) * n + z * h[i];
  }

  return next;
}

/**
 * Scatter messages onto destination nodes.
 *
 * ``max`` is the trained default: reachability is a logical OR over incoming
 * neighbours, and max is its differentiable analogue -- the same role the
 * noisy-OR plays in the exact mechanism. A node with no incoming edge receives
 * no message, which PyTorch reaches by initialising to -inf and replacing what
 * stays infinite with zero. That is reproduced literally here.
 */
function aggregate(messages, dst, numNodes, dim, aggregation) {
  if (aggregation === "max") {
    const out = Array.from({ length: numNodes }, () =>
      new Float64Array(dim).fill(-Infinity)
    );

    for (let e = 0; e < dst.length; e++) {
      const row = out[dst[e]];
      const message = messages[e];
      for (let k = 0; k < dim; k++) {
        if (message[k] > row[k]) row[k] = message[k];
      }
    }

    for (const row of out) {
      for (let k = 0; k < dim; k++) {
        if (!Number.isFinite(row[k])) row[k] = 0;
      }
    }

    return out;
  }

  const out = Array.from({ length: numNodes }, () => new Float64Array(dim));
  const counts = new Float64Array(numNodes);

  for (let e = 0; e < dst.length; e++) {
    const row = out[dst[e]];
    const message = messages[e];
    for (let k = 0; k < dim; k++) row[k] += message[k];
    counts[dst[e]] += 1;
  }

  if (aggregation === "mean") {
    for (let v = 0; v < numNodes; v++) {
      const divisor = Math.max(1, counts[v]);
      for (let k = 0; k < dim; k++) out[v][k] /= divisor;
    }
  }

  return out;
}

// ------------------------------------------------------------------- model

export class RecurrentGNN {
  /**
   * @param {object} bundle the parsed contents of public/data/model.json
   */
  constructor(bundle) {
    this.meta = bundle;
    this.aggregation = bundle.aggregation || "max";
    this.hiddenDim = bundle.hiddenDim || 32;
    this.sharedWeights = bundle.sharedWeights !== false;

    const w = bundle.weights;

    this.encoder = [
      { weight: w["encoder.0.weight"], bias: w["encoder.0.bias"] },
      { weight: w["encoder.2.weight"], bias: w["encoder.2.bias"] },
    ];
    this.decoder = [
      { weight: w["decoder.0.weight"], bias: w["decoder.0.bias"] },
      { weight: w["decoder.2.weight"], bias: w["decoder.2.bias"] },
    ];

    // One block for the shared model; a list of blocks for the ablation, whose
    // whole point is that it runs out of them.
    this.blocks = this.sharedWeights
      ? [readBlock(w, "block")]
      : countBlocks(w).map((index) => readBlock(w, `blocks.${index}`));
  }

  /** Shared weights impose no ceiling; the ablation's ceiling is real. */
  get maxInferenceDepth() {
    return this.sharedWeights ? Infinity : this.blocks.length;
  }

  get depthIsBounded() {
    return !this.sharedWeights;
  }

  blockAt(r) {
    if (this.sharedWeights) return this.blocks[0];
    if (r >= this.blocks.length) {
      throw new Error(
        `This checkpoint has ${this.blocks.length} update blocks and cannot ` +
          `run at depth ${r + 1}. That is the ablation's point: without ` +
          `shared weights, inference depth is fixed when training ends.`
      );
    }
    return this.blocks[r];
  }

  /** Node features are [is_source, is_target] and nothing else. */
  encode(numNodes, source, target) {
    const states = [];

    for (let v = 0; v < numNodes; v++) {
      const x = new Float64Array([v === source ? 1 : 0, v === target ? 1 : 0]);
      const first = applyReLU(linear(x, this.encoder[0].weight, this.encoder[0].bias));
      states.push(linear(first, this.encoder[1].weight, this.encoder[1].bias));
    }

    return states;
  }

  decode(state) {
    const first = applyReLU(linear(state, this.decoder[0].weight, this.decoder[0].bias));
    return linear(first, this.decoder[1].weight, this.decoder[1].bias)[0];
  }

  step(states, edges, r) {
    const block = this.blockAt(r);
    const numNodes = states.length;
    const dim = block.messageDim;

    let aggregated;

    if (edges.length === 0) {
      aggregated = Array.from({ length: numNodes }, () => new Float64Array(dim));
    } else {
      const messages = edges.map(([u, v]) => {
        const input = concat(states[u], states[v]);
        const first = applyReLU(
          linear(input, block.message_fn[0].weight, block.message_fn[0].bias)
        );
        return linear(first, block.message_fn[1].weight, block.message_fn[1].bias);
      });

      const dst = edges.map(([, v]) => v);
      aggregated = aggregate(messages, dst, numNodes, dim, this.aggregation);
    }

    return states.map((state, v) => gruCell(aggregated[v], state, block.update_cell));
  }

  /**
   * The full depth sweep on one graph: same weights, same input, every R.
   *
   * Returns the readout after every step, not only the last, so a page can
   * show the prediction firming up rather than only its final value.
   */
  forward({ graph, source, target, R }) {
    let states = this.encode(graph.n, source, target);

    const targetStateNormPerStep = [l2Norm(states[target])];
    const meanStateNormPerStep = [meanNorm(states)];
    const logitsPerStep = [this.decode(states[target])];
    const trajectory = [states];

    for (let r = 0; r < R; r++) {
      states = this.step(states, graph.edges, r);
      trajectory.push(states);
      targetStateNormPerStep.push(l2Norm(states[target]));
      meanStateNormPerStep.push(meanNorm(states));
      logitsPerStep.push(this.decode(states[target]));
    }

    const logit = logitsPerStep[logitsPerStep.length - 1];
    const probability = sigmoid(logit);

    return {
      R,
      logit,
      probability,
      prediction: probability > 0.5 ? 1 : 0,
      confidence: probability > 0.5 ? probability : 1 - probability,
      probabilityPerStep: logitsPerStep.map(sigmoid),
      targetStateNormPerStep,
      meanStateNormPerStep,
      trajectory,
    };
  }
}

function meanNorm(states) {
  let sum = 0;
  for (const state of states) sum += l2Norm(state);
  return sum / states.length;
}

function readBlock(weights, prefix) {
  const messageWeight = weights[`${prefix}.message_fn.2.weight`];

  return {
    messageDim: messageWeight.length,
    message_fn: [
      {
        weight: weights[`${prefix}.message_fn.0.weight`],
        bias: weights[`${prefix}.message_fn.0.bias`],
      },
      { weight: messageWeight, bias: weights[`${prefix}.message_fn.2.bias`] },
    ],
    update_cell: {
      weight_ih: weights[`${prefix}.update_cell.weight_ih`],
      weight_hh: weights[`${prefix}.update_cell.weight_hh`],
      bias_ih: weights[`${prefix}.update_cell.bias_ih`],
      bias_hh: weights[`${prefix}.update_cell.bias_hh`],
    },
  };
}

function countBlocks(weights) {
  const indices = new Set();
  for (const key of Object.keys(weights)) {
    const match = key.match(/^blocks\.(\d+)\./);
    if (match) indices.add(Number(match[1]));
  }
  return [...indices].sort((a, b) => a - b);
}
