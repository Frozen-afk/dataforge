// The lab's data and compute layer.
//
// Everything a page needs comes from here. The bundle in public/data is fetched
// once, then the two computational layers run locally: the noisy-OR recurrence
// and the BFS oracle from engine/, and the learned forward pass from the frozen
// checkpoint's weights. There is no server.
//
// Three things follow from that, and all three are things the lesson needs:
//
//   * The artifact opens from a static URL. A learner is never told to install
//     Python before they can see the mechanism.
//   * Moving the depth slider redraws in the same frame. The one interaction
//     the whole lesson rests on has no network in it.
//   * What "live computation" means stays honest. The forward pass really runs
//     from the real weights; Backend/tests/test_js_parity.py checks it against
//     PyTorch to within 2e-7 relative on seeded cases.
//
// The graph instances are the exception, and the interface says so: they were
// built by the seeded Python generator and shipped as data, because keeping a
// second copy of 500 lines of rejection sampling in JavaScript would risk the
// two drifting apart and quietly changing what the experiment measures.

import { RecurrentGNN } from "../engine/gnn.js";
import { runExact, runExactTrajectory, MAX_R } from "../engine/exact.js";
import { bfsDistance, graphStats } from "../engine/graph.js";

const BASE = `${import.meta.env?.BASE_URL ?? "/"}data`.replace(/\/+/g, "/");

let bundlePromise = null;

async function fetchJson(name) {
  const response = await fetch(`${BASE}/${name}`);
  if (!response.ok) {
    throw new Error(
      `Could not load ${name} (HTTP ${response.status}). The data bundle is ` +
        `built by "python export_web.py" in the Backend directory and is ` +
        `served from Frontend/public/data.`
    );
  }
  return response.json();
}

/**
 * Load the bundle once and keep it.
 *
 * The two learned files are optional: pages 1 to 3 are the exact mechanism and
 * must work even if nobody has trained a checkpoint, so a missing model is
 * reported rather than thrown.
 */
export function loadLab() {
  if (bundlePromise) return bundlePromise;

  const pending = (async () => {
    const [cases, manifest] = await Promise.all([
      fetchJson("cases.json"),
      fetchJson("manifest.json").catch(() => ({ files: {} })),
    ]);

    const [modelBundle, experiment, examples] = await Promise.all([
      fetchJson("model.json").catch(() => null),
      fetchJson("experiment.json").catch(() => null),
      fetchJson("examples.json").catch(() => null),
    ]);

    const model = modelBundle ? new RecurrentGNN(modelBundle) : null;

    return {
      cases,
      manifest,
      model,
      modelBundle,
      experiment,
      examples,
      status: {
        liveInferenceAvailable: model !== null,
        experimentPresent: experiment !== null,
        examplesPresent: examples !== null,
        computeLocation: "browser",
        meta: modelBundle
          ? {
              checkpointHash: modelBundle.checkpointHash,
              numParameters: modelBundle.numParameters,
              hiddenDim: modelBundle.hiddenDim,
              aggregation: modelBundle.aggregation,
              trainingMaxPathLength: modelBundle.trainingMaxPathLength,
              trainingMaxR: modelBundle.trainingMaxR,
              trainingDistances: modelBundle.trainingDistances,
              seed: modelBundle.seed,
              depthIsBounded: model ? model.depthIsBounded : false,
              maxInferenceDepth: model && model.depthIsBounded
                ? model.maxInferenceDepth
                : null,
            }
          : {},
      },
    };
  })();

  // Cache the success, never the failure. A rejected promise left in the cache
  // would make one dropped request permanent: every later page would replay the
  // same error and the only cure would be a full reload. Clearing it means the
  // next section to ask for the bundle simply tries again.
  bundlePromise = pending.catch((error) => {
    bundlePromise = null;
    throw error;
  });

  return bundlePromise;
}

// ------------------------------------------------------------- exact layer

const CASE_EVIDENCE = {
  evidenceType: "Synthetic data",
  source:
    "Graph built by the seeded generator in core/generator.py and shipped " +
    "as data. Everything computed on it below runs live in your browser.",
};

/** A preset teaching graph, with its whole trajectory precomputed to depth 12. */
export async function exactPreset(name, { alpha = 1, maxR = MAX_R } = {}) {
  const lab = await loadLab();
  const preset = lab.cases.presets?.[name];

  if (!preset) {
    throw new Error(`Unknown preset "${name}".`);
  }

  const result = runExactTrajectory({
    graph: { n: preset.n, edges: preset.edges },
    source: preset.source,
    target: preset.target,
    alpha,
    maxR,
  });

  return { ...result, preset: name, description: preset.description };
}

export async function presetList() {
  const lab = await loadLab();
  return Object.entries(lab.cases.presets || {}).map(([name, preset]) => ({
    name,
    description: preset.description,
    distance: preset.distance,
    numNodes: preset.n,
    numEdges: preset.edges.length,
  }));
}

/** Run the exact recurrence on any graph the caller already has. */
export function exactRun(graph, source, target, R, alpha = 1) {
  return runExact({ graph, source, target, R, alpha });
}

// ----------------------------------------------------------- the case bank

/** Node counts the bank actually holds for a distance, so no control lies. */
export async function nodeCountsFor(distance) {
  const lab = await loadLab();
  return Object.keys(lab.cases.bank)
    .filter((key) => Number(key.split(":")[0]) === distance)
    .map((key) => Number(key.split(":")[1]))
    .sort((a, b) => a - b);
}

export async function bankMatching() {
  const lab = await loadLab();
  return lab.cases.matching;
}

/**
 * Pull one instance out of the bank.
 *
 * A "variant" is one matched pair: a reachable case at the requested distance,
 * and the unreachable case cut from it. Asking for the unreachable half gives
 * you a graph with the identical node and edge count, which is what stops the
 * sandbox from being an easier problem than the measured experiment.
 */
export async function pickCase({ distance, nodes, reachable = true, variant = 0 }) {
  const lab = await loadLab();
  const entries = lab.cases.bank[`${distance}:${nodes}`];

  if (!entries || entries.length === 0) {
    const available = await nodeCountsFor(distance);
    throw new Error(
      `No graph in the bank with distance ${distance} on ${nodes} nodes. ` +
        `Available sizes at this distance: ${available.join(", ") || "none"}.`
    );
  }

  const entry = entries[((variant % entries.length) + entries.length) % entries.length];
  const record = reachable ? entry.reachable : entry.unreachable;

  return {
    graph: { n: record.n, edges: record.edges },
    source: record.source,
    target: record.target,
    // Re-derived in the browser rather than trusted from the file, so the
    // number on screen is one this session actually computed.
    bfsDistance: bfsDistance(record.n, record.edges, record.source, record.target),
    declaredDistance: record.distance,
    cutFromDistance: record.cutFromDistance,
    label: record.reachable ? 1 : 0,
    reachable: record.reachable,
    variantCount: entries.length,
    ...graphStats({ n: record.n, edges: record.edges }, record.source, record.target),
    evidence: CASE_EVIDENCE,
  };
}

// ----------------------------------------------------------- learned layer

function learnedEvidence(checkpointHash) {
  return {
    evidenceType: "Live computation",
    source:
      "Forward pass of the frozen toy checkpoint, run in your browser from " +
      "the exported weights. An independent reimplementation written for " +
      "this lab, not BDH and not BDH-CQ.",
    experimentId: `learned-browser-${checkpointHash ?? "unknown"}`,
  };
}

/**
 * The core experiment on one graph: identical weights, identical input,
 * every depth from 1 to maxR.
 *
 * Computed in one pass so the depth slider cannot change the input as well as
 * the depth, which would confound the only variable the page is varying.
 */
export async function learnedSweep({
  distance,
  reachable = true,
  nodes,
  variant = 0,
  maxR = 10,
}) {
  const lab = await loadLab();

  if (!lab.model) {
    throw new Error(
      "No exported checkpoint in the data bundle, so this page has nothing " +
        "to show. Run 'python reproduce.py' then 'python export_web.py' in " +
        "the Backend directory. Pages 1 to 3 work regardless."
    );
  }

  const testCase = await pickCase({ distance, nodes, reachable, variant });
  const ceiling = lab.model.maxInferenceDepth;

  const results = [];
  for (let R = 1; R <= maxR; R++) {
    if (R > ceiling) {
      results.push({
        R,
        unavailable: true,
        reason:
          `This checkpoint has ${ceiling} update blocks. Without shared ` +
          `weights, inference depth is fixed when training ends.`,
      });
      continue;
    }

    const learned = lab.model.forward({
      graph: testCase.graph,
      source: testCase.source,
      target: testCase.target,
      R,
    });
    const exact = runExact({
      graph: testCase.graph,
      source: testCase.source,
      target: testCase.target,
      R: Math.min(R, MAX_R),
    });

    results.push({
      R,
      probability: learned.probability,
      prediction: learned.prediction,
      confidence: learned.confidence,
      correct: learned.prediction === testCase.label,
      probabilityPerStep: learned.probabilityPerStep,
      targetStateNormPerStep: learned.targetStateNormPerStep,
      meanStateNormPerStep: learned.meanStateNormPerStep,
      exact: {
        estimate: exact.estimate,
        targetActivation: exact.targetActivation,
        trajectory: exact.trajectory,
        activeNodeCount: exact.activeNodeCount,
        activationMass: exact.activationMass,
        correct: exact.estimate === (testCase.label === 1),
      },
    });
  }

  // The depth from which the model *stays* committed to "reachable". A single
  // crossing is not a commitment: probabilities hover near 0.5 while the answer
  // is still unknowable, so an early 0.504 would otherwise be reported as the
  // flip and make the model look like it decided before it could.
  const available = results.filter((entry) => !entry.unavailable);
  let learnedFlipDepth = null;
  for (let i = 0; i < available.length; i++) {
    if (available.slice(i).every((later) => later.prediction === 1)) {
      learnedFlipDepth = available[i].R;
      break;
    }
  }

  return {
    case: testCase,
    results,
    maxInferenceDepth: lab.model.depthIsBounded ? ceiling : null,
    depthIsBounded: lab.model.depthIsBounded,
    learnedFlipDepth,
    exactFlipDepth: testCase.bfsDistance,
    flipDepthsAgree: learnedFlipDepth === testCase.bfsDistance,
    meta: lab.status.meta,
    evidence: learnedEvidence(lab.status.meta?.checkpointHash),
  };
}

/** One depth on one graph, for the smaller readouts. */
export async function learnedRun({ graph, source, target, R }) {
  const lab = await loadLab();
  if (!lab.model) throw new Error("No exported checkpoint in the data bundle.");

  const learned = lab.model.forward({ graph, source, target, R });
  return {
    ...learned,
    meta: lab.status.meta,
    evidence: learnedEvidence(lab.status.meta?.checkpointHash),
  };
}

export async function learnedExperiment() {
  const lab = await loadLab();
  if (!lab.experiment) {
    throw new Error(
      "experiment.json is missing from the data bundle. Run " +
        "'python reproduce.py' then 'python export_web.py'."
    );
  }
  return lab.experiment;
}

export async function learnedExamples() {
  const lab = await loadLab();
  if (!lab.examples) throw new Error("examples.json is missing from the data bundle.");
  return lab.examples;
}

export async function labStatus() {
  const lab = await loadLab();
  return lab.status;
}

export async function labManifest() {
  const lab = await loadLab();
  return lab.manifest;
}

export { MAX_R };
