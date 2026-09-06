// Runs the browser engine under Node so PyTorch can be compared against it.
//
// Reads a JSON job on stdin:
//   { modelPath, cases: [{ graph, source, target, R }] }
//
// Writes one result per case on stdout. The Python side of the comparison
// lives in tests/test_js_parity.py, which is what you should run.

import { readFileSync } from "node:fs";
import { RecurrentGNN } from "../../Frontend/src/engine/gnn.js";
import { runExact } from "../../Frontend/src/engine/exact.js";
import { bfsDistance } from "../../Frontend/src/engine/graph.js";

const job = JSON.parse(readFileSync(0, "utf8"));
const model = new RecurrentGNN(JSON.parse(readFileSync(job.modelPath, "utf8")));

const results = job.cases.map((testCase) => {
  const { graph, source, target, R } = testCase;

  const learned = model.forward({ graph, source, target, R });
  const exact = runExact({ graph, source, target, R });

  return {
    probability: learned.probability,
    logit: learned.logit,
    probabilityPerStep: learned.probabilityPerStep,
    targetStateNormPerStep: learned.targetStateNormPerStep,
    meanStateNormPerStep: learned.meanStateNormPerStep,
    exactTargetActivation: exact.targetActivation,
    exactEstimate: exact.estimate,
    exactActiveNodeCount: exact.activeNodeCount,
    bfsDistance: bfsDistance(graph.n, graph.edges, source, target),
  };
});

process.stdout.write(JSON.stringify(results));
