// Render smoke test.
//
// Renders every page to a string under Node with the data bundle served from
// disk, which catches the failures a type-free React app otherwise only shows
// at runtime in front of a reader: a missing import, a component called with
// the wrong shape, a null dereference on a loaded payload.
//
// It is not a substitute for looking at the thing. It is a floor: the build
// passing and this passing together mean every page at least mounts and
// renders with real data.
//
// Run:  npm run smoke

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import React from "react";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(HERE, "..", "public", "data");

// The pages fetch their bundle. Serve it from disk so the render sees the same
// payloads a browser would.
globalThis.fetch = async (url) => {
  const name = String(url).split("/").pop();
  try {
    const body = readFileSync(path.join(DATA, name), "utf8");
    return { ok: true, status: 200, json: async () => JSON.parse(body) };
  } catch {
    return { ok: false, status: 404, json: async () => ({}) };
  }
};

const { PAGES } = await import("../src/App.jsx");
const lab = await import("../src/lib/lab.js");

let failures = 0;

function check(name, fn) {
  try {
    fn();
    console.log(`  ok    ${name}`);
  } catch (error) {
    failures++;
    console.log(`  FAIL  ${name}\n        ${error.message}`);
  }
}

console.log("Rendering every page:");
for (const page of PAGES) {
  check(page.id, () => {
    const markup = renderToStaticMarkup(
      React.createElement(page.component, {
        onNext: () => {},
        goTo: () => {},
        status: null,
        pages: PAGES,
      })
    );
    if (!markup || markup.length < 40) {
      throw new Error(`rendered ${markup.length} characters, which is empty`);
    }
  });
}

console.log("\nExercising the compute layer:");

const preset = await lab.exactPreset("line");
check("exact preset trajectory has 13 states", () => {
  if (preset.trajectory.length !== 13) {
    throw new Error(`got ${preset.trajectory.length}`);
  }
});
check("the invariant holds at every depth of the preset", () => {
  const broken = preset.perDepth.filter((entry) => !entry.invariantPass);
  if (broken.length) throw new Error(`${broken.length} depths disagree with BFS`);
});
check("the four-hop line has distance 4", () => {
  if (preset.bfsDistance !== 4) throw new Error(`got ${preset.bfsDistance}`);
});

const disconnected = await lab.exactPreset("disconnected");
check("a disconnected target never activates", () => {
  const lit = disconnected.perDepth.filter((entry) => entry.estimate);
  if (lit.length) throw new Error(`activated at ${lit.length} depths`);
});

const status = await lab.labStatus();
check("a checkpoint is present in the bundle", () => {
  if (!status.liveInferenceAvailable) throw new Error("no model.json");
});

const sweep = await lab.learnedSweep({ distance: 6, nodes: 16, maxR: 10 });
check("the learned sweep returns a result at every depth", () => {
  if (sweep.results.length !== 10) throw new Error(`got ${sweep.results.length}`);
});
check("the sweep case really has the distance it was asked for", () => {
  if (sweep.case.bfsDistance !== 6) throw new Error(`got ${sweep.case.bfsDistance}`);
});
check("probabilities are probabilities", () => {
  for (const entry of sweep.results) {
    if (!(entry.probability >= 0 && entry.probability <= 1)) {
      throw new Error(`R=${entry.R} gave ${entry.probability}`);
    }
  }
});

const negative = await lab.learnedSweep({
  distance: 6, nodes: 16, reachable: false, maxR: 6,
});
check("the matched negative is genuinely unreachable", () => {
  if (negative.case.bfsDistance !== null) {
    throw new Error(`BFS found a path of length ${negative.case.bfsDistance}`);
  }
});
check("the negative matches its positive on node and edge count", () => {
  const positive = sweep.case;
  if (
    negative.case.numNodes !== positive.numNodes ||
    negative.case.numEdges !== positive.numEdges
  ) {
    throw new Error(
      `positive ${positive.numNodes}n/${positive.numEdges}e vs ` +
        `negative ${negative.case.numNodes}n/${negative.case.numEdges}e`
    );
  }
});

// Every distance the interface offers must resolve to a real graph, or a
// control would lead somewhere empty.
console.log("\nEvery offered distance and size resolves:");
for (let distance = 1; distance <= 10; distance++) {
  const sizes = await lab.nodeCountsFor(distance);
  check(`distance ${distance} (${sizes.join(", ")} nodes)`, async () => {
    if (!sizes.length) throw new Error("no sizes in the bank");
  });
  for (const nodes of sizes) {
    const picked = await lab.pickCase({ distance, nodes });
    if (picked.bfsDistance !== distance) {
      failures++;
      console.log(
        `  FAIL  distance ${distance} on ${nodes} nodes returned ` +
          `distance ${picked.bfsDistance}`
      );
    }
  }
}

console.log(
  failures === 0
    ? "\nAll checks passed."
    : `\n${failures} check${failures === 1 ? "" : "s"} failed.`
);
process.exit(failures === 0 ? 0 : 1);
