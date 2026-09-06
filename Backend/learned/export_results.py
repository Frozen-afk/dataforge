"""
Aggregate per-seed experiments into the JSON the web interface consumes.

The interface must never recompute the experiment, and must never present a
single seed as if it were the result. This script combines the per-seed sweeps
into mean curves with 95% confidence intervals, carries the per-seed curves
through so the page can draw seed-to-seed spread, and attaches the shared
versus unshared ablation.

Outputs:
    results/learned_experiment.json   depth curves, ablation, training curves
    results/learned_examples.json     categorised success and failure cases

Usage:
    python -m learned.export_results --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Student t critical values at 95% for small samples, indexed by n - 1.
T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def mean_ci(values: List[float]) -> Dict[str, float]:
    """
    Mean with a 95% confidence interval over seeds.

    With five seeds the normal approximation is too optimistic, so a Student t
    critical value is used. A single seed reports a zero-width interval, which
    the interface labels as such rather than drawing a misleading band.
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "std": 0.0, "n": 0}

    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "low": mean, "high": mean, "std": 0.0, "n": 1}

    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    half = T_95.get(n - 1, 2.228) * std / math.sqrt(n)

    return {
        "mean": mean,
        "low": max(0.0, mean - half),
        "high": min(1.0, mean + half),
        "std": std,
        "n": n,
    }


def load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def aggregate_depth_curve(experiments: List[dict]) -> List[dict]:
    """Mean +/- CI of every depth metric across seeds, plus per-seed values."""
    by_depth: Dict[int, List[dict]] = defaultdict(list)

    for experiment in experiments:
        for entry in experiment.get("inferenceDepthResults", []):
            if entry.get("unavailable"):
                continue
            by_depth[int(entry["R"])].append(entry)

    curve: List[dict] = []

    for R in sorted(by_depth):
        entries = by_depth[R]

        overall = [float(e["accuracy"]) for e in entries]
        seen = [float(e["accuracySeenDistances"]) for e in entries]
        unseen = [float(e["accuracyUnseenDistances"]) for e in entries]
        confidence = [float(e["confidence"]) for e in entries]
        norms = [float(e["meanStateNorm"]) for e in entries]

        # Per-distance accuracy, averaged over seeds.
        per_distance_values: Dict[str, List[float]] = defaultdict(list)
        for entry in entries:
            for key, value in entry.get("perDistance", {}).items():
                per_distance_values[key].append(float(value["accuracy"]))

        categories: Dict[str, int] = defaultdict(int)
        for entry in entries:
            for key, count in entry.get("failureCategories", {}).items():
                categories[key] += int(count)

        curve.append({
            "R": R,
            "accuracy": mean_ci(overall),
            "accuracySeenDistances": mean_ci(seen),
            "accuracyUnseenDistances": mean_ci(unseen),
            "confidence": mean_ci(confidence),
            "meanStateNorm": mean_ci(norms),
            "perDistance": {
                key: mean_ci(values)
                for key, values in sorted(
                    per_distance_values.items(),
                    key=lambda kv: (not kv[0].isdigit(),
                                    int(kv[0]) if kv[0].isdigit() else 0),
                )
            },
            "failureCategories": dict(categories),
            "perSeed": [
                {"seed": e.get("seed"), "accuracy": e["accuracy"],
                 "accuracySeenDistances": e["accuracySeenDistances"],
                 "accuracyUnseenDistances": e["accuracyUnseenDistances"]}
                for e in entries
            ],
        })

    return curve


def aggregate_exact(experiments: List[dict]) -> List[dict]:
    """
    The exact mechanism's reference curve.

    The exact solver is deterministic, so every seed sees the same behaviour on
    its own test split and the seeds are averaged only to line the curve up with
    the learned one on identical axes.
    """
    by_depth: Dict[int, List[dict]] = defaultdict(list)
    for experiment in experiments:
        for entry in experiment.get("exactReference", []):
            by_depth[int(entry["R"])].append(entry)

    curve = []
    for R in sorted(by_depth):
        entries = by_depth[R]
        per_distance_values: Dict[str, List[float]] = defaultdict(list)
        for entry in entries:
            for key, value in entry.get("perDistance", {}).items():
                per_distance_values[key].append(float(value["accuracy"]))

        curve.append({
            "R": R,
            "accuracy": mean_ci([float(e["accuracy"]) for e in entries]),
            "perDistance": {
                key: mean_ci(values)
                for key, values in sorted(
                    per_distance_values.items(),
                    key=lambda kv: (not kv[0].isdigit(),
                                    int(kv[0]) if kv[0].isdigit() else 0),
                )
            },
        })
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-seed results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--ablation-tag", type=str, default="unshared")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments: List[dict] = []
    examples: List[dict] = []
    missing: List[int] = []

    for seed in args.seeds:
        experiment = load(results_dir / f"learned_experiment_seed{seed}.json")
        if experiment is None:
            missing.append(seed)
            print(f"WARNING: missing experiment for seed {seed}")
        else:
            experiments.append(experiment)
            print(f"Loaded seed {seed} (checkpoint {experiment.get('checkpointHash')})")

        example_file = load(results_dir / f"learned_examples_seed{seed}.json")
        if example_file:
            for example in example_file.get("examples", []):
                example["seed"] = seed
                examples.append(example)

    if not experiments:
        raise SystemExit("No per-seed experiments found. Run learned.evaluate first.")

    # Shared-versus-unshared ablation.
    ablation: List[dict] = []
    for seed in args.seeds:
        unshared = load(
            results_dir / f"learned_experiment_seed{seed}_{args.ablation_tag}.json"
        )
        if unshared:
            ablation.append(unshared)
    if ablation:
        print(f"Loaded {len(ablation)} unshared-weight ablation run(s)")

    # Aggregation ablation. The default aggregator is max because reachability
    # is a logical OR over incoming neighbours, and max is its differentiable
    # analogue -- the same role the noisy-OR plays in the exact layer. That is
    # a claim, so it is measured: sum and mean are trained identically and
    # their depth curves are exported beside the default's.
    aggregation_runs: Dict[str, List[dict]] = {}
    for name in ("sum", "mean"):
        runs = [
            run for run in (
                load(results_dir / f"learned_experiment_seed{seed}_agg{name}.json")
                for seed in args.seeds
            ) if run
        ]
        if runs:
            aggregation_runs[name] = runs
            print(f"Loaded {len(runs)} '{name}'-aggregation run(s)")

    # Restrict the default's curve to the seeds the variants actually have, so
    # the comparison varies the aggregator and nothing else.
    ablation_seeds = sorted({
        run.get("seed")
        for runs in aggregation_runs.values()
        for run in runs
    })
    matched_max = [e for e in experiments if e.get("seed") in ablation_seeds]

    reference = experiments[0]
    depth_curve = aggregate_depth_curve(experiments)
    ceiling = max((entry["R"] for entry in depth_curve), default=0)

    combined = {
        "description": (
            "Inference-depth experiment: identical weights and identical inputs, "
            "evaluated at different recurrent depths R."
        ),
        "seeds": [e.get("seed") for e in experiments],
        "numSeeds": len(experiments),
        "missingSeeds": missing,
        "checkpointHashes": {
            str(e.get("seed")): e.get("checkpointHash") for e in experiments
        },
        "trainingMaxPathLength": reference.get("trainingMaxPathLength"),
        "trainingDistances": reference.get("trainingDistances"),
        "trainingMaxR": reference.get("trainingMaxR"),
        "testDistances": reference.get("testDistances"),
        "maxInferenceDepth": ceiling,
        "config": reference.get("config", {}),
        "trainingConfig": reference.get("trainingConfig", {}),
        "depthCurve": depth_curve,
        "exactReference": aggregate_exact(experiments),
        "trainingCurves": [
            {"seed": e.get("seed"), "log": e.get("trainingLog", [])}
            for e in experiments
        ],
        "ablation": {
            "description": (
                "Same architecture and training data, but a separate update "
                "block per step instead of one shared block. Parameter count "
                "grows with depth and the model has no weights to run beyond "
                "its trained depth."
            ),
            "sharedWeights": {
                "numParameters": reference.get("config", {}).get("numParameters"),
                "maxInferenceDepth": ceiling,
            },
            "unsharedWeights": (
                {
                    "numParameters": ablation[0].get("config", {}).get("numParameters"),
                    "maxInferenceDepth": ablation[0].get("maxInferenceDepth"),
                    "seeds": [a.get("seed") for a in ablation],
                    "depthCurve": aggregate_depth_curve(ablation),
                }
                if ablation
                else None
            ),
        },
        "aggregationAblation": (
            {
                "description": (
                    "Same architecture, same data, same seeds; only the "
                    "neighbour aggregator changes. Max is the default because "
                    "reachability is a logical OR over incoming neighbours, "
                    "and max is its differentiable analogue."
                ),
                "default": "max",
                # The variants are trained on fewer seeds than the headline
                # model, so the max curve here is recomputed over exactly the
                # seeds the variants have. Comparing a five-seed mean against a
                # two-seed mean would put the seed count inside the difference
                # the table is supposed to be about.
                "comparedOnSeeds": ablation_seeds,
                "variants": [
                    {
                        "aggregation": name,
                        "seeds": [run.get("seed") for run in runs],
                        "numParameters": runs[0].get("config", {}).get("numParameters"),
                        "depthCurve": aggregate_depth_curve(runs),
                    }
                    for name, runs in [
                        ("max", matched_max),
                        *sorted(aggregation_runs.items()),
                    ]
                ],
            }
            if aggregation_runs and matched_max
            else None
        ),
        "evidence": {
            "evidenceType": "Precomputed result",
            "source": (
                "Local sweeps over frozen toy checkpoints. These are Latent Loop "
                "Lab measurements, not BDH or BDH-CQ measurements."
            ),
            "experimentId": "depth-sweep-aggregate",
        },
    }

    experiment_path = results_dir / "learned_experiment.json"
    experiment_path.write_text(json.dumps(combined, indent=2))

    examples_path = results_dir / "learned_examples.json"
    examples_path.write_text(json.dumps({
        "description": "Categorised successful and failed learned-model cases",
        "trainingMaxPathLength": reference.get("trainingMaxPathLength"),
        "examples": examples,
        "evidence": {
            "evidenceType": "Precomputed result",
            "source": "Frozen toy checkpoints on held-out generated graphs",
            "experimentId": "examples-aggregate",
        },
    }, indent=2))

    print(f"\nWrote {experiment_path}")
    print(f"Wrote {examples_path}  ({len(examples)} examples)")

    print("\nMean accuracy versus inference depth:")
    for entry in depth_curve:
        accuracy = entry["accuracy"]
        print(f"  R={entry['R']:2d}  {accuracy['mean']:.3f} "
              f"[{accuracy['low']:.3f}, {accuracy['high']:.3f}]  "
              f"seen {entry['accuracySeenDistances']['mean']:.3f}  "
              f"unseen {entry['accuracyUnseenDistances']['mean']:.3f}")


if __name__ == "__main__":
    main()
