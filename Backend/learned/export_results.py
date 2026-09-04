"""
Export combined results from all seeds into final JSON files for the frontend.

Reads individual seed results and produces:
    - results/learned_experiment.json
    - results/learned_examples.json

Usage:
    python -m learned.export_results --seeds 1 2 3 4 5
"""

import argparse
import json
from pathlib import Path


def load_json(filepath: Path) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Export combined results")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5]
    )
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_experiments = []
    all_examples = []

    for seed in args.seeds:
        exp_path = results_dir / f"learned_experiment_seed{seed}.json"
        ex_path = results_dir / f"learned_examples_seed{seed}.json"

        if exp_path.exists():
            all_experiments.append(load_json(exp_path))
            print(f"Loaded experiment for seed {seed}")
        else:
            print(f"WARNING: Missing experiment for seed {seed}: {exp_path}")

        if ex_path.exists():
            all_examples.extend(load_json(ex_path))
            print(f"Loaded examples for seed {seed}")
        else:
            print(f"WARNING: Missing examples for seed {seed}: {ex_path}")

    # Combine into final output
    final_experiment = {
        "description": "Inference-depth experiment: same weights, different R",
        "trainingMaxPathLength": 4,
        "seeds": args.seeds,
        "numSeeds": len(args.seeds),
        "experiments": all_experiments,
    }

    final_examples = {
        "description": "Successful and failed learned model examples",
        "examples": all_examples,
    }

    # Save
    experiment_path = results_dir / "learned_experiment.json"
    with open(experiment_path, "w") as f:
        json.dump(final_experiment, f, indent=2)
    print(f"\nFinal experiment saved: {experiment_path}")

    examples_path = results_dir / "learned_examples.json"
    with open(examples_path, "w") as f:
        json.dump(final_examples, f, indent=2)
    print(f"Final examples saved: {examples_path}")

    # Print summary
    print("\n--- Summary ---")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Total examples: {len(all_examples)}")

    if all_experiments:
        # Compute average accuracy at each R across seeds
        print("\nAverage accuracy by R (across seeds):")
        max_R = len(all_experiments[0]["inference_depth_results"])
        for r_idx in range(max_R):
            accs = [
                exp["inference_depth_results"][r_idx]["accuracy"]
                for exp in all_experiments
                if r_idx < len(exp["inference_depth_results"])
            ]
            avg_acc = sum(accs) / len(accs) if accs else 0
            print(f"  R={r_idx + 1:2d}: {avg_acc:.3f}")


if __name__ == "__main__":
    main()
