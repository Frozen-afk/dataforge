"""
One-command reproduction of every number the interface displays.

    python reproduce.py              # full pipeline
    python reproduce.py --quick      # smaller sweep, for a smoke check
    python reproduce.py --tests-only # validation suites without retraining

Stages:
    1. exact-invariant suite (10,000 stratified seeded cases)
    2. generator self-checks
    3. unit tests
    4. train the shared-weight model on five seeds
    5. train the unshared-weight ablation
    6. train the sum and mean aggregation variants
    7. inference-depth sweep for every checkpoint
    8. aggregate into results/learned_experiment.json and learned_examples.json
    9. export the browser bundle into Frontend/public/data
   10. check the browser engine against PyTorch

Everything is seeded. Rerunning on the same machine reproduces the same
checkpoint hashes and the same curves.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
SEEDS = [1, 2, 3, 4, 5]


def run(description: str, command: list[str]) -> None:
    """Run one stage, echoing the exact command so it can be repeated by hand."""
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"  $ {' '.join(command)}")
    print("=" * 70)

    started = time.time()
    result = subprocess.run(command, cwd=BACKEND)

    if result.returncode != 0:
        raise SystemExit(f"\nFAILED: {description} (exit {result.returncode})")

    print(f"  -> ok ({time.time() - started:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce all results")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer cases, epochs and seeds")
    parser.add_argument("--tests-only", action="store_true",
                        help="Run validation suites and stop")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Go straight to training")
    parser.add_argument("--skip-aggregation", action="store_true",
                        help="Skip the sum/mean aggregation ablation")
    args = parser.parse_args()

    python = sys.executable
    seeds = SEEDS[:2] if args.quick else SEEDS
    cases = 2000 if args.quick else 10000
    epochs = 15 if args.quick else 40
    train_pairs = 60 if args.quick else 150
    eval_pairs = 20 if args.quick else 40

    started = time.time()

    if not args.skip_tests:
        run(
            f"Exact-invariant suite ({cases} stratified cases)",
            [python, "-m", "tests.test_exact_invariant", "--large", "--cases", str(cases)],
        )
        run(
            "Generator distance guarantees",
            [python, "-m", "tests.test_exact_invariant", "--generator"],
        )
        run(
            "Unit tests",
            [python, "-m", "tests.test_exact_invariant"],
        )

    if args.tests_only:
        print(f"\nValidation complete in {time.time() - started:.1f}s.")
        return

    for seed in seeds:
        run(
            f"Train shared-weight model, seed {seed}",
            [python, "-m", "learned.train",
             "--seed", str(seed),
             "--epochs", str(epochs),
             "--pairs-per-distance", str(train_pairs)],
        )

    # The ablation only needs enough seeds to show the depth ceiling is real.
    for seed in seeds[:2]:
        run(
            f"Train unshared-weight ablation, seed {seed}",
            [python, "-m", "learned.train",
             "--seed", str(seed),
             "--epochs", str(epochs),
             "--pairs-per-distance", str(train_pairs),
             "--unshared"],
        )

    # Aggregation ablation. "Max because reachability is an OR" is a claim
    # about the architecture, so it is trained against its alternatives rather
    # than asserted in a docstring. Two seeds are enough to show the ordering.
    if not args.skip_aggregation:
        for aggregation in ("sum", "mean"):
            for seed in seeds[:2]:
                run(
                    f"Train {aggregation}-aggregation variant, seed {seed}",
                    [python, "-m", "learned.train",
                     "--seed", str(seed),
                     "--epochs", str(epochs),
                     "--pairs-per-distance", str(train_pairs),
                     "--aggregation", aggregation,
                     "--tag", f"agg{aggregation}"],
                )

    for seed in seeds:
        run(
            f"Inference-depth sweep, seed {seed}",
            [python, "-m", "learned.evaluate",
             "--seed", str(seed),
             "--checkpoint", f"results/model_seed{seed}.pt",
             "--pairs-per-distance", str(eval_pairs)],
        )

    for seed in seeds[:2]:
        run(
            f"Inference-depth sweep, unshared ablation, seed {seed}",
            [python, "-m", "learned.evaluate",
             "--seed", str(seed),
             "--checkpoint", f"results/model_seed{seed}_unshared.pt",
             "--pairs-per-distance", str(eval_pairs),
             "--tag", "unshared"],
        )

    if not args.skip_aggregation:
        for aggregation in ("sum", "mean"):
            for seed in seeds[:2]:
                run(
                    f"Inference-depth sweep, {aggregation} aggregation, seed {seed}",
                    [python, "-m", "learned.evaluate",
                     "--seed", str(seed),
                     "--checkpoint", f"results/model_seed{seed}_agg{aggregation}.pt",
                     "--pairs-per-distance", str(eval_pairs),
                     "--tag", f"agg{aggregation}"],
                )

    run(
        "Aggregate seeds into one results file",
        [python, "-m", "learned.export_results", "--seeds", *[str(s) for s in seeds]],
    )

    # The interface reads none of results/ directly. It reads the bundle in
    # Frontend/public/data, so the export is part of the pipeline rather than a
    # step someone has to remember.
    run(
        "Export the browser bundle",
        [python, "export_web.py"] + (["--quick"] if args.quick else []),
    )

    run(
        "Browser engine against PyTorch",
        [python, "-m", "tests.test_js_parity"],
    )

    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {time.time() - started:.1f}s")
    print(f"  Results in {BACKEND / 'results'}")
    print(f"  Browser bundle in {BACKEND.parent / 'Frontend' / 'public' / 'data'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
