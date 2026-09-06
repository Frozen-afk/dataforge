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
    6. inference-depth sweep for every checkpoint
    7. aggregate into results/learned_experiment.json and learned_examples.json

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

    run(
        "Aggregate seeds into frontend JSON",
        [python, "-m", "learned.export_results", "--seeds", *[str(s) for s in seeds]],
    )

    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {time.time() - started:.1f}s")
    print(f"  Results in {BACKEND / 'results'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
