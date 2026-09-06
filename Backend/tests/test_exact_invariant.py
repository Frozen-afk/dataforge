"""
Validation of the exact graph recurrence.

The invariant under test (architecture equation 11):

    h_q^(R) > epsilon   <=>   d(s, q) <= R

BFS is an independent implementation (``core/bfs.py``) and never consults the
recurrence, so agreement between the two is evidence rather than tautology.

    python -m tests.test_exact_invariant                      # unit tests
    python -m tests.test_exact_invariant --large --cases 10000 # full suite
    python -m tests.test_exact_invariant --generator           # generator checks
"""

from __future__ import annotations

import argparse
import random
import sys
import unittest
from collections import defaultdict

from core.bfs import bfs_distance
from core.generator import make_balanced_pair, make_case, stratified_cases
from core.graph import make_disconnected, make_line
from core.recurrent import run_exact
from experiments.generate import generate_cases


class TestHandCases(unittest.TestCase):
    """Cases a reader can verify by eye."""

    def test_line_distance_3(self):
        graph, source, target = make_line(3)

        result = run_exact(graph, source, target, R=2)
        self.assertFalse(result["estimate"])
        self.assertEqual(result["bfsDistance"], 3)
        self.assertFalse(result["expected"])
        self.assertTrue(result["invariantPass"])

        result = run_exact(graph, source, target, R=3)
        self.assertTrue(result["estimate"])
        self.assertTrue(result["expected"])
        self.assertTrue(result["invariantPass"])

    def test_source_equals_target(self):
        graph, source, _ = make_line(3)
        result = run_exact(graph, source, source, R=0)

        self.assertTrue(result["estimate"])
        self.assertEqual(result["bfsDistance"], 0)
        self.assertTrue(result["invariantPass"])

    def test_disconnected_graph(self):
        graph, source, target = make_disconnected()

        for R in range(0, 13):
            result = run_exact(graph, source, target, R=R)
            self.assertFalse(result["estimate"])
            self.assertIsNone(result["bfsDistance"])
            self.assertFalse(result["reachable"])
            self.assertTrue(result["invariantPass"])

    def test_alpha_below_one_preserves_invariant(self):
        """
        The invariant is stated for 0 < alpha <= 1, not only alpha = 1.

        A smaller alpha shrinks the activation but must never move the
        threshold crossing, since epsilon is far below any reachable value.
        """
        graph, source, target = make_line(4)

        for alpha in (0.25, 0.5, 0.75, 1.0):
            for R in range(0, 9):
                result = run_exact(graph, source, target, R=R, alpha=alpha)
                self.assertTrue(
                    result["invariantPass"],
                    f"invariant failed at alpha={alpha}, R={R}",
                )
                self.assertEqual(result["estimate"], R >= 4)


class TestGenerator(unittest.TestCase):
    """The generator must deliver the distances it promises."""

    def test_requested_distance_is_exact(self):
        rng = random.Random(7)

        for distance in range(1, 8):
            case = make_case(rng, n=16, distance=distance, density=0.15)
            self.assertIsNotNone(case, f"no case at distance {distance}")

            graph = case["graph"]
            actual = bfs_distance(graph.outgoing(), case["source"], case["target"])
            self.assertEqual(actual, distance)

    def test_unreachable_cases_are_unreachable(self):
        rng = random.Random(11)

        for _ in range(40):
            case = make_case(rng, n=12, distance=None, density=0.15)
            if case is None:
                continue
            graph = case["graph"]
            self.assertIsNone(
                bfs_distance(graph.outgoing(), case["source"], case["target"])
            )

    def test_hard_negatives_match_positive_size(self):
        """
        A negative must be indistinguishable from its positive by counting.

        Matching node and edge counts is what forces the model to propagate
        instead of reading the answer off graph statistics.
        """
        rng = random.Random(13)
        checked = 0

        for distance in (2, 3, 4, 6, 8):
            for _ in range(10):
                pair = make_balanced_pair(rng, n=16, distance=distance, density=0.15)
                if pair is None:
                    continue
                positive, negative = pair

                self.assertEqual(positive["graph"].n, negative["graph"].n)
                self.assertEqual(
                    len(positive["graph"].edges), len(negative["graph"].edges)
                )
                self.assertEqual(negative["cutFromDistance"], distance)
                checked += 1

        self.assertGreater(checked, 20, "too few pairs generated to be meaningful")

    def test_source_has_outgoing_edges_in_negatives(self):
        """A negative with an isolated source would be a one-hop giveaway."""
        rng = random.Random(17)
        isolated = 0
        total = 0

        for _ in range(40):
            pair = make_balanced_pair(rng, n=12, distance=3, density=0.15)
            if pair is None:
                continue
            _, negative = pair
            graph = negative["graph"]
            out_degree = sum(1 for u, _ in graph.edges if u == negative["source"])
            isolated += int(out_degree == 0)
            total += 1

        self.assertGreater(total, 10)
        self.assertLess(isolated / total, 0.35)


class TestStratifiedInvariant(unittest.TestCase):
    def test_small_stratified_suite(self):
        for case in stratified_cases(num_cases=200, seed=3):
            distance = case["distance"]
            probes = [0, 12] if distance is None else [
                max(0, distance - 1), distance, min(12, distance + 1)
            ]

            for R in probes:
                result = run_exact(
                    case["graph"], case["source"], case["target"], R=R
                )
                self.assertTrue(
                    result["invariantPass"],
                    f"invariant failed: n={case['graph'].n}, d={distance}, R={R}",
                )

    def test_generated_suite(self):
        failures = [
            case
            for case in generate_cases(num_cases=500, seed=0)
            if not run_exact(
                case["graph"], case["source"], case["target"], R=case["R"]
            )["invariantPass"]
        ]
        self.assertEqual(failures, [])


def run_large_suite(cases: int = 10000, seed: int = 0) -> None:
    """
    The full suite, with per-stratum reporting.

    A pass line alone would hide which regimes were actually exercised, so the
    breakdown by stratum is printed: it is the evidence that the boundary cases
    are covered, not just the easy ones.
    """
    print(f"Exact-invariant suite: {cases} seeded cases (seed {seed})")

    generated = generate_cases(num_cases=cases, seed=seed)
    by_stratum: dict = defaultdict(lambda: {"passed": 0, "failed": 0})
    failures = []

    for index, case in enumerate(generated):
        result = run_exact(
            case["graph"], case["source"], case["target"], R=case["R"]
        )
        stratum = case.get("stratum", "unknown")

        if result["invariantPass"]:
            by_stratum[stratum]["passed"] += 1
        else:
            by_stratum[stratum]["failed"] += 1
            if len(failures) < 10:
                failures.append({
                    "caseIndex": index,
                    "stratum": stratum,
                    "n": case["graph"].n,
                    "source": case["source"],
                    "target": case["target"],
                    "R": case["R"],
                    "estimate": result["estimate"],
                    "expected": result["expected"],
                    "bfsDistance": result["bfsDistance"],
                })

    distance_strata = sorted(
        (key for key in by_stratum if key.startswith("d=")),
        key=lambda key: (key == "d=inf", key),
    )
    other_strata = sorted(key for key in by_stratum if not key.startswith("d="))

    print(f"\n{'stratum':<16}{'passed':>10}{'failed':>10}")
    print("-" * 36)
    for key in distance_strata + other_strata:
        counts = by_stratum[key]
        print(f"{key:<16}{counts['passed']:>10}{counts['failed']:>10}")

    total_failed = sum(c["failed"] for c in by_stratum.values())

    if failures:
        print("\nInvariant failures:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)

    print(f"\nOK: {len(generated)} cases passed, {total_failed} failures.")


def run_generator_checks(cases: int = 2000, seed: int = 0) -> None:
    """Verify the generator's own promises about distance and matching."""
    print(f"Generator checks: {cases} cases (seed {seed})")

    mismatched = 0
    counted = defaultdict(int)

    for case in stratified_cases(num_cases=cases, seed=seed):
        graph = case["graph"]
        actual = bfs_distance(graph.outgoing(), case["source"], case["target"])
        if actual != case["distance"]:
            mismatched += 1
        counted[case["distance"] if case["distance"] is not None else "inf"] += 1

    print(f"  distance mismatches: {mismatched}")
    for key in sorted(counted, key=lambda k: (k == "inf", k)):
        print(f"  d={key}: {counted[key]}")

    if mismatched:
        raise SystemExit(1)
    print("OK: every generated case has the distance it claims.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true",
                        help="Run the full seeded invariant suite")
    parser.add_argument("--generator", action="store_true",
                        help="Verify generator distance guarantees")
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()

    if args.large:
        run_large_suite(cases=args.cases, seed=args.seed)
    elif args.generator:
        run_generator_checks(cases=min(args.cases, 2000), seed=args.seed)
    else:
        unittest.main(argv=[sys.argv[0]], exit=True)
