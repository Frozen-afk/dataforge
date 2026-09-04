import argparse
import sys
import unittest

from core.graph import make_disconnected, make_line
from core.recurrent import run_exact
from experiments.generate import generate_cases


class TestHandCases(unittest.TestCase):
    def test_line_distance_3(self):
        graph, source, target = make_line(3)

        # R too small
        result = run_exact(graph, source, target, R=2)
        self.assertFalse(result["estimate"])
        self.assertEqual(result["bfsDistance"], 3)
        self.assertFalse(result["expected"])
        self.assertTrue(result["invariantPass"])

        # R exactly distance
        result = run_exact(graph, source, target, R=3)
        self.assertTrue(result["estimate"])
        self.assertEqual(result["bfsDistance"], 3)
        self.assertTrue(result["expected"])
        self.assertTrue(result["invariantPass"])

    def test_source_equals_target(self):
        graph, source, target = make_line(3)
        target = source

        result = run_exact(graph, source, target, R=0)

        self.assertTrue(result["estimate"])
        self.assertEqual(result["bfsDistance"], 0)
        self.assertTrue(result["expected"])
        self.assertTrue(result["invariantPass"])

    def test_disconnected_graph(self):
        graph, source, target = make_disconnected()

        for R in range(0, 13):
            result = run_exact(graph, source, target, R=R)

            self.assertFalse(result["estimate"])
            self.assertIsNone(result["bfsDistance"])
            self.assertFalse(result["reachable"])
            self.assertFalse(result["expected"])
            self.assertTrue(result["invariantPass"])


class TestGeneratedInvariant(unittest.TestCase):
    def test_small_generated_suite(self):
        cases = generate_cases(num_cases=300, seed=0)
        failures = []

        for i, case in enumerate(cases):
            result = run_exact(
                graph=case["graph"],
                source=case["source"],
                target=case["target"],
                R=case["R"],
            )

            if not result["invariantPass"]:
                failures.append(
                    {
                        "caseIndex": i,
                        "n": case["graph"].n,
                        "source": case["source"],
                        "target": case["target"],
                        "R": case["R"],
                        "estimate": result["estimate"],
                        "expected": result["expected"],
                        "bfsDistance": result["bfsDistance"],
                    }
                )

            if len(failures) >= 5:
                break

        self.assertEqual(failures, [])


def run_large_suite(cases: int = 10000, seed: int = 0) -> None:
    print(f"Running exact invariant suite with {cases} cases...")

    failures = []

    for i, case in enumerate(generate_cases(num_cases=cases, seed=seed)):
        result = run_exact(
            graph=case["graph"],
            source=case["source"],
            target=case["target"],
            R=case["R"],
        )

        if not result["invariantPass"]:
            failures.append(
                {
                    "caseIndex": i,
                    "n": case["graph"].n,
                    "source": case["source"],
                    "target": case["target"],
                    "R": case["R"],
                    "estimate": result["estimate"],
                    "expected": result["expected"],
                    "bfsDistance": result["bfsDistance"],
                }
            )

        if len(failures) >= 10:
            break

    if failures:
        print("Invariant failures detected:")
        for failure in failures:
            print(failure)
        raise SystemExit(1)

    print(f"OK: {cases} cases passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--large",
        action="store_true",
        help="Run large invariant suite instead of unit tests.",
    )

    parser.add_argument(
        "--cases",
        type=int,
        default=10000,
        help="Number of cases for large suite.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for large suite.",
    )

    args, _ = parser.parse_known_args()

    if args.large:
        run_large_suite(cases=args.cases, seed=args.seed)
    else:
        unittest.main(argv=[sys.argv[0]], exit=True)
