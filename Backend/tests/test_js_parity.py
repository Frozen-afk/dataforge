"""
Cross-language parity: the browser engine against PyTorch and core/recurrent.py.

The artifact runs its computation in the learner's browser. That is only
honest if the browser is running the same thing the Python pipeline measured,
so this suite takes seeded cases, runs them through both implementations, and
fails if any number differs by more than a tolerance far below anything the
interface displays.

It covers the two claims that would otherwise be taken on trust:

  1. The JavaScript forward pass reproduces the PyTorch forward pass over the
     exported weights, at every depth, including the per-step readouts the
     interface plots.
  2. The JavaScript noisy-OR recurrence reproduces core/recurrent.py, and both
     satisfy the invariant against the BFS oracle.

Run:
    python -m tests.test_js_parity            # needs node and an export
    python -m tests.test_js_parity --verbose  # print the largest deviations
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from core.generator import make_balanced_pair  # noqa: E402
from core.recurrent import run_exact  # noqa: E402
from core.graph import Graph  # noqa: E402

RUNNER = BACKEND_DIR / "tests" / "js_parity_runner.mjs"
MODEL_JSON = BACKEND_DIR.parent / "Frontend" / "public" / "data" / "model.json"
CHECKPOINT = BACKEND_DIR / "results" / "model_seed1.pt"

# PyTorch runs the checkpoint in float32 and the browser runs it in float64, so
# the two accumulate rounding differently and the gap grows with the magnitude
# of the value. The tolerance is therefore relative, scaled by the PyTorch
# value but never tighter than 1e-6 absolute. At 1e-6 relative a logit of -12.5
# may differ in its seventh significant digit, which is float32 noise; anything
# a learner can see on screen is three orders of magnitude coarser.
TOLERANCE = 1e-6

MAX_R = 10


def node_available() -> bool:
    return shutil.which("node") is not None


def build_cases(count: int = 24, seed: int = 777) -> List[Dict[str, Any]]:
    """Seeded matched pairs across the whole distance range, plus the presets."""
    import random

    rng = random.Random(seed)
    cases: List[Dict[str, Any]] = []

    distances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    node_counts = {1: 8, 2: 8, 3: 12, 4: 12, 5: 16, 6: 16, 7: 16, 8: 24, 9: 24, 10: 24}

    while len(cases) < count:
        distance = distances[len(cases) % len(distances)]
        pair = make_balanced_pair(
            rng, n=node_counts[distance], distance=distance, density=0.15
        )
        if pair is None:
            continue

        for item in pair:
            graph: Graph = item["graph"]
            cases.append({
                "graph": graph.to_dict(),
                "source": int(item["source"]),
                "target": int(item["target"]),
                "R": MAX_R,
                "distance": item["distance"],
            })

    return cases[:count]


def run_node(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    job = json.dumps({"modelPath": str(MODEL_JSON), "cases": cases})
    completed = subprocess.run(
        ["node", str(RUNNER)],
        input=job.encode(),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "node runner failed:\n" + completed.stderr.decode()[-4000:]
        )
    return json.loads(completed.stdout.decode())


def run_torch(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import torch

    from learned.evaluate import load_checkpoint
    from learned.dataset import case_to_sample, collate

    model, _ = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    model.eval()

    results = []
    for case in cases:
        graph = Graph.from_dict(case["graph"])
        sample = case_to_sample({
            "graph": graph,
            "source": case["source"],
            "target": case["target"],
            "distance": case["distance"],
            "density": None,
            "targetInDegree": 0,
        })
        batch = collate([sample])

        with torch.no_grad():
            logit, trajectory, logits_per_step = model.forward_with_trajectory(
                batch.x, batch.edge_index, batch.target_idx, case["R"]
            )

        results.append({
            "logit": logit[0].item(),
            "probability": torch.sigmoid(logit)[0].item(),
            "probabilityPerStep": [
                torch.sigmoid(step)[0].item() for step in logits_per_step
            ],
            "targetStateNormPerStep": [
                step[batch.target_idx[0]].norm().item() for step in trajectory
            ],
            "meanStateNormPerStep": [
                step.norm(dim=-1).mean().item() for step in trajectory
            ],
        })

    return results


class TestJsParity(unittest.TestCase):
    """The browser engine must agree with the Python pipeline it stands in for."""

    @classmethod
    def setUpClass(cls) -> None:
        if not node_available():
            raise unittest.SkipTest("node is not installed")
        if not MODEL_JSON.exists():
            raise unittest.SkipTest(
                f"{MODEL_JSON} missing. Run: python export_web.py"
            )
        if not CHECKPOINT.exists():
            raise unittest.SkipTest(
                f"{CHECKPOINT} missing. Run: python reproduce.py"
            )

        cls.cases = build_cases()
        cls.js = run_node(cls.cases)
        cls.torch_results = run_torch(cls.cases)
        cls.worst: Dict[str, float] = {}

    def _compare(self, field: str, js_value, torch_value, index: int) -> None:
        js_list = js_value if isinstance(js_value, list) else [js_value]
        torch_list = torch_value if isinstance(torch_value, list) else [torch_value]

        self.assertEqual(
            len(js_list), len(torch_list),
            f"case {index}: {field} has {len(js_list)} entries in JavaScript "
            f"and {len(torch_list)} in PyTorch",
        )

        for step, (a, b) in enumerate(zip(js_list, torch_list)):
            deviation = abs(a - b)
            relative = deviation / max(1.0, abs(b))
            self.worst[field] = max(self.worst.get(field, 0.0), relative)
            self.assertLessEqual(
                relative, TOLERANCE,
                f"case {index}, {field}[{step}]: browser {a!r} vs PyTorch "
                f"{b!r}, relative deviation {relative:.3e} exceeds "
                f"{TOLERANCE:.0e}",
            )

    def test_final_probability_matches(self) -> None:
        for index, (js, torch_result) in enumerate(zip(self.js, self.torch_results)):
            self._compare("probability", js["probability"], torch_result["probability"], index)
            self._compare("logit", js["logit"], torch_result["logit"], index)

    def test_per_step_readout_matches(self) -> None:
        for index, (js, torch_result) in enumerate(zip(self.js, self.torch_results)):
            self._compare(
                "probabilityPerStep",
                js["probabilityPerStep"], torch_result["probabilityPerStep"], index,
            )

    def test_state_norms_match(self) -> None:
        for index, (js, torch_result) in enumerate(zip(self.js, self.torch_results)):
            self._compare(
                "targetStateNormPerStep",
                js["targetStateNormPerStep"],
                torch_result["targetStateNormPerStep"], index,
            )
            self._compare(
                "meanStateNormPerStep",
                js["meanStateNormPerStep"],
                torch_result["meanStateNormPerStep"], index,
            )

    def test_exact_recurrence_matches(self) -> None:
        """The hand-designed layer must agree across languages too."""
        for index, (case, js) in enumerate(zip(self.cases, self.js)):
            graph = Graph.from_dict(case["graph"])
            expected = run_exact(graph, case["source"], case["target"], R=case["R"])

            self._compare(
                "exactTargetActivation",
                js["exactTargetActivation"], expected["targetActivation"], index,
            )
            self.assertEqual(js["exactEstimate"], expected["estimate"])
            self.assertEqual(js["bfsDistance"], expected["bfsDistance"])

    def test_browser_invariant_holds(self) -> None:
        """h(R)[q] > eps <=> d(s,q) <= R, as computed in the browser."""
        for index, (case, js) in enumerate(zip(self.cases, self.js)):
            distance = js["bfsDistance"]
            expected = distance is not None and distance <= case["R"]
            self.assertEqual(
                js["exactEstimate"], expected,
                f"case {index}: browser recurrence broke the invariant at "
                f"R={case['R']} with distance {distance}",
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if "--verbose" in sys.argv and getattr(cls, "worst", None):
            print("\nLargest browser-versus-PyTorch relative deviation per field:")
            for field, deviation in sorted(cls.worst.items()):
                print(f"  {field:28s} {deviation:.3e}")


if __name__ == "__main__":
    unittest.main(argv=[a for a in sys.argv if a != "--verbose"])
