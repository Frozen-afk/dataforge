"""
Export everything the web artifact needs to run without a backend.

Why this exists
---------------
The lab used to require a local FastAPI process. That is fine on a laptop and
useless as a public artifact: a learner who opens the URL sees a red banner
instead of the mechanism. It also put a network round trip between the depth
slider and the picture, which is the one interaction the whole lesson rests on.

So the two computational layers are moved into the browser, and this script
ships them the data they need:

    model.json      the frozen seed-1 checkpoint as plain arrays, so the
                    learned forward pass can run in JavaScript
    cases.json      a bank of graph instances built by the seeded Python
                    generator in core/generator.py
    experiment.json the multi-seed depth sweep (unchanged, precomputed)
    examples.json   categorised successes and failures (unchanged)
    manifest.json   hashes and provenance for everything above

What this does and does not change about evidence labels
--------------------------------------------------------
Nothing is turned into an animation. The BFS oracle, the noisy-OR recurrence
and the learned forward pass all still *run* -- they run in the browser instead
of on a server, from the same weights, and ``tests/test_js_parity.py`` checks
the JavaScript against PyTorch to eight decimal places on seeded cases.

The graph instances are the one thing that becomes data rather than
computation. Reproducing ``core/generator.py`` in JavaScript would mean
maintaining two copies of 500 lines of subtle rejection sampling, and any drift
between them would silently change what the experiment measures. Shipping the
Python generator's own output removes that risk. The cases are therefore
labelled "Synthetic data" in the interface, which is what they were all along.

Usage:
    python export_web.py                  # full bank
    python export_web.py --quick          # small bank, for a fast check
    python export_web.py --out ../Frontend/public/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.bfs import bfs_distance
from core.generator import (
    DENSITIES,
    make_balanced_pair,
    make_case,
    sample_target_in_degree,
)
from core.graph import PRESETS, Graph, get_preset

BACKEND_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BACKEND_DIR / "results"
DEFAULT_OUT = BACKEND_DIR.parent / "Frontend" / "public" / "data"

# Node counts offered per distance. A distance-d backbone consumes d + 1
# distinct nodes, so short distances get small graphs and long ones need room.
NODE_COUNTS_BY_DISTANCE: Dict[int, List[int]] = {
    1: [8, 12], 2: [8, 12], 3: [8, 12, 16], 4: [12, 16],
    5: [12, 16, 24], 6: [16, 24], 7: [16, 24], 8: [16, 24],
    9: [16, 24], 10: [16, 24],
}

MAX_DISTANCE = 10


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


# ----------------------------------------------------------------------
# Model weights
# ----------------------------------------------------------------------

def _round_nested(value: Any, places: int = 8) -> Any:
    """
    Round a nested list of floats before serialising.

    The checkpoint is float32, which carries about seven decimal digits, so
    eight places is lossless in practice and roughly halves the file the
    learner has to download. ``tests/test_js_parity.py`` compares the browser
    forward pass against PyTorch on the *rounded* weights, so any loss this
    introduces would show up there rather than passing unnoticed.
    """
    if isinstance(value, list):
        return [_round_nested(item, places) for item in value]
    return round(float(value), places)


def export_model(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Dump a checkpoint to nested plain lists.

    Layer names are kept exactly as PyTorch stores them, so a reader can line
    the JSON up against ``learned/model.py`` without a translation table.
    """
    import torch

    # weights_only=True refuses to unpickle arbitrary objects. These
    # checkpoints hold tensors and plain scalars only, so nothing is lost, and
    # a reader who clones this repository is never one torch.load away from
    # executing whatever a downloaded .pt file happens to contain.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]

    weights = {name: _round_nested(tensor.tolist()) for name, tensor in state.items()}

    return {
        "description": (
            "Frozen toy checkpoint for the shared-weight recurrent GNN. An "
            "independent reimplementation written for this lab; not BDH, not "
            "BDH-CQ, not derived from any Pathway checkpoint."
        ),
        "checkpointFile": checkpoint_path.name,
        "checkpointHash": file_hash(checkpoint_path),
        "seed": checkpoint.get("seed"),
        "sharedWeights": checkpoint.get("shared_weights", True),
        "hiddenDim": checkpoint.get("hidden_dim", 32),
        "messageDim": checkpoint.get("config", {}).get("messageDim", 32),
        "aggregation": checkpoint.get("aggregation", "max"),
        "numParameters": checkpoint.get("config", {}).get("numParameters"),
        "trainingDistances": checkpoint.get("training_distances", [1, 2, 3, 4]),
        "trainingMaxPathLength": checkpoint.get("trainingMaxPathLength", 4),
        "trainingMaxR": checkpoint.get("max_R", 4),
        "numSteps": checkpoint.get("config", {}).get("numSteps"),
        "weights": weights,
        "evidence": {
            "evidenceType": "Live computation",
            "source": (
                "Weights of the frozen toy checkpoint. The forward pass runs "
                "in your browser; tests/test_js_parity.py checks it against "
                "PyTorch."
            ),
        },
    }


# ----------------------------------------------------------------------
# Case bank
# ----------------------------------------------------------------------

def _case_record(case: Dict[str, Any]) -> Dict[str, Any]:
    """One generator case, flattened, with its BFS distance re-verified."""
    graph: Graph = case["graph"]
    source = int(case["source"])
    target = int(case["target"])

    verified = bfs_distance(graph.outgoing(), source, target)
    if verified != case["distance"]:
        raise AssertionError(
            f"Generator returned distance {case['distance']} but BFS says "
            f"{verified}. Refusing to export an unverified case."
        )

    return {
        "n": graph.n,
        "edges": [[u, v] for u, v in graph.edges],
        "source": source,
        "target": target,
        "distance": verified,
        "reachable": verified is not None,
        "cutFromDistance": case.get("cutFromDistance"),
        "targetInDegree": case["targetInDegree"],
        "density": case["density"],
        "numEdges": len(graph.edges),
    }


def build_case_bank(variants: int, seed: int = 4242) -> Dict[str, Any]:
    """
    Matched (reachable, unreachable) pairs at every distance and node count.

    Each unreachable case is cut from the reachable one it sits beside, so the
    two share a node count and an edge count exactly. That is the same
    construction the training and evaluation sets use, which is what stops the
    page-4 sandbox from being an easier problem than the measured experiment.
    """
    rng = random.Random(seed)
    bank: Dict[str, List[Dict[str, Any]]] = {}
    made = 0
    dropped = 0

    for distance in range(1, MAX_DISTANCE + 1):
        for n in NODE_COUNTS_BY_DISTANCE[distance]:
            key = f"{distance}:{n}"
            entries: List[Dict[str, Any]] = []
            attempts = 0

            while len(entries) < variants and attempts < variants * 40:
                attempts += 1
                density = rng.choice(list(DENSITIES))
                pair = make_balanced_pair(rng, n=n, distance=distance, density=density)
                if pair is None:
                    dropped += 1
                    continue

                positive, negative = pair
                entries.append({
                    "reachable": _case_record(positive),
                    "unreachable": _case_record(negative),
                })
                made += 1

            if entries:
                bank[key] = entries

    return {
        "description": (
            "Graph instances produced by the seeded generator in "
            "core/generator.py and verified against the BFS oracle before "
            "export. Each unreachable case is cut from the reachable case "
            "beside it, so the pair matches on node count and edge count "
            "exactly."
        ),
        "generatorSeed": seed,
        "variantsPerKey": variants,
        "maxDistance": MAX_DISTANCE,
        "nodeCountsByDistance": NODE_COUNTS_BY_DISTANCE,
        "pairsExported": made,
        "pairsDropped": dropped,
        "keys": sorted(bank.keys()),
        "bank": bank,
        "evidence": {
            "evidenceType": "Synthetic data",
            "source": (
                "Seeded distance-stratified generator, core/generator.py. "
                "Reproducible with python export_web.py."
            ),
        },
    }


def export_presets() -> Dict[str, Any]:
    """The hand-built teaching graphs, with their distances recomputed."""
    out: Dict[str, Any] = {}

    for name in PRESETS:
        graph, source, target, description = get_preset(name)
        distance = bfs_distance(graph.outgoing(), source, target)
        out[name] = {
            "n": graph.n,
            "edges": [[u, v] for u, v in graph.edges],
            "source": source,
            "target": target,
            "distance": distance,
            "description": description,
        }

    return out


# ----------------------------------------------------------------------
# Statistics the interface quotes, computed rather than hard-coded
# ----------------------------------------------------------------------

def matching_report(bank: Dict[str, Any]) -> Dict[str, Any]:
    """
    How well the negatives are matched to the positives they came from.

    The interface states these gaps out loud. Computing them here means the
    number on screen cannot drift away from the bank actually shipped.
    """
    stats = {
        "positive": {"nodes": [], "edges": [], "targetInDegree": [], "sourceOutDegree": []},
        "negative": {"nodes": [], "edges": [], "targetInDegree": [], "sourceOutDegree": []},
    }

    for entries in bank.values():
        for entry in entries:
            for role, key in (("reachable", "positive"), ("unreachable", "negative")):
                case = entry[role]
                out_degree = sum(1 for u, _ in case["edges"] if u == case["source"])
                stats[key]["nodes"].append(case["n"])
                stats[key]["edges"].append(case["numEdges"])
                stats[key]["targetInDegree"].append(case["targetInDegree"])
                stats[key]["sourceOutDegree"].append(out_degree)

    def mean(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "description": (
            "Mean node-local statistics of the exported bank. Node and edge "
            "counts match exactly by construction; the remaining gaps are "
            "reported rather than hidden."
        ),
        "positive": {k: mean(v) for k, v in stats["positive"].items()},
        "negative": {k: mean(v) for k, v in stats["negative"].items()},
        "numPairs": sum(len(v) for v in bank.values()),
    }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BACKEND_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="output directory (default: Frontend/public/data)")
    parser.add_argument("--variants", type=int, default=6,
                        help="matched pairs per (distance, node count)")
    parser.add_argument("--quick", action="store_true",
                        help="two variants per key, for a fast smoke check")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    variants = 2 if args.quick else args.variants

    print(f"Exporting the static web bundle to {out_dir}")

    manifest: Dict[str, Any] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gitRevision": git_revision(),
        "python": platform.python_version(),
        "files": {},
    }

    # --- model -------------------------------------------------------
    checkpoint = RESULTS_DIR / "model_seed1.pt"
    if checkpoint.exists():
        model = export_model(checkpoint)
        size = write_json(out_dir / "model.json", model)
        manifest["files"]["model.json"] = {
            "bytes": size,
            "checkpointHash": model["checkpointHash"],
            "numParameters": model["numParameters"],
            "evidence": "Live computation (weights; forward pass runs in the browser)",
        }
        print(f"  model.json          {size / 1024:7.1f} KB  "
              f"checkpoint {model['checkpointHash']}")
    else:
        print(f"  model.json          SKIPPED, no {checkpoint}")

    # The unshared ablation checkpoint is deliberately not exported. Nothing
    # in the interface runs it: the ablation's point is a parameter count and a
    # hard depth ceiling, both of which come from the precomputed sweep. Its
    # weights would add roughly 450 KB to every page load for no interaction.

    # --- case bank ---------------------------------------------------
    print(f"  building the case bank ({variants} pairs per key)...")
    bank = build_case_bank(variants)
    bank["presets"] = export_presets()
    bank["matching"] = matching_report(bank["bank"])
    size = write_json(out_dir / "cases.json", bank)
    manifest["files"]["cases.json"] = {
        "bytes": size,
        "pairs": bank["pairsExported"],
        "hash": payload_hash(bank["bank"]),
        "evidence": "Synthetic data",
    }
    print(f"  cases.json          {size / 1024:7.1f} KB  "
          f"{bank['pairsExported']} matched pairs, {bank['pairsDropped']} dropped")
    print(f"    matched means: {bank['matching']}")

    # --- precomputed experiment outputs ------------------------------
    for name in ("learned_experiment.json", "learned_examples.json"):
        source = RESULTS_DIR / name
        if not source.exists():
            print(f"  {name:19s} SKIPPED, not found. Run python reproduce.py")
            continue
        payload = json.loads(source.read_text())
        short = name.replace("learned_", "")
        size = write_json(out_dir / short, payload)
        manifest["files"][short] = {
            "bytes": size,
            "sourceFile": name,
            "evidence": "Precomputed result",
        }
        print(f"  {short:19s} {size / 1024:7.1f} KB")

    write_json(out_dir / "manifest.json", manifest)
    print(f"  manifest.json")
    print("Done. The frontend now needs no backend to run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
