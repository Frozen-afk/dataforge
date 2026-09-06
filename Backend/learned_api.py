"""
Learned-model endpoints.

    GET  /learned/status       what is trained and available right now
    GET  /learned/experiment   aggregated depth curves, ablation, training logs
    GET  /learned/examples     categorised success and failure cases
    POST /learned/run          live inference on one graph at a chosen depth
    POST /learned/sweep        the same graph at every depth 1..maxR

The split between "live computation" and "precomputed result" is enforced here
rather than left to the frontend: every payload carries an evidence object
naming what produced it, so a page cannot silently present a stored sweep as a
live measurement.

If no checkpoint exists the endpoints say so explicitly instead of inventing
numbers. Pages 1 to 3 do not depend on any of this.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

RESULTS_DIR = Path(__file__).resolve().parent / "results"

router = APIRouter(prefix="/learned", tags=["learned"])

try:
    import torch

    from core.graph import Graph
    from core.recurrent import run_exact
    from learned.dataset import case_to_sample, collate, make_demo_case
    from learned.evaluate import file_hash, load_checkpoint

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the install
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)


_model = None
_meta: Dict[str, Any] = {}


def _checkpoint_path() -> Path:
    return RESULTS_DIR / "model_seed1.pt"


def _load_model():
    """Lazy-load the frozen seed-1 checkpoint. Returns None when unavailable."""
    global _model, _meta

    if not TORCH_AVAILABLE or _model is not None:
        return _model

    path = _checkpoint_path()
    if not path.exists():
        return None

    import torch as _torch

    model, checkpoint = load_checkpoint(path, _torch.device("cpu"))

    _model = model
    _meta = {
        "checkpointHash": file_hash(path),
        "sharedWeights": checkpoint.get("shared_weights", True),
        "hiddenDim": checkpoint.get("hidden_dim", 32),
        "aggregation": checkpoint.get("aggregation", "max"),
        "trainingMaxPathLength": checkpoint.get("trainingMaxPathLength", 4),
        "trainingDistances": checkpoint.get("training_distances", [1, 2, 3, 4]),
        "trainingMaxR": checkpoint.get("max_R", 4),
        "numParameters": checkpoint.get("config", {}).get("numParameters"),
        "seed": checkpoint.get("seed", 1),
        # None means "unbounded": shared weights can be reapplied indefinitely.
        "maxInferenceDepth": (
            model.max_inference_depth if model.depth_is_bounded else None
        ),
        "depthIsBounded": model.depth_is_bounded,
    }
    return _model


def _read_json(name: str) -> Dict[str, Any]:
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(
            404,
            f"{name} not found. Run the learned pipeline first: "
            f"python reproduce.py",
        )
    return json.loads(path.read_text())


@router.get("/status")
def status() -> Dict[str, Any]:
    """
    What the learned layer can actually do in this deployment.

    The interface uses this to decide whether to label learned numbers as live
    computation or as a precomputed result, so the badge always matches
    reality instead of being hard-coded in the page.
    """
    checkpoint = _checkpoint_path()
    model = _load_model()

    return {
        "torchAvailable": TORCH_AVAILABLE,
        "torchImportError": TORCH_IMPORT_ERROR or None,
        "checkpointPresent": checkpoint.exists(),
        "liveInferenceAvailable": model is not None,
        "experimentPresent": (RESULTS_DIR / "learned_experiment.json").exists(),
        "examplesPresent": (RESULTS_DIR / "learned_examples.json").exists(),
        "meta": _meta if model is not None else {},
        "evidenceTypeForLearnedNumbers": (
            "Live computation" if model is not None else "Precomputed result"
        ),
    }


@router.get("/experiment")
def get_experiment() -> Dict[str, Any]:
    """Aggregated multi-seed depth curves. Always a precomputed result."""
    return _read_json("learned_experiment.json")


@router.get("/examples")
def get_examples() -> Dict[str, Any]:
    """Categorised successes and failures from the frozen checkpoints."""
    return _read_json("learned_examples.json")


class LearnedRunRequest(BaseModel):
    """
    One graph, one inference depth.

    Either a distance is requested and the server generates a matching case, or
    an explicit graph is supplied so the learned model and the exact mechanism
    can be compared on exactly the same input.
    """

    distance: Optional[int] = Field(None, ge=1, le=10)
    reachable: bool = True
    nodes: int = Field(12, ge=2, le=24)
    density: float = Field(0.15, gt=0.0, le=1.0)
    caseSeed: int = 0

    graph: Optional[Dict[str, Any]] = None
    source: Optional[int] = None
    target: Optional[int] = None

    R: int = Field(4, ge=0, le=12)


def _require_model():
    if not TORCH_AVAILABLE:
        raise HTTPException(
            503,
            f"PyTorch is not installed on the server, so live learned inference "
            f"is unavailable. The precomputed sweep is still served from "
            f"/learned/experiment. ({TORCH_IMPORT_ERROR})",
        )

    model = _load_model()
    if model is None:
        raise HTTPException(
            503,
            "No checkpoint at results/model_seed1.pt. Train one with "
            "'python -m learned.train --seed 1', or use /learned/experiment "
            "for the precomputed sweep.",
        )
    return model


def _build_sample(request: LearnedRunRequest) -> Dict[str, Any]:
    """Resolve the request into one sample, explicit graph or generated case."""
    if request.graph is not None:
        if request.source is None or request.target is None:
            raise HTTPException(
                400, "source and target are required when supplying a graph."
            )
        try:
            graph = Graph.from_dict(request.graph)
        except Exception as exc:
            raise HTTPException(400, f"Invalid graph: {exc}")

        from core.bfs import bfs_distance

        distance = bfs_distance(graph.outgoing(), request.source, request.target)
        return case_to_sample({
            "graph": graph,
            "source": request.source,
            "target": request.target,
            "distance": distance,
            "density": None,
            "targetInDegree": sum(1 for _, v in graph.edges if v == request.target),
        })

    distance = request.distance if request.reachable else None
    if request.reachable and distance is None:
        distance = 4

    sample = make_demo_case(
        distance=distance,
        n=request.nodes,
        density=request.density,
        seed=request.caseSeed,
    )
    if sample is None:
        raise HTTPException(
            400,
            f"Could not generate a graph with distance {distance} on "
            f"{request.nodes} nodes. Try more nodes or a shorter distance.",
        )

    # A generated negative inherits the distance it was cut from, so the page
    # can still say how far apart the pair would have been.
    if distance is None and request.distance is not None:
        sample["bucket_distance"] = request.distance

    return sample


def _infer(model, sample: Dict[str, Any], R: int) -> Dict[str, Any]:
    """One forward pass, plus the exact mechanism on the identical graph."""
    import torch as _torch

    batch = collate([sample])

    with _torch.no_grad():
        logit, trajectory, logits_per_step = model.forward_with_trajectory(
            batch.x, batch.edge_index, batch.target_idx, R
        )

    probability = _torch.sigmoid(logit)[0].item()
    prediction = 1.0 if probability > 0.5 else 0.0
    label = float(sample["label"])

    graph = Graph.from_dict(sample["graph"])
    source = int(sample["source_idx"])
    target = int(sample["target_idx"])
    exact = run_exact(graph, source, target, R=min(R, 12))

    return {
        "R": R,
        "probability": probability,
        "prediction": prediction,
        "confidence": probability if prediction == 1.0 else 1.0 - probability,
        "correct": prediction == label,
        "probabilityPerStep": [
            _torch.sigmoid(step)[0].item() for step in logits_per_step
        ],
        "targetStateNormPerStep": [
            step[batch.target_idx[0]].norm().item() for step in trajectory
        ],
        "meanStateNormPerStep": [
            step.norm(dim=-1).mean().item() for step in trajectory
        ],
        "exact": {
            "estimate": exact["estimate"],
            "targetActivation": exact["targetActivation"],
            "trajectory": exact["trajectory"],
            "activeNodeCount": sum(1 for value in exact["trajectory"][-1] if value > 0),
            "activationMass": sum(exact["trajectory"][-1]),
        },
    }


def _sample_summary(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "graph": sample["graph"],
        "source": sample["source_idx"],
        "target": sample["target_idx"],
        "bfsDistance": sample["distance"],
        "label": sample["label"],
        "reachable": sample["label"] == 1.0,
        "numNodes": sample["num_nodes"],
        "numEdges": sample["num_edges"],
    }


@router.post("/run")
def learned_run(request: LearnedRunRequest) -> Dict[str, Any]:
    """Live inference at one depth, beside the exact mechanism's answer."""
    model = _require_model()
    sample = _build_sample(request)

    ceiling = model.max_inference_depth
    if request.R > ceiling:
        raise HTTPException(
            400,
            f"This checkpoint has only {ceiling} update blocks and cannot run at "
            f"depth {request.R}. Without shared weights, inference depth is "
            f"fixed at training time.",
        )

    result = _infer(model, sample, request.R)

    return {
        "case": _sample_summary(sample),
        "result": result,
        "meta": _meta,
        "evidence": {
            "evidenceType": "Live computation",
            "source": (
                "Live forward pass of the toy shared-weight recurrent GNN "
                "(independent reimplementation, not BDH or BDH-CQ)"
            ),
            "experimentId": f"learned-live-{_meta.get('checkpointHash', 'unknown')}",
        },
    }


class LearnedSweepRequest(LearnedRunRequest):
    maxR: int = Field(10, ge=1, le=12)


@router.post("/sweep")
def learned_sweep(request: LearnedSweepRequest) -> Dict[str, Any]:
    """
    The core experiment on one graph: same weights, same input, every depth.

    This is what Page 4 animates. Holding the graph and the parameters fixed
    while only R varies is the whole point, so the sweep is computed in one
    request rather than assembled from separate calls that might drift.
    """
    model = _require_model()
    sample = _build_sample(request)
    ceiling = model.max_inference_depth

    results: List[Dict[str, Any]] = []
    for R in range(1, request.maxR + 1):
        if R > ceiling:
            results.append({
                "R": R,
                "unavailable": True,
                "reason": (
                    f"Model has {ceiling} update blocks. Without shared weights, "
                    f"inference depth cannot exceed training depth."
                ),
            })
            continue
        results.append(_infer(model, sample, R))

    available = [r for r in results if not r.get("unavailable")]
    distance = sample["distance"]

    # The depth from which the model *stays* committed to "reachable". A single
    # crossing is not a commitment: probabilities hover near 0.5 while the
    # answer is still unknowable, so an early 0.504 would otherwise be reported
    # as the flip and make the model look like it decided before it could.
    flip_depth = None
    for index, entry in enumerate(available):
        if all(later["prediction"] == 1.0 for later in available[index:]):
            flip_depth = entry["R"]
            break

    return {
        "case": _sample_summary(sample),
        "results": results,
        "maxInferenceDepth": ceiling if model.depth_is_bounded else None,
        "depthIsBounded": model.depth_is_bounded,
        "learnedFlipDepth": flip_depth,
        "exactFlipDepth": distance,
        "flipDepthsAgree": flip_depth == distance,
        "meta": _meta,
        "evidence": {
            "evidenceType": "Live computation",
            "source": (
                "Live depth sweep over one frozen toy checkpoint on a single "
                "fixed graph"
            ),
            "experimentId": f"learned-sweep-{_meta.get('checkpointHash', 'unknown')}",
        },
    }
