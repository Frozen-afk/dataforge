"""
Learned-model API endpoints for Latent Loop Lab Pages 4 and 5.

Provides:
    GET  /learned/experiment   -> results/learned_experiment.json
    GET  /learned/examples     -> results/learned_examples.json
    POST /learned/run          -> live inference with frozen checkpoint (if available)
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

RESULTS_DIR = Path(__file__).resolve().parent / "results"

router = APIRouter(prefix="/learned", tags=["learned"])

try:
    import torch
    from learned.model import SharedRecurrentGNN
    from learned.dataset import make_path_graph

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


_model = None
_meta: Dict[str, Any] = {}


def _load_model():
    """Lazy-load the frozen seed-1 checkpoint."""
    global _model, _meta

    if not TORCH_AVAILABLE:
        return None

    if _model is not None:
        return _model

    ckpt_path = RESULTS_DIR / "model_seed1.pt"
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("hidden_dim", 32))

    model = SharedRecurrentGNN(
        input_dim=2,
        hidden_dim=hidden,
        message_dim=hidden,
        num_classes=1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sha = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    _model = model
    _meta = {
        "checkpointHash": sha.hexdigest()[:16],
        "hiddenDim": hidden,
        "trainingMaxPathLength": 4,
        "trainingMaxR": int(ckpt.get("max_R", 4)),
        "seed": int(ckpt.get("seed", 1)),
    }
    return _model


@router.get("/experiment")
def get_experiment() -> Dict[str, Any]:
    path = RESULTS_DIR / "learned_experiment.json"
    if not path.exists():
        raise HTTPException(
            404,
            "learned_experiment.json not found. Run the learned pipeline first.",
        )
    return json.loads(path.read_text())


@router.get("/examples")
def get_examples() -> Dict[str, Any]:
    path = RESULTS_DIR / "learned_examples.json"
    if not path.exists():
        raise HTTPException(
            404,
            "learned_examples.json not found. Run the learned pipeline first.",
        )
    return json.loads(path.read_text())


class LearnedRunRequest(BaseModel):
    pathLength: int = Field(..., ge=1, le=10)
    reachable: bool = True
    R: int = Field(..., ge=1, le=10)


@router.post("/run")
def learned_run(req: LearnedRunRequest) -> Dict[str, Any]:
    model = _load_model()

    if model is None:
        raise HTTPException(
            503,
            "Learned model not available. Train it first, or use precomputed examples.",
        )

    sample = make_path_graph(req.pathLength, reachable=req.reachable)

    with torch.no_grad():
        logit, trajectory = model.forward_with_trajectory(
            sample["node_features"],
            sample["edge_index"],
            sample["target_idx"],
            req.R,
        )
        prob = torch.sigmoid(logit).item()

    pred = 1.0 if prob > 0.5 else 0.0
    label = sample["label"]

    node_norms = trajectory[-1].norm(dim=-1).tolist()
    target_norm_series = [z[sample["target_idx"]].norm().item() for z in trajectory]

    return {
        "pathLength": req.pathLength,
        "reachable": req.reachable,
        "R": req.R,
        "label": label,
        "prediction": pred,
        "probability": prob,
        "confidence": prob if pred == 1.0 else 1.0 - prob,
        "correct": abs(pred - label) < 0.5,
        "numNodes": sample["node_features"].size(0),
        "nodeStateNorms": node_norms,
        "targetNormSeries": target_norm_series,
        "meta": _meta,
        "evidence": {
            "evidenceType": "Live computation",
            "source": "Frozen shared-weight recurrent GNN (live inference)",
            "experimentId": f"learned-live-{_meta.get('checkpointHash', 'unknown')}",
        },
    }
