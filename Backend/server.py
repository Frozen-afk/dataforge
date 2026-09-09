import random
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.generator import make_case, sample_target_in_degree
from core.graph import PRESETS, Graph, get_preset, preset_names
from core.recurrent import run_exact


app = FastAPI(
    title="Latent Loop Lab Backend",
    description=(
        "Reference implementation for Latent Loop Lab.\n\n"
        "The published artifact does not use this server. Both computational "
        "layers run in the visitor's browser from the bundle built by "
        "export_web.py, so the lab opens from a static URL with nothing "
        "running behind it.\n\n"
        "This API stays for two reasons. It is the implementation the browser "
        "engine is checked against by tests/test_js_parity.py, and it is a "
        "convenient way to exercise the mechanism from a terminal.\n\n"
        "The /exact routes are the mechanism layer: live noisy-OR graph "
        "recurrence with an independent BFS oracle. The /learned routes are "
        "the AI layer: a small shared-weight recurrent GNN used as an "
        "experimental bridge. The two are never the same model."
    ),
    version="1.0.0",
)

# This server carries no session, cookie or authentication of any kind, so it
# asks for no credentials. The pairing of a wildcard origin with
# allow_credentials=True is the combination browsers refuse outright, and it
# would be claiming a trust relationship that does not exist here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

from learned_api import router as learned_router

app.include_router(learned_router)

class ExactRunRequest(BaseModel):
    """
    Request body for /exact/run.

    If graph is provided, source and target are required.
    If graph is not provided, a preset is used.
    """

    preset: Optional[str] = "line"
    graph: Optional[Dict[str, Any]] = None
    source: Optional[int] = None
    target: Optional[int] = None

    R: int = Field(..., ge=0, le=12)
    alpha: float = Field(1.0, gt=0.0, le=1.0)


class ExactGenerateRequest(BaseModel):
    """
    Ask for a graph with a known source-to-target distance.

    This is what lets a learner set the answer first and then discover the
    depth it costs, instead of being limited to a handful of fixed presets.
    """

    distance: Optional[int] = Field(None, ge=1, le=10,
                                    description="None requests an unreachable pair")
    nodes: int = Field(12, ge=2, le=24)
    density: float = Field(0.15, gt=0.0, le=1.0)
    caseSeed: int = 0
    R: int = Field(4, ge=0, le=12)
    alpha: float = Field(1.0, gt=0.0, le=1.0)


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Latent Loop Lab backend is running.",
        "docs": "/docs",
        "presets": "/presets",
        "example": "/exact/preset/line/4",
        "learnedStatus": "/learned/status",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/presets")
def list_presets() -> Dict[str, Dict[str, str]]:
    return {
        name: {
            "description": item["description"],
        }
        for name, item in PRESETS.items()
    }


@app.get("/exact/preset/{name}/{R}")
def exact_preset(
    name: str,
    R: int = Path(..., ge=0, le=12),
    alpha: float = Query(1.0, gt=0.0, le=1.0),
    source: Optional[int] = Query(None),
    target: Optional[int] = Query(None),
) -> Dict[str, Any]:
    """
    Easy endpoint for browser testing.

    Example:
        /exact/preset/line/4
    """
    try:
        graph, preset_source, preset_target, description = get_preset(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown preset: {name}")

    if source is not None:
        preset_source = source

    if target is not None:
        preset_target = target

    try:
        result = run_exact(
            graph=graph,
            source=preset_source,
            target=preset_target,
            R=R,
            alpha=alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result["preset"] = name
    result["description"] = description

    return result


@app.post("/exact/run")
def exact_run(request: ExactRunRequest) -> Dict[str, Any]:
    """
    Main endpoint used by the frontend.

    The frontend can either request a preset:

        {
            "preset": "line",
            "R": 4
        }

    or send a custom graph:

        {
            "graph": {
                "n": 5,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4]]
            },
            "source": 0,
            "target": 4,
            "R": 4
        }
    """
    if request.graph is not None:
        try:
            graph = Graph.from_dict(request.graph)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid graph: {exc}")

        if request.source is None or request.target is None:
            raise HTTPException(
                status_code=400,
                detail="source and target are required when using a custom graph.",
            )

        source = request.source
        target = request.target
        description = "Custom graph"

    else:
        preset_name = request.preset or "line"

        if preset_name not in preset_names():
            raise HTTPException(
                status_code=404,
                detail=f"Unknown preset: {preset_name}",
            )

        try:
            graph, source, target, description = get_preset(preset_name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if request.source is not None:
            source = request.source

        if request.target is not None:
            target = request.target

    try:
        result = run_exact(
            graph=graph,
            source=source,
            target=target,
            R=request.R,
            alpha=request.alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result["description"] = description

    return result


@app.post("/exact/generate")
def exact_generate(request: ExactGenerateRequest) -> Dict[str, Any]:
    """
    Generate a graph with a requested distance, then run the exact recurrence.

    The same generator feeds the learned model's training and test sets, so a
    graph produced here is drawn from exactly the distribution the learned
    layer was measured on. That is what makes the two layers comparable rather
    than merely adjacent.
    """
    rng = random.Random(request.caseSeed)

    case = make_case(
        rng,
        n=request.nodes,
        distance=request.distance,
        density=request.density,
        target_in_degree=sample_target_in_degree(rng),
    )

    if case is None:
        raise HTTPException(
            400,
            f"Could not build a graph with distance "
            f"{request.distance if request.distance is not None else 'unreachable'} "
            f"on {request.nodes} nodes. Try more nodes or a shorter distance.",
        )

    result = run_exact(
        graph=case["graph"],
        source=int(case["source"]),
        target=int(case["target"]),
        R=request.R,
        alpha=request.alpha,
    )

    result["description"] = (
        f"Generated graph, {request.nodes} nodes, "
        + (
            f"shortest path {case['distance']}"
            if case["distance"] is not None
            else "target unreachable"
        )
    )
    result["requestedDistance"] = request.distance
    result["caseSeed"] = request.caseSeed

    return result


@app.get("/exact/trajectory/summary")
def trajectory_summary(
    preset: str = Query("line"),
    R: int = Query(4, ge=0, le=12),
    alpha: float = Query(1.0, gt=0.0, le=1.0),
) -> Dict[str, Any]:
    """
    Per-step activation statistics for one preset.

    Reports the active-node count and total activation mass after each update,
    which are the two metrics the architecture asks the interface to show for
    the exact model.
    """
    try:
        graph, source, target, description = get_preset(preset)
    except KeyError:
        raise HTTPException(404, f"Unknown preset: {preset}")

    result = run_exact(graph, source, target, R=R, alpha=alpha)
    epsilon = result["epsilon"]

    return {
        "preset": preset,
        "description": description,
        "R": R,
        "bfsDistance": result["bfsDistance"],
        "steps": [
            {
                "r": step,
                "activeNodeCount": sum(1 for value in state if value > epsilon),
                "activationMass": sum(state),
                "targetActivation": state[target],
            }
            for step, state in enumerate(result["trajectory"])
        ],
        "evidence": result["evidence"],
    }
