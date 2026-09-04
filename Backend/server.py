from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.graph import PRESETS, Graph, get_preset, preset_names
from core.recurrent import run_exact


app = FastAPI(
    title="Latent Loop Lab Exact Backend",
    description=(
        "Backend for the exact graph recurrence layer of Latent Loop Lab. "
        "This provides exact latent-state trajectories and independent BFS truth."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Latent Loop Lab exact backend is running.",
        "docs": "/docs",
        "presets": "/presets",
        "example": "/exact/preset/line/4",
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
