import argparse
import json
from pathlib import Path

from core.graph import get_preset, preset_names
from core.recurrent import run_exact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact latent graph recurrence from the command line."
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="line",
        choices=preset_names(),
        help="Graph preset to run.",
    )

    parser.add_argument(
        "--R",
        type=int,
        default=4,
        help="Number of recurrent updates. Must be between 0 and 12.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Propagation factor. Must satisfy 0 < alpha <= 1.",
    )

    parser.add_argument(
        "--source",
        type=int,
        default=None,
        help="Optional override for source node.",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Optional override for target node.",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save JSON result.",
    )

    args = parser.parse_args()

    graph, source, target, description = get_preset(args.preset)

    if args.source is not None:
        source = args.source

    if args.target is not None:
        target = args.target

    result = run_exact(
        graph=graph,
        source=source,
        target=target,
        R=args.R,
        alpha=args.alpha,
    )

    result["preset"] = args.preset
    result["description"] = description

    print(json.dumps(result, indent=2))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved result to {save_path}")


if __name__ == "__main__":
    main()
