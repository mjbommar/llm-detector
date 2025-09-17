"""Command-line interface for building baseline artifacts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import compute_baselines_to_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute and store baseline distributions.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .json.gz path for the baseline cache",
    )
    parser.add_argument(
        "--samples-per-source",
        type=int,
        default=None,
        help="Maximum samples to stream from each registered source",
    )
    parser.add_argument(
        "--version", default="v1", help="Version label stored alongside the artifacts"
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include sources marked as disabled in the registry",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if the output file already exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    samples = args.samples_per_source
    if samples is not None and samples < 0:
        parser.error("--samples-per-source must be non-negative")

    try:
        compute_baselines_to_path(
            args.output,
            samples_per_source=samples,
            version=args.version,
            enabled_only=not args.include_disabled,
            overwrite=args.overwrite,
        )
    except Exception as exc:  # pragma: no cover - CLI delivery
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
