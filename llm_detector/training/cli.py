"""Command-line entry point for training the logistic classifier."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .pipeline import train_logistic_from_registry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the logistic LLM detector")
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Output path for the trained model artifact"
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=None,
        help="Optional path for storing/loading baseline distributions",
    )
    parser.add_argument(
        "--samples-per-source",
        type=int,
        default=None,
        help="Maximum samples to draw from each source for feature training",
    )
    parser.add_argument(
        "--baseline-samples-per-source",
        type=int,
        default=None,
        help="Sample limit per source when computing baselines (defaults to --samples-per-source)",
    )
    parser.add_argument(
        "--version", default="v1", help="Version label stored alongside baseline artifacts"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling and splits")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of the dataset reserved for evaluation (set to 0 to disable)",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing between human and LLM samples",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before feature extraction",
    )
    parser.add_argument(
        "--include-length-dependent",
        action="store_true",
        help="Include scale-sensitive features during training",
    )
    parser.add_argument(
        "--include-disabled-sources",
        action="store_true",
        help="Stream from sources that are marked disabled in the registry",
    )
    parser.add_argument(
        "--overwrite-baseline",
        action="store_true",
        help="Force recomputation of baseline artifacts when --baseline-path exists",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a tqdm progress bar during feature extraction",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.samples_per_source is not None and args.samples_per_source < 0:
        parser.error("--samples-per-source must be non-negative")
    if args.baseline_samples_per_source is not None and args.baseline_samples_per_source < 0:
        parser.error("--baseline-samples-per-source must be non-negative")
    if args.test_ratio < 0:
        parser.error("--test-ratio must be non-negative")

    baseline_path = args.baseline_path
    if baseline_path is not None and args.overwrite_baseline and baseline_path.exists():
        baseline_path.unlink()

    test_ratio: float | None
    if args.test_ratio == 0:
        test_ratio = None
    else:
        test_ratio = args.test_ratio

    artifacts = train_logistic_from_registry(
        model_path=args.model_path,
        baseline_path=baseline_path,
        samples_per_source=args.samples_per_source,
        baseline_samples_per_source=args.baseline_samples_per_source,
        version=args.version,
        enabled_only=not args.include_disabled_sources,
        balance=not args.no_balance,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        scale_invariant_only=not args.include_length_dependent,
        test_ratio=test_ratio,
        show_progress=args.progress,
    )

    print(f"model saved to: {artifacts.model_path}")
    if artifacts.baselines_path is not None:
        print(f"baselines cached at: {artifacts.baselines_path}")
    print(f"train accuracy: {artifacts.train_accuracy:.4f}")
    for key, value in sorted(artifacts.metrics.items()):
        print(f"{key}: {value:.4f}")
    print(f"features: {len(artifacts.feature_names)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
