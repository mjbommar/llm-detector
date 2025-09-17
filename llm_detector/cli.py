"""Command-line interface for runtime scoring."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from . import assets
from .runtime import DetectorRuntime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score text using a trained LLM detector")
    parser.add_argument(
        "--model", required=False, type=Path, help="Path to the trained model artifact"
    )
    parser.add_argument(
        "--baselines",
        required=False,
        type=Path,
        help="Path to the baseline cache (JSON.gz) matching the model",
    )
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="Explicit text input (can be provided multiple times)",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        type=Path,
        help="File whose entire contents should be scored (can be repeated)",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read additional newline-delimited samples from standard input",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON instead of human-readable text",
    )
    parser.add_argument(
        "--return-features",
        action="store_true",
        help="Include the feature vector in JSON output",
    )
    parser.add_argument(
        "--cohort",
        choices=["human", "llm"],
        default="human",
        help="Select which side of the baselines to use for divergence priors",
    )
    parser.add_argument(
        "--show-diagnostics",
        action="store_true",
        help="Print auxiliary aggregation metrics alongside the primary score",
    )
    return parser


def _read_inputs(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []

    if args.texts:
        texts.extend([text for text in args.texts if text])

    if args.files:
        for path in args.files:
            content = Path(path).read_text(encoding="utf-8")
            if content:
                texts.append(content)

    if args.stdin or not texts:
        stdin_data = sys.stdin.read()
        if stdin_data:
            texts.extend([line for line in stdin_data.splitlines() if line.strip()])

    return texts


def _format_human(readings: Sequence[dict], *, show_diagnostics: bool = False) -> str:
    lines: list[str] = []
    for idx, item in enumerate(readings, start=1):
        label = "LLM" if item["is_llm"] else "HUMAN"
        confidence = f"{item['confidence']:.4f}"
        p_llm = f"{item['p_llm']:.4f}"
        primary = item.get("primary_metric") or item.get("details", {}).get("primary_metric")
        snippet = item["text"].replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        if primary:
            lines.append(
                f"[{idx}] {label} (p_llm={p_llm} [{primary}], confidence={confidence}) :: {snippet}"
            )
        else:
            lines.append(f"[{idx}] {label} (p_llm={p_llm}, confidence={confidence}) :: {snippet}")

        if show_diagnostics:
            metrics = item.get("aggregators")
            if metrics is None:
                metrics = item.get("details", {}).get("document_metrics")
            if metrics:
                diag_parts: list[str] = []
                for key in ("simple_mean", "max_score", "vote_fraction"):
                    if key in metrics:
                        diag_parts.append(f"{key}={metrics[key]:.4f}")
                if diag_parts:
                    lines.append("    diagnostics: " + ", ".join(diag_parts))
    if not lines:
        return ""
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    inputs = _read_inputs(args)
    if not inputs:
        parser.error("No input text provided; use --text/--file or pipe data via --stdin")

    artifact_cm = None
    default_model: Path | None = None
    default_baseline: Path | None = None
    if args.model is None or args.baselines is None:
        artifact_cm = assets.default_artifacts()
        defaults = artifact_cm.__enter__()
        if defaults is None:
            parser.error("Bundled model/baselines not available; specify --model and --baselines")
        default_model, default_baseline = defaults

    model_path = Path(args.model) if args.model is not None else default_model
    baseline_path = Path(args.baselines) if args.baselines is not None else default_baseline
    assert model_path is not None and baseline_path is not None

    try:
        runtime = DetectorRuntime(
            model_path=model_path,
            baseline_path=baseline_path,
            cohort=args.cohort,
            return_features=args.return_features,
        )

        outputs: list[dict] = []
        for text in inputs:
            result = runtime.predict(text)
            payload = {
                "is_llm": result.is_llm,
                "confidence": result.confidence,
                "p_llm": result.p_llm,
                "p_human": result.p_human,
                "text": text,
            }
            if args.return_features and result.features is not None:
                payload["features"] = result.features
            if result.details is not None:
                payload["details"] = result.details
                payload["aggregators"] = result.details.get("document_metrics")
                payload["primary_metric"] = result.details.get("primary_metric")
                payload["diagnostic_metrics"] = result.details.get("diagnostic_metrics")
            else:
                payload["primary_metric"] = getattr(runtime, "primary_metric", None)
                payload["diagnostic_metrics"] = list(getattr(runtime, "diagnostic_metrics", ()))
            outputs.append(payload)

        if args.json:
            json.dump(outputs, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            sys.stdout.write(_format_human(outputs, show_diagnostics=args.show_diagnostics) + "\n")
        return 0
    finally:
        if artifact_cm is not None:
            artifact_cm.__exit__(None, None, None)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
