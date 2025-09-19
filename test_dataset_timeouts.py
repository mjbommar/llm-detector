#!/usr/bin/env python3
"""Test each dataset with timeout to identify problematic sources."""

import signal
import sys
from contextlib import contextmanager

from llm_detector.training.sources.registry import DEFAULT_REGISTRY


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds):
    """Timeout context manager."""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")

    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable alarm


def test_source_with_timeout(source_name, timeout_seconds=10, sample_count=10):
    """Test if a source can provide samples within timeout."""
    source_def = DEFAULT_REGISTRY.get(source_name)
    if not source_def:
        return f"NOT FOUND", None

    try:
        source = source_def.factory()

        with timeout(timeout_seconds):
            samples = []
            for i, sample in enumerate(source):
                samples.append(sample)
                if i >= sample_count - 1:
                    break

            if len(samples) < sample_count:
                return f"INSUFFICIENT ({len(samples)}/{sample_count})", samples
            return "OK", samples

    except TimeoutException as e:
        return f"TIMEOUT ({timeout_seconds}s)", None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)[:50]}", None


def main():
    print("Testing all enabled data sources with 10-second timeout...")
    print("=" * 80)

    # Test all enabled sources
    enabled_sources = [s.name for s in DEFAULT_REGISTRY.all(enabled_only=True)]

    problematic = []
    working = []

    for source_name in enabled_sources:
        print(f"Testing {source_name}...", end=" ", flush=True)

        status, samples = test_source_with_timeout(source_name, timeout_seconds=10, sample_count=10)

        if status == "OK":
            print(f"✓ {status} - Got {len(samples)} samples")
            working.append(source_name)
        else:
            print(f"✗ {status}")
            problematic.append((source_name, status))

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Working sources ({len(working)}):")
    for name in working:
        print(f"  ✓ {name}")

    if problematic:
        print(f"\nProblematic sources ({len(problematic)}):")
        for name, status in problematic:
            print(f"  ✗ {name}: {status}")

        print("\nRECOMMENDATION: Disable problematic sources before training!")
        print("Run: uv run python test_dataset_timeouts.py")

        return 1  # Exit with error
    else:
        print("\nAll sources working! Safe to train.")
        return 0


if __name__ == "__main__":
    sys.exit(main())