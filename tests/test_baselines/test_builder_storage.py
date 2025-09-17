from pathlib import Path

from llm_detector.baselines import (
    BaselineCache,
    build_from_samples,
)
from llm_detector.types import TextSample


def test_build_and_roundtrip(tmp_path: Path):
    samples = [
        TextSample(text="Human text. Natural language.", is_llm=False, source="unit"),
        TextSample(text="Another human sentence!", is_llm=False, source="unit"),
        TextSample(text="LLM output with patterns.", is_llm=True, source="unit"),
        TextSample(text="More LLM text; consistent punctuation.", is_llm=True, source="unit"),
    ]
    baselines = build_from_samples(samples)
    path = tmp_path / "baselines.npz"
    BaselineCache.save(baselines, path)
    loaded = BaselineCache.load(path)

    # Check distributions exist and sum to ~1
    for pair in (loaded.unicode, loaded.regex, loaded.punctws):
        for art in pair:
            if art.distribution:
                s = sum(art.distribution)
                assert 0.99 <= s <= 1.01
