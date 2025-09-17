# llm-detector

**Research WIP**: Transparent, probabilistic classification of text as human-generated or LLM-generated.

## Installation

```bash
pip install llm-detector
```

## Quick Start

```python
from llm_detector import classify_text

# Simple classification
result = classify_text("Your text here")
print(f"LLM probability: {result['p_llm']:.2%}")
print(f"Classification: {'LLM' if result['is_llm'] else 'Human'}")
```

## Advanced Usage

### Using the Runtime API

```python
from llm_detector import DetectorRuntime
from llm_detector.assets import default_artifacts

# Initialize detector with default models
with default_artifacts() as (model_path, baseline_path):
    detector = DetectorRuntime(
        model_path=model_path,
        baseline_path=baseline_path
    )

    # Single text classification
    result = detector.predict("This is a sample text.")
    print(f"LLM: {result.p_llm:.2%}, Human: {result.p_human:.2%}")

    # Access detailed metrics
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Document metrics:", result.details['document_metrics'])
```

### Detailed Results with Diagnostics

```python
from llm_detector import classify_text

result = classify_text(
    "Your text here",
    include_diagnostics=True
)

# Access classification
print(f"Classification: {'LLM' if result['is_llm'] else 'Human'}")
print(f"LLM probability: {result['p_llm']:.4f}")
print(f"Confidence: {result['confidence']:.4f}")

# Diagnostic metrics for analysis
if 'diagnostics' in result:
    diag = result['diagnostics']
    print(f"Simple mean: {diag.get('simple_mean', 0):.4f}")
    print(f"Max score: {diag.get('max_score', 0):.4f}")
```

## CLI Usage

```bash
# Classify text from command line
llm-detector --text "Your text here"

# Classify from file
llm-detector --file input.txt

# Get detailed output with diagnostics
llm-detector --text "Your text" --show-diagnostics --json
```

## Research Notes

This is an active research project exploring transparent statistical methods for LLM detection. The approach combines:

- **Statistical features**: Lexical diversity, punctuation patterns, repetition metrics
- **Tokenizer divergence**: Cross-tokenizer efficiency and consistency metrics
- **Ensemble aggregation**: Logit-weighted mean with diagnostic fallbacks

Current limitations:
- Performance varies by text length (best with 3+ sentences)
- Optimized for general English text
- Continuous model updates as LLM capabilities evolve

## Development

```bash
# Install with development dependencies
pip install -e ".[dev,training]"

# Run tests
pytest

# Train custom models (requires training extras)
python -m llm_detector.training.cli --help
```

## License

MIT
