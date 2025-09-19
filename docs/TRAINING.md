# Training LLM Detector Models

This document explains how to train custom LLM detector models and understand the model format.

## Prerequisites

To train models, you need the training dependencies:

```bash
# Install with training extras
pip install llm-detector[training]

# Or with uv
uv pip install -e ".[training]"
```

This installs:
- `scikit-learn` (for model training)
- `datasets` (for streaming data sources)
- `numpy` (for numerical operations during training)

**Note**: These dependencies are **only** required for training. Inference works without them using the pure Python implementation.

## Training a Model

### Using the CLI (Recommended)

The simplest way to train a model is using the built-in CLI:

```bash
llm-detector-train \
  --model-path my_model.json.gz \
  --baseline-path my_baselines.json.gz \
  --samples-per-source 5000 \
  --test-ratio 0.2 \
  --progress
```

Key parameters:
- `--model-path`: Where to save the trained model (JSON format)
- `--baseline-path`: Where to cache baseline distributions
- `--samples-per-source`: Number of samples per data source (6 sources total)
- `--test-ratio`: Fraction of data for testing (0.2 = 20%)
- `--progress`: Show progress bars during training

### Using Python API

```python
from pathlib import Path
from llm_detector.training.pipeline import train_logistic_from_registry

# Train a model
artifacts = train_logistic_from_registry(
    model_path=Path("model.json.gz"),
    baseline_path=Path("baselines.json.gz"),

    # Data configuration
    samples_per_source=5000,  # Per source (6 sources = 30k samples)
    baseline_samples_per_source=5000,

    # Model configuration
    scale_invariant_only=True,  # Use only scale-invariant features
    test_ratio=0.2,  # 20% test split

    # Training configuration
    balance=True,  # Balance classes
    shuffle=True,  # Shuffle data
    seed=42,  # Random seed for reproducibility

    show_progress=True,
)

print(f"Train accuracy: {artifacts.train_accuracy:.4f}")
print(f"Test metrics: {artifacts.metrics}")
```

## Data Sources

The training pipeline uses 6 data sources by default:

### Human Sources
1. **FinePDFs**: Academic/technical PDF documents
2. **Project Gutenberg**: Classic literature
3. **Wikipedia**: Encyclopedia articles

### LLM Sources
1. **Cosmopedia**: Synthetic educational content
2. **LMSYS Chat**: Real LLM conversations
3. **UltraChat**: Generated dialogues

## Model Format

The trained model is saved as a compressed JSON file with the following structure:

```json
{
  "format": "llm-detector/logistic-regression",
  "version": 1,

  "feature_names": [
    "div.char_jsd",
    "div.punct_jsd",
    "stat.mean_word_length",
    ...
  ],

  "class_weight": "balanced",
  "max_iter": 1000,
  "random_state": 42,

  "scaler": {
    "mean": [0.123, 0.456, ...],  // Feature means
    "scale": [0.789, 0.234, ...]  // Feature std devs
  },

  "model": {
    "coefficients": [1.23, -0.45, ...],  // Feature weights
    "intercept": -0.123,  // Model bias term
    "classes": [0, 1]  // 0=human, 1=LLM
  }
}
```

### Model Components

1. **Feature Names**: List of 80 features used by the model
2. **Scaler Parameters**: Mean and standard deviation for Z-score normalization
3. **Model Coefficients**: Learned weights for each feature
4. **Intercept**: Bias term for the logistic regression

### File Sizes

- Compressed: ~3-5 KB (gzip)
- Uncompressed: ~7-10 KB (JSON)
- Compression ratio: ~2-3x

## Feature Importance

After training, you can examine feature importance:

```python
import json
import gzip

# Load model
with gzip.open('model.json.gz', 'rt') as f:
    model = json.load(f)

# Get feature importance (absolute coefficients)
features = model['feature_names']
coeffs = model['model']['coefficients']

importance = sorted(
    zip(features, coeffs),
    key=lambda x: abs(x[1]),
    reverse=True
)

# Top 10 most important features
for name, coef in importance[:10]:
    print(f"{name:40} {coef:+.4f}")
```

Typical important features:
- `tok.*.word_match_rate`: Token-word alignment
- `stat.period_ratio`: Sentence ending frequency
- `stat.repeated_punctuation_ratio`: Punctuation patterns
- `tok.*.tokenization_efficiency`: Encoding efficiency
- `stat.herdan_c`: Lexical diversity

## Custom Training Data

To train on custom data sources:

```python
from llm_detector.training.sources.base import BaseDataSource
from llm_detector.types import TextSample

class CustomSource(BaseDataSource):
    def __init__(self, texts, is_llm=False):
        super().__init__(
            name="custom",
            category="llm" if is_llm else "human"
        )
        self.texts = texts
        self.is_llm = is_llm

    def stream_samples(self, limit=None):
        for i, text in enumerate(self.texts[:limit]):
            yield TextSample(
                text=text,
                is_llm=self.is_llm,
                source=self.name,
                metadata={"index": i}
            )

# Register and use
from llm_detector.training.sources.registry import SourceRegistry

registry = SourceRegistry()
registry.register("custom_human", CustomSource(human_texts, False))
registry.register("custom_llm", CustomSource(llm_texts, True))

# Train with custom registry
artifacts = train_logistic_from_registry(
    registry=registry,
    model_path=Path("custom_model.json.gz"),
    ...
)
```

## Performance Tuning

### Sample Size
- Minimum: 1000 samples per source
- Recommended: 5000-10000 per source
- Production: 10000+ per source

### Feature Selection
- Default: 80 scale-invariant features
- Optional: Include length-dependent features with `scale_invariant_only=False`
- Custom: Modify feature registry before training

### Hyperparameters
- `class_weight`: "balanced" (default) or custom weights
- `max_iter`: 1000 (usually sufficient)
- `C`: Regularization strength (not exposed, uses sklearn default)

## Evaluation

The training pipeline automatically computes:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True + False positives)
- **Recall**: True positives / (True + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

Typical performance:
- Accuracy: 85-95%
- F1 Score: 0.85-0.95
- Precision: 85-95%
- Recall: 85-95%

## Migration from joblib to JSON

Old models (v0.1.0) used joblib format. New models (v0.1.1+) use JSON:

```python
from llm_detector.models import LogisticRegressionModel

# Load old joblib model
model = LogisticRegressionModel.load("old_model.joblib")

# Save as JSON
model.save("new_model.json.gz")
```

The JSON format enables:
- Inference without numpy/scikit-learn
- Smaller file sizes
- Cross-platform compatibility
- Human-readable format

## Deployment

For production deployment:

1. **Train on representative data**: Include diverse text types
2. **Validate on held-out data**: Use separate test sets
3. **Monitor performance**: Track accuracy over time
4. **Update regularly**: Retrain as LLMs evolve

### Model Serving

```python
# Load once at startup
from llm_detector import DetectorRuntime

runtime = DetectorRuntime(
    model_path="model.json.gz",
    baseline_path="baselines.json.gz"
)

# Use for many predictions
def classify(text):
    result = runtime.predict(text)
    return {
        "is_llm": result.is_llm,
        "confidence": result.confidence,
        "probability": result.p_llm
    }
```

## Troubleshooting

### Out of Memory
- Reduce `samples_per_source`
- Process in smaller batches
- Use machine with more RAM

### Poor Performance
- Increase training samples
- Check data quality
- Ensure balanced classes
- Verify feature extraction

### Slow Training
- Reduce number of features
- Use fewer tokenizers
- Decrease sample size
- Disable progress bars