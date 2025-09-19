# New Datasets for LLM Detector (2024-2025)

Based on comprehensive research of Hugging Face datasets, here are recommended datasets to add for improved LLM detection across diverse contexts and recent models.

## Priority 1: Recent LLM-Generated Datasets (2024-2025)

### 1. GPT-4/GPT-4o Datasets
- **vicgalle/alpaca-gpt4** - 52K GPT-4 instruction-following examples
- **QuietImpostor/Claude-3-Opus-Claude-3.5-Sonnnet-9k** - Claude 3/3.5 synthetic data
- **leafspark/DetailedReflection-Claude-v3_5-Sonnet** - Reflection-based Claude 3.5 outputs
- **nothingiisreal/Claude-3-Opus-Instruct-15K** - 15K Claude-3 Opus instructions

**Integration rationale**: These represent outputs from the most advanced 2024-2025 models (GPT-4, Claude 3/3.5)

### 2. Llama 3.x Synthetic Data
- **Llama 3.1/3.2/3.3 outputs** - Need to find/create dataset from Meta's 2024 models
- Models explicitly allow synthetic data generation in their licenses

**Integration rationale**: Llama 3.x models (released April 2024 - December 2024) are widely deployed

### 3. Gemini Pro/Ultra Datasets
- **deepkaria/extreme-toxic-human-dataset-synthetic** - Gemini 2.5 Pro generated (June 2025)
- Limited public Gemini datasets available

**Integration rationale**: Google's latest models need representation

## Priority 2: Diverse Human Text Sources

### 1. Social Media (Short-form, Informal)
- **bit0/reddit_dataset_12** - 388M Reddit posts/comments (Nov 2024 - Jan 2025)
- **Exorde/exorde-social-media-december-2024-week1** - 65M multi-platform entries
- **cardiffnlp/tweet_eval** - Twitter classification datasets
- **SocialGrep/the-reddit-dataset-dataset** - Large Reddit corpus

**Integration rationale**: Informal, short-form human text with modern slang/expressions

### 2. Legal Documents (Long-form, Formal)
- **joelniklaus/Multi_Legal_Pile** - 689GB multilingual legal corpus
- **pile-of-law/pile-of-law** - 256GB US legal documents
  - Contracts, court opinions, bills, IRS memos
- **nguha/legalbench** - Legal reasoning benchmark

**Integration rationale**: Highly formal, structured human writing

### 3. Academic Papers (Long-form, Technical)
- **armanc/scientific_papers** - ArXiv and PubMed papers
- **CShorten/ML-ArXiv-Papers** - Machine learning papers

**Integration rationale**: Technical, citation-heavy human writing

## Priority 3: Content Length Diversity

### Short-form Content (<280 characters)
- **Twitter datasets** (tweet_eval, tweetner7)
- **Chat messages** (lmsys-chat-1m single turns)
- **Reddit comments** (typically brief)

### Medium-form Content (280-2000 characters)
- **Reddit posts**
- **Chat conversations** (ultrachat_200k)
- **News snippets**

### Long-form Content (>2000 characters)
- **Legal documents** (contracts, opinions)
- **Academic papers** (full articles)
- **Wikipedia articles**
- **Books** (Project Gutenberg - already included)

## Implementation Plan

### Phase 1: Add Recent LLM Sources
```python
# llm_detector/training/sources/llm_2024.py
class GPT4AlpacaSource(BaseDataSource):
    """GPT-4 generated instruction-following data."""
    HF_DATASET = "vicgalle/alpaca-gpt4"

class Claude3OpusSource(BaseDataSource):
    """Claude 3 Opus synthetic instructions."""
    HF_DATASET = "nothingiisreal/Claude-3-Opus-Instruct-15K"

class Claude35SonnetSource(BaseDataSource):
    """Claude 3.5 Sonnet reflection data."""
    HF_DATASET = "leafspark/DetailedReflection-Claude-v3_5-Sonnet"
```

### Phase 2: Add Human Social Media Sources
```python
# llm_detector/training/sources/social_media.py
class RedditRecentSource(BaseDataSource):
    """Recent Reddit posts/comments (2024-2025)."""
    HF_DATASET = "bit0/reddit_dataset_12"

    def stream_samples(self, limit=None):
        # Filter for posts only (not comments)
        # Sample across different subreddits
        pass

class TwitterSource(BaseDataSource):
    """Twitter/X posts for short-form text."""
    HF_DATASET = "cardiffnlp/tweet_eval"
```

### Phase 3: Add Formal/Technical Sources
```python
# llm_detector/training/sources/formal.py
class LegalDocumentsSource(BaseDataSource):
    """US legal documents - contracts, opinions, bills."""
    HF_DATASET = "pile-of-law/pile-of-law"

    def stream_samples(self, limit=None):
        # Sample different document types
        # Filter for English only
        pass

class ScientificPapersSource(BaseDataSource):
    """ArXiv and PubMed academic papers."""
    HF_DATASET = "armanc/scientific_papers"
```

### Phase 4: Length-Based Sampling
```python
# llm_detector/training/sources/utils.py
def categorize_by_length(text: str) -> str:
    """Categorize text by character length."""
    length = len(text)
    if length < 280:
        return "short"
    elif length < 2000:
        return "medium"
    else:
        return "long"

class LengthBalancedSampler:
    """Ensure balanced sampling across text lengths."""
    def __init__(self, sources, length_ratios=None):
        self.sources = sources
        self.length_ratios = length_ratios or {
            "short": 0.33,
            "medium": 0.33,
            "long": 0.34
        }
```

## Configuration Updates

### Registry Updates
```python
# llm_detector/training/sources/registry.py
DEFAULT_REGISTRY_2025 = SourceRegistry()

# Recent LLM sources
DEFAULT_REGISTRY_2025.register("gpt4_alpaca", GPT4AlpacaSource, enabled=True)
DEFAULT_REGISTRY_2025.register("claude3_opus", Claude3OpusSource, enabled=True)
DEFAULT_REGISTRY_2025.register("claude35_sonnet", Claude35SonnetSource, enabled=True)

# Human sources - informal
DEFAULT_REGISTRY_2025.register("reddit_recent", RedditRecentSource, enabled=True)
DEFAULT_REGISTRY_2025.register("twitter", TwitterSource, enabled=True)

# Human sources - formal
DEFAULT_REGISTRY_2025.register("legal_docs", LegalDocumentsSource, enabled=True)
DEFAULT_REGISTRY_2025.register("scientific_papers", ScientificPapersSource, enabled=True)

# Keep existing sources but mark some as legacy
DEFAULT_REGISTRY_2025.register("ultrachat", UltraChatSource, enabled=False)  # Older GPT-3.5
DEFAULT_REGISTRY_2025.register("cosmopedia", CosmopediaSource, enabled=True)  # Still relevant
```

### Training Configuration
```yaml
# config/training_2025.yaml
data_sources:
  llm:
    - gpt4_alpaca: 5000
    - claude3_opus: 5000
    - claude35_sonnet: 5000
    - cosmopedia: 5000  # Keep for Mixtral representation
    - lmsys_chat: 5000  # Keep for chat variety

  human:
    - reddit_recent: 5000
    - twitter: 5000
    - legal_docs: 5000
    - scientific_papers: 5000
    - wikipedia: 5000  # Keep for encyclopedic
    - gutenberg: 5000  # Keep for literature

balance_by_length: true
length_ratios:
  short: 0.3
  medium: 0.4
  long: 0.3
```

## Testing Strategy

### 1. Domain-Specific Evaluation
```python
def evaluate_by_domain(model, test_sets):
    results = {}
    for domain in ["social_media", "legal", "academic", "chat", "instruction"]:
        test_data = test_sets[domain]
        accuracy = model.evaluate(test_data)
        results[domain] = accuracy
    return results
```

### 2. Length-Based Evaluation
```python
def evaluate_by_length(model, test_sets):
    results = {}
    for length_category in ["short", "medium", "long"]:
        test_data = test_sets[length_category]
        accuracy = model.evaluate(test_data)
        results[length_category] = accuracy
    return results
```

### 3. Model-Specific Evaluation
```python
def evaluate_by_llm_model(model, test_sets):
    results = {}
    for llm in ["gpt4", "claude3", "llama3", "gemini", "mixtral"]:
        if llm in test_sets:
            accuracy = model.evaluate(test_sets[llm])
            results[llm] = accuracy
    return results
```

## Expected Improvements

### Coverage
- **Model diversity**: GPT-4, Claude 3.x, Llama 3.x, Gemini (vs. current GPT-3.5 heavy)
- **Domain diversity**: Legal, academic, social media (vs. current general text)
- **Length diversity**: Explicit short/medium/long balancing

### Performance Targets
- Overall accuracy: 90-95% (from current 85%)
- Short text (<280 chars): 85%+ accuracy
- Legal/formal text: 95%+ accuracy
- Recent model detection: 90%+ for GPT-4/Claude 3

### Robustness
- Better generalization to unseen domains
- Improved handling of mixed human/AI content
- More reliable on very short or very long texts

## Implementation Timeline

1. **Week 1**: Implement new LLM source classes
2. **Week 2**: Implement human source classes
3. **Week 3**: Add length-based sampling
4. **Week 4**: Train and evaluate new models
5. **Week 5**: Fine-tune and optimize
6. **Week 6**: Release v0.2.0 with new datasets

## Notes

- Some datasets require authentication or have size limitations
- Consider data licensing for commercial use
- May need to implement streaming/chunking for very large datasets
- Consider adding data filtering for quality control
- Monitor for dataset updates and new releases