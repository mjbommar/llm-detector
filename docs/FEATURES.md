# LLM Detector Features Documentation

This document provides a comprehensive overview of all features extracted by the LLM detector for text classification. Features are organized into three main categories: Statistical, Divergence, and Tokenizer-based features.

## Feature Categories

### 1. Statistical Features
Statistical features capture lexical diversity, text structure, and stylistic patterns directly from raw text.

### 2. Divergence Features
Divergence features measure distributional differences using Jensen-Shannon divergence between text characteristics and baseline distributions.

### 3. Tokenizer Features
Features derived from multiple tokenizers to capture encoding efficiency and tokenization patterns across different NLP models.

---

## Statistical Features (Scale-Invariant)

These features are normalized to be independent of text length, making them suitable for comparing texts of different sizes.

### Lexical Diversity Metrics

#### `stat.type_token_ratio`
- **Calculation**: Approximates type-token ratio using fixed windows (default: 100 tokens) to reduce length bias
- **Formula**: For each window: `unique_tokens / total_tokens`, then averaged across windows
- **Purpose**: Measures vocabulary richness; lower values indicate more repetitive language
- **Range**: [0, 1]

#### `stat.herdan_c`
- **Calculation**: Herdan's C lexical diversity metric
- **Formula**: `log10(V) / log10(N)` where V = vocabulary size, N = total tokens
- **Purpose**: Length-adjusted measure of lexical diversity
- **Range**: [0, 1] (clamped)

#### `stat.mattr`
- **Calculation**: Moving-Average Type-Token Ratio using sliding window (default: 50 tokens)
- **Formula**: Average TTR across all possible windows of fixed size
- **Purpose**: More stable lexical diversity measure for varying text lengths
- **Range**: [0, 1]

### Length and Variation Metrics

#### `stat.mean_word_length`
- **Calculation**: Average number of characters per word token
- **Formula**: `sum(len(word) for word in tokens) / num_tokens`
- **Purpose**: Captures word complexity; technical/formal text often has longer words
- **Range**: [0, ∞) (typically 3-10)

#### `stat.mean_sentence_length`
- **Calculation**: Average number of word tokens per sentence
- **Formula**: `total_words / num_sentences`
- **Purpose**: Measures sentence complexity and writing style
- **Range**: [0, ∞) (typically 5-30)

#### `stat.word_length_cv`
- **Calculation**: Coefficient of variation for word lengths
- **Formula**: `std_dev(word_lengths) / mean(word_lengths)`
- **Purpose**: Measures consistency in word length usage
- **Range**: [0, ∞) (typically 0-2)

#### `stat.sentence_length_cv`
- **Calculation**: Coefficient of variation for sentence lengths (in words)
- **Formula**: `std_dev(sentence_lengths) / mean(sentence_lengths)`
- **Purpose**: Measures sentence length variability
- **Range**: [0, ∞) (typically 0-2)

### Entropy Metrics

#### `stat.char_entropy_norm`
- **Calculation**: Normalized Shannon entropy over character distribution
- **Formula**: `H(chars) / log2(num_unique_chars)`
- **Purpose**: Measures character distribution uniformity
- **Range**: [0, 1]

#### `stat.word_entropy_norm`
- **Calculation**: Normalized Shannon entropy over word token distribution
- **Formula**: `H(words) / log2(num_unique_words)`
- **Purpose**: Measures word usage uniformity
- **Range**: [0, 1]

### Character Class Ratios

#### `stat.punctuation_ratio`
- **Calculation**: Ratio of punctuation characters to total characters
- **Characters**: `.,;:!?-—'"\\/()[]{}<>`
- **Purpose**: Captures punctuation density
- **Range**: [0, 1]

#### `stat.lowercase_char_ratio`
- **Calculation**: Ratio of lowercase letters to total characters
- **Purpose**: Measures text case distribution
- **Range**: [0, 1]

#### `stat.uppercase_char_ratio`
- **Calculation**: Ratio of uppercase letters to total characters
- **Purpose**: Captures emphasis and proper noun usage
- **Range**: [0, 1]

#### `stat.digit_char_ratio`
- **Calculation**: Ratio of digit characters to total characters
- **Purpose**: Measures numerical content density
- **Range**: [0, 1]

#### `stat.whitespace_char_ratio`
- **Calculation**: Ratio of whitespace characters to total characters
- **Purpose**: Captures text spacing patterns
- **Range**: [0, 1]

#### `stat.other_char_ratio`
- **Calculation**: Ratio of non-alphanumeric, non-punctuation characters
- **Purpose**: Captures special symbols and unicode characters
- **Range**: [0, 1]

### Specific Punctuation Ratios

#### `stat.comma_ratio`
- **Calculation**: Frequency of commas per character
- **Purpose**: Measures list usage and sentence complexity
- **Range**: [0, 1]

#### `stat.semicolon_ratio`
- **Calculation**: Frequency of semicolons per character
- **Purpose**: Indicates formal or complex writing style
- **Range**: [0, 1]

#### `stat.question_ratio`
- **Calculation**: Frequency of question marks per character
- **Purpose**: Measures interrogative content
- **Range**: [0, 1]

#### `stat.exclamation_ratio`
- **Calculation**: Frequency of exclamation marks per character
- **Purpose**: Captures emphasis and emotional tone
- **Range**: [0, 1]

#### `stat.period_ratio`
- **Calculation**: Frequency of periods per character
- **Purpose**: Relates to sentence density
- **Range**: [0, 1]

#### `stat.colon_ratio`
- **Calculation**: Frequency of colons per character
- **Purpose**: Indicates explanatory or list-based content
- **Range**: [0, 1]

#### `stat.quote_ratio`
- **Calculation**: Frequency of quote characters (single + double) per character
- **Purpose**: Measures dialogue or citation density
- **Range**: [0, 1]

### Linguistic Pattern Features

#### `stat.function_word_ratio`
- **Calculation**: Fraction of tokens that are common English function words
- **Function words**: Articles (the, a, an), conjunctions (and, or, but), prepositions (in, on, at, to, for, of, with, by, from), auxiliaries (is, was, are, were, been, be, have, has, had, do, does, did, will, would, could, should, may, might, can, shall, must), demonstratives (that, this, these, those), pronouns (i, you, he, she, it, we, they, them, their), interrogatives (what, which, who, when, where, why, how)
- **Purpose**: Function words are style markers; their frequency differs between human and AI text
- **Range**: [0, 1]

#### `stat.capitalized_word_ratio`
- **Calculation**: Ratio of capitalized words excluding sentence-initial positions
- **Purpose**: Measures proper noun usage and emphasis
- **Range**: [0, 1]

### Repetition and Pattern Features

#### `stat.max_char_run_ratio`
- **Calculation**: Maximum repeated-character run length normalized by text length
- **Purpose**: Detects character repetition patterns (e.g., "ooooh", "!!!!")
- **Range**: [0, 1]

#### `stat.repeated_punctuation_ratio`
- **Calculation**: Fraction of punctuation characters appearing in repeated runs (≥2)
- **Purpose**: Captures stylistic emphasis patterns
- **Range**: [0, 1]

#### `stat.whitespace_burstiness`
- **Calculation**: Coefficient of variation of whitespace run lengths (clamped at 10)
- **Purpose**: Measures irregularity in spacing patterns
- **Range**: [0, 10]

---

## Statistical Features (Scale-Dependent)

These features depend on text length and are disabled by default in the registry but can be enabled for specific use cases.

#### `stat.total_words`
- **Calculation**: Total number of word tokens
- **Purpose**: Document length in words

#### `stat.total_sentences`
- **Calculation**: Total number of sentences (split by `.!?`)
- **Purpose**: Document structure size

#### `stat.unique_words`
- **Calculation**: Number of unique word types (case-insensitive)
- **Purpose**: Vocabulary size

#### `stat.hapax_legomena_count`
- **Calculation**: Count of words appearing exactly once
- **Purpose**: Measures vocabulary uniqueness

#### `stat.total_characters`
- **Calculation**: Total character count
- **Purpose**: Raw text length

---

## Divergence Features

All divergence features use Jensen-Shannon Divergence (JSD) to measure distributional differences. JSD is symmetric and bounded [0, 1], where 0 indicates identical distributions and 1 indicates maximally different distributions.

### Jensen-Shannon Divergence Calculation
```
JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
where M = 0.5 * (P + Q)
and KL is Kullback-Leibler divergence
```

#### `div.char_jsd`
- **Calculation**: JSD between character frequency distribution and baseline
- **Baseline**: Either uniform distribution over observed characters or learned from human corpus
- **Purpose**: Captures character usage patterns
- **Range**: [0, 1]

#### `div.punct_jsd`
- **Calculation**: JSD between punctuation frequency distribution and baseline
- **Characters considered**: `.,;:!?-—'\"()[]{}<>/\\`
- **Baseline**: Uniform or learned from human corpus
- **Purpose**: Measures punctuation style divergence
- **Range**: [0, 1]

#### `div.regex_jsd`
- **Calculation**: JSD between regex-derived category distributions and baseline
- **Categories**:
  - `uppercase`: `[A-Z]` matches
  - `lowercase`: `[a-z]` matches
  - `digit`: `\d` matches
  - `whitespace`: `\s` matches
  - `word`: `\w+` matches
  - `sentence_end`: `[.!?]` matches
  - `comma`: `,` matches
  - `quote`: `['\"]` matches
- **Purpose**: Coarse-grained structural pattern divergence
- **Range**: [0, 1]

---

## Tokenizer Features

Features extracted using four different tokenizers to capture cross-model encoding patterns. Each tokenizer produces 12 metrics, plus 2 aggregated cross-tokenizer metrics.

### Tokenizers Used

1. **GPT-2** (`tok.gpt2.*`)
   - OpenAI GPT-2 byte-pair encoding
   - Vocabulary size: 50,257

2. **BERT** (`tok.bert.*`)
   - BERT WordPiece tokenizer (uncased)
   - Vocabulary size: 30,522

3. **RoBERTa** (`tok.roberta.*`)
   - RoBERTa byte-pair tokenizer
   - Vocabulary size: 50,265

4. **GPT-OSS-20B** (`tok.gpt_oss_20b.*`)
   - OpenAI GPT-OSS 20B tokenizer
   - Large-scale model tokenizer

### Per-Tokenizer Metrics

For each tokenizer (prefix: `tok.{tokenizer_name}.`):

#### Efficiency Metrics

##### `tokenization_efficiency`
- **Calculation**: `text_length / num_non_special_tokens`
- **Purpose**: Characters per token; higher = more efficient encoding
- **Interpretation**: LLMs often produce text optimized for their tokenizers

##### `word_match_rate`
- **Calculation**: Fraction of cleaned tokens matching actual words in text
- **Purpose**: Alignment between tokenization and word boundaries
- **Range**: [0, 1]

#### Token Type Ratios

##### `whole_token_ratio`
- **Calculation**: Fraction of tokens that begin a word
- **Purpose**: Measures subword splitting frequency
- **Range**: [0, 1]

##### `single_char_ratio`
- **Calculation**: Fraction of cleaned tokens with length 1
- **Purpose**: Captures granularity of tokenization
- **Range**: [0, 1]

##### `start_word_ratio`
- **Calculation**: Tokens marked as word-initial (by tokenizer conventions)
- **Purpose**: Word boundary detection
- **Range**: [0, 1]

##### `subword_ratio`
- **Calculation**: Tokens marked as subwords (e.g., "##" prefix in BERT)
- **Purpose**: Measures word fragmentation
- **Range**: [0, 1]

#### Token Characteristics

##### `avg_token_length`
- **Calculation**: Average length of cleaned tokens in characters
- **Purpose**: Token granularity measure

##### `token_length_cv`
- **Calculation**: Coefficient of variation of cleaned token lengths
- **Purpose**: Token length consistency

##### `non_alnum_ratio`
- **Calculation**: Fraction of cleaned tokens that are non-alphanumeric
- **Purpose**: Special character handling
- **Range**: [0, 1]

##### `digit_token_ratio`
- **Calculation**: Fraction of cleaned tokens that are pure digits
- **Purpose**: Numerical content tokenization
- **Range**: [0, 1]

##### `unique_token_ratio`
- **Calculation**: `unique_cleaned_tokens / total_cleaned_tokens`
- **Purpose**: Token diversity measure
- **Range**: [0, 1]

##### `short_token_ratio_le2`
- **Calculation**: Fraction of cleaned tokens with length ≤ 2
- **Purpose**: Captures fine-grained tokenization
- **Range**: [0, 1]

### Aggregated Cross-Tokenizer Metrics

#### `tok.efficiency_variance`
- **Calculation**: Variance of `tokenization_efficiency` across all tokenizers
- **Purpose**: Measures encoding consistency across models
- **Interpretation**: Human text often shows higher variance

#### `tok.efficiency_std`
- **Calculation**: Standard deviation of `tokenization_efficiency` across tokenizers
- **Purpose**: Spread of encoding efficiency
- **Interpretation**: AI text may be optimized for specific tokenizers

---

## Feature Selection and Usage

### Default Configuration

- **Scale-invariant features only**: By default, only scale-invariant features are used
- **Enabled tokenizers**: All four tokenizers are enabled by default
- **Baseline distributions**: Learned from human corpus during training

### Feature Vector Composition

The final feature vector for a text sample consists of:
1. 28 scale-invariant statistical features
2. 3 divergence features (with learned baselines)
3. 48 tokenizer features (12 per tokenizer × 4 tokenizers)
4. 2 aggregated tokenizer metrics

**Total: 81 features** in the default configuration

### Customization Options

Features can be customized by:
1. Enabling scale-dependent features (`scale_invariant_only=False`)
2. Selecting specific tokenizers
3. Modifying baseline distributions for divergence features
4. Enabling/disabling specific feature categories

---

## Implementation Details

### Text Preprocessing

1. **Word tokenization**: Unicode word boundaries (`\b\w+\b`)
2. **Sentence splitting**: Split on `.!?` characters
3. **Case handling**: Lowercase conversion for vocabulary analysis
4. **Special token handling**: Removal of tokenizer-specific markers

### Numerical Stability

- Probabilities clamped to [1e-6, 1-1e-6] to avoid log(0)
- Zero-division protection in all ratio calculations
- Coefficient of variation capped at reasonable bounds

### Performance Optimizations

- LRU caching for expensive character class calculations
- Single-pass computation where possible
- Lazy tokenizer loading (loaded on first use)
- Result caching for repeated text analysis

---

## Feature Interpretation Guidelines

### Human Text Indicators
- Higher lexical diversity (type_token_ratio, mattr)
- More variable sentence lengths (higher sentence_length_cv)
- Natural function word distribution
- Higher cross-tokenizer efficiency variance

### LLM Text Indicators
- Lower lexical diversity (more repetitive)
- More consistent sentence lengths
- Optimized for specific tokenizers (low efficiency_variance)
- Different punctuation patterns (divergence features)
- More uniform character distributions

### Domain-Specific Considerations
- Technical texts: longer words, more digits
- Creative writing: more punctuation variety, quotes
- Academic writing: longer sentences, semicolons
- Social media: more exclamations, questions

---

## References

1. **Herdan's C**: Herdan, G. (1960). Type-token mathematics. The Hague: Mouton.
2. **MATTR**: Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type-token ratio (MATTR).
3. **Jensen-Shannon Divergence**: Lin, J. (1991). Divergence measures based on the Shannon entropy.
4. **Tokenization**: Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer.