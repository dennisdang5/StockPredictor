# Phase 1 NLP Features - Implementation Complete

## Overview

Phase 1 of NLP integration is now implemented! This module extracts daily NLP features from NYT articles and prepares them for LSTM input.

## What Was Created

### `nlp_features.py`
Main module that:
- ✅ Reads NYT article CSVs
- ✅ Runs FinBERT sentiment analysis on headlines
- ✅ Aggregates variable-length articles into fixed-size daily features
- ✅ Handles "no news" vs "neutral sentiment" days properly
- ✅ Aligns features with trading days
- ✅ Produces clean 10-dimensional feature vectors

### `test_nlp_features.py`
Test script to verify the implementation works.

## Installation

Make sure you have the required dependencies:

```bash
pip install transformers torch pandas numpy tqdm
```

## Quick Start

### 1. Test the Implementation

```bash
cd logan-version
python test_nlp_features.py
```

This will:
- Load one NYT CSV file
- Extract NLP features
- Test alignment with trading days
- Show example feature vectors

### 2. Extract Features for All Years

```python
from nlp_features import extract_daily_nlp_features, align_nlp_with_trading_days
from pathlib import Path
import glob

# Find all NYT CSV files
nyt_dir = Path("../other_models/huggingface_nyt_articles")
csv_files = sorted(glob.glob(str(nyt_dir / "new_york_times_stories_*.csv")))

# Extract daily NLP features
nlp_df = extract_daily_nlp_features(
    csv_paths=csv_files,
    filter_finance=True,  # Only finance-related articles
    batch_size=32,
    progress=True
)

# Save for later use
nlp_df.to_csv("daily_nlp_features.csv", index=False)
print(f"Saved features for {len(nlp_df)} days")
```

### 3. Align with Trading Days

```python
from nlp_features import align_nlp_with_trading_days
import pandas as pd

# Load your trading days (from price data)
# Example: trading_days = pd.bdate_range(start='1990-01-01', end='2015-12-31')

# Align NLP features
aligned = align_nlp_with_trading_days(
    nlp_df,
    trading_days=trading_days,
    fill_method='zero'  # or 'sentinel'
)

# Now aligned has one row per trading day
# Missing days are filled with: has_news=0, n_articles=0, sentiment=0
```

## Feature Vector Format

Each day produces a **10-dimensional feature vector**:

```python
[
    has_news,      # 1 if articles exist, 0 otherwise
    n_articles,    # Number of articles that day
    mean_pos,      # Mean positive sentiment probability
    mean_neg,      # Mean negative sentiment probability
    mean_neu,      # Mean neutral sentiment probability
    max_pos,       # Maximum positive sentiment
    max_neg,       # Maximum negative sentiment
    std_pos,       # Std dev of positive sentiment
    std_neg,       # Std dev of negative sentiment
    net_sent       # Net sentiment (mean_pos - mean_neg)
]
```

### Key Design Decisions

1. **"No News" vs "Neutral Sentiment"**:
   - Days with no articles: `has_news=0`, `n_articles=0`, all sentiment=0
   - Days with neutral articles: `has_news=1`, `n_articles>0`, sentiment≈0
   - The model can learn the difference!

2. **Variable Articles → Fixed Features**:
   - Multiple articles per day are aggregated into statistics
   - One fixed-size vector per day, perfect for LSTM

3. **Missing Days**:
   - Filled with zeros (or sentinel values)
   - `has_news=0` flag distinguishes from actual neutral sentiment

## Integration with Existing Pipeline

This Phase 1 implementation is **standalone** and doesn't modify your existing code. To integrate:

1. **Extract NLP features** (this module)
2. **Modify `util.py`** to accept NLP features (Phase 2)
3. **Update `model.py`** to accept variable input dimensions (Phase 3)
4. **Update `trainer.py`** to load and use NLP features (Phase 4)

See `NLP_INTEGRATION_PLAN.md` for the full integration plan.

## Configuration Options

### Sentiment Model
- **Default**: `ProsusAI/finbert` (FinBERT - better for financial text)
- **Fallback**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (general sentiment)

You can specify a custom model:
```python
nlp_df = extract_daily_nlp_features(
    csv_paths=csv_files,
    sentiment_model="your-model-name"
)
```

### Filtering
- `filter_finance=True`: Only keeps finance/business articles
- Filters by section (`Business Day`, `Financial Desk`, etc.)
- Filters by keywords in headlines

### Batch Processing
- `batch_size=32`: Adjust based on GPU memory
- Larger batches = faster but more memory

## Output Format

The `extract_daily_nlp_features()` function returns a DataFrame:

```
   date       n_articles  has_news  mean_pos  mean_neg  mean_neu  ...
1990-01-02          5         1      0.45      0.30      0.25   ...
1990-01-03          3         1      0.50      0.25      0.25   ...
1990-01-04          0         0      0.00      0.00      0.00   ...
```

After alignment with trading days, every trading day has a row (missing days filled).

## Next Steps

1. ✅ **Phase 1 Complete**: NLP feature extraction working
2. **Phase 2**: Integrate into `util.py` data pipeline
3. **Phase 3**: Update model architecture
4. **Phase 4**: Update trainer to use NLP features

## Troubleshooting

### "transformers library not available"
```bash
pip install transformers torch
```

### "No articles remaining after filtering"
- Try `filter_finance=False` to include all articles
- Check that your CSV files have the expected columns

### Out of Memory
- Reduce `batch_size` (e.g., `batch_size=16` or `batch_size=8`)
- Process fewer CSV files at once

### Slow Processing
- FinBERT is slower but more accurate for financial text
- Consider using the fallback model for faster processing
- Process files in smaller batches

## Example: Full Workflow

```python
from nlp_features import (
    extract_daily_nlp_features,
    align_nlp_with_trading_days,
    get_nlp_feature_vector
)
from pathlib import Path
import glob
import pandas as pd

# 1. Extract NLP features
nyt_dir = Path("../other_models/huggingface_nyt_articles")
csv_files = sorted(glob.glob(str(nyt_dir / "*.csv")))

nlp_df = extract_daily_nlp_features(
    csv_paths=csv_files,
    filter_finance=True,
    progress=True
)

# 2. Get trading days from your price data
# (This would come from your actual price data loading)
trading_days = pd.bdate_range(start='1990-01-01', end='2015-12-31')

# 3. Align NLP features with trading days
aligned_nlp = align_nlp_with_trading_days(
    nlp_df,
    trading_days=trading_days,
    fill_method='zero'
)

# 4. Extract feature vectors for specific dates
for idx, row in aligned_nlp.iterrows():
    feature_vec = get_nlp_feature_vector(row)
    # feature_vec is ready for LSTM input!
    # Shape: (10,) - can be concatenated with price features
```

## Questions?

See `NLP_INTEGRATION_PLAN.md` for the full integration strategy and architecture decisions.

