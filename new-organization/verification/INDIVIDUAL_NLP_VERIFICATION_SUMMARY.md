# Individual NLP Feature Extraction Verification Summary

## Overview
This document summarizes the verification of the individual NLP feature extraction method (`nlp_method="individual"`) in the StockPredictor codebase.

## Key Findings

### 1. yfinance News API Structure
- **Current API Format**: yfinance news API now returns articles with a nested structure
  - Top-level keys: `['id', 'content']`
  - Article data is nested in `article['content']` dictionary
  - Fields include: `title`, `pubDate`, `displayTime`, `summary`, etc.

### 2. Code Fix Applied
**File**: `nlp_features.py`  
**Function**: `extract_daily_nlp_features_yfinance()`

**Issue**: The original code looked for `providerPublishTime` and `pubDate` directly in the article object, but these fields are now nested in `article['content']`.

**Fix**: Updated the code to:
- Check if `content` exists and is a dictionary (new format)
- Extract data from `content` if available, otherwise use article directly (backward compatibility)
- Try multiple date fields: `providerPublishTime`, `pubDate`, `displayTime`
- Try multiple headline fields: `title`, `headline`

### 3. News Availability
- **Recent News**: ✅ Available (last few months)
- **Historical News**: ❌ Limited availability
  - yfinance news API typically returns only recent news (last 1-3 months)
  - Historical data (e.g., 1990-2015) is generally not available via yfinance
  - **Recommendation**: Use `aggregated` method with NYT articles for historical data

### 4. Feature Extraction Flow

#### Step 1: News Download (`extract_daily_nlp_features_yfinance`)
```
For each stock:
  1. Fetch news from yfinance: ticker.news
  2. Parse article structure (handle nested 'content')
  3. Extract date and headline from each article
  4. Filter by date range (if specified)
  5. Run FinBERT sentiment analysis on headlines
  6. Aggregate to daily features per stock
```

#### Step 2: Alignment (`align_nlp_with_trading_days`)
```
For each stock's NLP DataFrame:
  1. Align calendar dates with trading days
  2. Forward-fill news from weekends/holidays to next trading day
  3. Fill missing trading days with zeros
```

#### Step 3: Dictionary Creation (`get_data` in util.py)
```
Create dictionary mapping:
  (stock_ticker, date) -> NLP feature row
  
Example:
  ('AAPL', date(2025, 11, 22)) -> {
    'date': date(2025, 11, 22),
    'has_news': 1,
    'p_pos': 0.7,
    'p_neg': 0.2,
    'p_neu': 0.1,
    ...
  }
```

#### Step 4: Feature Window Construction (`get_feature_input_classification`)
```
For each sample (stock n, end time t):
  1. Detect method: Check if keys are tuples -> individual method
  2. Get stock ticker: successfully_downloaded_stocks[n]
  3. For each timestep in period:
     - Get date: date_index[t]
     - Lookup: (stock_ticker, date) in nlp_features_dict
     - Extract 4 features: [has_news, p_pos, p_neg, p_neu]
     - If missing: use zeros
  4. Concatenate: price features (3) + NLP features (4) = 7 total
```

### 5. Stock Index Alignment
**Verified**: ✅ Stock indices are correctly aligned
- `op.iloc[n, t]` corresponds to `successfully_downloaded_stocks[n]`
- This ensures correct NLP feature lookup for individual method
- Alignment is maintained because `handle_yfinance_errors` preserves stock order

### 6. Feature Dimensions

| Method | NLP Features | Total Features |
|--------|-------------|----------------|
| **Individual** | 4: `[has_news, p_pos, p_neg, p_neu]` | 7 (3 price + 4 NLP) |
| **Aggregated** | 10: `[has_news, n_articles, mean_pos, mean_neg, mean_neu, max_pos, max_neg, std_pos, std_neg, net_sent]` | 13 (3 price + 10 NLP) |

### 7. Test Results

#### ✅ Working Correctly
- News download with updated API structure
- Date parsing from nested content structure
- Sentiment analysis on headlines
- Daily aggregation per stock
- Alignment with trading days
- Feature vector extraction (4 features for individual method)
- Feature window construction logic
- Stock index alignment

#### ⚠️ Limitations
- Historical news not available (only recent news)
- Rate limiting needed for API calls
- Some stocks may have no news data

## Recommendations

1. **For Historical Data (1990-2015)**:
   - Use `nlp_method="aggregated"` with NYT articles
   - Individual method will not work due to lack of historical news

2. **For Recent Data (last few months)**:
   - Use `nlp_method="individual"` for stock-specific news
   - Provides per-stock sentiment features

3. **Error Handling**:
   - Code gracefully handles missing news data
   - Returns empty DataFrame if no news available
   - Continues processing other stocks

4. **Performance**:
   - Rate limiting (0.1s sleep) between stock requests
   - Batch processing for sentiment analysis
   - Progress bars for long operations

## Code Locations

- **NLP Extraction**: `nlp_features.py::extract_daily_nlp_features_yfinance()`
- **Feature Integration**: `util.py::get_feature_input_classification()`
- **Feature Vector**: `nlp_features.py::get_nlp_feature_vector()`
- **Alignment**: `nlp_features.py::align_nlp_with_trading_days()`

## Verification Scripts

- `verification/verify_individual_nlp.py` - Full feature extraction flow
- `verification/verify_news_download.py` - News API verification
- `verification/check_stock_alignment.py` - Stock index alignment check

