# NLP Feature Integration Plan for LSTM Stock Predictor

## Current Architecture Overview

### Data Flow
1. **Data Loading** (`util.py`):
   - `get_data()` fetches stock prices from yfinance
   - Creates features: (num_samples, 31, 3)
     - 31 time steps (lookback window)
     - 3 features: intraday returns (ir), closing price returns (cpr), opening price returns (opr)
   - Returns: X_train, X_val, X_test, Y_train, Y_val, Y_test, dates, revenues, returns, S&P500

2. **Model Architecture** (`model.py`):
   - `LSTMModel` / `CNNLSTMModel`: Expects `input_dim=(31, 3)`
   - Processes through LSTM → Linear layer → single output

3. **Training Pipeline** (`trainer.py`):
   - `Trainer` class loads data, creates DataLoaders, trains model
   - Uses `IndexedDataset` that returns `(X, Y, idx)`

### Key Constraints
- Data is cached in `.npz` files with specific naming conventions
- Model expects fixed input dimensions
- Training pipeline is well-established and working

---

## Integration Strategy Options

### Option 1: Feature Concatenation (Recommended for Quick Start)
**Approach**: Add NLP features as additional feature dimensions

**Pros**:
- Minimal code changes
- Easy to implement incrementally
- Backward compatible (can disable NLP features)
- Works with existing caching system

**Cons**:
- NLP features treated same as price features (may not be optimal)
- Model needs to learn feature importance

**Implementation**:
- Modify `get_feature_input_classification()` to include NLP features
- Change input_dim from `(31, 3)` to `(31, 3 + n_nlp_features)`
- Update model initialization

### Option 2: Multi-Modal Architecture (Better Performance)
**Approach**: Separate input streams for price and NLP features

**Pros**:
- Can use different architectures for different modalities
- Better feature separation
- More flexible

**Cons**:
- More significant code changes
- More complex model architecture
- Harder to maintain backward compatibility

### Option 3: Time-Aligned NLP Features (Most Realistic)
**Approach**: One NLP feature vector per time step (aligned with price data)

**Pros**:
- Natural alignment with time series
- Can capture temporal sentiment changes
- Matches existing data structure

**Cons**:
- Requires NLP data for each time step
- More complex data preparation

---

## Recommended Implementation Plan (Option 1 + 3 Hybrid)

### Phase 1: Create NLP Feature Extraction Module
**Goal**: Build a standalone module that can extract NLP features for given dates/stocks

**Files to Create**:
- `nlp_features.py`: Main NLP feature extraction
  - Function: `get_nlp_features(stocks, dates, nlp_source='news')`
  - Returns: Dictionary mapping (stock, date) → NLP feature vector
  - Should handle missing data gracefully (return zeros or NaN)

**Key Functions**:
```python
def extract_nlp_features(stocks, dates, source='news'):
    """
    Extract NLP features for given stocks and dates.
    
    Args:
        stocks: List of stock symbols
        dates: List of datetime objects
        source: Source of NLP data ('news', 'sentiment', etc.)
    
    Returns:
        Dict mapping (stock, date) -> np.array of NLP features
    """
    pass

def align_nlp_with_price_data(nlp_features, price_dates, stocks):
    """
    Align NLP features with price data dates.
    Handles missing dates, forward-fills if needed.
    """
    pass
```

### Phase 2: Modify Data Pipeline (Backward Compatible)
**Goal**: Add NLP features to existing data pipeline without breaking current functionality

**Changes to `util.py`**:

1. **Add optional parameter to `get_data()`**:
```python
def get_data(stocks, args, ..., use_nlp=False, nlp_features=None):
    """
    use_nlp: Boolean flag to enable NLP features
    nlp_features: Optional pre-computed NLP features dict
    """
```

2. **Modify `get_feature_input_classification()`**:
```python
def get_feature_input_classification(op, cp, lookback, study_period, 
                                     num_stocks, date_index, 
                                     nlp_features=None):
    """
    nlp_features: Optional dict mapping (stock_idx, date) -> nlp_feature_vector
    """
    # Existing feature computation...
    # If nlp_features provided, concatenate to window
    if nlp_features is not None:
        # Extract NLP features for this window
        nlp_vec = extract_nlp_for_window(nlp_features, stock_idx, period)
        window = np.concatenate([window, nlp_vec], axis=1)  # Add NLP features
```

3. **Update cache naming**:
   - Include NLP flag in cache ID: `data_{id}_nlp_{nlp_type}.npz`
   - Or add separate NLP cache file: `data_{id}_nlp_features.npz`

### Phase 3: Update Model Architecture (Flexible)
**Goal**: Make model accept variable input dimensions

**Changes to `model.py`**:

1. **Make input_dim flexible**:
```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim=(31, 3), hidden_size=25, ..., use_nlp=False, nlp_dim=0):
        super().__init__()
        self.input_dim = input_dim
        self.use_nlp = use_nlp
        feature_dim = input_dim[1] + (nlp_dim if use_nlp else 0)
        
        self.input_norm = nn.LayerNorm(feature_dim)
        self.lstm = nn.LSTM(input_size=feature_dim, ...)
        # ... rest of model
```

2. **Update CNNLSTMModel similarly**

### Phase 4: Update Trainer (Optional Flag)
**Goal**: Add NLP support to training pipeline

**Changes to `trainer.py`**:

1. **Add NLP parameters to Trainer.__init__**:
```python
def __init__(self, ..., use_nlp=False, nlp_source='news'):
    self.use_nlp = use_nlp
    self.nlp_source = nlp_source
```

2. **Load NLP features if enabled**:
```python
# In __init__, after loading price data:
if self.use_nlp:
    from nlp_features import extract_nlp_features, align_nlp_with_price_data
    nlp_features = extract_nlp_features(stocks, all_dates, source=self.nlp_source)
    # Align with price data
    # Pass to get_data or merge after loading
```

3. **Update model initialization**:
```python
if self.use_nlp:
    nlp_dim = get_nlp_feature_dim(self.nlp_source)  # e.g., 5 for sentiment scores
    self.Model = CNNLSTMModel(input_dim=(31, 3), use_nlp=True, nlp_dim=nlp_dim)
else:
    self.Model = CNNLSTMModel(input_dim=(31, 3))
```

### Phase 5: Update Main Script
**Goal**: Add easy way to enable/disable NLP

**Changes to `main.py`**:
```python
# Add flag at top
USE_NLP = False  # Set to True to enable NLP features
NLP_SOURCE = 'news'  # or 'sentiment', etc.

# Pass to Trainer
train_obj = trainer.Trainer(
    stocks=stocks, 
    time_args=[start, end], 
    ...,
    use_nlp=USE_NLP,
    nlp_source=NLP_SOURCE
)
```

---

## Implementation Checklist

### Step 1: Create NLP Feature Module
- [ ] Create `nlp_features.py` with basic structure
- [ ] Implement `extract_nlp_features()` function
- [ ] Implement `align_nlp_with_price_data()` function
- [ ] Add error handling for missing data
- [ ] Test with sample dates/stocks

### Step 2: Modify Data Pipeline (Backward Compatible)
- [ ] Add `use_nlp` parameter to `get_data()` in `util.py`
- [ ] Modify `get_feature_input_classification()` to accept NLP features
- [ ] Update cache naming to include NLP flag
- [ ] Test that existing code still works (use_nlp=False)
- [ ] Test with NLP enabled (use_nlp=True)

### Step 3: Update Model Architecture
- [ ] Make `LSTMModel` accept variable feature dimensions
- [ ] Make `CNNLSTMModel` accept variable feature dimensions
- [ ] Test model with and without NLP features
- [ ] Verify model can load old checkpoints (backward compatibility)

### Step 4: Update Trainer
- [ ] Add NLP parameters to `Trainer.__init__()`
- [ ] Add NLP feature loading logic
- [ ] Update model initialization based on NLP flag
- [ ] Test training with and without NLP

### Step 5: Integration Testing
- [ ] Test end-to-end with NLP disabled (should work as before)
- [ ] Test end-to-end with NLP enabled
- [ ] Verify cache works correctly
- [ ] Test with missing NLP data (graceful degradation)

---

## Backward Compatibility Strategy

1. **Default Behavior**: `use_nlp=False` by default
   - Existing code continues to work without changes
   - Model defaults to `input_dim=(31, 3)`

2. **Cache Separation**: 
   - Old cache: `data_{id}_classification_train.npz` (no NLP)
   - New cache: `data_{id}_classification_nlp_train.npz` (with NLP)
   - Both can coexist

3. **Model Checkpoints**:
   - Old checkpoints: `input_dim=(31, 3)`
   - New checkpoints: `input_dim=(31, 3+nlp_dim)`
   - Model can detect and handle both

4. **Graceful Degradation**:
   - If NLP data missing for some dates: use zeros or forward-fill
   - If NLP extraction fails: fall back to price-only features

---

## File Structure

```
logan-version/
├── model.py              # Update: flexible input_dim
├── trainer.py            # Update: add NLP support
├── util.py               # Update: add NLP to data pipeline
├── main.py               # Update: add NLP flag
├── nlp_features.py       # NEW: NLP feature extraction
└── NLP_INTEGRATION_PLAN.md  # This file
```

---

## Next Steps

1. **Start with Phase 1**: Create `nlp_features.py` as a standalone module
   - This can be developed and tested independently
   - Doesn't require changes to existing code

2. **Test NLP Feature Extraction**:
   - Verify you can extract NLP features for your date range
   - Check data quality and coverage

3. **Incremental Integration**:
   - Add NLP features to data pipeline (Phase 2)
   - Test with small dataset first
   - Gradually enable for full training

4. **Model Updates** (Phase 3):
   - Only after NLP features are working in data pipeline
   - Test model can handle variable dimensions

---

## Questions to Consider

1. **NLP Data Source**: Where will NLP features come from?
   - News articles? (you have NYT data in `other_models/`)
   - Sentiment scores?
   - Social media?
   - Pre-computed embeddings?

2. **NLP Feature Format**:
   - Single scalar per time step? (e.g., sentiment score)
   - Vector per time step? (e.g., [sentiment, topic1, topic2, ...])
   - How many dimensions?

3. **Temporal Alignment**:
   - Should NLP features be for the same day as price data?
   - Or previous day (news from yesterday affects today's price)?
   - How to handle weekends/holidays?

4. **Missing Data Handling**:
   - What if no news for a particular stock/date?
   - Use zeros? Forward-fill? Skip that sample?

---

## Example Code Skeleton

### `nlp_features.py` (New File)
```python
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def extract_nlp_features(stocks: List[str], 
                         dates: List[datetime],
                         source: str = 'news') -> Dict[Tuple[str, datetime], np.ndarray]:
    """
    Extract NLP features for given stocks and dates.
    
    Returns:
        Dict mapping (stock, date) -> np.array of NLP features
    """
    nlp_features = {}
    
    # TODO: Implement actual NLP extraction
    # For now, return empty dict or placeholder
    
    for stock in stocks:
        for date in dates:
            # Extract features for this stock/date
            features = _extract_for_stock_date(stock, date, source)
            nlp_features[(stock, date)] = features
    
    return nlp_features

def _extract_for_stock_date(stock: str, date: datetime, source: str) -> np.ndarray:
    """Extract NLP features for a single stock/date pair."""
    # TODO: Implement based on your NLP source
    # Return np.array of shape (nlp_dim,)
    return np.zeros(5)  # Placeholder: 5-dimensional NLP features

def get_nlp_feature_dim(source: str = 'news') -> int:
    """Return the dimensionality of NLP features for given source."""
    # TODO: Return actual dimension
    return 5
```

### Modified `util.py` snippet
```python
def get_feature_input_classification(op, cp, lookback, study_period, 
                                     num_stocks, date_index,
                                     nlp_features: Optional[Dict] = None):
    # ... existing code ...
    
    # In the window building loop:
    for end_t in range(lookback + 2, T):
        for n in range(num_stocks):
            # ... existing window building ...
            
            # Add NLP features if available
            if nlp_features is not None:
                nlp_vec = _extract_nlp_for_window(nlp_features, n, period, date_index)
                if nlp_vec is not None:
                    window = np.concatenate([window, nlp_vec], axis=1)
                else:
                    # Missing NLP data: use zeros
                    nlp_dim = _get_nlp_dim(nlp_features)
                    nlp_zeros = np.zeros((len(period), nlp_dim))
                    window = np.concatenate([window, nlp_zeros], axis=1)
            
            # ... rest of code ...
```

---

## Testing Strategy

1. **Unit Tests**:
   - Test NLP feature extraction with sample data
   - Test alignment with price data
   - Test missing data handling

2. **Integration Tests**:
   - Test data pipeline with NLP disabled (should match current behavior)
   - Test data pipeline with NLP enabled
   - Test model with variable input dimensions

3. **End-to-End Tests**:
   - Train model without NLP (baseline)
   - Train model with NLP (compare performance)
   - Verify backward compatibility

---

This plan allows you to:
- ✅ Add NLP features incrementally
- ✅ Maintain backward compatibility
- ✅ Test each component independently
- ✅ Roll back easily if needed
- ✅ Enable/disable NLP with a simple flag

Start with Phase 1 (NLP feature extraction module) and work your way through the phases!

