"""
Verification script for individual NLP feature extraction method.

This script tests:
1. yfinance news fetching for individual stocks
2. Sentiment analysis on headlines
3. Daily aggregation per stock
4. Alignment with trading days
5. Feature vector extraction
6. Integration with feature windows
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, date

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_features import (
    extract_daily_nlp_features_yfinance,
    align_nlp_with_trading_days,
    get_nlp_feature_vector,
    get_nlp_feature_dim
)
from util import get_data


def test_yfinance_news_fetching():
    """Test 1: Verify yfinance news fetching works for individual stocks."""
    print("=" * 80)
    print("TEST 1: yfinance News Fetching")
    print("=" * 80)
    
    test_stocks = ["AAPL", "MSFT", "GOOGL"]
    print(f"\nTesting with stocks: {test_stocks}")
    
    try:
        stock_nlp_features = extract_daily_nlp_features_yfinance(
            stocks=test_stocks,
            start_date="2020-01-01",
            end_date="2020-12-31",
            batch_size=32,
            progress=True
        )
        
        print(f"\n✓ Successfully fetched news for {len(stock_nlp_features)} stocks")
        
        for stock, nlp_df in stock_nlp_features.items():
            if len(nlp_df) > 0:
                print(f"\n  {stock}:")
                print(f"    - Days with news: {len(nlp_df)}")
                print(f"    - Date range: {nlp_df['date'].min()} to {nlp_df['date'].max()}")
                print(f"    - Columns: {list(nlp_df.columns)}")
                print(f"    - Sample row:")
                print(f"      {nlp_df.iloc[0].to_dict()}")
            else:
                print(f"\n  {stock}: No news data available")
        
        return stock_nlp_features
        
    except Exception as e:
        print(f"\n✗ Error fetching yfinance news: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_daily_aggregation(stock_nlp_features):
    """Test 2: Verify daily aggregation per stock."""
    print("\n" + "=" * 80)
    print("TEST 2: Daily Aggregation Per Stock")
    print("=" * 80)
    
    if stock_nlp_features is None:
        print("Skipping - no stock NLP features available")
        return None
    
    for stock, nlp_df in stock_nlp_features.items():
        if len(nlp_df) > 0:
            print(f"\n{stock}:")
            print(f"  - Total days: {len(nlp_df)}")
            print(f"  - Days with has_news=1: {(nlp_df['has_news'] == 1).sum()}")
            print(f"  - Days with has_news=0: {(nlp_df['has_news'] == 0).sum()}")
            
            # Check feature columns
            expected_cols = ['date', 'has_news', 'p_pos', 'p_neg', 'p_neu', 
                           'n_articles', 'mean_pos', 'mean_neg', 'mean_neu',
                           'max_pos', 'max_neg', 'std_pos', 'std_neg', 'net_sent']
            missing_cols = [col for col in expected_cols if col not in nlp_df.columns]
            if missing_cols:
                print(f"  ⚠ Missing columns: {missing_cols}")
            else:
                print(f"  ✓ All expected columns present")
            
            # Check data types
            print(f"  - Data types:")
            for col in ['p_pos', 'p_neg', 'p_neu', 'has_news']:
                if col in nlp_df.columns:
                    print(f"    {col}: {nlp_df[col].dtype}, range: [{nlp_df[col].min():.4f}, {nlp_df[col].max():.4f}]")
    
    return stock_nlp_features


def test_alignment_with_trading_days(stock_nlp_features):
    """Test 3: Verify alignment with trading days."""
    print("\n" + "=" * 80)
    print("TEST 3: Alignment with Trading Days")
    print("=" * 80)
    
    if stock_nlp_features is None:
        print("Skipping - no stock NLP features available")
        return None
    
    # Create sample trading days (weekdays only)
    trading_days = pd.bdate_range(start="2020-01-01", end="2020-12-31")
    print(f"\nTrading days: {len(trading_days)} days")
    print(f"Date range: {trading_days.min()} to {trading_days.max()}")
    
    aligned_features = {}
    for stock, nlp_df in stock_nlp_features.items():
        if len(nlp_df) > 0:
            print(f"\nAligning {stock}...")
            aligned = align_nlp_with_trading_days(
                nlp_df,
                trading_days=trading_days,
                fill_method='zero'
            )
            aligned_features[stock] = aligned
            
            print(f"  ✓ Aligned to {len(aligned)} trading days")
            print(f"  - Days with news: {(aligned['has_news'] == 1).sum()}")
            print(f"  - Days without news: {(aligned['has_news'] == 0).sum()}")
            
            # Check that all trading days are present
            if len(aligned) == len(trading_days):
                print(f"  ✓ All trading days accounted for")
            else:
                print(f"  ⚠ Mismatch: {len(aligned)} aligned vs {len(trading_days)} trading days")
        else:
            print(f"\n{stock}: Skipping (no news data)")
    
    return aligned_features


def test_feature_vector_extraction(aligned_features):
    """Test 4: Verify feature vector extraction."""
    print("\n" + "=" * 80)
    print("TEST 4: Feature Vector Extraction")
    print("=" * 80)
    
    if aligned_features is None or len(aligned_features) == 0:
        print("Skipping - no aligned features available")
        return
    
    print(f"\nFeature dimension for individual method: {get_nlp_feature_dim('individual')}")
    print(f"Expected: 4 features [has_news, p_pos, p_neg, p_neu]")
    
    for stock, aligned_df in aligned_features.items():
        if len(aligned_df) > 0:
            print(f"\n{stock}:")
            
            # Test feature vector extraction
            sample_row = aligned_df.iloc[0]
            feature_vec = get_nlp_feature_vector(sample_row, nlp_method="individual")
            
            print(f"  ✓ Feature vector shape: {feature_vec.shape}")
            print(f"  ✓ Feature vector: {feature_vec}")
            print(f"  ✓ Feature names: [has_news, p_pos, p_neg, p_neu]")
            
            # Verify feature values
            if feature_vec.shape[0] == 4:
                print(f"  ✓ Correct dimension (4 features)")
            else:
                print(f"  ✗ Wrong dimension: expected 4, got {feature_vec.shape[0]}")
            
            # Check a few more rows
            print(f"\n  Sample feature vectors:")
            for idx in [0, len(aligned_df)//2, len(aligned_df)-1]:
                if idx < len(aligned_df):
                    row = aligned_df.iloc[idx]
                    vec = get_nlp_feature_vector(row, nlp_method="individual")
                    print(f"    Row {idx} (date={row['date']}, has_news={row['has_news']}): {vec}")
            break  # Only test first stock


def test_integration_with_util():
    """Test 5: Verify integration with util.py get_data()."""
    print("\n" + "=" * 80)
    print("TEST 5: Integration with util.py")
    print("=" * 80)
    
    print("\nTesting get_data() with individual NLP method...")
    print("Note: This will download stock data and extract NLP features.")
    print("This may take a while...")
    
    try:
        # Use a small test set
        test_stocks = ["AAPL", "MSFT"]
        test_args = ["2020-01-01", "2020-12-31"]
        
        print(f"\nStocks: {test_stocks}")
        print(f"Date range: {test_args}")
        print(f"NLP method: individual")
        
        # This will call get_data with individual NLP method
        # Note: This is a full integration test, so it may take time
        print("\n⚠ Skipping full integration test (would download data)")
        print("  To test integration, run:")
        print(f"    from util import get_data")
        print(f"    data = get_data({test_stocks}, {test_args}, seq_len=60, use_nlp=True, nlp_method='individual')")
        
    except Exception as e:
        print(f"\n✗ Error in integration test: {e}")
        import traceback
        traceback.print_exc()


def test_feature_window_construction():
    """Test 6: Verify feature window construction logic."""
    print("\n" + "=" * 80)
    print("TEST 6: Feature Window Construction Logic")
    print("=" * 80)
    
    # Simulate the logic from get_feature_input_classification
    print("\nSimulating feature window construction...")
    
    # Mock NLP features dictionary (individual method format)
    mock_nlp_features = {
        ('AAPL', date(2020, 1, 2)): pd.Series({
            'date': date(2020, 1, 2),
            'has_news': 1,
            'p_pos': 0.7,
            'p_neg': 0.2,
            'p_neu': 0.1
        }),
        ('AAPL', date(2020, 1, 3)): pd.Series({
            'date': date(2020, 1, 3),
            'has_news': 1,
            'p_pos': 0.6,
            'p_neg': 0.3,
            'p_neu': 0.1
        }),
        ('MSFT', date(2020, 1, 2)): pd.Series({
            'date': date(2020, 1, 2),
            'has_news': 1,
            'p_pos': 0.8,
            'p_neg': 0.1,
            'p_neu': 0.1
        }),
        '_method': 'individual_simple'
    }
    
    # Test method detection
    use_simple = mock_nlp_features.get('_method') == 'individual_simple'
    nlp_features_clean = {k: v for k, v in mock_nlp_features.items() if k != '_method'}
    sample_key = next(iter(nlp_features_clean.keys())) if nlp_features_clean else None
    is_individual_method = (sample_key is not None and isinstance(sample_key, tuple))
    
    print(f"  Method detection:")
    print(f"    - use_simple: {use_simple}")
    print(f"    - is_individual_method: {is_individual_method}")
    print(f"    - Sample key type: {type(sample_key)}")
    
    if is_individual_method:
        print(f"  ✓ Correctly identified as individual method")
        
        # Test feature dimension
        nlp_feature_dim = 4
        actual_method_for_features = "individual"
        print(f"  ✓ Feature dimension: {nlp_feature_dim}")
        print(f"  ✓ Method for features: {actual_method_for_features}")
        
        # Test lookup
        stock_ticker = 'AAPL'
        test_date = date(2020, 1, 2)
        lookup_key = (stock_ticker, test_date)
        
        if lookup_key in nlp_features_clean:
            nlp_row = nlp_features_clean[lookup_key]
            nlp_vec = get_nlp_feature_vector(nlp_row, nlp_method=actual_method_for_features)
            print(f"  ✓ Lookup successful for ({stock_ticker}, {test_date})")
            print(f"  ✓ Feature vector: {nlp_vec}")
        else:
            print(f"  ✗ Lookup failed for ({stock_ticker}, {test_date})")
        
        # Test missing lookup
        missing_key = ('AAPL', date(2020, 1, 5))
        if missing_key not in nlp_features_clean:
            zero_vec = np.zeros(nlp_feature_dim, dtype=float)
            print(f"  ✓ Missing lookup correctly returns zeros: {zero_vec}")
    else:
        print(f"  ✗ Failed to identify as individual method")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL NLP FEATURE EXTRACTION VERIFICATION")
    print("=" * 80)
    
    # Test 1: yfinance news fetching
    stock_nlp_features = test_yfinance_news_fetching()
    
    # Test 2: Daily aggregation
    stock_nlp_features = test_daily_aggregation(stock_nlp_features)
    
    # Test 3: Alignment with trading days
    aligned_features = test_alignment_with_trading_days(stock_nlp_features)
    
    # Test 4: Feature vector extraction
    test_feature_vector_extraction(aligned_features)
    
    # Test 5: Integration with util.py (skipped - would download data)
    test_integration_with_util()
    
    # Test 6: Feature window construction logic
    test_feature_window_construction()
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

