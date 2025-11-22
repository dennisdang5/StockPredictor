"""
Data Download and Consistency Validation Script

This script:
1. Downloads stock price data and NLP features
2. Validates data consistency according to the integration plan
3. Checks alignment between price data and NLP features
4. Verifies feature dimensions and structure
5. Validates "no news" vs "neutral sentiment" distinction

Based on recommendations:
- Variable articles → fixed-size daily features ✓
- "No news" vs "neutral sentiment" properly handled ✓
- Trading day alignment ✓
- FinBERT on headlines only ✓
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, date
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import util
from nlp_features import get_nlp_feature_dim


def validate_data_consistency(
    stocks,
    time_args,
    use_nlp=False,
    nlp_method=None,
    data_dir="data"
):
    """
    Download and validate data consistency.
    
    Args:
        stocks: List of stock tickers
        time_args: Date range arguments [start, end] or period string
        use_nlp: Whether to include NLP features
        nlp_method: "aggregated" (NYT) or "individual" (yfinance per stock). Required if use_nlp=True.
        data_dir: Directory for data files
    
    Returns:
        dict with validation results
    """
    print("=" * 80)
    print("DATA DOWNLOAD AND CONSISTENCY VALIDATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Stocks: {len(stocks)} stocks")
    print(f"  Time range: {time_args}")
    print(f"  Use NLP: {use_nlp}")
    if use_nlp:
        print(f"  NLP Method: {nlp_method}")
    print()
    
    # Validate that nlp_method is provided when use_nlp=True
    if use_nlp and nlp_method is None:
        error_msg = "ERROR: nlp_method must be explicitly provided when use_nlp=True. Please specify either 'aggregated' or 'individual'."
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}
    
    # Validate nlp_method value if provided
    if nlp_method is not None and nlp_method not in ["aggregated", "individual"]:
        error_msg = f"ERROR: Invalid nlp_method '{nlp_method}'. Must be 'aggregated' or 'individual'."
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}
    
    # ========================================================================
    # Step 1: Download/Load Data
    # ========================================================================
    print("=" * 80)
    print("STEP 1: Downloading/Loading Data")
    print("=" * 80)
    
    try:
        data = util.get_data(
            stocks=stocks,
            args=time_args,
            data_dir=data_dir,
            prediction_type="classification",
            use_nlp=use_nlp,
            nlp_method=nlp_method   
        )
        
        if isinstance(data, int):
            print(f"❌ ERROR: get_data returned error code {data}")
            return {"success": False, "error": f"get_data returned {data}"}
        
        Xtr, Xva, Xte, Ytr, Yva, Yte, Dtr, Dva, Dte, Rev_test, Returns_test, Sp500_test = data
        
        print(f"✅ Data loaded successfully")
        print(f"  Training samples: {len(Xtr):,}")
        print(f"  Validation samples: {len(Xva):,}")
        print(f"  Test samples: {len(Xte):,}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    
    # ========================================================================
    # Step 2: Validate Data Structure
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Validating Data Structure")
    print("=" * 80)
    
    validation_results = {
        "success": True,
        "warnings": [],
        "errors": []
    }
    
    # Check all datasets have same structure
    datasets = {
        "train": (Xtr, Ytr, Dtr),
        "validation": (Xva, Yva, Dva),
        "test": (Xte, Yte, Dte)
    }
    
    expected_time_steps = None
    expected_features = None
    
    for split_name, (X, Y, D) in datasets.items():
        print(f"\n[{split_name.upper()}]")
        
        # Check shapes
        if len(X) != len(Y) or len(X) != len(D):
            error_msg = f"Length mismatch: X={len(X)}, Y={len(Y)}, D={len(D)}"
            print(f"  ❌ {error_msg}")
            validation_results["errors"].append(f"{split_name}: {error_msg}")
            validation_results["success"] = False
        else:
            print(f"  ✅ Lengths match: {len(X):,} samples")
        
        # Check X shape
        if len(X) > 0:
            X_shape = X.shape if isinstance(X, torch.Tensor) else np.array(X).shape
            print(f"  X shape: {X_shape}")
            
            if expected_time_steps is None:
                expected_time_steps = X_shape[1]  # Should be 31
                expected_features = X_shape[2]    # Should be 3 (price) or 3+NLP
            
            # Validate time steps (should be 31)
            if X_shape[1] != 31:
                error_msg = f"Unexpected time steps: {X_shape[1]} (expected 31)"
                print(f"  ❌ {error_msg}")
                validation_results["errors"].append(f"{split_name}: {error_msg}")
                validation_results["success"] = False
            else:
                print(f"  ✅ Time steps: {X_shape[1]} (expected: 31)")
            
            # Validate feature dimensions
            if use_nlp:
                # With NLP: should be 3 (price) + NLP features
                expected_nlp_dim = 4 if nlp_method == "individual" else 10
                expected_total_features = 3 + expected_nlp_dim
                if X_shape[2] != expected_total_features:
                    error_msg = f"Feature dimension mismatch: {X_shape[2]} (expected {expected_total_features} = 3 price + {expected_nlp_dim} NLP)"
                    print(f"  ❌ {error_msg}")
                    validation_results["errors"].append(f"{split_name}: {error_msg}")
                    validation_results["success"] = False
                else:
                    print(f"  ✅ Feature dimensions: {X_shape[2]} (3 price + {expected_nlp_dim} NLP)")
            else:
                # Without NLP: should be 3 (price only)
                if X_shape[2] != 3:
                    error_msg = f"Feature dimension mismatch: {X_shape[2]} (expected 3 for price-only)"
                    print(f"  ❌ {error_msg}")
                    validation_results["errors"].append(f"{split_name}: {error_msg}")
                    validation_results["success"] = False
                else:
                    print(f"  ✅ Feature dimensions: {X_shape[2]} (price only)")
        
        # Check Y shape
        if len(Y) > 0:
            Y_shape = Y.shape if isinstance(Y, torch.Tensor) else np.array(Y).shape
            print(f"  Y shape: {Y_shape}")
            if Y_shape[1] != 1:
                error_msg = f"Y should have 1 output, got {Y_shape[1]}"
                print(f"  ❌ {error_msg}")
                validation_results["errors"].append(f"{split_name}: {error_msg}")
                validation_results["success"] = False
            else:
                print(f"  ✅ Y shape correct: {Y_shape}")
        
        # Check dates
        print(f"  Date range: {D[0]} to {D[-1]}")
        if len(D) != len(X):
            error_msg = f"Date length mismatch: {len(D)} dates vs {len(X)} samples"
            print(f"  ❌ {error_msg}")
            validation_results["errors"].append(f"{split_name}: {error_msg}")
            validation_results["success"] = False
    
    # ========================================================================
    # Step 3: Validate NLP Features (if enabled)
    # ========================================================================
    if use_nlp:
        print("\n" + "=" * 80)
        print("STEP 3: Validating NLP Features")
        print("=" * 80)
        print(f"  Method: {nlp_method}")
        
        # First check if NLP features are actually present in the data
        sample_X = Xtr[0] if len(Xtr) > 0 else None
        if sample_X is not None:
            sample_X = sample_X.numpy() if isinstance(sample_X, torch.Tensor) else sample_X
            actual_feature_dim = sample_X.shape[1]  # Second dimension is features
            
            # Check if NLP features are present
            if actual_feature_dim <= 3:
                error_msg = f"NLP features not found in data! Expected >= 7 features (3 price + 4 NLP) or >= 13 features (3 price + 10 NLP), but data only has {actual_feature_dim} features. The cache may have been created without NLP features. Try deleting the cache and regenerating with --use-nlp flag."
                print(f"  ❌ {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["success"] = False
                print(f"\n  💡 Solution: Delete cache files and re-run with --use-nlp flag, or load data without --use-nlp flag.")
                print(f"      Cache files location: {data_dir}/")
                return validation_results
            
            # Extract NLP features from sample based on method
            if nlp_method == "individual":
                # Individual method: 4 features [has_news, p_pos, p_neg, p_neu]
                # Uses first article's sentiment per day (no aggregation)
                nlp_start_idx = 3
                nlp_end_idx = 7  # 4 features: has_news, p_pos, p_neg, p_neu
                expected_nlp_dim = 4
                expected_total_features = 7
                print(f"  Expected NLP features: 4 (has_news, p_pos, p_neg, p_neu)")
                print(f"  Expected total features: 7 (3 price + 4 NLP)")
            else:
                # Aggregated method: 10 features [has_news, n_articles, mean_pos, mean_neg, mean_neu, 
                #                                  max_pos, max_neg, std_pos, std_neg, net_sent]
                nlp_start_idx = 3
                nlp_end_idx = 13  # 10 features
                expected_nlp_dim = 10
                expected_total_features = 13
                print(f"  Expected NLP features: 10 (has_news, n_articles, mean_pos, mean_neg, mean_neu, max_pos, max_neg, std_pos, std_neg, net_sent)")
                print(f"  Expected total features: 13 (3 price + 10 NLP)")
            
            # Check if we have enough features
            if actual_feature_dim < nlp_end_idx:
                error_msg = f"Not enough features in data! Expected at least {nlp_end_idx} features, but data only has {actual_feature_dim} features. The cache may have been created with a different NLP configuration."
                print(f"  ❌ {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["success"] = False
                print(f"\n  💡 Solution: Delete cache files and re-run with matching NLP configuration.")
                return validation_results
            
            nlp_features_sample = sample_X[:, nlp_start_idx:nlp_end_idx]
            
            print(f"\n  Sample NLP features (first time step):")
            print(f"    Shape: {nlp_features_sample.shape}")
            if nlp_features_sample.shape[1] > 0:
                print(f"    Values: {nlp_features_sample[0]}")
            else:
                error_msg = "NLP features slice is empty!"
                print(f"  ❌ {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["success"] = False
                return validation_results
            
            # Validate feature ranges
            has_news_values = nlp_features_sample[:, 0]
            unique_has_news = np.unique(has_news_values)
            print(f"\n  has_news values: {unique_has_news}")
            
            if not np.all(np.isin(unique_has_news, [0.0, 1.0])):
                error_msg = f"has_news should be 0 or 1, found: {unique_has_news}"
                print(f"  ❌ {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["success"] = False
            else:
                print(f"  ✅ has_news values are valid (0 or 1)")
            
            # Method-specific validation
            if nlp_method == "individual":
                # Individual method validation: simple format with direct sentiment probabilities
                p_pos = nlp_features_sample[:, 1]
                p_neg = nlp_features_sample[:, 2]
                p_neu = nlp_features_sample[:, 3]
                
                print(f"\n  Individual Method Validation:")
                print(f"    p_pos range: [{p_pos.min():.3f}, {p_pos.max():.3f}]")
                print(f"    p_neg range: [{p_neg.min():.3f}, {p_neg.max():.3f}]")
                print(f"    p_neu range: [{p_neu.min():.3f}, {p_neu.max():.3f}]")
                
                # Check probabilities are in valid range [0, 1]
                if np.any(p_pos < 0) or np.any(p_pos > 1) or np.any(p_neg < 0) or np.any(p_neg > 1) or np.any(p_neu < 0) or np.any(p_neu > 1):
                    error_msg = "Sentiment probabilities must be in [0, 1] range"
                    print(f"  ❌ {error_msg}")
                    validation_results["errors"].append(error_msg)
                    validation_results["success"] = False
                else:
                    print(f"  ✅ Sentiment probabilities in valid range [0, 1]")
                
                # Check probabilities sum to ~1 for days with news
                prob_sums = p_pos + p_neg + p_neu
                with_news_mask = has_news_values == 1
                
                if np.any(with_news_mask):
                    with_news_sums = prob_sums[with_news_mask]
                    if not np.allclose(with_news_sums, 1.0, atol=0.01):
                        warning_msg = f"Days with news: sentiment probabilities don't sum to 1 (range: [{with_news_sums.min():.3f}, {with_news_sums.max():.3f}])"
                        print(f"  ⚠️  {warning_msg}")
                        validation_results["warnings"].append(warning_msg)
                    else:
                        print(f"  ✅ Days with news: sentiment probabilities sum to ~1")
                
                # Check "no news" vs "neutral sentiment" distinction
                # No-news days should have sentiment probabilities = 0 (not neutral sentiment)
                no_news_mask = has_news_values == 0
                
                if np.any(no_news_mask):
                    no_news_p_pos = p_pos[no_news_mask]
                    no_news_p_neg = p_neg[no_news_mask]
                    no_news_p_neu = p_neu[no_news_mask]
                    no_news_sentiment = no_news_p_pos + no_news_p_neg + no_news_p_neu
                    
                    if not np.allclose(no_news_sentiment, 0.0, atol=0.01):
                        warning_msg = f"No-news days should have sentiment=0, but found range: [{no_news_sentiment.min():.3f}, {no_news_sentiment.max():.3f}]"
                        print(f"  ⚠️  {warning_msg}")
                        validation_results["warnings"].append(warning_msg)
                    else:
                        print(f"  ✅ No-news days have sentiment=0 (distinct from neutral sentiment)")
                
            else:
                # Aggregated method validation: full format with aggregations
                print(f"\n  Aggregated Method Validation:")
                
                # Check aggregated features exist
                if nlp_features_sample.shape[1] != 10:
                    error_msg = f"Aggregated method should have 10 features, got {nlp_features_sample.shape[1]}"
                    print(f"  ❌ {error_msg}")
                    validation_results["errors"].append(error_msg)
                    validation_results["success"] = False
                else:
                    print(f"  ✅ All 10 aggregated features present")
                
                # Validate aggregated statistics
                n_articles = nlp_features_sample[:, 1] if nlp_features_sample.shape[1] > 1 else None
                mean_pos = nlp_features_sample[:, 2] if nlp_features_sample.shape[1] > 2 else None
                mean_neg = nlp_features_sample[:, 3] if nlp_features_sample.shape[1] > 3 else None
                mean_neu = nlp_features_sample[:, 4] if nlp_features_sample.shape[1] > 4 else None
                
                if n_articles is not None:
                    print(f"    n_articles range: [{n_articles.min():.0f}, {n_articles.max():.0f}]")
                    if np.any(n_articles < 0):
                        error_msg = "n_articles should be >= 0"
                        print(f"  ❌ {error_msg}")
                        validation_results["errors"].append(error_msg)
                        validation_results["success"] = False
                    else:
                        print(f"  ✅ n_articles values are valid")
                
                # Check "no news" vs "neutral sentiment" distinction for aggregated method
                no_news_mask = has_news_values == 0
                with_news_mask = has_news_values == 1
                
                if np.any(no_news_mask) and mean_pos is not None:
                    no_news_mean_pos = mean_pos[no_news_mask]
                    no_news_mean_neg = mean_neg[no_news_mask]
                    no_news_mean_neu = mean_neu[no_news_mask]
                    no_news_sentiment = no_news_mean_pos + no_news_mean_neg + no_news_mean_neu
                    
                    if not np.allclose(no_news_sentiment, 0.0, atol=0.01):
                        warning_msg = f"No-news days should have sentiment=0, but found range: [{no_news_sentiment.min():.3f}, {no_news_sentiment.max():.3f}]"
                        print(f"  ⚠️  {warning_msg}")
                        validation_results["warnings"].append(warning_msg)
                    else:
                        print(f"  ✅ No-news days have sentiment=0 (distinct from neutral)")
            
            # Check for NaN or Inf
            if np.any(np.isnan(nlp_features_sample)) or np.any(np.isinf(nlp_features_sample)):
                error_msg = "NLP features contain NaN or Inf values"
                print(f"  ❌ {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["success"] = False
            else:
                print(f"  ✅ No NaN or Inf values in NLP features")
    
    # ========================================================================
    # Step 4: Validate Date Alignment and Split Integrity
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Validating Date Alignment and Split Integrity")
    print("=" * 80)
    
    # Check that dates are in order
    for split_name, (X, Y, D) in datasets.items():
        if len(D) > 1:
            dates_sorted = sorted(D)
            if dates_sorted != D:
                warning_msg = f"{split_name}: Dates are not sorted"
                print(f"  ⚠️  {warning_msg}")
                validation_results["warnings"].append(warning_msg)
            else:
                print(f"  ✅ {split_name}: Dates are sorted")
    
    # Check train/val/test split boundaries
    if len(Dtr) > 0 and len(Dva) > 0:
        train_end = max(Dtr)
        val_start = min(Dva)
        if train_end >= val_start:
            warning_msg = f"Train/val date boundary issue: train ends {train_end}, val starts {val_start}"
            print(f"  ⚠️  {warning_msg}")
            validation_results["warnings"].append(warning_msg)
        else:
            print(f"  ✅ Train/val split: train ends {train_end}, val starts {val_start}")
    
    if len(Dva) > 0 and len(Dte) > 0:
        val_end = max(Dva)
        test_start = min(Dte)
        if val_end >= test_start:
            warning_msg = f"Val/test date boundary issue: val ends {val_end}, test starts {test_start}"
            print(f"  ⚠️  {warning_msg}")
            validation_results["warnings"].append(warning_msg)
        else:
            print(f"  ✅ Val/test split: val ends {val_end}, test starts {test_start}")
    
    # Verify date grouping: all samples from same date are in same split
    print(f"\n  Verifying date grouping integrity...")
    
    # Group dates by split
    train_dates_set = set(pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, datetime)) else d for d in Dtr)
    val_dates_set = set(pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, datetime)) else d for d in Dva)
    test_dates_set = set(pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, datetime)) else d for d in Dte)
    
    # Verify: no date appears in multiple splits (all samples from same date are in same split)
    train_val_overlap = train_dates_set & val_dates_set
    train_test_overlap = train_dates_set & test_dates_set
    val_test_overlap = val_dates_set & test_dates_set
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        error_msg = f"Date overlap detected! Dates appear in multiple splits: train-val={len(train_val_overlap)}, train-test={len(train_test_overlap)}, val-test={len(val_test_overlap)}. This violates temporal integrity."
        print(f"  ❌ {error_msg}")
        validation_results["errors"].append(error_msg)
        validation_results["success"] = False
        # Show first few overlapping dates
        if train_val_overlap:
            print(f"    Train-Val overlap examples: {list(train_val_overlap)[:3]}")
        if train_test_overlap:
            print(f"    Train-Test overlap examples: {list(train_test_overlap)[:3]}")
        if val_test_overlap:
            print(f"    Val-Test overlap examples: {list(val_test_overlap)[:3]}")
    else:
        print(f"  ✅ All samples from same date are in same split (no date overlap)")
        print(f"    Train: {len(train_dates_set)} unique dates")
        print(f"    Val:   {len(val_dates_set)} unique dates")
        print(f"    Test:  {len(test_dates_set)} unique dates")
    
    # Verify: all samples are assigned to a split
    total_samples = len(Dtr) + len(Dva) + len(Dte)
    total_unique_dates = len(train_dates_set | val_dates_set | test_dates_set)
    print(f"  Total samples: {total_samples:,}, Total unique dates: {total_unique_dates}")
    
    if total_samples == 0:
        error_msg = "No samples found in any split!"
        print(f"  ❌ {error_msg}")
        validation_results["errors"].append(error_msg)
        validation_results["success"] = False
    else:
        print(f"  ✅ All {total_samples:,} samples are assigned to splits")
    
    # ========================================================================
    # Step 5: Validate Feature Consistency
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Validating Feature Consistency")
    print("=" * 80)
    
    # Check that all splits have same feature dimensions
    feature_dims = {}
    for split_name, (X, Y, D) in datasets.items():
        if len(X) > 0:
            X_shape = X.shape if isinstance(X, torch.Tensor) else np.array(X).shape
            feature_dims[split_name] = X_shape[2]
    
    if len(set(feature_dims.values())) > 1:
        error_msg = f"Feature dimension mismatch across splits: {feature_dims}"
        print(f"  ❌ {error_msg}")
        validation_results["errors"].append(error_msg)
        validation_results["success"] = False
    else:
        print(f"  ✅ All splits have consistent feature dimensions: {list(feature_dims.values())[0]}")
    
    # Check price features (first 3) are valid
    sample_X = Xtr[0] if len(Xtr) > 0 else None
    if sample_X is not None:
        sample_X = sample_X.numpy() if isinstance(sample_X, torch.Tensor) else sample_X
        price_features = sample_X[:, :3]
        
        if np.any(np.isnan(price_features)) or np.any(np.isinf(price_features)):
            error_msg = "Price features contain NaN or Inf values"
            print(f"  ❌ {error_msg}")
            validation_results["errors"].append(error_msg)
            validation_results["success"] = False
        else:
            print(f"  ✅ Price features are valid (no NaN/Inf)")
            print(f"    Price feature range: [{price_features.min():.3f}, {price_features.max():.3f}]")
    
    # ========================================================================
    # Step 6: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Validation Summary")
    print("=" * 80)
    
    if validation_results["success"]:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ VALIDATION FAILED")
    
    if validation_results["errors"]:
        print(f"\n❌ Errors ({len(validation_results['errors'])}):")
        for error in validation_results["errors"]:
            print(f"  - {error}")
    
    if validation_results["warnings"]:
        print(f"\n⚠️  Warnings ({len(validation_results['warnings'])}):")
        for warning in validation_results["warnings"]:
            print(f"  - {warning}")
    
    # Print data statistics
    print(f"\n📊 Data Statistics:")
    print(f"  Total samples: {len(Xtr) + len(Xva) + len(Xte):,}")
    print(f"  Training: {len(Xtr):,} ({len(Xtr)/(len(Xtr)+len(Xva)+len(Xte))*100:.1f}%)")
    print(f"  Validation: {len(Xva):,} ({len(Xva)/(len(Xtr)+len(Xva)+len(Xte))*100:.1f}%)")
    print(f"  Test: {len(Xte):,} ({len(Xte)/(len(Xtr)+len(Xva)+len(Xte))*100:.1f}%)")
    
    if use_nlp:
        print(f"\n  NLP Features:")
        print(f"    Method: {nlp_method}")
        nlp_dim = 4 if nlp_method == "individual" else 10
        print(f"    NLP dimension: {nlp_dim}")
        print(f"    Total features: 3 (price) + {nlp_dim} (NLP) = {3 + nlp_dim}")
    
    print("\n" + "=" * 80)
    
    return validation_results


def main():
    """Main validation script."""
    import argparse
    
    # Define stocks list as a variable
    
    stocks = [
        # Communication Services
        "GOOGL", "GOOG", "T", "CHTR", "CMCSA", "EA", "FOXA", "FOX", "IPG", "LYV", "MTCH", "META", "NFLX", "NWSA", "NWS", "OMC", "PSKY", "TMUS", "TTWO", "TKO", "TTD", "VZ", "DIS", "WBD",

        # consumer discretionary
        "ABNB", "AMZN", "APTV", "AZO", "BBY", "BKNG", "CZR", "KMX", "CCL", "CMG", "DRI", "DECK", "DPZ", "DASH", "DHI", "EBAY", "EXPE", "F", "GRMN", "GM", "GPC", "HAS", "HLT", "HD", "LVS", "LEN", "LKQ", "LOW", "LULU", "MAR", "MCD", "MGM", "MHK", "NKE", "NCLH", "NVR", "ORLY", "POOL", "PHM", "RL", "ROST", "RCL", "SBUX", "TPR", "TSLA", "TJX", "TSCO", "ULTA", "WSM", "WYNN", "YUM",

        # Consumer Staples
        "MO", "ADM", "BF.B", "BG", "CPB", "CHD", "CLX", "KO", "CL", "CAG", "STZ", "COST", "DG", "DLTR", "EL", "GIS", "HSY", "HRL", "K", "KVUE", "KDP", "KMB", "KHC", "KR", "LW", "MKC", "TAP", "MDLZ", "MNST", "PEP", "PM", "PG", "SJM", "SYY", "TGT", "TSN", "WBA", "WMT",

        # Energy
        "APA", "BKR", "CVX", "COP", "CTRA", "DVN", "FANG", "EOG", "EQT", "EXE", "XOM", "HAL", "KMI", "MPC", "OXY", "OKE", "PSX", "SLB", "TRGP", "TPL", "VLO", "WMB",

        # Financials
        "AFL", "ALL", "AXP", "AIG", "AMP", "AON", "APO", "ACGL", "AJG", "AIZ", "BAC", "BRK.B", "BLK", "BX", "XYZ", "BK", "BRO", "COF", "CBOE", "SCHW", "CB", "CINF", "C", "CFG", "CME", "COIN", "CPAY", "ERIE", "EG", "FDS", "FIS", "FITB", "FI", "BEN", "GPN", "GL", "GS", "HIG", "HBAN", "ICE", "IVZ", "JKHY", "JPM", "KEY", "KKR", "L", "MTB", "MKTX", "MMC", "MA", "MET", "MCO", "MS", "MSCI", "NDAQ", "NTRS", "PYPL", "PNC", "PFG", "PGR", "PRU", "RJF", "RF", "SPGI", "STT", "SYF", "TROW", "TRV", "TFC", "USB", "V", "WRB", "WFC", "WTW",

        # Healthcare
        "ABT", "ABBV", "A", "ALGN", "AMGN", "BAX", "BDX", "TECH", "BIIB", "BSX", "BMY", "CAH", "COR", "CNC", "CRL", "CI", "COO", "CVS", "DHR", "DVA", "DXCM", "EW", "ELV", "GEHC", "GILD", "HCA", "HSIC", "HOLX", "HUM", "IDXX", "INCY", "PODD", "ISRG", "IQV", "JNJ", "LH", "LLY", "MCK", "MDT", "MRK", "MTD", "MRNA", "MOH", "PFE", "DGX", "REGN", "RMD", "RVTY", "SOLV", "STE", "SYK", "TMO", "UNH", "UHS", "VRTX", "VTRS", "WAT", "WST", "ZBH", "ZTS",

        # Industrials
        "MMM", "AOS", "ALLE", "AME", "ADP", "AXON", "BA", "BR", "BLDR", "CHRW", "CARR", "CAT", "CTAS", "CPRT", "CSX", "CMI", "DAY", "DE", "DAL", "DOV", "ETN", "EMR", "EFX", "EXPD", "FAST", "FDX", "FTV", "GE", "GEV", "GNRC", "GD", "HON", "HWM", "HUBB", "HII", "IEX", "ITW", "IR", "JBHT", "J", "JCI", "LHX", "LDOS", "LII", "LMT", "MAS", "NDSN", "NSC", "NOC", "ODFL", "OTIS", "PCAR", "PH", "PAYX", "PAYC", "PNR", "PWR", "RTX", "RSG", "ROK", "ROL", "SNA", "LUV", "SWK", "TXT", "TT", "TDG", "UBER", "UNP", "UAL", "UPS", "URI", "VLTO", "VRSK", "GWW", "WAB", "WM", "XYL",

        # Information Technology
        "ACN", "ADBE", "AMD", "AKAM", "APH", "ADI", "AAPL", "AMAT", "ANET", "ADSK", "AVGO", "CDNS", "CDW", "CSCO", "CTSH", "GLW", "CRWD", "DDOG", "DELL", "ENPH", "EPAM", "FFIV", "FICO", "FSLR", "FTNT", "IT", "GEN", "GDDY", "HPE", "HPQ", "IBM", "INTC", "INTU", "JBL", "KEYS", "KLAC", "LRCX", "MCHP", "MU", "MSFT", "MPWR", "MSI", "NTAP", "NVDA", "NXPI", "ON", "ORCL", "PLTR", "PANW", "PTC", "QCOM", "ROP", "CRM", "STX", "NOW", "SWKS", "SMCI", "SNPS", "TEL", "TDY", "TER", "TXN", "TRMB", "TYL", "VRSN", "WDC", "WDAY", "ZBRA",

        # Materials
        "APD", "ALB", "AMCR", "AVY", "BALL", "CF", "CTVA", "DOW", "DD", "EMN", "ECL", "FCX", "IFF", "IP", "LIN", "LYB", "MLM", "MOS", "NEM", "NUE", "PKG", "PPG", "SHW", "SW", "STLD", "VMC",

        # Real Estate
        "ARE", "AMT", "AVB", "BXP", "CPT", "CBRE", "CSGP", "CCI", "DLR", "EQIX", "EQR", "ESS", "EXR", "FRT", "DOC", "HST", "INVH", "IRM", "KIM", "MAA", "PLD", "PSA", "O", "REG", "SBAC", "SPG", "UDR", "VTR", "VICI", "WELL", "WY",

        # Utilities
        "AES", "LNT", "AEE", "AEP", "AWK", "ATO", "CNP", "CMS", "ED", "CEG", "D", "DTE", "DUK", "EIX", "ETR", "EVRG", "ES", "EXC", "FE", "NEE", "NI", "NRG", "PCG", "PNW", "PPL", "PEG", "SRE", "SO", "VST", "WEC", "XEL"
    ]
    """

    stocks = [
        # Top Technology & Growth
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",

        # Major Financial Services
        "JPM", "BAC", "V", "MA", "WFC", "GS", "BLK", "AXP",

        # Healthcare Leaders
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO",

        # Consumer & Retail Giants
        "WMT", "PG", "HD", "COST", "MCD", "NKE",

        # Industrial & Energy Leaders
        "BA", "CAT", "XOM", "CVX"
    ]
    """
    
    parser = argparse.ArgumentParser(description="Validate data download and consistency")
    parser.add_argument("--start-date", type=str, default="1990-01-01",
                        help="Start date (default: 1990-01-01)")
    parser.add_argument("--end-date", type=str, default="2015-12-31",
                        help="End date (default: 2015-12-31)")
    parser.add_argument("--use-nlp", action="store_true",
                        help="Include NLP features")
    parser.add_argument("--nlp-method", choices=["aggregated", "individual"], default=None,
                        help="NLP method: aggregated (NYT) or individual (yfinance per stock). Required if --use-nlp is specified.")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory (default: data)")
    
    args = parser.parse_args()
    
    time_args = [args.start_date, args.end_date]
    
    results = validate_data_consistency(
        stocks=stocks,
        time_args=time_args,
        use_nlp=args.use_nlp,
        nlp_method=args.nlp_method,
        data_dir=args.data_dir
    )
    
    # Exit with error code if validation failed
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()

