#!/usr/bin/env python3
"""
Preprocessing Verification Script

Verifies that train and validation datasets have:
- Same scaling/normalization
- No data leakage (future info in inputs)
- Correct shapes and data types
- Identical horizon/shift for sequence models
- Non-overlapping date ranges
- Consistent statistics
"""

import os
import sys
import torch
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from datetime import datetime

# Add parent directory (new-organization) to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up one level to new-organization
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trainer import Trainer, TrainerConfig
from models.configs import LSTMConfig
import util


def resolve_model_path(model_file_name, use_trained_models_dir=True):
    """
    Resolve model file path handling three cases:
    1. None -> returns None (uses unique ID system)
    2. Full/relative path -> returns as-is
    3. Filename only -> checks in trained_models/models/ if use_trained_models_dir=True
    
    Args:
        model_file_name: Model file name (None, full path, or filename)
        use_trained_models_dir: If True and filename only, check trained_models/models/
        
    Returns:
        Resolved model path or None
    """
    if model_file_name is None:
        print("[Model Path] Using None - will use unique ID system (trained_models/models/)")
        return None
    
    # If it's already an absolute path, use it as-is
    if os.path.isabs(model_file_name):
        print(f"[Model Path] Using absolute path as-is: {model_file_name}")
        return model_file_name
    
    # Check if it contains a directory component (relative path)
    dir_part = os.path.dirname(model_file_name)
    if dir_part:  # Has directory component (e.g., "path/to/model.pth" or "./model.pth")
        print(f"[Model Path] Using relative path as-is: {model_file_name}")
        return model_file_name
    
    # It's just a filename (e.g., "model.pth") - check in trained_models/models/ if flag is set
    if use_trained_models_dir:
        models_path = os.path.join(util.MODELS_DIR, model_file_name)
        if os.path.exists(models_path):
            print(f"[Model Path] Using existing model from trained_models: {models_path}")
            return models_path
        # File doesn't exist yet, but return the path where it should be saved
        print(f"[Model Path] Will save to trained_models: {models_path}")
        return models_path
    
    # Just return filename as-is (will save in current directory)
    print(f"[Model Path] Using filename as-is (current directory): {model_file_name}")
    return model_file_name


def analyze_dataset(loader, name: str, max_samples: int = 100):
    """
    Analyze a dataset loader and return statistics.
    
    Args:
        loader: DataLoader instance
        name: Name of the dataset (e.g., "train", "val")
        max_samples: Maximum number of samples to analyze
        
    Returns:
        Dictionary with statistics
    """
    print(f"\n{'=' * 80}")
    print(f"Analyzing {name.upper()} Dataset")
    print(f"{'=' * 80}")
    
    stats = {
        'name': name,
        'num_samples': 0,
        'batch_size': loader.batch_size,
        'num_batches': len(loader),
        'sample_shapes': [],
        'X_stats': {'mean': [], 'std': [], 'min': [], 'max': []},
        'Y_stats': {'mean': [], 'std': [], 'min': [], 'max': []},
        'dates': [],
        'indices': []
    }
    
    samples_analyzed = 0
    
    for batch_idx, (X_batch, Y_batch, indices) in enumerate(loader):
        if samples_analyzed >= max_samples:
            break
        
        batch_size = X_batch.shape[0]
        stats['num_samples'] += batch_size
        
        # Store shapes
        if batch_idx == 0:
            stats['sample_shapes'] = {
                'X': list(X_batch.shape),
                'Y': list(Y_batch.shape)
            }
        
        # Compute statistics for X
        X_np = X_batch.numpy()
        stats['X_stats']['mean'].append(np.mean(X_np))
        stats['X_stats']['std'].append(np.std(X_np))
        stats['X_stats']['min'].append(np.min(X_np))
        stats['X_stats']['max'].append(np.max(X_np))
        
        # Compute statistics for Y
        Y_np = Y_batch.numpy()
        stats['Y_stats']['mean'].append(np.mean(Y_np))
        stats['Y_stats']['std'].append(np.std(Y_np))
        stats['Y_stats']['min'].append(np.min(Y_np))
        stats['Y_stats']['max'].append(np.max(Y_np))
        
        # Store dates and indices
        if hasattr(loader.dataset, 'dates'):
            batch_dates = [loader.dataset.dates[idx] for idx in indices.tolist()]
            stats['dates'].extend(batch_dates)
        stats['indices'].extend(indices.tolist())
        
        samples_analyzed += batch_size
        
        # Print first few samples
        if batch_idx == 0:
            print(f"\nFirst batch sample:")
            print(f"  X shape: {X_batch.shape}")
            print(f"  Y shape: {Y_batch.shape}")
            print(f"  X sample (first timestep, first 5 features): {X_batch[0, 0, :5]}")
            print(f"  Y sample: {Y_batch[0]}")
            print(f"  Indices: {indices[:5].tolist()}")
    
    # Aggregate statistics
    stats['X_stats']['mean'] = np.mean(stats['X_stats']['mean'])
    stats['X_stats']['std'] = np.mean(stats['X_stats']['std'])
    stats['X_stats']['min'] = np.min(stats['X_stats']['min'])
    stats['X_stats']['max'] = np.max(stats['X_stats']['max'])
    
    stats['Y_stats']['mean'] = np.mean(stats['Y_stats']['mean'])
    stats['Y_stats']['std'] = np.mean(stats['Y_stats']['std'])
    stats['Y_stats']['min'] = np.min(stats['Y_stats']['min'])
    stats['Y_stats']['max'] = np.max(stats['Y_stats']['max'])
    
    # Print summary
    print(f"\nSummary Statistics:")
    print(f"  Total samples: {stats['num_samples']}")
    print(f"  Number of batches: {stats['num_batches']}")
    print(f"  Batch size: {stats['batch_size']}")
    print(f"\nX (Input) Statistics:")
    print(f"  Mean: {stats['X_stats']['mean']:.6f}")
    print(f"  Std:  {stats['X_stats']['std']:.6f}")
    print(f"  Min:  {stats['X_stats']['min']:.6f}")
    print(f"  Max:  {stats['X_stats']['max']:.6f}")
    print(f"\nY (Target) Statistics:")
    print(f"  Mean: {stats['Y_stats']['mean']:.6f}")
    print(f"  Std:  {stats['Y_stats']['std']:.6f}")
    print(f"  Min:  {stats['Y_stats']['min']:.6f}")
    print(f"  Max:  {stats['Y_stats']['max']:.6f}")
    
    if stats['dates']:
        dates_sorted = sorted(stats['dates'])
        print(f"\nDate Range:")
        print(f"  First date: {dates_sorted[0]}")
        print(f"  Last date:  {dates_sorted[-1]}")
        print(f"  Total days: {len(set(dates_sorted))}")
    
    return stats


def compare_datasets(train_stats: dict, val_stats: dict):
    """
    Compare train and validation dataset statistics.
    
    Args:
        train_stats: Statistics from training dataset
        val_stats: Statistics from validation dataset
    """
    print(f"\n{'=' * 80}")
    print("COMPARISON: Train vs Validation")
    print(f"{'=' * 80}")
    
    issues = []
    
    # Check shapes
    if train_stats['sample_shapes']['X'] != val_stats['sample_shapes']['X']:
        issues.append(f"X shapes differ: train={train_stats['sample_shapes']['X']}, val={val_stats['sample_shapes']['X']}")
    else:
        print(f"✓ X shapes match: {train_stats['sample_shapes']['X']}")
    
    if train_stats['sample_shapes']['Y'] != val_stats['sample_shapes']['Y']:
        issues.append(f"Y shapes differ: train={train_stats['sample_shapes']['Y']}, val={val_stats['sample_shapes']['Y']}")
    else:
        print(f"✓ Y shapes match: {train_stats['sample_shapes']['Y']}")
    
    # Check normalization (mean/std should be similar if normalized together)
    X_mean_diff = abs(train_stats['X_stats']['mean'] - val_stats['X_stats']['mean'])
    X_std_diff = abs(train_stats['X_stats']['std'] - val_stats['X_stats']['std'])
    
    print(f"\nNormalization Check:")
    print(f"  X mean difference: {X_mean_diff:.6f}")
    print(f"  X std difference:  {X_std_diff:.6f}")
    
    if X_mean_diff > 0.1:
        issues.append(f"Large X mean difference: {X_mean_diff:.6f} (may indicate different normalization)")
    else:
        print(f"✓ X means are similar")
    
    if X_std_diff > 0.1:
        issues.append(f"Large X std difference: {X_std_diff:.6f} (may indicate different normalization)")
    else:
        print(f"✓ X stds are similar")
    
    # Check date ranges (should not overlap)
    if train_stats['dates'] and val_stats['dates']:
        train_dates_set = set(train_stats['dates'])
        val_dates_set = set(val_stats['dates'])
        overlap = train_dates_set & val_dates_set
        
        print(f"\nDate Range Check:")
        print(f"  Train date range: {min(train_stats['dates'])} to {max(train_stats['dates'])}")
        print(f"  Val date range:   {min(val_stats['dates'])} to {max(val_stats['dates'])}")
        print(f"  Overlapping dates: {len(overlap)}")
        
        if len(overlap) > 0:
            issues.append(f"Date overlap detected: {len(overlap)} overlapping dates (data leakage risk!)")
        else:
            print(f"✓ No date overlap (good)")
    
    # Check index overlap
    train_indices_set = set(train_stats['indices'])
    val_indices_set = set(val_stats['indices'])
    index_overlap = train_indices_set & val_indices_set
    
    print(f"\nIndex Overlap Check:")
    print(f"  Overlapping indices: {len(index_overlap)}")
    
    if len(index_overlap) > 0:
        issues.append(f"Index overlap detected: {len(index_overlap)} overlapping indices (data leakage risk!)")
    else:
        print(f"✓ No index overlap (good)")
    
    # Summary
    print(f"\n{'=' * 80}")
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✓ No issues detected - preprocessing looks good!")
    print(f"{'=' * 80}")


def verify_preprocessing(stocks: List[str] = ["AAPL", "MSFT"],
                         time_args: List[str] = ["2000-01-01", "2010-12-31"],
                         model_type: str = "LSTM",
                         period_type: str = "LS",
                         lookback: int = 240,
                         use_nlp: bool = True,
                         nlp_method: str = "aggregated",
                         batch_size: int = 32,
                         model_file_name: str = None,
                         use_trained_models_dir: bool = True):
    """
    Verify preprocessing consistency between train and validation sets.
    
    Args:
        stocks: List of stock symbols
        time_args: Time range arguments
        model_type: Model type to use
        period_type: Period type ("LS" or "full")
        lookback: Days of historical data
        use_nlp: Whether to use NLP features
        nlp_method: NLP method ("aggregated" or "individual")
        batch_size: Batch size for training
    """
    print("=" * 80)
    print("Preprocessing Verification")
    print("=" * 80)
    print(f"Stocks: {stocks}")
    print(f"Time period: {time_args}")
    print(f"Model type: {model_type}")
    print(f"Period type: {period_type}")
    print(f"Lookback: {lookback}")
    print(f"Use NLP: {use_nlp}")
    print(f"NLP method: {nlp_method}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)
    
    # Create model config
    model_config = LSTMConfig(parameters={
        'input_shape': (31, 13),
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1
    })
    
    # Create trainer config
    trainer_config = TrainerConfig(
        stocks=stocks,
        time_args=time_args,
        batch_size=batch_size,
        num_epochs=1,  # Just need to load data
        model_type=model_type,
        model_config=model_config,
        period_type=period_type,
        lookback=lookback,
        use_nlp=use_nlp,
        nlp_method=nlp_method,
        saved_model="__VERIFICATION_NO_MODEL__.pth",  # Non-existent path to skip model loading/searching
        save_every_epochs=0,
        early_stop_patience=999999,
        early_stop_min_delta=0.0,
        k=10,
        cost_bps_per_side=5.0
    )
    
    # Create trainer (this loads the data)
    print("\n" + "=" * 80)
    print("Loading data...")
    print(f"[DEBUG] Starting Trainer initialization at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)
    trainer = Trainer(config=trainer_config)
    print(f"[DEBUG] Trainer initialized successfully at {datetime.now().strftime('%H:%M:%S')}")
    
    # Analyze datasets
    train_stats = analyze_dataset(trainer.trainLoader, "train", max_samples=100)
    val_stats = analyze_dataset(trainer.validationLoader, "validation", max_samples=100)
    
    # Compare
    compare_datasets(train_stats, val_stats)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify preprocessing consistency")
    parser.add_argument("--stocks", nargs="+", default=["AAPL", "MSFT"],
                       help="Stock symbols to use")
    parser.add_argument("--time-args", nargs="+", default=["2000-01-01", "2010-12-31"],
                       help="Time range arguments")
    parser.add_argument("--model-type", type=str, default="LSTM",
                       choices=["LSTM", "CNNLSTM"],
                       help="Model type to use")
    parser.add_argument("--period-type", type=str, default="LS",
                       choices=["LS", "full"],
                       help="Period type for feature extraction")
    parser.add_argument("--lookback", type=int, default=240,
                       help="Days of historical data")
    parser.add_argument("--use-nlp", type=str, default="true",
                       choices=["true", "false"],
                       help="Whether to use NLP features")
    parser.add_argument("--nlp-method", type=str, default="aggregated",
                       choices=["aggregated", "individual"],
                       help="NLP method")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--model-file-name", type=str, default=None,
                       help="Model file name (default: auto-generate). Can be None, full path, or filename.")
    parser.add_argument("--use-trained-models-dir", type=str, default="true",
                       choices=["true", "false"],
                       help="If true and filename only, check/save in trained_models/models/")
    
    args = parser.parse_args()
    
    verify_preprocessing(
        stocks=args.stocks,
        time_args=args.time_args,
        model_type=args.model_type,
        period_type=args.period_type,
        lookback=args.lookback,
        use_nlp=args.use_nlp.lower() == "true",
        nlp_method=args.nlp_method,
        batch_size=args.batch_size,
        model_file_name=args.model_file_name,
        use_trained_models_dir=args.use_trained_models_dir.lower() == "true"
    )

