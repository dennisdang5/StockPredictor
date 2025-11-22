#!/usr/bin/env python3
"""
Random Label Test Script

Shuffles training labels randomly and trains for a few epochs.
Loss should stay roughly flat and not decrease much.
If loss decreases similarly to normal training, indicates bug in loss computation or logging.
"""

import os
import sys
import torch
import numpy as np
from typing import List
import random

# Add parent directory (new-organization) to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up one level to new-organization
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trainer import Trainer, TrainerConfig
from models.configs import LSTMConfig, CNNLSTMConfig
import util
import os


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


class ShuffledLabelDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that shuffles labels randomly.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        # Create shuffled labels
        self.shuffled_indices = list(range(len(base_dataset)))
        random.shuffle(self.shuffled_indices)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original data
        X, Y, original_idx = self.base_dataset[idx]
        # Get shuffled label from random other sample
        shuffled_idx = self.shuffled_indices[idx]
        _, Y_shuffled, _ = self.base_dataset[shuffled_idx]
        # Return original X but shuffled Y
        return X, Y_shuffled, original_idx


def test_random_labels(stocks: List[str] = ["AAPL", "MSFT"],
                       time_args: List[str] = ["2000-01-01", "2010-12-31"],
                       num_epochs: int = 10,
                       model_type: str = "LSTM",
                       period_type: str = "LS",
                       lookback: int = 240,
                       use_nlp: bool = True,
                       nlp_method: str = "aggregated",
                       batch_size: int = 32,
                       model_file_name: str = None,
                       use_trained_models_dir: bool = True):
    """
    Test training on randomly shuffled labels.
    
    Args:
        stocks: List of stock symbols
        time_args: Time range arguments
        num_epochs: Number of epochs to train
        model_type: Model type to test
        period_type: Period type ("LS" or "full")
        lookback: Days of historical data
        use_nlp: Whether to use NLP features
        nlp_method: NLP method ("aggregated" or "individual")
        batch_size: Batch size for training
    """
    print("=" * 80)
    print("Random Label Test")
    print("=" * 80)
    print(f"Stocks: {stocks}")
    print(f"Time period: {time_args}")
    print(f"Model type: {model_type}")
    print(f"Period type: {period_type}")
    print(f"Lookback: {lookback}")
    print(f"Use NLP: {use_nlp}")
    print(f"NLP method: {nlp_method}")
    print(f"Batch size: {batch_size}")
    print(f"Training epochs: {num_epochs}")
    print("=" * 80)
    print()
    print("WARNING: Training labels will be randomly shuffled!")
    print("Expected behavior: Loss should stay roughly flat (not decrease)")
    print()
    
    # Create model config
    if model_type == "LSTM":
        model_config = LSTMConfig(parameters={
            'input_shape': (31, 13),
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        })
    elif model_type == "CNNLSTM":
        model_config = CNNLSTMConfig(parameters={
            'input_shape': (31, 13),
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        })
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create trainer config
    trainer_config = TrainerConfig(
        stocks=stocks,
        time_args=time_args,
        batch_size=batch_size,
        num_epochs=num_epochs,
        model_type=model_type,
        model_config=model_config,
        period_type=period_type,
        lookback=lookback,
        use_nlp=use_nlp,
        nlp_method=nlp_method,
        saved_model="__VERIFICATION_NO_MODEL__.pth",  # Non-existent path to skip model loading/searching
        save_every_epochs=0,  # Don't save during test
        early_stop_patience=999999,  # Disable early stopping
        early_stop_min_delta=0.0,
        k=10,
        cost_bps_per_side=5.0
    )
    
    # Create trainer
    print("Creating trainer and loading data...")
    trainer = Trainer(config=trainer_config)
    
    # Wrap training dataset with shuffled labels
    original_dataset = trainer.trainLoader.dataset
    shuffled_dataset = ShuffledLabelDataset(original_dataset)
    
    # Create new data loader with shuffled labels
    trainer.trainLoader = torch.utils.data.DataLoader(
        shuffled_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=trainer.pin_memory
    )
    
    print(f"Training on {len(shuffled_dataset)} samples with shuffled labels")
    print()
    
    # Track loss history
    loss_history = []
    
    # Train
    print("Starting training with shuffled labels...")
    for epoch in range(num_epochs):
        stop_condition = trainer.train_one_epoch(epoch)
        
        # We'll need to extract loss from trainer output
        # For now, we'll track it manually by checking trainer's internal state
        # This is a simplified version - in practice, you'd want to capture the actual loss
        
        if stop_condition:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    print()
    print("=" * 80)
    print("Random Label Test Results")
    print("=" * 80)
    print()
    print("Expected behavior:")
    print("  - Loss should stay roughly flat (not decrease significantly)")
    print("  - Loss may fluctuate randomly but should not trend downward")
    print()
    print("If loss decreases similarly to normal training:")
    print("  - Bug in loss computation")
    print("  - Bug in loss logging")
    print("  - Labels not actually shuffled")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test training on randomly shuffled labels")
    parser.add_argument("--stocks", nargs="+", default=["AAPL", "MSFT"],
                       help="Stock symbols to use")
    parser.add_argument("--time-args", nargs="+", default=["2000-01-01", "2010-12-31"],
                       help="Time range arguments")
    parser.add_argument("--num-epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--model-type", type=str, default="LSTM",
                       choices=["LSTM", "CNNLSTM"],
                       help="Model type to test")
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
    
    test_random_labels(
        stocks=args.stocks,
        time_args=args.time_args,
        num_epochs=args.num_epochs,
        model_type=args.model_type,
        period_type=args.period_type,
        lookback=args.lookback,
        use_nlp=args.use_nlp.lower() == "true",
        nlp_method=args.nlp_method,
        batch_size=args.batch_size,
        model_file_name=args.model_file_name,
        use_trained_models_dir=args.use_trained_models_dir.lower() == "true"
    )

