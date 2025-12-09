#!/usr/bin/env python3
"""
Simple test script to verify the new-organization structure is working correctly.
Tests imports, model registry, configs, and basic functionality.
"""

import sys
import os

def test_imports():
    """Test that all main modules can be imported."""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    try:
        print("  [1/6] Testing models package...")
        from models import get_available_models, create_model
        from models.configs import LSTMConfig, CAELSTMConfig, AELSTMConfig
        print("      ✓ Models package imported successfully")
        
        print("  [2/6] Testing trainer...")
        from trainer import Trainer, TrainerConfig
        print("      ✓ Trainer imported successfully")
        
        print("  [3/6] Testing util...")
        import util
        print("      ✓ Util imported successfully")
        
        print("  [4/6] Testing evaluation package...")
        from evaluation.configs.evaluation_config import EvaluationConfig
        print("      ✓ Evaluation config imported successfully")
        
        print("  [5/6] Testing model registry...")
        from models.registry import ModelRegistry
        print("      ✓ Model registry imported successfully")
        
        print("  [6/6] Testing base model...")
        from models.base import BaseModel
        print("      ✓ Base model imported successfully")
        
        print("\n✓ All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_model_registry():
    """Test that models are registered correctly."""
    print("=" * 60)
    print("Testing Model Registry")
    print("=" * 60)
    
    try:
        from models import get_available_models
        
        available_models = get_available_models()
        print(f"  Available models: {', '.join(available_models)}")
        
        if len(available_models) == 0:
            print("  ⚠ Warning: No models registered")
            return False
        
        print(f"\n✓ Model registry working! Found {len(available_models)} model(s)\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Model registry test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_config_creation():
    """Test that configs can be created."""
    print("=" * 60)
    print("Testing Config Creation")
    print("=" * 60)
    
    try:
        from models.configs import LSTMConfig, CAELSTMConfig, AELSTMConfig
        
        print("  [1/3] Creating LSTMConfig...")
        lstm_config = LSTMConfig(parameters={
            'input_shape': (31, 13),  # 3 base + 10 NLP features (aggregated method)
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        })
        print(f"      ✓ LSTMConfig created: hidden_size={lstm_config.hidden_size}")
        
        print("  [2/3] Creating CAELSTMConfig...")
        cae_lstm_config = CAELSTMConfig(parameters={
            'input_shape': (31, 13),  # 3 base + 10 NLP features (aggregated method)
            'kernel_size': 3,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        })
        print(f"      ✓ CAELSTMConfig created: kernel_size={cae_lstm_config.kernel_size}")
        
        print("  [3/3] Creating AELSTMConfig...")
        ae_lstm_config = AELSTMConfig(parameters={
            'input_shape': (31, 13),  # 3 base + 10 NLP features (aggregated method)
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.15
        })
        print(f"      ✓ AELSTMConfig created: hidden_size={ae_lstm_config.hidden_size}")
        
        print("\n✓ All configs created successfully!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Config creation test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test that models can be created from configs."""
    print("=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    try:
        from models import create_model
        from models.configs import LSTMConfig
        
        print("  Creating LSTM model...")
        config = LSTMConfig(parameters={
            'input_shape': (31, 13),  # 3 base + 10 NLP features (aggregated method)
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        })
        
        model = create_model("LSTM", config)
        print(f"      ✓ Model created: {type(model).__name__}")
        print(f"      ✓ Model parameters: {sum(p.numel() for p in model.parameters())} total")
        
        # Test forward pass with dummy data
        import torch
        dummy_input = torch.randn(2, 31, 13)  # batch_size=2, seq_len=31, features=13 (3 base + 10 NLP)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"      ✓ Forward pass successful: output shape = {output.shape}")
        
        print("\n✓ Model creation test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Model creation test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_config():
    """Test that TrainerConfig can be created."""
    print("=" * 60)
    print("Testing TrainerConfig Creation")
    print("=" * 60)
    
    try:
        from trainer import TrainerConfig
        from models.configs import LSTMConfig
        
        print("  Creating TrainerConfig...")
        model_config = LSTMConfig(parameters={
            'input_shape': (31, 13),  # 3 base + 10 NLP features (aggregated method)
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        })
        
        trainer_config = TrainerConfig(
            stocks=["AAPL", "MSFT"],
            time_args=["2000-01-01", "2020-12-31"],
            batch_size=32,
            num_epochs=10,
            model_type="LSTM",
            model_config=model_config,
            period_type="LS",
            lookback=240
        )
        
        print(f"      ✓ TrainerConfig created")
        print(f"      ✓ Stocks: {trainer_config.stocks}")
        print(f"      ✓ Model type: {trainer_config.model_type}")
        print(f"      ✓ Batch size: {trainer_config.batch_size}")
        
        print("\n✓ TrainerConfig creation test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ TrainerConfig creation test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_config():
    """Test that EvaluationConfig can be created."""
    print("=" * 60)
    print("Testing EvaluationConfig Creation")
    print("=" * 60)
    
    try:
        from evaluation.configs.evaluation_config import EvaluationConfig
        
        print("  Creating EvaluationConfig...")
        eval_config = EvaluationConfig(
            model_path="test_model.pth",
            model_type="lstm",
            stocks=["AAPL", "MSFT"],
            time_args=["2000-01-01", "2020-12-31"],
            batch_size=32,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=True,  # Default uses NLP with aggregated method
            nlp_method="aggregated"
        )
        
        print(f"      ✓ EvaluationConfig created")
        print(f"      ✓ Model type: {eval_config.model_type}")
        print(f"      ✓ Stocks: {len(eval_config.stocks)} stocks")
        print(f"      ✓ Batch size: {eval_config.batch_size}")
        
        print("\n✓ EvaluationConfig creation test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ EvaluationConfig creation test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_timesnet():
    """Test that TimesNet can be imported and created."""
    print("=" * 60)
    print("Testing TimesNet")
    print("=" * 60)
    
    try:
        from models import create_model, get_available_models
        from models.configs import TimesNetConfig
        
        # Check if TimesNet is available
        available_models = get_available_models()
        if "TIMESNET" not in available_models:
            print("  ⚠ Warning: TimesNet not registered")
            print(f"  Available models: {', '.join(available_models)}")
            return False
        
        print("  [1/3] Creating TimesNetConfig...")
        timesnet_config = TimesNetConfig(parameters={
            'input_shape': (31, 13),  # 3 base + 10 NLP features (aggregated method)
            'task_name': 'classification',
            'enc_in': 13,  # 3 base + 10 NLP features (aggregated method)
            'num_class': 2,
            'd_model': 64,  # Smaller for testing
            'e_layers': 2,
            'top_k': 5,
            'num_kernels': 6,
            'dropout': 0.1
        })
        print(f"      ✓ TimesNetConfig created: d_model={timesnet_config.d_model}")
        
        print("  [2/3] Creating TimesNet model...")
        # Set seq_len from input_shape (TimesNetConfig inherits input_shape from BaseModelConfig)
        if hasattr(timesnet_config, 'input_shape') and timesnet_config.input_shape:
            timesnet_config.seq_len = timesnet_config.input_shape[0]
        else:
            # Default seq_len if input_shape not set
            timesnet_config.seq_len = 31
        model = create_model("TIMESNET", timesnet_config)
        print(f"      ✓ Model created: {type(model).__name__}")
        print(f"      ✓ Model parameters: {sum(p.numel() for p in model.parameters())} total")
        
        print("  [3/3] Testing forward pass...")
        import torch
        dummy_input = torch.randn(2, 31, 13)  # batch_size=2, seq_len=31, features=13 (3 base + 10 NLP)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"      ✓ Forward pass successful: output shape = {output.shape}")
        
        print("\n✓ TimesNet test passed!\n")
        return True
        
    except ImportError as e:
        print(f"\n✗ TimesNet import failed: {e}\n")
        print("  Note: TimesNet requires Time-Series-Library to be available")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ TimesNet test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing New Organization Structure")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Model Registry", test_model_registry()))
    results.append(("Config Creation", test_config_creation()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("TrainerConfig", test_trainer_config()))
    results.append(("EvaluationConfig", test_evaluation_config()))
    results.append(("TimesNet", test_timesnet()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! Structure is working correctly.\n")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

