"""
Main script for training multiple models based on config objects.

This script allows you to define multiple model configurations and train them sequentially.
Each model configuration consists of a model config and a trainer config.
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from trainer import Trainer, TrainerConfig
from models.configs import (
    LSTMConfig, 
    CNNLSTMConfig, 
    AELSTMConfig, 
    CNNAELSTMConfig, 
    TimesNetConfig
)
from models import get_available_models


class ModelTrainingConfig:
    """
    Container for a complete model training configuration.
    
    Combines a model-specific config with trainer config parameters.
    """
    def __init__(
        self,
        name: str,
        model_type: str,
        model_config,
        stocks: List[str],
        time_args: List[str],
        batch_size: int = 32,
        num_epochs: int = 100,
        period_type: str = "LS",
        lookback: int = 240,
        use_nlp: bool = True,  # Default to True
        nlp_method: Optional[str] = "aggregated",  # Default to aggregated
        saved_model: Optional[str] = None,
        save_every_epochs: int = 25,
        early_stop_patience: int = 7,
        early_stop_min_delta: float = 0.001,
        k: int = 10,
        cost_bps_per_side: float = 5.0,
        **kwargs
    ):
        """
        Initialize a model training configuration.
        
        Args:
            name: Unique name for this model configuration (used for saving/logging)
            model_type: Type of model ("LSTM", "CNNLSTM", "AELSTM", "CNNAELSTM", "TimesNet")
            model_config: Model-specific config object (e.g., LSTMConfig instance)
            stocks: List of stock tickers
            time_args: Time range arguments (e.g., ["3y"] or ["1990-01-01", "2015-12-31"])
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            period_type: Period type ("LS" or "full")
            lookback: Days of historical data used for feature extraction
            use_nlp: Whether to use NLP features
            nlp_method: NLP method ("aggregated" or "individual") if use_nlp=True
            saved_model: Path to saved model to load (optional). If None, uses unique ID system
                         to find/create model based on config (saves to trained_models/models/{model_id}.pth)
            save_every_epochs: Save model every N epochs
            early_stop_patience: Early stopping patience
            early_stop_min_delta: Early stopping minimum delta
            k: Number of top/bottom positions for portfolio
            cost_bps_per_side: Transaction costs per side in basis points
            **kwargs: Additional arguments passed to TrainerConfig
        """
        self.name = name
        self.model_type = model_type
        self.model_config = model_config
        self.stocks = stocks
        self.time_args = time_args
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.period_type = period_type
        self.lookback = lookback
        self.use_nlp = use_nlp
        self.nlp_method = nlp_method
        self.saved_model = saved_model
        self.save_every_epochs = save_every_epochs
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.k = k
        self.cost_bps_per_side = cost_bps_per_side
        self.kwargs = kwargs
    
    def create_trainer_config(self) -> TrainerConfig:
        """Create a TrainerConfig from this model training config."""
        return TrainerConfig(
            stocks=self.stocks,
            time_args=self.time_args,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            model_type=self.model_type,
            model_config=self.model_config,
            period_type=self.period_type,
            lookback=self.lookback,
            use_nlp=self.use_nlp,
            nlp_method=self.nlp_method,
            saved_model=self.saved_model,
            save_every_epochs=self.save_every_epochs,
            early_stop_patience=self.early_stop_patience,
            early_stop_min_delta=self.early_stop_min_delta,
            k=self.k,
            cost_bps_per_side=self.cost_bps_per_side,
            **self.kwargs
        )


def create_model_configs() -> List[ModelTrainingConfig]:
    """
    Create a list of model training configurations.
    
    Add or modify configurations here to train different models.
    """
    configs = []
    
    # Common stock list
    common_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    common_time_args = ["2000-01-01", "2020-12-31"]
    
    # Example 1: Basic LSTM
    lstm_config = LSTMConfig(parameters={
        'input_shape': (31, 13),  # 3 base + 10 NLP features (aggregated method) - will be overridden by actual data
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1
    })
    configs.append(ModelTrainingConfig(
        name="lstm_basic",
        model_type="LSTM",
        model_config=lstm_config,
        stocks=common_stocks,
        time_args=common_time_args,
        batch_size=32,
        num_epochs=100,
        use_nlp=True,
        nlp_method="aggregated"
    ))
    
    # Example 2: CNN-LSTM with NLP (aggregated)
    cnn_lstm_config = CNNLSTMConfig(parameters={
        'input_shape': (31, 13),  # 3 base + 10 NLP features
        'kernel_size': 3,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2
    })
    configs.append(ModelTrainingConfig(
        name="cnn_lstm_nlp_agg",
        model_type="CNNLSTM",
        model_config=cnn_lstm_config,
        stocks=common_stocks,
        time_args=common_time_args,
        batch_size=32,
        num_epochs=150,
        use_nlp=True,
        nlp_method="aggregated"
    ))
    
    # Example 3: AutoEncoder-LSTM
    ae_lstm_config = AELSTMConfig(parameters={
        'input_shape': (31, 13),  # 3 base + 10 NLP features (aggregated method) - will be overridden by actual data
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.15
    })
    configs.append(ModelTrainingConfig(
        name="ae_lstm",
        model_type="AELSTM",
        model_config=ae_lstm_config,
        stocks=common_stocks,
        time_args=common_time_args,
        batch_size=32,
        num_epochs=150,
        use_nlp=True,
        nlp_method="aggregated"
    ))
    
    # Example 4: TimesNet with full period
    timesnet_config = TimesNetConfig(parameters={
        'input_shape': (240, 13),  # Full period: all timesteps, 3 base + 10 NLP features (aggregated method)
        'task_name': 'classification',
        'enc_in': 13,  # 3 base + 10 NLP features (aggregated method)
        'num_class': 2,
        'd_model': 256,
        'e_layers': 2,
        'top_k': 5,
        'num_kernels': 6,
        'dropout': 0.1,
        'embed': 'timeF',
        'freq': 'd'
    })
    configs.append(ModelTrainingConfig(
        name="timesnet_full",
        model_type="TimesNet",
        model_config=timesnet_config,
        stocks=common_stocks[:3],  # Fewer stocks for TimesNet (more computationally intensive)
        time_args=common_time_args,
        batch_size=16,  # Smaller batch for full period
        num_epochs=150,
        period_type="full",  # Use all consecutive days
        use_nlp=True,
        nlp_method="aggregated"
    ))
    
    return configs


def train_model(config: ModelTrainingConfig, log_dir: str = "training_logs") -> Dict:
    """
    Train a single model configuration.
    
    Args:
        config: ModelTrainingConfig instance
        log_dir: Directory for training logs
    
    Returns:
        Dictionary with training results and metadata
    """
    print("\n" + "=" * 80)
    print(f"Training Model: {config.name}")
    print("=" * 80)
    print(f"Model Type: {config.model_type}")
    print(f"Stocks: {config.stocks}")
    print(f"Time Range: {config.time_args}")
    print(f"Batch Size: {config.batch_size}, Epochs: {config.num_epochs}")
    print(f"Period Type: {config.period_type}, Lookback: {config.lookback}")
    print(f"NLP: {config.use_nlp} ({config.nlp_method if config.use_nlp else 'N/A'})")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    result = {
        'name': config.name,
        'model_type': config.model_type,
        'start_time': datetime.now().isoformat(),
        'success': False,
        'error': None,
        'training_time': None
    }
    
    try:
        # Create trainer config
        # If saved_model is None, trainer will use unique ID system to find/create model
        trainer_config = config.create_trainer_config()
        
        # Create trainer
        # Trainer will automatically:
        # - Check for existing model with matching config
        # - Use existing model if found, or create new one with unique ID
        # - Save to trained_models/models/{model_id}.pth
        trainer = Trainer(config=trainer_config)
        
        # Get the actual save path from trainer (set by unique ID system)
        actual_save_path = trainer.save_path
        
        # Train: run training loop
        # Trainer uses train_one_epoch which handles early stopping internally
        for epoch in range(trainer.num_epochs):
            stop_condition = trainer.train_one_epoch(epoch)
            if stop_condition:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Evaluate
        if hasattr(trainer, 'evaluate'):
            trainer.evaluate()
        
        result['success'] = True
        result['training_time'] = time.time() - start_time
        result['saved_model'] = actual_save_path  # Include actual save path from trainer
        
        print(f"\n✓ Successfully trained {config.name}")
        print(f"  Training time: {result['training_time']:.2f} seconds")
        print(f"  Model saved to: {actual_save_path}")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        result['training_time'] = time.time() - start_time
        
        print(f"\n✗ Failed to train {config.name}")
        print(f"  Error: {e}")
        print(f"  Time elapsed: {result['training_time']:.2f} seconds")
        
        import traceback
        traceback.print_exc()
    
    return result


def train_all_models(
    configs: List[ModelTrainingConfig],
    log_dir: str = "training_logs",
    continue_on_error: bool = True
) -> List[Dict]:
    """
    Train multiple model configurations sequentially.
    
    Args:
        configs: List of ModelTrainingConfig instances
        log_dir: Directory for training logs
        continue_on_error: Whether to continue training other models if one fails
    
    Returns:
        List of result dictionaries, one per model
    """
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'=' * 80}")
    print(f"Starting Training Session")
    print(f"{'=' * 80}")
    print(f"Total models to train: {len(configs)}")
    print(f"Log directory: {log_dir}")
    print(f"Continue on error: {continue_on_error}")
    print(f"{'=' * 80}\n")
    
    results = []
    session_start = time.time()
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Processing: {config.name}")
        
        try:
            result = train_model(config, log_dir)
            results.append(result)
            
            if not result['success'] and not continue_on_error:
                print(f"\nStopping training due to error in {config.name}")
                break
                
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            break
        except Exception as e:
            print(f"\nUnexpected error processing {config.name}: {e}")
            results.append({
                'name': config.name,
                'model_type': config.model_type,
                'success': False,
                'error': str(e),
                'training_time': None
            })
            if not continue_on_error:
                break
    
    # Summary
    session_time = time.time() - session_start
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'=' * 80}")
    print(f"Training Session Complete")
    print(f"{'=' * 80}")
    print(f"Total models: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {session_time:.2f} seconds ({session_time/60:.2f} minutes)")
    print(f"{'=' * 80}\n")
    
    # Save results summary
    import json
    summary_path = os.path.join(log_dir, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'session_start': datetime.fromtimestamp(session_start).isoformat(),
            'session_time': session_time,
            'total_models': len(results),
            'successful': successful,
            'failed': failed,
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to: {summary_path}")
    
    return results


if __name__ == "__main__":
    """
    Main entry point.
    
    Modify create_model_configs() to add/remove model configurations.
    """
    
    # List available models
    print("Available models:", ", ".join(get_available_models()))
    print()
    
    # Create model configurations
    model_configs = create_model_configs()
    
    print(f"Created {len(model_configs)} model configurations:")
    for cfg in model_configs:
        print(f"  - {cfg.name} ({cfg.model_type})")
    print()
    
    # Train all models
    results = train_all_models(
        configs=model_configs,
        log_dir="training_logs",
        continue_on_error=True  # Continue training other models if one fails
    )
    
    # Print final summary
    print("\nFinal Results:")
    for result in results:
        status = "✓" if result['success'] else "✗"
        time_str = f"{result['training_time']:.2f}s" if result['training_time'] else "N/A"
        print(f"  {status} {result['name']}: {time_str}")
        if result['success'] and 'saved_model' in result:
            print(f"    Saved to: {result['saved_model']}")
        if not result['success']:
            print(f"    Error: {result['error']}")
