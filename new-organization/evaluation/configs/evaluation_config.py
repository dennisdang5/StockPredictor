"""
Configuration class for evaluation runs.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class EvaluationConfig:
    """
    Configuration for model evaluation.
    
    Attributes:
        model_path: Path to the saved model file
        model_type: Type of model (e.g., "LSTM", "CNNLSTM")
        stocks: List of stock symbols to evaluate on
        time_args: Time arguments for data loading (e.g., ["1990-01-01", "2015-12-31"])
        log_dir: Directory for TensorBoard logs
        device: Device to run evaluation on (optional, auto-detect if None)
        use_nlp: Whether to use NLP features
        nlp_method: NLP method to use - "aggregated" or "individual"
        input_shape: Input shape tuple (lookback_window, num_features). If None, will be determined from data.
        batch_size: Batch size for evaluation
        k: Number of top/bottom positions for portfolio construction
        cost_bps_per_side: Transaction costs per side in basis points
        create_plots: Whether to create visualization plots
    """
    model_path: str
    model_type: str
    stocks: List[str]
    time_args: List[str]
    log_dir: str = "runs/evaluation"
    device: Optional[str] = None
    use_nlp: bool = True  # Default to True
    nlp_method: str = "aggregated"  # Default to aggregated
    input_shape: Optional[Tuple[int, int]] = None
    batch_size: int = 32
    k: int = 10
    cost_bps_per_side: float = 5.0
    create_plots: bool = True
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'stocks': self.stocks,
            'time_args': self.time_args,
            'log_dir': self.log_dir,
            'device': self.device,
            'use_nlp': self.use_nlp,
            'nlp_method': self.nlp_method,
            'input_shape': self.input_shape,
            'batch_size': self.batch_size,
            'k': self.k,
            'cost_bps_per_side': self.cost_bps_per_side,
            'create_plots': self.create_plots,
        }

