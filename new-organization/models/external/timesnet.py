from ..base import BaseModel
from ..configs import TimesNetConfig
import torch
import torch.nn as nn
import sys
import os

# Lazy import function for TimesNet Model
def _get_timesnet_model():
    """Lazy import of TimesNet Model to ensure path is set correctly."""
    # Add Time-Series-Library to path
    _lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Time-Series-Library'))
    if _lib_path not in sys.path:
        sys.path.insert(0, _lib_path)
    
    # Use importlib to avoid conflicts with our own 'models' package
    import importlib.util
    timesnet_path = os.path.join(_lib_path, 'models', 'TimesNet.py')
    if os.path.exists(timesnet_path):
        spec = importlib.util.spec_from_file_location("timesnet_model", timesnet_path)
        timesnet_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(timesnet_module)
        return timesnet_module.Model
    else:
        # Fallback to regular import
        try:
            # Temporarily rename our models package to avoid conflict
            import models as our_models
            sys.modules['_our_models'] = our_models
            del sys.modules['models']
            
            from models.TimesNet import Model
            
            # Restore our models package
            sys.modules['models'] = our_models
            del sys.modules['_our_models']
            
            return Model
        except Exception:
            # Restore models if it was deleted
            if '_our_models' in sys.modules:
                sys.modules['models'] = sys.modules['_our_models']
                del sys.modules['_our_models']
            return None

# Try to import at module level, but Model may be None if import fails
Model = _get_timesnet_model()

class TimesNetAdapter(BaseModel):
    """
    Adapter for TimesNet model from Time-Series-Library.
    
    Args:
        model_config: TimesNetConfig instance containing model parameters.
        
    Expected model_config attributes:
        - input_shape (tuple): Shape of input data (timesteps, features). Default: (31, 3)
        - task_name (str): Task type. Options: 'classification', 'long_term_forecast', 
                        'short_term_forecast', 'imputation', 'anomaly_detection'. Default: 'classification'
        - enc_in (int): Encoder input size = number of features per timestep. Default: 3
        - num_class (int): Number of classification classes. Default: 2
        - d_model (int): Dimension of model embeddings (hidden dimension). Default: 256
        - d_ff (int): Dimension of feed-forward network (usually 4x d_model). Default: 1024
        - e_layers (int): Number of encoder layers (TimesBlock layers). Default: 2
        - top_k (int): Number of top frequencies to consider in FFT. Default: 5
        - num_kernels (int): Number of kernels in Inception_Block_V1. Default: 6
    """
    def __init__(self, model_config):
        super(TimesNetAdapter, self).__init__(model_config)
        if model_config is None:
            raise ValueError("model_config is required for TimesNetAdapter")
        
        if not isinstance(model_config, TimesNetConfig):
            raise TypeError(
                f"TimesNetAdapter requires TimesNetConfig instance, "
                f"got {type(model_config).__name__}. "
                f"Use TimesNetConfig(input_shape=..., task_name=..., enc_in=..., num_class=..., d_model=..., d_ff=..., e_layers=..., top_k=..., num_kernels=...) to create the config."
            )

        if Model is None:
            raise ImportError(
                "TimesNet model not available. Please ensure Time-Series-Library is properly installed."
            )

        # TimesNet Model expects configs with seq_len attribute
        # Set seq_len from input_shape if not already set
        if not hasattr(model_config, 'seq_len') or model_config.seq_len is None:
            if hasattr(model_config, 'input_shape') and model_config.input_shape:
                model_config.seq_len = model_config.input_shape[0]
            else:
                model_config.seq_len = 31  # Default fallback

        self.timesnet = Model(model_config)
    
    @classmethod
    def from_config(cls, model_config):
        """Create model instance from config."""
        return cls(model_config)
        
    def forward(self, x_enc, params=None):
        # no specific mask assume all are valid timesteps
        if params is None:
            x_mark_enc=torch.ones(x_enc.shape[0], x_enc.shape[1], device=x_enc.device, dtype=x_enc.dtype)
        else:
            x_mark_enc = params['mask'] # mask of valid timesteps
        
        # Get raw output from TimesNet (shape: [batch_size, num_classes])
        output = self.timesnet(x_enc, x_mark_enc, None, None, None)
        
        # If classification with 3 classes, convert to single value using top_k ranking method
        # similar to how LSTM uses cross-sectional ranking for classification
        # Formula: (positive - negative) * (1 - neutral)
        # where: class 0 = negative, class 1 = neutral, class 2 = positive
        if (hasattr(self.model_config, 'task_name') and 
            self.model_config.task_name == 'classification' and
            hasattr(self.model_config, 'num_class') and 
            self.model_config.num_class == 3 and
            output.shape[1] == 3):
            
            # Convert logits to probabilities using softmax
            probs = torch.nn.functional.softmax(output, dim=1)
            
            # Extract probabilities for each class
            negative = probs[:, 0]  # Class 0: negative return
            neutral = probs[:, 1]   # Class 1: neutral return
            positive = probs[:, 2]   # Class 2: positive return
            
            # Use ranking-based approach: rank classes by probability (similar to top_k method)
            # The ranking gives us confidence in each class
            # Apply conversion formula: (positive - negative) * (1 - neutral)
            # This formula:
            # - When positive > negative: gives positive value
            # - When negative > positive: gives negative value  
            # - (1 - neutral) scales by confidence (high neutral → low confidence in direction)
            single_output = (positive - negative) * (1 - neutral)
            
            # Reshape to (batch_size, 1) to match expected output shape
            return single_output.unsqueeze(1)
        
        # For other cases (2 classes, regression, etc.), return as-is
        # If it's classification with 2 classes, we might want to convert to single value too
        if (hasattr(self.model_config, 'task_name') and 
            self.model_config.task_name == 'classification' and
            output.shape[1] > 1):
            # For binary classification, take the difference or first class
            # This handles cases where num_class != 3
            if output.shape[1] == 2:
                probs = torch.nn.functional.softmax(output, dim=1)
                # Convert binary to single value: prob[1] - prob[0]
                single_output = probs[:, 1] - probs[:, 0]
                return single_output.unsqueeze(1)
            else:
                # For other multi-class cases, take first class or use argmax
                # Default: use first class logit
                return output[:, 0:1]
        
        return output


# Register the model
from ..registry import ModelRegistry
ModelRegistry.register("TIMESNET", lambda config: TimesNetAdapter(config), TimesNetConfig)