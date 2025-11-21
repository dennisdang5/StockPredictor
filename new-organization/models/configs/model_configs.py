from .base_config import BaseModelConfig

class LSTMConfig(BaseModelConfig):
    """
    Configuration class for LSTMModel.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features). Default: (31, 13) for NLP aggregated, (31, 3) without NLP
        hidden_size (int): Hidden dimension of LSTM layers. Default: 25
        num_layers (int): Number of stacked LSTM layers. Default: 1
        batch_first (bool): Whether batch is first dimension. Default: True
        dropout (float): Dropout rate (0.0 to 1.0). Default: 0.1
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.input_shape = parameters.get('input_shape', (31, 13))  # Default includes NLP features (3 base + 10 NLP)
        self.hidden_size = parameters.get('hidden_size', 25)
        self.num_layers = parameters.get('num_layers', 1)
        self.batch_first = parameters.get('batch_first', True)
        self.dropout = parameters.get('dropout', 0.1)


class CNNLSTMConfig(BaseModelConfig):
    """
    Configuration class for CNNLSTMModel.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features). Default: (31, 13) for NLP aggregated, (31, 3) without NLP
        kernel_size (int): Size of CNN convolution kernel. Default: 3
        hidden_size (int): Hidden dimension for CNN and LSTM. Default: 25
        num_layers (int): Number of stacked LSTM layers. Default: 1
        batch_first (bool): Whether batch is first dimension. Default: True
        dropout (float): Dropout rate (0.0 to 1.0). Default: 0.1
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.input_shape = parameters.get('input_shape', (31, 13))  # Default includes NLP features (3 base + 10 NLP)
        self.kernel_size = parameters.get('kernel_size', 3)
        self.hidden_size = parameters.get('hidden_size', 25)
        self.num_layers = parameters.get('num_layers', 1)
        self.batch_first = parameters.get('batch_first', True)
        self.dropout = parameters.get('dropout', 0.1)


class AutoEncoderConfig(BaseModelConfig):
    """
    Configuration class for AutoEncoder.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features). Default: (31, 3)
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)


class CNNAutoEncoderConfig(BaseModelConfig):
    """
    Configuration class for CNNAutoEncoder.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features). Default: (31, 3)
        kernel_size (int): Size of CNN convolution kernel. Default: 3
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.kernel_size = parameters.get('kernel_size', 3)


class AELSTMConfig(BaseModelConfig):
    """
    Configuration class for AELSTM.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features). Default: (31, 13) for NLP aggregated, (31, 3) without NLP
        hidden_size (int): Hidden dimension of LSTM layers. Default: 25
        num_layers (int): Number of stacked LSTM layers. Default: 1
        batch_first (bool): Whether batch is first dimension. Default: True
        dropout (float): Dropout rate (0.0 to 1.0). Default: 0.1
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.input_shape = parameters.get('input_shape', (31, 13))  # Default includes NLP features (3 base + 10 NLP)
        self.hidden_size = parameters.get('hidden_size', 25)
        self.num_layers = parameters.get('num_layers', 1)
        self.batch_first = parameters.get('batch_first', True)
        self.dropout = parameters.get('dropout', 0.1)


class CNNAELSTMConfig(BaseModelConfig):
    """
    Configuration class for CNNAELSTM.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features). Default: (31, 3)
        kernel_size (int): Size of CNN convolution kernel. Default: 3
        hidden_size (int): Hidden dimension of LSTM layers. Default: 25
        num_layers (int): Number of stacked LSTM layers. Default: 1
        batch_first (bool): Whether batch is first dimension. Default: True
        dropout (float): Dropout rate (0.0 to 1.0). Default: 0.1
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.kernel_size = parameters.get('kernel_size', 3)
        self.hidden_size = parameters.get('hidden_size', 25)
        self.num_layers = parameters.get('num_layers', 1)
        self.batch_first = parameters.get('batch_first', True)
        self.dropout = parameters.get('dropout', 0.1)


class TimesNetConfig(BaseModelConfig):
    """
    Configuration class for TimesNet model.
    
    TimesNet is a time series forecasting/classification model that uses FFT-based
    period detection and 2D convolution blocks.
    
    IMPORTANT: seq_len is NOT a parameter in TimesNetConfig. It must be provided via 
    TrainerConfig.seq_len. The seq_len parameter controls both data generation (feature 
    extraction window) and model architecture, so it belongs in TrainerConfig, not model_config.
    When the model is initialized, seq_len will be automatically set from TrainerConfig.seq_len.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features). Default: (31, 13) for NLP aggregated, (31, 3) without NLP
        task_name (str): Task type. Options: 'classification', 'long_term_forecast', 
                        'short_term_forecast', 'imputation', 'anomaly_detection'. Default: 'classification'
        enc_in (int): Encoder input size = number of features per timestep. Default: 13 for NLP aggregated, 3 without NLP
        num_class (int): Number of classification classes. Default: 2
        d_model (int): Dimension of model embeddings (hidden dimension). Default: 256
        d_ff (int): Dimension of feed-forward network (usually 4x d_model). Default: 1024
        e_layers (int): Number of encoder layers (TimesBlock layers). Default: 2
        top_k (int): Number of top frequencies to consider in FFT. Default: 5
        num_kernels (int): Number of kernels in Inception_Block_V1. Default: 6
        embed (str): Time features encoding type. Options: 'timeF', 'fixed', 'learned'. Default: 'timeF'
        freq (str): Frequency for time features. Options: 's', 't', 'h', 'd', 'b', 'w', 'm'. Default: 'd'
        dropout (float): Dropout rate (0.0 to 1.0). Default: 0.1
        pred_len (int): Prediction sequence length (for forecast tasks). Default: 0
        label_len (int): Start token length (for forecast tasks). Default: 0
        c_out (int): Output size (for forecast/imputation/anomaly). Default: 3
        freeze_encoder (bool): If True, freezes encoder (enc_embedding, TimesBlock layers, layer_norm)
                              and only trains the classifier head. Default: False
    
    Example:
        from types import SimpleNamespace
        from trainer import TrainerConfig
        
        # Create TimesNet config WITHOUT seq_len
        timesnet_config = SimpleNamespace(
            task_name='classification',
            enc_in=13,
            num_class=2,
            d_model=256,
            # ... other parameters
        )
        
        # Create TrainerConfig WITH seq_len
        config = TrainerConfig(
            model_type="TimesNet",
            model_config=timesnet_config,
            seq_len=240,  # Set seq_len here, not in timesnet_config
            period_type="full",
            # ... other TrainerConfig parameters
        )
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.input_shape = parameters.get('input_shape', (31, 13))  # Default includes NLP features (3 base + 10 NLP)
        self.task_name = parameters.get('task_name', 'classification')
        # seq_len will be set from TrainerConfig.seq_len during model initialization
        # But can also be set from input_shape[0] if not provided
        self.seq_len = parameters.get('seq_len', None)  # Can be None, will be set from input_shape if needed
        self.enc_in = parameters.get('enc_in', 13)  # Default includes NLP features (3 base + 10 NLP)
        self.num_class = parameters.get('num_class', 2)
        self.d_model = parameters.get('d_model', 256)
        self.d_ff = parameters.get('d_ff', 1024)
        self.e_layers = parameters.get('e_layers', 2)
        self.top_k = parameters.get('top_k', 5)
        self.num_kernels = parameters.get('num_kernels', 6)
        self.embed = parameters.get('embed', 'timeF')
        self.freq = parameters.get('freq', 'd')
        self.dropout = parameters.get('dropout', 0.1)
        self.pred_len = parameters.get('pred_len', 0)
        self.label_len = parameters.get('label_len', 0)
        self.c_out = parameters.get('c_out', 3)
        self.freeze_encoder = parameters.get('freeze_encoder', False)
