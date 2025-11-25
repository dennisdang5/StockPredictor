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
        self.output_dim = parameters.get('output_dim', 1)

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
        self.output_dim = parameters.get('output_dim', 1)
        if parameters.get('lstm_config') is not None:
            self.lstm_config = LSTMConfig(parameters=parameters.get('lstm_config'))
        else:
            self.lstm_config = LSTMConfig(parameters={
                'input_shape': self.input_shape,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            })


class AutoEncoderConfig(BaseModelConfig):
    """
    Configuration class for AutoEncoder.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features). Default: (31, 3)
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.input_shape = parameters.get('input_shape', (31, 3))


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
        self.input_shape = parameters.get('input_shape', (31, 3))


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
        self.output_dim = parameters.get('output_dim', 1)
        if parameters.get('lstm_config') is not None:
            self.lstm_config = LSTMConfig(parameters=parameters.get('lstm_config'))
        else:
            self.lstm_config = LSTMConfig(parameters={
                'input_shape': self.input_shape,
                'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout
                })
        if parameters.get('ae_config') is not None:
            self.ae_config = AutoEncoderConfig(parameters=parameters.get('ae_config'))
        else:
            self.ae_config = AutoEncoderConfig(parameters={
                'input_shape': self.input_shape,
                })

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
        self.output_dim = parameters.get('output_dim', 1)
        if parameters.get('cnn_ae_config') is not None:
            self.cnn_ae_config = CNNAutoEncoderConfig(parameters=parameters.get('cnn_ae_config'))
        else:
            self.cnn_ae_config = CNNAutoEncoderConfig(parameters={
                'input_shape': self.input_shape,
                'kernel_size': self.kernel_size,
                })
        if parameters.get('lstm_config') is not None:
            self.lstm_config = LSTMConfig(parameters=parameters.get('lstm_config'))
        else:
            self.lstm_config = LSTMConfig(parameters={
                'input_shape': self.input_shape,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
                })

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


class TabPFNConfig(BaseModelConfig):
    """
    Lightweight configuration holder for TabPFN adapters.

    Args:
        backend (str): Which TabPFN backend to use ("client" or "local"). Default: "client"
        max_samples (int): Safety limit for the number of tabular rows. Default: 50_000
        random_state (int): Random seed passed to TabPFN estimators. Default: 42
        model_params (dict): Extra keyword arguments forwarded to the underlying estimator.
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.backend = parameters.get('backend', 'client')
        self.max_samples = parameters.get('max_samples', 50_000)
        self.random_state = parameters.get('random_state', 42)
        self.model_params = parameters.get('model_params', {})


class MLPConfig(BaseModelConfig):
    """
    Configuration class for MLPModel.
    
    MLP acts as a classifier on outputs from individual models (e.g., portfolio architecture).
    Input should be feature vectors from individual model outputs, not raw time series data.
    
    Args:
        input_dim (int): Dimension of input data (feature vectors from individual models). Required.
        hidden_dims (list[int]): List of hidden layer dimensions. Default: [input_dim//2]
        activation (str): Activation name supported by torch.nn (default: "relu").
        dropout (float): Dropout rate applied between MLP layers (0.0 to 1.0). Default: 0.1
        output_dim (int): Output dimension. Default: 1
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.input_dim = parameters.get('input_dim')
        # Default hidden_dims to [input_dim//2] if not provided, but only if input_dim exists
        default_hidden = [self.input_dim // 2] if self.input_dim else None
        self.hidden_dims = parameters.get('hidden_dims', default_hidden)
        self.activation = parameters.get('activation', 'relu')
        self.dropout = parameters.get('dropout', 0.1)
        self.output_dim = parameters.get('output_dim', 1)


class PortfolioConfig(BaseModelConfig):
    """
    Configuration container for PortfolioArchitecture.

    Args:
        stocks (list[str]): Ordered list of stock tickers represented in each batch.
        base_model_type (str): Name of the single-stock model to instantiate (e.g., "TabPFN", "LSTM").
        base_model_config (BaseModelConfig): Config object forwarded to each base model.
        strategy (str): Composition strategy. Options: "independent", "shared".
        mlp_hidden_dims (list[int]): Hidden layer sizes for the portfolio MLP head.
        activation (str): Activation name supported by torch.nn (default: "relu").
        dropout (float): Dropout applied between MLP layers.
        embedding_dim (int): Dimensionality of stock embeddings (shared strategy only).
        use_stock_embeddings (bool): Whether to concatenate stock embeddings before the MLP.
        freeze_base_models (bool): Freeze inner models during portfolio training.
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.stocks = parameters.get('stocks', [])
        self.base_model_type = parameters.get('base_model_type', 'TabPFN')
        self.base_model_config = parameters.get('base_model_config')
        self.strategy = parameters.get('strategy', 'independent').lower()
        self.mlp_hidden_dims = parameters.get('mlp_hidden_dims', [128, 64])
        self.activation = parameters.get('activation', 'relu')
        self.dropout = parameters.get('dropout', 0.1)
        self.embedding_dim = parameters.get('embedding_dim', 32)
        self.use_stock_embeddings = parameters.get('use_stock_embeddings', True)
        self.freeze_base_models = parameters.get('freeze_base_models', False)
