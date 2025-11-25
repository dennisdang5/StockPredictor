"""
Main script for training multiple models based on config objects.

This script allows you to define multiple model configurations and train them sequentially.
Each model configuration consists of a model config and a trainer config.
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Optional

from trainer import Trainer, TrainerConfig
from models.configs import (
    LSTMConfig,
    AELSTMConfig,
    TimesNetConfig,
    TabPFNConfig,
    PortfolioConfig,
)
from models import get_available_models
from data_sources import YFinanceDataSource

# Full stock universe (mirrors STOCKS in download_data.py)
LARGE_STOCKS = [
    # Communication Services
    "GOOGL", "GOOG", "T", "CHTR", "CMCSA", "EA", "FOXA", "FOX", "IPG", "LYV", "MTCH",
    "META", "NFLX", "NWSA", "NWS", "OMC", "PSKY", "TMUS", "TTWO", "TKO", "TTD", "VZ",
    "DIS", "WBD",

    # Consumer Discretionary
    "ABNB", "AMZN", "APTV", "AZO", "BBY", "BKNG", "CZR", "KMX", "CCL", "CMG", "DRI",
    "DECK", "DPZ", "DASH", "DHI", "EBAY", "EXPE", "F", "GRMN", "GM", "GPC", "HAS",
    "HLT", "HD", "LVS", "LEN", "LKQ", "LOW", "LULU", "MAR", "MCD", "MGM", "MHK",
    "NKE", "NCLH", "NVR", "ORLY", "POOL", "PHM", "RL", "ROST", "RCL", "SBUX", "TPR",
    "TSLA", "TJX", "TSCO", "ULTA", "WSM", "WYNN", "YUM",

    # Consumer Staples
    "MO", "ADM", "BF.B", "BG", "CPB", "CHD", "CLX", "KO", "CL", "CAG", "STZ", "COST",
    "DG", "DLTR", "EL", "GIS", "HSY", "HRL", "K", "KVUE", "KDP", "KMB", "KHC", "KR",
    "LW", "MKC", "TAP", "MDLZ", "MNST", "PEP", "PM", "PG", "SJM", "SYY", "TGT", "TSN",
    "WBA", "WMT",

    # Energy
    "APA", "BKR", "CVX", "COP", "CTRA", "DVN", "FANG", "EOG", "EQT", "EXE", "XOM",
    "HAL", "KMI", "MPC", "OXY", "OKE", "PSX", "SLB", "TRGP", "TPL", "VLO", "WMB",

    # Financials
    "AFL", "ALL", "AXP", "AIG", "AMP", "AON", "APO", "ACGL", "AJG", "AIZ", "BAC",
    "BRK.B", "BLK", "BX", "XYZ", "BK", "BRO", "COF", "CBOE", "SCHW", "CB", "CINF",
    "C", "CFG", "CME", "COIN", "CPAY", "ERIE", "EG", "FDS", "FIS", "FITB", "FI",
    "BEN", "GPN", "GL", "GS", "HIG", "HBAN", "ICE", "IVZ", "JKHY", "JPM", "KEY",
    "KKR", "L", "MTB", "MKTX", "MMC", "MA", "MET", "MCO", "MS", "MSCI", "NDAQ",
    "NTRS", "PYPL", "PNC", "PFG", "PGR", "PRU", "RJF", "RF", "SPGI", "STT", "SYF",
    "TROW", "TRV", "TFC", "USB", "V", "WRB", "WFC", "WTW",

    # Healthcare
    "ABT", "ABBV", "A", "ALGN", "AMGN", "BAX", "BDX", "TECH", "BIIB", "BSX", "BMY",
    "CAH", "COR", "CNC", "CRL", "CI", "COO", "CVS", "DHR", "DVA", "DXCM", "EW", "ELV",
    "GEHC", "GILD", "HCA", "HSIC", "HOLX", "HUM", "IDXX", "INCY", "PODD", "ISRG",
    "IQV", "JNJ", "LH", "LLY", "MCK", "MDT", "MRK", "MTD", "MRNA", "MOH", "PFE",
    "DGX", "REGN", "RMD", "RVTY", "SOLV", "STE", "SYK", "TMO", "UNH", "UHS", "VRTX",
    "VTRS", "WAT", "WST", "ZBH", "ZTS",

    # Industrials
    "MMM", "AOS", "ALLE", "AME", "ADP", "AXON", "BA", "BR", "BLDR", "CHRW", "CARR",
    "CAT", "CTAS", "CPRT", "CSX", "CMI", "DAY", "DE", "DAL", "DOV", "ETN", "EMR",
    "EFX", "EXPD", "FAST", "FDX", "FTV", "GE", "GEV", "GNRC", "GD", "HON", "HWM",
    "HUBB", "HII", "IEX", "ITW", "IR", "JBHT", "J", "JCI", "LHX", "LDOS", "LII", "LMT",
    "MAS", "NDSN", "NSC", "NOC", "ODFL", "OTIS", "PCAR", "PH", "PAYX", "PAYC", "PNR",
    "PWR", "RTX", "RSG", "ROK", "ROL", "SNA", "LUV", "SWK", "TXT", "TT", "TDG",
    "UBER", "UNP", "UAL", "UPS", "URI", "VLTO", "VRSK", "GWW", "WAB", "WM", "XYL",

    # Information Technology
    "ACN", "ADBE", "AMD", "AKAM", "APH", "ADI", "AAPL", "AMAT", "ANET", "ADSK", "AVGO",
    "CDNS", "CDW", "CSCO", "CTSH", "GLW", "CRWD", "DDOG", "DELL", "ENPH", "EPAM",
    "FFIV", "FICO", "FSLR", "FTNT", "IT", "GEN", "GDDY", "HPE", "HPQ", "IBM", "INTC",
    "INTU", "JBL", "KEYS", "KLAC", "LRCX", "MCHP", "MU", "MSFT", "MPWR", "MSI",
    "NTAP", "NVDA", "NXPI", "ON", "ORCL", "PLTR", "PANW", "PTC", "QCOM", "ROP", "CRM",
    "STX", "NOW", "SWKS", "SMCI", "SNPS", "TEL", "TDY", "TER", "TXN", "TRMB", "TYL",
    "VRSN", "WDC", "WDAY", "ZBRA",

    # Materials
    "APD", "ALB", "AMCR", "AVY", "BALL", "CF", "CTVA", "DOW", "DD", "EMN", "ECL",
    "FCX", "IFF", "IP", "LIN", "LYB", "MLM", "MOS", "NEM", "NUE", "PKG", "PPG", "SHW",
    "SW", "STLD", "VMC",

    # Real Estate
    "ARE", "AMT", "AVB", "BXP", "CPT", "CBRE", "CSGP", "CCI", "DLR", "EQIX", "EQR",
    "ESS", "EXR", "FRT", "DOC", "HST", "INVH", "IRM", "KIM", "MAA", "PLD", "PSA", "O",
    "REG", "SBAC", "SPG", "UDR", "VTR", "VICI", "WELL", "WY",

    # Utilities
    "AES", "LNT", "AEE", "AEP", "AWK", "ATO", "CNP", "CMS", "ED", "CEG", "D", "DTE",
    "DUK", "EIX", "ETR", "EVRG", "ES", "EXC", "FE", "NEE", "NI", "NRG", "PCG", "PNW",
    "PPL", "PEG", "SRE", "SO", "VST", "WEC", "XEL"
]


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
        batch_size: Optional[int] = None,
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
        data_source: Optional[YFinanceDataSource] = None,
        enabled: bool = True,
        notes: Optional[str] = None,
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
            batch_size: Batch size for training (optional, defaults to 32). Not used for TabPFN models.
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
            data_source: Optional DataSource instance (defaults to YFinanceDataSource)
            enabled: Toggle to include/exclude this config when training
            notes: Optional string describing the intent/requirements for this config
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
        self.data_source = data_source if data_source is not None else YFinanceDataSource()
        self.enabled = enabled
        self.notes = notes
        self.kwargs = kwargs
    
    def create_trainer_config(self) -> TrainerConfig:
        """Create a TrainerConfig from this model training config."""
        # For TabPFN models, batch_size is not used, so set a dummy value if None
        # For other models, use default of 32 if not specified
        effective_batch_size = self.batch_size if self.batch_size is not None else 32
        return TrainerConfig(
            stocks=self.stocks,
            time_args=self.time_args,
            batch_size=effective_batch_size,
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
            data_source=self.data_source,
            **self.kwargs
        )


def create_model_configs() -> List[ModelTrainingConfig]:
    """
    Create a list of model training configurations.
    
    Add or modify configurations here to train different models.
    """
    configs = []
    
    # Shared stock/time splits
    stock_tiers = {
        "large": list(LARGE_STOCKS),  # Full S&P-like universe from download_data
        "base": [
            # Core diversified basket (≈34 names) from download_data.py doc block
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",
            "JPM", "BAC", "V", "MA", "WFC", "GS", "BLK", "AXP",
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO",
            "WMT", "PG", "HD", "COST", "MCD", "NKE",
            "BA", "CAT", "XOM", "CVX"
        ],
        "small": [
            # Growth-heavy subset between base and micro tiers
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AVGO", "ORCL"
        ],
        "micro": ["AAPL", "MSFT", "NVDA"]
    }
    large_stocks = stock_tiers["large"]
    base_stocks = stock_tiers["base"]
    small_stocks = stock_tiers["small"]
    micro_stocks = stock_tiers["micro"]
    
    short_history = ["1990-01-01", "1999-01-01"]
    long_history = ["1990-01-01", "2015-12-31"]
   
    # ---------------------------------------------------------------------
    # 1. Base LSTM (no NLP features)
    # ---------------------------------------------------------------------
    configs.append(ModelTrainingConfig(
        name="lstm_base",
        model_type="LSTM",
        model_config=LSTMConfig(parameters={
            'input_shape': (31, 3),
            'hidden_size': 25,
            'num_layers': 1,
            'dropout': 0.1
        }),
        stocks=large_stocks,
        time_args=long_history,
        batch_size=64,
        num_epochs=1000,
        period_type="LS",
        lookback=240,
        use_nlp=False,
        nlp_method=None
    ))
    
    # ---------------------------------------------------------------------
    # 2. Base LSTM + aggregated NLP
    # ---------------------------------------------------------------------
    configs.append(ModelTrainingConfig(
        name="lstm_base_nlp",
        model_type="LSTM",
        model_config=LSTMConfig(parameters={
            'input_shape': (31, 13),  # 3 price + 10 aggregated NLP features
            'hidden_size': 25,
            'num_layers': 1,
            'dropout': 0.1
        }),
        stocks=large_stocks,
        time_args=long_history,
        batch_size=64,
        num_epochs=1000,
        period_type="LS",
        lookback=240,
        use_nlp=True,
        nlp_method="aggregated"
    ))
    
    # ---------------------------------------------------------------------
    # 3. Base AELSTM (no NLP)
    # ---------------------------------------------------------------------
    configs.append(ModelTrainingConfig(
        name="aelstm_base",
        model_type="AELSTM",
        model_config=AELSTMConfig(parameters={
            'input_shape': (31, 3),
            'hidden_size': 25,
            'num_layers': 1,
            'dropout': 0.1
        }),
        stocks=large_stocks,
        time_args=long_history,
        batch_size=64,
        num_epochs=1000,
        period_type="LS",
        lookback=240,
        use_nlp=False,
        nlp_method=None
    ))
    
    # ---------------------------------------------------------------------
    # 4. Base AELSTM + aggregated NLP
    # ---------------------------------------------------------------------
    configs.append(ModelTrainingConfig(
        name="aelstm_base_nlp",
        model_type="AELSTM",
        model_config=AELSTMConfig(parameters={
            'input_shape': (31, 13),
            'hidden_size': 25,
            'num_layers': 1,
            'dropout': 0.1
        }),
        stocks=large_stocks,
        time_args=long_history,
        batch_size=64,
        num_epochs=1000,
        period_type="LS",
        lookback=240,
        use_nlp=True,
        nlp_method="aggregated"
    ))
 
    """
    
    # ---------------------------------------------------------------------
    # 6. TabFPN + individual NLP on smaller dataset (placeholder, disabled)
    # ---------------------------------------------------------------------
    small_tabfpn_note = (
        "TabPFN portfolio with individual NLP features. Keep per-stock samples <=50k rows."
    )
    configs.append(ModelTrainingConfig(
        name="tabfpn_nlp_portfolio_small",
        model_type="TabPFN",
        model_config=TabPFNConfig(parameters={
            'backend': 'client',
            'max_samples': 50000,
            'random_state': 42
        }),
        stocks=large_stocks,
        time_args=long_history,
        num_epochs=2,
        period_type="LS",
        lookback=240,
        use_nlp=True,
        nlp_method="aggregated",
        enabled=True,
        notes=small_tabfpn_note
    ))

    # ---------------------------------------------------------------------
    # 7. Portfolio architecture (independent per-stock LSTMs) - disabled
    # ---------------------------------------------------------------------
    shared_portfolio_base = LSTMConfig(parameters={
        'input_shape': (31, 13),
        'hidden_size': 48,
        'num_layers': 1,
        'dropout': 0.1
    })
    configs.append(ModelTrainingConfig(
        name="portfolio_lstm_independent",
        model_type="Portfolio",
        model_config=PortfolioConfig(parameters={
            'stocks': micro_stocks,
            'base_model_type': 'LSTM',
            'base_model_config': shared_portfolio_base,
            'strategy': 'independent',
            'mlp_hidden_dims': [128],
            'embedding_dim': 32,
            'use_stock_embeddings': False,
            'dropout': 0.1,
        }),
        stocks=micro_stocks,
        time_args=long_history,
        batch_size=64,
        num_epochs=2,
        period_type="LS",
        lookback=240,
        use_nlp=True,
        nlp_method="aggregated",
        enabled=True,
        notes="Builds one LSTM backbone per stock and feeds their outputs into a shared MLP head."
    ))
    
    # ---------------------------------------------------------------------
    # 8. Portfolio architecture (shared backbone + embeddings) - disabled
    # ---------------------------------------------------------------------
    configs.append(ModelTrainingConfig(
        name="portfolio_lstm_shared",
        model_type="Portfolio",
        model_config=PortfolioConfig(parameters={
            'stocks': micro_stocks,
            'base_model_type': 'LSTM',
            'base_model_config': shared_portfolio_base,
            'strategy': 'shared',
            'mlp_hidden_dims': [128],
            'embedding_dim': 64,
            'use_stock_embeddings': True,
            'dropout': 0.15,
        }),
        stocks=large_stocks,
        time_args=long_history,
        batch_size=64,
        num_epochs=2,
        period_type="LS",
        lookback=240,
        use_nlp=True,
        nlp_method="aggregated",
        enabled=True,
        notes="Shared LSTM backbone across stocks with learnable embeddings before the MLP portfolio head."
    ))
     """
    
    # ---------------------------------------------------------------------
    # 7. TimesNet + aggregated NLP on smaller dataset
    # ---------------------------------------------------------------------
    configs.append(ModelTrainingConfig(
        name="timesnet_small_agg",
        model_type="TimesNet",
        model_config=TimesNetConfig(parameters={
            'input_shape': (240, 13),  # full window with NLP
            'task_name': 'classification',
            'enc_in': 13,
            'num_class': 3,
            'd_model': 256,
            'd_ff': 1024,
            'e_layers': 2,
            'top_k': 5,
            'num_kernels': 6,
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': 'd'
        }),
        stocks=large_stocks,
        time_args=long_history,
        batch_size=64,
        num_epochs=1000,
        period_type="LS",
        lookback=240,
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
    if not config.enabled:
        raise ValueError(f"Config '{config.name}' is disabled. Enable it before training.")
    
    print("\n" + "=" * 80)
    print(f"Training Model: {config.name}")
    print("=" * 80)
    print(f"Model Type: {config.model_type}")
    print(f"Stocks: {config.stocks}")
    print(f"Time Range: {config.time_args}")
    # Batch size is not relevant for TabPFN models
    if config.model_type.upper() == "TABPFN":
        print(f"Batch Size: N/A (not used for TabPFN), Epochs: {config.num_epochs}")
    else:
        batch_size_str = str(config.batch_size) if config.batch_size is not None else "32 (default)"
        print(f"Batch Size: {batch_size_str}, Epochs: {config.num_epochs}")
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
    enabled_count = sum(1 for cfg in configs if cfg.enabled)
    skipped_count = len(configs) - enabled_count
    print(f"Total models defined: {len(configs)}")
    print(f"Models scheduled to train: {enabled_count}")
    if skipped_count:
        print(f"Models skipped (disabled): {skipped_count}")
    print(f"Log directory: {log_dir}")
    print(f"Continue on error: {continue_on_error}")
    print(f"{'=' * 80}\n")
    
    results = []
    session_start = time.time()
    
    for i, config in enumerate(configs, 1):
        status_prefix = "[SKIP]" if not config.enabled else "[RUN]"
        print(f"\n[{i}/{len(configs)}] {status_prefix} {config.name}")
        
        if not config.enabled:
            reason = config.notes or "Disabled via configuration"
            print(f"  ↳ Skipping (disabled). Reason: {reason}")
            results.append({
                'name': config.name,
                'model_type': config.model_type,
                'success': None,
                'error': None,
                'training_time': None,
                'skipped': True,
                'reason': reason
            })
            continue
        
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
    successful = sum(1 for r in results if r.get('success'))
    failed = sum(1 for r in results if r.get('success') is False)
    skipped = sum(1 for r in results if r.get('skipped'))
    
    print(f"\n{'=' * 80}")
    print(f"Training Session Complete")
    print(f"{'=' * 80}")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
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
        status = "enabled" if cfg.enabled else "disabled"
        extra = f" | notes: {cfg.notes}" if cfg.notes else ""
        print(f"  - {cfg.name} ({cfg.model_type}) [{status}]{extra}")
    print()
    
    # Optionally override number of epochs for quick tests via env var
    override_epochs = os.environ.get("QUICK_TEST_EPOCHS")
    if override_epochs:
        try:
            override_epochs = int(override_epochs)
            print(f"\n[quick-test] Overriding num_epochs to {override_epochs} for all configs\n")
            for cfg in model_configs:
                cfg.num_epochs = override_epochs
        except ValueError:
            print(f"[quick-test] Invalid QUICK_TEST_EPOCHS value: {override_epochs}. Ignoring override.")

    # Train all models
    results = train_all_models(
        configs=model_configs,
        log_dir="training_logs",
        continue_on_error=True  # Continue training other models if one fails
    )
    
    # Print final summary
    print("\nFinal Results:")
    for result in results:
        if result.get('skipped'):
            print(f"  - {result['name']}: skipped ({result.get('reason', 'disabled')})")
            continue
        status = "✓" if result.get('success') else "✗"
        time_str = f"{result['training_time']:.2f}s" if result.get('training_time') else "N/A"
        print(f"  {status} {result['name']}: {time_str}")
        if result.get('success') and result.get('saved_model'):
            print(f"    Saved to: {result['saved_model']}")
        if not result.get('success'):
            print(f"    Error: {result.get('error')}")
