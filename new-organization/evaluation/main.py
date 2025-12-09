"""
Main script for evaluating multiple trained models based on config objects.

This script allows you to define multiple evaluation configurations and evaluate them sequentially.
Each evaluation configuration specifies a model path, model type, and evaluation parameters.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict, Optional

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import from parent directory's main module (after path is set up)
from main import ModelTrainingConfig, CAELSTMConfig, AELSTMConfig, TimesNetConfig

from evaluation.evaluator import ModelEvaluator
from evaluation.configs.evaluation_config import EvaluationConfig
import util

# Define results directory path (deliverables/results/)
project_root = os.path.dirname(parent_dir)
RESULTS_DIR = os.path.join(project_root, "deliverables", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Boolean to choose mapping file location
USE_DELIVERABLES_MAPPING = True  # True for deliverables/models/, False for trained_models/models/

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

configs = []

# Common stock list (can be customized)
common_stocks = [
    # Communication Services
    "GOOGL", "GOOG", "T", "CHTR", "CMCSA", "EA", "FOXA", "FOX", "IPG", "LYV", "MTCH", "META", "NFLX", "NWSA", "NWS", "OMC", "PSKY", "TMUS", "TTWO", "TKO", "TTD", "VZ", "DIS", "WBD",
    # Consumer Discretionary
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

def load_model_mapping(use_deliverables: bool = True) -> Dict:
    """
    Load the model mapping file from the specified location.
    Reuses logic from util._load_model_mapping() but allows choosing location.
    
    Args:
        use_deliverables: If True, load from deliverables/models/, else from trained_models/models/
    
    Returns:
        Dictionary mapping model_id -> {'config_class': str, 'parameters': dict, 'full_name': str}
    
    Exits:
        If mapping file doesn't exist, exits with error message
    """
    if use_deliverables:
        # Use deliverables/models/ directory
        models_dir = os.path.join(project_root, "deliverables", "models")
    else:
        # Use trained_models/models/ directory (via util)
        models_dir = util.MODELS_DIR
    
    mapping_path = os.path.join(models_dir, "_model_mapping.json")
    
    if not os.path.exists(mapping_path):
        print(f"Error: Model mapping file not found at: {mapping_path}")
        print(f"Please ensure the mapping file exists or set USE_DELIVERABLES_MAPPING appropriately.")
        sys.exit(1)
    
    # Reuse loading logic from util._load_model_mapping()
    try:
        with open(mapping_path, 'r') as f:
            content = f.read()
            if not content.strip():
                print(f"Warning: Model mapping file is empty: {mapping_path}")
                return {}
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error: Could not load model mapping (JSON decode error): {e}")
        print(f"  File: {mapping_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Could not load model mapping: {e}")
        print(f"  File: {mapping_path}")
        sys.exit(1)


def find_matching_model(model_config, mapping: Dict, models_dir: str) -> Optional[tuple]:
    """
    Find a matching model in the mapping file given a model config.
    Reuses matching logic from util.find_model_by_config().
    
    Args:
        model_config: Model configuration object to match
        mapping: Dictionary mapping model_id -> config info
        models_dir: Directory where model files are stored
    
    Returns:
        Tuple (model_id, model_path, mapping_data) if match found, None otherwise
    """
    # Get config class name
    config_class_name = model_config.__class__.__name__
    
    # Extract parameters from input config using util._config_to_dict() to match saved format
    # This ensures nested configs are converted to the same __config_class__/__config_params__ format
    if hasattr(model_config, 'parameters'):
        input_params = model_config.parameters
        if not isinstance(input_params, dict):
            # Build dict from __dict__ and convert nested configs
            input_params = {k: util._config_to_dict(v) for k, v in model_config.__dict__.items() if not k.startswith('_')}
        else:
            # Convert nested configs in parameters dict
            input_params = util._config_to_dict(input_params)
    else:
        # Build dict from __dict__ and convert nested configs
        input_params = {k: util._config_to_dict(v) for k, v in model_config.__dict__.items() if not k.startswith('_')}
    
    # Normalize parameters for comparison (convert to JSON-serializable format)
    # Reuse normalization logic from util.find_model_by_config()
    def normalize_value(v):
        """Convert value to JSON-serializable format for comparison."""
        if isinstance(v, (list, tuple)):
            return tuple(normalize_value(item) for item in v)
        elif isinstance(v, dict):
            # Handle the special __config_class__/__config_params__ format used by _config_to_dict
            if '__config_class__' in v and '__config_params__' in v:
                # This is a nested config - normalize both the class name and params
                return (v['__config_class__'], normalize_value(v['__config_params__']))
            else:
                # Regular dict - recursively normalize values
                return {k: normalize_value(val) for k, val in sorted(v.items())}
        elif isinstance(v, (int, float, str, bool, type(None))):
            return v
        else:
            return str(v)
    
    input_params_normalized = {k: normalize_value(v) for k, v in sorted(input_params.items())}
    
    # Search for matching config in mapping
    for model_id, cached_info in mapping.items():
        cached_config_class = cached_info.get('config_class', '')
        cached_params = cached_info.get('parameters', {})
        
        # Step 1: Check if config class types match
        if cached_config_class != config_class_name:
            continue
        
        # Step 2: Check if all parameters are equivalent
        cached_params_normalized = {k: normalize_value(v) for k, v in sorted(cached_params.items())}
        
        if input_params_normalized == cached_params_normalized:
            # Found matching config - verify model file exists
            model_path = os.path.join(models_dir, f"{model_id}.pth")
            if os.path.exists(model_path):
                return (model_id, model_path, cached_info)
            else:
                # Model file doesn't exist, but config matches
                print(f"Warning: Model config matches but file not found: {model_path}")
    
    # No matching model found
    return None


# Evaluation-specific parameters (not in model config)
# These can be reused for multiple model evaluations
EVALUATION_PARAMS = {
    'time_args': ["1990-01-01", "2015-12-31"],
    'batch_size': 32,
    'k': 10,
    'cost_bps_per_side': 5.0,
    'use_nlp': True,
    'nlp_method': "aggregated",
    'create_plots': True,
    'log_dir': "runs/evaluation",
}


def create_evaluation_config_from_mapping(
    model_config, 
    mapping_data: Dict, 
    eval_params: Dict,
    model_path: str
) -> EvaluationConfig:
    """
    Create EvaluationConfig from matched mapping data and evaluation parameters.
    
    Args:
        model_config: Original model config object
        mapping_data: Mapping data from _model_mapping.json
        eval_params: Evaluation-specific parameters dict
        model_path: Path to the model file
    
    Returns:
        EvaluationConfig object
    """
    # Extract parameters from mapping_data
    config_class_name = mapping_data.get('config_class', '')
    parameters = mapping_data.get('parameters', {})
    
    # Determine model_type from config_class name
    model_type = config_class_name.replace('Config', '').lower()
    
    # Extract stocks if available (for PortfolioConfig)
    stocks = eval_params.get('stocks')  # Default from eval_params
    if 'stocks' in parameters and isinstance(parameters['stocks'], list):
        stocks = parameters['stocks']
    
    # Extract input_shape if available
    input_shape = None
    if 'input_shape' in parameters:
        input_shape = parameters['input_shape']
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)
    
    # Create EvaluationConfig
    return EvaluationConfig(
        model_path=model_path,
        model_type=model_type,
        stocks=stocks if stocks else eval_params.get('stocks', []),
        time_args=eval_params.get('time_args', ["1990-01-01", "2015-12-31"]),
        log_dir=eval_params.get('log_dir', "runs/evaluation"),
        use_nlp=eval_params.get('use_nlp', True),
        nlp_method=eval_params.get('nlp_method', "aggregated"),
        input_shape=input_shape,
        batch_size=eval_params.get('batch_size', 32),
        k=eval_params.get('k', 10),
        cost_bps_per_side=eval_params.get('cost_bps_per_side', 5.0),
        create_plots=eval_params.get('create_plots', True),
    )


def evaluate_model(config: EvaluationConfig, log_dir: Optional[str] = None) -> Dict:
    """
    Evaluate a single model configuration.
    
    Args:
        config: EvaluationConfig instance
        log_dir: Optional override for log directory (uses config.log_dir if None)
    
    Returns:
        Dictionary with evaluation results and metadata
    """
    print("\n" + "=" * 80)
    print(f"Evaluating Model: {os.path.basename(config.model_path)}")
    print("=" * 80)
    print(f"Model Type: {config.model_type}")
    print(f"Model Path: {config.model_path}")
    print(f"Stocks: {len(config.stocks)} stocks")
    print(f"Time Range: {config.time_args}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Portfolio k: {config.k}, Cost: {config.cost_bps_per_side} bps")
    print(f"NLP: {config.use_nlp} ({config.nlp_method if config.use_nlp else 'N/A'})")
    print(f"Create Plots: {config.create_plots}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    result = {
        'model_path': config.model_path,
        'model_type': config.model_type,
        'start_time': datetime.now().isoformat(),
        'success': False,
        'error': None,
        'evaluation_time': None,
        'results_file': None,
        'metrics': None
    }
    
    evaluator = None
    try:
        # Use provided log_dir or config's log_dir
        actual_log_dir = log_dir if log_dir is not None else config.log_dir
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=config.model_path,
            stocks=config.stocks,
            time_args=config.time_args,
            log_dir=actual_log_dir,
            device=config.device,
            use_nlp=config.use_nlp,
            nlp_method=config.nlp_method,
            model_type=config.model_type.lower(),  # ModelEvaluator expects lowercase
            input_shape=config.input_shape
        )
        
        # Run comprehensive evaluation
        print("[Evaluation] Starting comprehensive evaluation...")
        metrics = evaluator.evaluate_all_metrics(
            batch_size=config.batch_size,
            create_plots=config.create_plots,
            k=config.k,
            cost_bps_per_side=config.cost_bps_per_side,
            use_paper_aligned=True
        )
        
        # Generate results filename
        model_name = os.path.basename(config.model_path).replace('.pth', '').replace('.pt', '')
        results_filename = f"evaluation_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file = os.path.join(RESULTS_DIR, results_filename)
        
        # Save results
        evaluator.save_results(results_file)
        
        result['success'] = True
        result['evaluation_time'] = time.time() - start_time
        result['results_file'] = results_file
        result['metrics'] = metrics
        
        print(f"\n✓ Successfully evaluated {os.path.basename(config.model_path)}")
        print(f"  Evaluation time: {result['evaluation_time']:.2f} seconds")
        print(f"  Results saved to: {results_file}")
        print(f"  TensorBoard logs: {actual_log_dir}")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        result['evaluation_time'] = time.time() - start_time
        
        print(f"\n✗ Failed to evaluate {os.path.basename(config.model_path)}")
        print(f"  Error: {e}")
        print(f"  Time elapsed: {result['evaluation_time']:.2f} seconds")
        
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up evaluator
        if evaluator is not None:
            try:
                evaluator.close()
            except Exception as e:
                print(f"Warning: Error closing evaluator: {e}")
    
    return result


def evaluate_all_models(
    configs: List[EvaluationConfig],
    log_dir: Optional[str] = None,
    continue_on_error: bool = True
) -> List[Dict]:
    """
    Evaluate multiple model configurations sequentially.
    
    Args:
        configs: List of EvaluationConfig instances
        log_dir: Optional override for log directory (uses config.log_dir if None)
        continue_on_error: Whether to continue evaluating other models if one fails
    
    Returns:
        List of result dictionaries, one per model
    """
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'=' * 80}")
    print(f"Starting Evaluation Session")
    print(f"{'=' * 80}")
    print(f"Total models to evaluate: {len(configs)}")
    if log_dir:
        print(f"Log directory: {log_dir}")
    print(f"Continue on error: {continue_on_error}")
    print(f"{'=' * 80}\n")
    
    results = []
    session_start = time.time()
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Processing: {os.path.basename(config.model_path)}")
        
        try:
            result = evaluate_model(config, log_dir=log_dir)
            results.append(result)
            
            if not result['success'] and not continue_on_error:
                print(f"\nStopping evaluation due to error in {os.path.basename(config.model_path)}")
                break
                
        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted by user")
            break
        except Exception as e:
            print(f"\nUnexpected error processing {os.path.basename(config.model_path)}: {e}")
            results.append({
                'model_path': config.model_path,
                'model_type': config.model_type,
                'success': False,
                'error': str(e),
                'evaluation_time': None
            })
            if not continue_on_error:
                break
    
    # Summary
    session_time = time.time() - session_start
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'=' * 80}")
    print(f"Evaluation Session Complete")
    print(f"{'=' * 80}")
    print(f"Total models: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {session_time:.2f} seconds ({session_time/60:.2f} minutes)")
    print(f"{'=' * 80}\n")
    
    # Save results summary
    summary_filename = f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path = os.path.join(RESULTS_DIR, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump({
            'session_start': datetime.fromtimestamp(session_start).isoformat(),
            'session_time': session_time,
            'total_models': len(results),
            'successful': successful,
            'failed': failed,
            'results': results
        }, f, indent=2)
    
    print(f"Results summary saved to: {summary_path}")
    
    return results


def get_evaluation_configs(
    model_configs: List,
    eval_params: Dict = None,
    use_deliverables_mapping: bool = True
) -> List[EvaluationConfig]:
    """
    Create a list of evaluation configurations by matching model configs against mapping file.
    
    Args:
        model_configs: List of model config objects to evaluate (e.g., [PortfolioConfig(...), LSTMConfig(...)])
        eval_params: Dictionary of evaluation-specific parameters (defaults to EVALUATION_PARAMS)
        use_deliverables_mapping: Whether to use deliverables/models/ or trained_models/models/ mapping file
    
    Returns:
        List of EvaluationConfig objects
    """
    if eval_params is None:
        eval_params = EVALUATION_PARAMS
    
    # Load mapping file
    mapping = load_model_mapping(use_deliverables=use_deliverables_mapping)
    
    # Determine models directory
    if use_deliverables_mapping:
        models_dir = os.path.join(project_root, "deliverables", "models")
    else:
        models_dir = util.MODELS_DIR
    
    configs = []
    
    # Process each model config
    for model_config in model_configs:
        # Find matching model in mapping
        match_result = find_matching_model(model_config, mapping, models_dir)
        
        if match_result is None:
            print(f"Warning: No matching model found for config: {model_config.__class__.__name__}")
            print(f"  Skipping this model configuration.")
            continue
        
        model_id, model_path, mapping_data = match_result
        print(f"Found matching model: {model_id} at {model_path}")
        
        # Create EvaluationConfig from mapping data and eval params
        eval_config = create_evaluation_config_from_mapping(
            model_config=model_config,
            mapping_data=mapping_data,
            eval_params=eval_params,
            model_path=model_path
        )
        
        configs.append(eval_config)
    
    return configs


def get_evaluation_configs_legacy() -> List[EvaluationConfig]:
    """
    Legacy function: Create a list of evaluation configurations using models from deliverables/models/.
    
    This uses the actual model files created by test.sh and the training scripts.
    Uses the same parameters as the training configs from main.py.
    """
    configs = []
    
    # Get stocks and time_args from main.py (same as training)

    
    # Models directory (deliverables/models/)
    models_dir = os.path.join(project_root, "deliverables", "models")

    
    # Model 3: LSTM Base (lstm_base.pth)
    lstmpath = os.path.join(models_dir, "lstm_base.pth")
    if os.path.exists(lstmpath):
        configs.append(EvaluationConfig(
            model_path=lstmpath,
            model_type="lstm",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=False,
            nlp_method=None,
            create_plots=True,
            log_dir="deliverables/evaluation"
        ))

    
    # Model 1: AELSTM Base (aelstm_base.pth)
    aelstm_path = os.path.join(models_dir, "aelstm_base.pth")
    if os.path.exists(aelstm_path):
        configs.append(EvaluationConfig(
            model_path=aelstm_path,
            model_type="aelstm",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=False,
            nlp_method=None,
            create_plots=True,
            log_dir="deliverables/evaluation"
        ))
    
    # Model 2: CAELSTM Base (caelstm_base.pth)
    caelstm_path = os.path.join(models_dir, "caelstm_base.pth")
    if os.path.exists(caelstm_path):
        configs.append(EvaluationConfig(
            model_path=caelstm_path,
            model_type="caelstm",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=False,
            nlp_method=None,
            create_plots=True,
            log_dir="deliverables/evaluation"
        ))

    # Model 3: CAELSTM NLP (caelstm_nlp.pth)
    caelstm_nlp_path = os.path.join(models_dir, "caelstm_nlp.pth")
    if os.path.exists(caelstm_nlp_path):
        configs.append(EvaluationConfig(
            model_path=caelstm_nlp_path,
            model_type="caelstm",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=True,
            nlp_method="aggregated",
            create_plots=True,
            log_dir="deliverables/evaluation"
        ))
    
    # Model 2: AELSTM nlp (aelstm_nlp.pth)
    aelstm_nlp_path = os.path.join(models_dir, "aelstm_nlp.pth")
    if os.path.exists(aelstm_nlp_path):
        configs.append(EvaluationConfig(
            model_path=aelstm_nlp_path,
            model_type="aelstm",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=True,
            nlp_method="aggregated",
            create_plots=True,
            log_dir="deliverables/evaluation"
        ))
    
    # Model 4: LSTM NLP (lstm_nlp.pth)
    lstm_nlp_path = os.path.join(models_dir, "lstm_nlp.pth")
    if os.path.exists(lstm_nlp_path):
        configs.append(EvaluationConfig(
            model_path=lstm_nlp_path,
            model_type="lstm",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=True,
            nlp_method=None,
            create_plots=True,
            log_dir="deliverables/evaluation",
            input_shape=(31,13)
        ))
    
    
    # ---------------------------------------------------------------------
    # 9. TimesNet Base
    # ---------------------------------------------------------------------
    timesnet_path = os.path.join(models_dir, "timesnet_base.pth")
    if os.path.exists(timesnet_path):
        configs.append(EvaluationConfig(
            model_path=timesnet_path,
            model_type="timesnet",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=False,
            nlp_method=None,
            create_plots=True,
            log_dir="deliverables/evaluation",
            input_shape=(31,3)
        ))

    # ---------------------------------------------------------------------
    # 10. Portfolio Shared LSTM Base
    # ---------------------------------------------------------------------
    portfolio_shared_lstm_path = os.path.join(models_dir, "portfolio_shared_lstm_base.pth")
    if os.path.exists(portfolio_shared_lstm_path):
        configs.append(EvaluationConfig(
            model_path=portfolio_shared_lstm_path,
            model_type="portfolio_shared_lstm",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=False,
            nlp_method=None,
            create_plots=True,
            log_dir="deliverables/evaluation",
            input_shape=(31,3)
        ))

    # ---------------------------------------------------------------------
    # 11. Portfolio Independent LSTM Base
    # ---------------------------------------------------------------------
    portfolio_independent_lstm_path = os.path.join(models_dir, "portfolio_independent_lstm_base.pth")
    if os.path.exists(portfolio_independent_lstm_path):
        configs.append(EvaluationConfig(
            model_path=portfolio_independent_lstm_path,
            model_type="portfolio_independent_lstm",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=False,
            nlp_method=None,
            create_plots=True,
            log_dir="deliverables/evaluation",
            input_shape=(31,3)
        ))

    """
    # ---------------------------------------------------------------------
    # 12. LSTM Full Base
    # ---------------------------------------------------------------------
    lstm_full_base_path = os.path.join(models_dir, "lstm_full_base.pth")
    if os.path.exists(lstm_full_base_path):
        configs.append(EvaluationConfig(
            model_path=lstm_full_base_path,
            model_type="lstm",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=False,
            nlp_method=None,
            create_plots=True,
            log_dir="deliverables/evaluation",
            input_shape=(31,3)
        ))

    # ---------------------------------------------------------------------
    # 13. LSTM Full NLP (aggregated)
    # ---------------------------------------------------------------------
    lstm_full_nlp_path = os.path.join(models_dir, "lstm_full_nlp.pth")
    if os.path.exists(lstm_full_nlp_path):
        configs.append(EvaluationConfig(
            model_path=lstm_full_nlp_path,
            model_type="lstm",
            stocks=large_stocks,
            time_args=long_history,
            batch_size=64,
            k=10,
            cost_bps_per_side=5.0,
            use_nlp=True,
            nlp_method="aggregated",
            create_plots=True,
            log_dir="deliverables/evaluation",
            input_shape=(31,13)
        ))
    """

    return configs


# Hard-coded configuration variables
# Modify these variables to customize evaluation behavior

# Model configs to evaluate (define your model configs here)
# Example:
# from models.configs import PortfolioConfig, LSTMConfig
# MODEL_CONFIGS_TO_EVALUATE = [
#     PortfolioConfig(parameters={'stocks': [...], 'base_model_type': 'LSTM', ...}),
#     LSTMConfig(parameters={'input_shape': (31, 13), 'hidden_size': 25, ...}),
# ]
configs = []


# ---------------------------------------------------------------------
# 5. CAELSTM Base
# ---------------------------------------------------------------------
configs.append(ModelTrainingConfig(
    name="caelstm_base",
    model_type="CAELSTM",
    model_config=CAELSTMConfig(parameters={
        'input_shape': (31, 3),
        'kernel_size': 3,
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
    nlp_method=None,
    enabled=True,
)) # Set to list of model configs, or None to use legacy mode

# ---------------------------------------------------------------------
# 3. AELSTM Base
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
# 9. TimesNet Base
# ---------------------------------------------------------------------
configs.append(ModelTrainingConfig(
    name="timesnet_base",
    model_type="TimesNet",
    model_config=TimesNetConfig(parameters={
        'input_shape': (31, 3),
        'task_name': 'classification',
        'seq_len': None,
        'enc_in': 3,
        'num_class': 3,
        'd_model': 256,
        'd_ff': 1024,
        'e_layers': 2,
        'top_k': 3,
        'num_kernels': 3,
        'embed': 'timeF',
        'freq': 'd',
        'dropout': 0.1,
        'pred_len': 0,
        'label_len': 0,
        'c_out': None,
        'freeze_encoder': False,
    }),
    stocks=large_stocks,
    time_args=long_history,
    batch_size=64,
    num_epochs=1000,
    period_type="LS",
    lookback=240,
    use_nlp=False,
    nlp_method=None,
    enabled=True,
))


# Set to None to use legacy evaluation mode (default)
# Set to a list of model configs to use mapping file mode
MODEL_CONFIGS_TO_EVALUATE = None  # Use legacy mode by default

# Evaluation settings
LOG_DIR = "runs/evaluation"
CONTINUE_ON_ERROR = True
CREATE_PLOTS = True

# Default evaluation parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_K = 10
DEFAULT_COST_BPS_PER_SIDE = 5.0
DEFAULT_USE_NLP = False
DEFAULT_NLP_METHOD = "aggregated"

# Single model evaluation mode (set to None to use get_evaluation_configs() instead)
SINGLE_MODEL_PATH = None  # e.g., "savedmodel_classification.pth"
SINGLE_MODEL_TYPE = None  # e.g., "lstm"
SINGLE_MODEL_STOCKS = None  # e.g., ["AAPL", "MSFT", "GOOGL"]
SINGLE_MODEL_TIME_ARGS = None  # e.g., ["1989-12-01", "2015-09-30"]


if __name__ == "__main__":
    """
    Main entry point.
    
    Modify the hard-coded configuration variables at the top of this file
    to customize evaluation behavior.
    """
    
    # Determine evaluation mode
    # Default to legacy mode (use get_evaluation_configs_legacy)
    # Set MODEL_CONFIGS_TO_EVALUATE = None to use legacy mode
    USE_LEGACY_MODE = (MODEL_CONFIGS_TO_EVALUATE is None or len(MODEL_CONFIGS_TO_EVALUATE) == 0)
    
    if USE_LEGACY_MODE:
        # Legacy mode: Use hardcoded examples from get_evaluation_configs_legacy()
        print("Using legacy evaluation mode...")
        configs = get_evaluation_configs_legacy()
        print(f"Legacy evaluation mode: {len(configs)} models")
        
    elif MODEL_CONFIGS_TO_EVALUATE is not None:
        # New mode: Use model configs and mapping file
        print("Using model configs with mapping file...")
        
        # Prepare evaluation parameters
        eval_params = {
            'time_args': SINGLE_MODEL_TIME_ARGS if SINGLE_MODEL_TIME_ARGS else EVALUATION_PARAMS['time_args'],
            'batch_size': DEFAULT_BATCH_SIZE,
            'k': DEFAULT_K,
            'cost_bps_per_side': DEFAULT_COST_BPS_PER_SIDE,
            'use_nlp': DEFAULT_USE_NLP,
            'nlp_method': DEFAULT_NLP_METHOD,
            'create_plots': CREATE_PLOTS,
            'log_dir': LOG_DIR,
        }
        
        # If stocks specified for single model, use them
        if SINGLE_MODEL_STOCKS:
            eval_params['stocks'] = SINGLE_MODEL_STOCKS
        
        # Get evaluation configs from model configs
        configs = get_evaluation_configs(
            model_configs=MODEL_CONFIGS_TO_EVALUATE if isinstance(MODEL_CONFIGS_TO_EVALUATE, list) else [MODEL_CONFIGS_TO_EVALUATE],
            eval_params=eval_params,
            use_deliverables_mapping=USE_DELIVERABLES_MAPPING
        )
        
        if not configs:
            print("\nError: No valid evaluation configs created from model configs.")
            print("Please check that:")
            print("  1. Model configs are correctly defined")
            print("  2. Matching models exist in the mapping file")
            print("  3. Model files exist in the models directory")
            sys.exit(1)
        
        print(f"Created {len(configs)} evaluation config(s) from model configs")
        
    elif SINGLE_MODEL_PATH:
        # Single model evaluation mode
        if not SINGLE_MODEL_TYPE:
            print("Error: SINGLE_MODEL_TYPE must be set when SINGLE_MODEL_PATH is provided")
            sys.exit(1)
        
        # Use provided stocks or default
        stocks = SINGLE_MODEL_STOCKS if SINGLE_MODEL_STOCKS else ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        # Use provided time_args or default
        time_args = SINGLE_MODEL_TIME_ARGS if SINGLE_MODEL_TIME_ARGS else ["1989-12-01", "2015-09-30"]
        
        # Create single evaluation config
        config = EvaluationConfig(
            model_path=SINGLE_MODEL_PATH,
            model_type=SINGLE_MODEL_TYPE,
            stocks=stocks,
            time_args=time_args,
            batch_size=DEFAULT_BATCH_SIZE,
            k=DEFAULT_K,
            cost_bps_per_side=DEFAULT_COST_BPS_PER_SIDE,
            use_nlp=DEFAULT_USE_NLP,
            nlp_method=DEFAULT_NLP_METHOD,
            log_dir=LOG_DIR,
            create_plots=CREATE_PLOTS
        )
        
        configs = [config]
        print(f"Single model evaluation mode: {os.path.basename(SINGLE_MODEL_PATH)}")
        
    else:
        # Fallback to legacy mode if nothing else is set
        print("No evaluation mode specified, using legacy evaluation mode...")
        configs = get_evaluation_configs_legacy()
        print(f"Legacy evaluation mode: {len(configs)} models")
    
    # Filter configs to only those with existing model files
    valid_configs = []
    for config in configs:
        if os.path.exists(config.model_path):
            valid_configs.append(config)
        else:
            print(f"Warning: Model file not found: {config.model_path}")
            print(f"  Skipping evaluation for this model")
    
    if not valid_configs:
        print("\nError: No valid model files found for evaluation")
        print("Please check model paths or train models first")
        sys.exit(1)
    
    print(f"\nFound {len(valid_configs)} valid model(s) to evaluate:")
    for cfg in valid_configs:
        print(f"  - {os.path.basename(cfg.model_path)} ({cfg.model_type})")
    print()
    
    # Run evaluation(s) using hard-coded settings
    results = evaluate_all_models(
        configs=valid_configs,
        log_dir=LOG_DIR,
        continue_on_error=CONTINUE_ON_ERROR
    )
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Final Evaluation Results Summary")
    print("=" * 80)
    for result in results:
        status = "✓" if result['success'] else "✗"
        model_name = os.path.basename(result['model_path'])
        time_str = f"{result['evaluation_time']:.2f}s" if result['evaluation_time'] else "N/A"
        print(f"  {status} {model_name}: {time_str}")
        if result.get('results_file'):
            print(f"    Results: {result['results_file']}")
        if not result['success']:
            print(f"    Error: {result['error']}")
    print("=" * 80)
    
    # Print next steps
    print("\nNext steps:")
    print("1. Check the generated JSON files for detailed metrics")
    print("2. Launch TensorBoard to view visualizations:")
    print(f"   tensorboard --logdir {LOG_DIR}")
    print("3. Analyze results using evaluation_analysis.py:")
    print("   python evaluation/evaluation_analysis.py --input <results_file> --output <analysis_file>")

