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

from evaluation.evaluator import ModelEvaluator
from evaluation.configs.evaluation_config import EvaluationConfig


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
        results_file = f"evaluation_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
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
    summary_path = f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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


def get_evaluation_configs() -> List[EvaluationConfig]:
    """
    Create a list of evaluation configurations.
    
    Add or modify configurations here to evaluate different models.
    Modify model paths, stocks, time_args, and other parameters as needed.
    """
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
    
    # Common time arguments (can be customized)
    common_time_args = ["1989-12-01", "2015-09-30"]  # Default evaluation period
    
    # Example 1: Basic LSTM model
    configs.append(EvaluationConfig(
        model_path="savedmodel_classification.pth",
        model_type="lstm",
        stocks=common_stocks,
        time_args=common_time_args,
        batch_size=32,
        k=10,
        cost_bps_per_side=5.0,
        use_nlp=True,  # Default uses NLP with aggregated method
        create_plots=True
    ))
    
    # Example 2: CNN-LSTM model with NLP
    configs.append(EvaluationConfig(
        model_path="savedmodel_classification_cnn_lstm.pth",
        model_type="cnn_lstm",
        stocks=common_stocks,
        time_args=common_time_args,
        batch_size=32,
        k=10,
        cost_bps_per_side=5.0,
        use_nlp=True,
        nlp_method="aggregated",
        create_plots=True
    ))
    
    # Example 3: AutoEncoder-LSTM model
    configs.append(EvaluationConfig(
        model_path="savedmodel_classification_aelstm.pth",
        model_type="aelstm",
        stocks=common_stocks,
        time_args=common_time_args,
        batch_size=32,
        k=10,
        cost_bps_per_side=5.0,
        use_nlp=True,  # Default uses NLP with aggregated method
        create_plots=True
    ))
    
    # Example 4: CNN-AutoEncoder-LSTM model
    configs.append(EvaluationConfig(
        model_path="savedmodel_classification_cnnaelstm.pth",
        model_type="cnnaelstm",
        stocks=common_stocks,
        time_args=common_time_args,
        batch_size=32,
        k=10,
        cost_bps_per_side=5.0,
        use_nlp=True,  # Default uses NLP with aggregated method
        create_plots=True
    ))
    
    # Example 5: TimesNet model (if available)
    configs.append(EvaluationConfig(
        model_path="savedmodel_classification_timesnet.pth",
        model_type="timesnet",
        stocks=common_stocks[:100],  # Fewer stocks for TimesNet (more computationally intensive)
        time_args=common_time_args,
        batch_size=16,  # Smaller batch for TimesNet
        k=10,
        cost_bps_per_side=5.0,
        use_nlp=True,  # Default uses NLP with aggregated method
        create_plots=True
    ))
    
    return configs


# Hard-coded configuration variables
# Modify these variables to customize evaluation behavior

# Evaluation settings
LOG_DIR = "runs/evaluation"
CONTINUE_ON_ERROR = True
CREATE_PLOTS = True

# Default evaluation parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_K = 10
DEFAULT_COST_BPS_PER_SIDE = 5.0
DEFAULT_USE_NLP = True
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
    
    # Determine evaluation mode based on hard-coded variables
    if SINGLE_MODEL_PATH:
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
        # Multiple model evaluation mode (use get_evaluation_configs)
        configs = get_evaluation_configs()
        print(f"Multiple model evaluation mode: {len(configs)} models")
    
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

