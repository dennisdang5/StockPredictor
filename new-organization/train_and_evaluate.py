#!/usr/bin/env python3
"""
Simple script to train and evaluate a model on a few stocks over 1 year.
"""

import os
import sys
import time
from datetime import datetime

# Ensure we're importing from the current directory (new-organization)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Explicit imports from current directory
from trainer import Trainer, TrainerConfig
from models.configs import LSTMConfig
from evaluation.configs.evaluation_config import EvaluationConfig
from evaluation.evaluator import ModelEvaluator


def train_model():
    """Train a simple LSTM model on a few stocks for 1 year."""
    print("=" * 80)
    print("Training Model")
    print("=" * 80)
    
    # Configuration: 3 stocks, 1 year period
    stocks = ["AAPL", "MSFT", "GOOGL"]
    time_args = ["1990-01-01", "1999-12-31"]  # 1 year period
    model_save_path = "savedmodel_test_1year.pth"
    
    print(f"Stocks: {stocks}")
    print(f"Time period: {time_args[0]} to {time_args[1]}")
    print(f"Model will be saved to: {model_save_path}")
    print()
    
    # Create model config
    model_config = LSTMConfig(parameters={
        'input_shape': (31, 13),  # Will be overridden by actual data
        'hidden_size': 25,
        'num_layers': 2,
        'dropout': 0.1
    })
    
    # Create trainer config
    trainer_config = TrainerConfig(
        stocks=stocks,
        time_args=time_args,
        batch_size=32,
        num_epochs=10,  # Small number for quick test
        model_type="LSTM",
        model_config=model_config,
        period_type="LS",
        lookback=240,
        use_nlp=True,
        nlp_method="aggregated",
        saved_model=model_save_path,  # Save path
        save_every_epochs=5,  # Save every 5 epochs
        early_stop_patience=7,
        early_stop_min_delta=0.001,
        k=10,
        cost_bps_per_side=5.0
    )
    
    # Create and train
    print("Creating trainer...")
    trainer = Trainer(config=trainer_config)
    
    print(f"Starting training for {trainer.num_epochs} epochs...")
    start_time = time.time()
    
    # Train
    for epoch in range(trainer.num_epochs):
        stop_condition = trainer.train_one_epoch(epoch)
        if stop_condition:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time:.2f} seconds")
    print(f"✓ Model saved to: {model_save_path}")
    
    return model_save_path


def evaluate_model(model_path):
    """Evaluate the trained model."""
    print("\n" + "=" * 80)
    print("Evaluating Model")
    print("=" * 80)
    
    # Use same stocks and time period for evaluation
    # Top 30 S&P 500 stocks by market cap (subset for testing)

    
    stocks = [
        # Top Technology & Growth
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",

        # Major Financial Services
        "JPM", "BAC", "V", "MA", "WFC", "GS", "BLK", "AXP",

        # Healthcare Leaders
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO",

        # Consumer & Retail Giants
        "WMT", "PG", "HD", "COST", "MCD", "NKE",

        # Industrial & Energy Leaders
        "BA", "CAT", "XOM", "CVX"
    ]
    """

    stocks = [
    # Communication Services
    "GOOGL", "GOOG", "T", "CHTR", "CMCSA", "EA", "FOXA", "FOX", "IPG", "LYV", "MTCH", "META", "NFLX", "NWSA", "NWS", "OMC", "PSKY", "TMUS", "TTWO", "TKO", "TTD", "VZ", "DIS", "WBD",

    # consumer discretionary
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
    """
    
    time_args = ["2020-01-01", "2020-06-01"]
    
    print(f"Model: {model_path}")
    print(f"Stocks: {stocks}")
    print(f"Time period: {time_args[0]} to {time_args[1]}")
    print()
    
    # Create model config matching training config
    model_config = LSTMConfig(parameters={
        'input_shape': (31, 13),  # Will be determined from data
        'hidden_size': 25,  # Match training config
        'num_layers': 2,  # Match training config
        'dropout': 0.1
    })
    
    # Create evaluation config
    # Note: k must be <= (number of stocks) / 2 to allow trading
    # With 3 stocks, we can use k=1 (requires 2 stocks minimum)
    eval_config = EvaluationConfig(
        model_path=model_path,
        model_type="lstm",
        stocks=stocks,
        time_args=time_args,
        batch_size=32,
        k=10,
        cost_bps_per_side=5.0,
        use_nlp=True,  # Default uses NLP with aggregated method
        nlp_method="aggregated",
        create_plots=False,  # Skip plots for quick test
        log_dir="runs/test_evaluation"
    )
    
    # Create evaluator with model config
    print("Creating evaluator...")
    evaluator = ModelEvaluator(
        model_path=eval_config.model_path,
        stocks=eval_config.stocks,
        time_args=eval_config.time_args,
        log_dir=eval_config.log_dir,
        device=eval_config.device,
        use_nlp=eval_config.use_nlp,
        nlp_method=eval_config.nlp_method,
        model_type=eval_config.model_type.lower(),
        input_shape=eval_config.input_shape,
        model_config=model_config  # Pass model config to match training
    )
    
    # Run evaluation
    print("Running evaluation...")
    start_time = time.time()
    
    metrics = evaluator.evaluate_all_metrics(
        batch_size=eval_config.batch_size,
        create_plots=eval_config.create_plots,
        k=eval_config.k,
        cost_bps_per_side=eval_config.cost_bps_per_side,
        use_paper_aligned=True
    )
    
    eval_time = time.time() - start_time
    
    # Save results
    results_file = f"evaluation_results_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evaluator.save_results(results_file)
    
    print(f"\n✓ Evaluation completed in {eval_time:.2f} seconds")
    print(f"✓ Results saved to: {results_file}")
    
    # Print key metrics
    print("\n" + "-" * 80)
    print("Key Metrics:")
    print("-" * 80)
    if metrics:
        if 'accuracy' in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if 'precision' in metrics:
            print(f"  Precision: {metrics['precision']:.4f}")
        if 'recall' in metrics:
            print(f"  Recall: {metrics['recall']:.4f}")
        if 'f1_score' in metrics:
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
        if 'sharpe_ratio' in metrics:
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        if 'total_return' in metrics:
            print(f"  Total Return: {metrics['total_return']:.4f}")
    
    # Clean up
    evaluator.close()
    
    return results_file


def main():
    """Main function to run training and evaluation."""
    print("\n" + "=" * 80)
    print("Train and Evaluate - 1 Year Test")
    print("=" * 80 + "\n")
    
    try:
        # Step 1: Train
        model_path = train_model()
        
        # Step 2: Evaluate
        if os.path.exists(model_path):
            results_file = evaluate_model(model_path)
            print("\n" + "=" * 80)
            print("✓ Complete! Training and evaluation finished successfully.")
            print("=" * 80)
            print(f"\nModel: {model_path}")
            print(f"Results: {results_file}")
        else:
            print(f"\n✗ Error: Model file not found at {model_path}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

