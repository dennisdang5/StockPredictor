"""
Example script for evaluating a trained stock prediction model.

This script demonstrates how to use the ModelEvaluator class to comprehensively
evaluate a saved model with multiple metrics and visualizations.
"""

import os
import sys

# Add parent directory and logan-version to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logan_version_dir = os.path.join(parent_dir, "logan-version")
trained_model_dir = os.path.join(parent_dir, "trained_models")
if logan_version_dir not in sys.path:
    sys.path.insert(0, logan_version_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if trained_model_dir not in sys.path:
    sys.path.insert(0, trained_model_dir)

from evaluator import ModelEvaluator
from nlp_features import get_nlp_feature_dim

def evaluate_saved_model(model_path: str = "savedmodel_classification.pth", 
                        stocks: list = None,
                        time_args: list = None,
                        log_dir: str = "runs/evaluation",
                        nlp_method: str = "aggregated"):
    """
    Evaluate a saved model using the ModelEvaluator.
    
    Args:
        model_path: Path to the saved model file
        stocks: List of stock symbols to evaluate on
        time_args: Time arguments for data loading
        log_dir: Directory for TensorBoard logs
        nlp_method: NLP method to use for evaluation
    """
    # Get logan-version directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logan_version_dir = os.path.join(parent_dir, "logan-version")
    
    # Default parameters
    if stocks is None:
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
        ]  # Default stocks for evaluation
        
    if time_args is None:
        time_args = ["1989-12-01", "2015-09-30"]  # Default time period
    
    # Check if model exists (try both current directory and logan-version)
    if not os.path.exists(model_path):
        # Try in logan-version directory
        logan_model_path = os.path.join(logan_version_dir, model_path)
        if os.path.exists(logan_model_path):
            model_path = logan_model_path
        else:
            print(f"❌ Model file '{model_path}' not found!")
            print("Please train a model first or check the model path.")
            return False
    
    print(f"🔍 Evaluating model: {model_path}")
    print(f"📊 Stocks: {stocks}")
    print(f"📅 Time period: {time_args}")
    print(f"📝 Log directory: {log_dir}")
    print("-" * 50)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=model_path,
            stocks=stocks,
            time_args=time_args,
            log_dir=log_dir
        )
        
        # Run comprehensive evaluation
        print("🚀 Starting evaluation...")
        results = evaluator.evaluate_all_metrics(
            batch_size=32,  # Adjust based on your GPU memory
            create_plots=True
        )
        
        # Save results to JSON
        results_file = f"evaluation_results_{os.path.basename(model_path).replace('.pth', '')}.json"
        evaluator.save_results(results_file)
        
        # Clean up
        evaluator.close()
        
        print(f"✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📈 TensorBoard logs: {log_dir}")
        print(f"💡 Run 'tensorboard --logdir {log_dir}' to view visualizations")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        return False

def main():
    """Main function with example usage."""
    
    # Example 1: Evaluate with default parameters
    print("=" * 60)
    print("STOCK PREDICTION MODEL EVALUATION")
    print("=" * 60)
    
    model_name = "savedmodel_classification.pth"
    success = evaluate_saved_model(model_path=model_name)
    
    if success:
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Check the generated JSON file for detailed metrics")
        print("2. Launch TensorBoard to view visualizations:")
        print("   tensorboard --logdir runs/evaluation")
        print("3. Analyze the results to understand model performance")
    else:
        print("\n" + "=" * 60)
        print("EVALUATION FAILED!")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Ensure you have trained a model first")
        print("2. Check that savedmodel.pth exists in the current directory")
        print("3. Verify data files are available in the data/ directory")
        print("4. Check error messages above for specific issues")

if __name__ == "__main__":
    main()
