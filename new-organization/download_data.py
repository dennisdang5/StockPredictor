#!/usr/bin/env python3
"""
Simple script to download and cache stock data.

This script downloads stock data from yfinance and caches it for later use.
The cached data can be used by training and evaluation scripts without re-downloading.

Edit the configuration variables below to customize the download.
"""

import os
import sys

# Ensure we're importing from the current directory (new-organization)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from util import get_data, DATA_DIR
from data_sources import YFinanceDataSource

# ============================================================================
# Configuration Variables - Edit these to customize the download
# ============================================================================

# Stock tickers to download
STOCKS = [
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

# Time range arguments
# Option 1: Date range (start and end dates)
TIME_ARGS = ["1990-01-01", "2015-12-31"]

# Option 2: Period string (uncomment to use instead of date range)
# TIME_ARGS = ["3y"]  # Options: "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"

# Sequence length (lookback window size)
SEQ_LEN = 240

# Period type: "LS" (long-short periods) or "full" (full sequence)
PERIOD_TYPE = "LS"

# Prediction type: "classification" or "regression"
PREDICTION_TYPE = "classification"

# NLP options
USE_NLP = True  # Set to True to include NLP features
NLP_METHOD = "individual"  # "aggregated" (NYT headlines) or "individual" (yfinance per stock)

# Cache options
FORCE = True  # Set to True to force re-download even if cache exists


def download_and_cache(
    stocks,
    time_args,
    seq_len=240,
    use_nlp=True,
    nlp_method="aggregated",
    period_type="LS",
    force=False,
    prediction_type="classification"
):
    """
    Download and cache stock data.
    
    Args:
        stocks: List of stock tickers
        time_args: Time range arguments (e.g., ["1990-01-01", "2015-12-31"] or ["3y"])
        seq_len: Sequence length (lookback window size). Default: 240
        use_nlp: Whether to include NLP features. Default: False
        nlp_method: NLP method - "aggregated" (NYT) or "individual" (yfinance). Default: "aggregated"
        period_type: Period type - "LS" or "full". Default: "LS"
        force: Force re-download even if cache exists. Default: False
        prediction_type: Prediction type - "classification" or "regression". Default: "classification"
    
    Returns:
        True if successful, False otherwise
    """
    print("=" * 80)
    print("Downloading and Caching Stock Data")
    print("=" * 80)
    print(f"Stocks: {len(stocks)} stocks")
    print(f"Time period: {time_args}")
    print(f"Sequence length: {seq_len}")
    print(f"Period type: {period_type}")
    print(f"Use NLP: {use_nlp}")
    if use_nlp:
        print(f"NLP method: {nlp_method}")
    print(f"Prediction type: {prediction_type}")
    print(f"Force re-download: {force}")
    print(f"Cache directory: {DATA_DIR}")
    print("=" * 80)
    print()
    
    try:
        # Create data source (explicitly specify YFinanceDataSource)
        data_source = YFinanceDataSource()
        
        # Call get_data which will download and cache automatically
        print("Downloading data...")
        data = get_data(
            stocks=stocks,
            args=time_args,
            seq_len=seq_len,
            data_source=data_source,
            force=force,
            prediction_type=prediction_type,
            use_nlp=use_nlp,
            nlp_method=nlp_method,
            period_type=period_type
        )
        
        if data is None:
            print("ERROR: Failed to download data")
            return False
        
        # Unpack data tuple
        Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Dtrain, Dval, Dtest, Rev_test, Returns_test, Sp500_test = data
        
        print()
        print("=" * 80)
        print("✓ Data Successfully Downloaded and Cached")
        print("=" * 80)
        print(f"Training samples: {Xtrain.shape[0]}")
        print(f"Validation samples: {Xval.shape[0]}")
        print(f"Test samples: {Xtest.shape[0]}")
        print(f"Feature shape: {Xtrain.shape[1:]}")
        print(f"Cache location: {DATA_DIR}")
        print("=" * 80)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Error downloading data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    # Validate configuration
    if not STOCKS:
        print("ERROR: No stocks specified. Please edit STOCKS variable.")
        return 1
    
    if not TIME_ARGS:
        print("ERROR: No time arguments specified. Please edit TIME_ARGS variable.")
        return 1
    
    # Download and cache
    success = download_and_cache(
        stocks=STOCKS,
        time_args=TIME_ARGS,
        seq_len=SEQ_LEN,
        use_nlp=USE_NLP,
        nlp_method=NLP_METHOD if USE_NLP else "aggregated",
        period_type=PERIOD_TYPE,
        force=FORCE,
        prediction_type=PREDICTION_TYPE
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

