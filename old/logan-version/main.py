import trainer
from trainer import TrainerConfig
from model import LSTMConfig, CNNLSTMConfig, AELSTMConfig, CNNAELSTMConfig
import torch
import os
import sys
import traceback

def main():
    print("my code")
    
    
    # Top 30 S&P 500 stocks by market cap (subset for testing)

    """
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
    
    
    start = "1990-01-01"
    #end = "1991-01-01"
    end = "2015-12-31"
    prediction_type = "classification"
    period_type = "LS"  # Period type for feature extraction ("LS" or "full")
    
    models = {
        "LSTM": {
            "model_type": "LSTM",
            "model_path": "savedmodel_classification_nlp_lstm.pth",
            "use_nlp": True,
            "nlp_method": "aggregated"
        },
        "AE_NLP_LSTM": {
            "model_type": "AELSTM",
            "nlp_method": "aggregated",
            "use_nlp": True,
            "model_path": "savedmodel_classification_ae_nlp_lstm.pth"
        },
        "AE_NLP_CNN_LSTM": {
            "model_type": "CNNAELSTM",
            "nlp_method": "aggregated",
            "use_nlp": True,
            "model_path": "savedmodel_classification_ae_cnn_nlp_lstm.pth",
            "kernel_size": 3
        },
    }
    
    # Training phase
    print("=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    
    total_epochs = 1000
    for model_name, model_args in models.items():
        try:
            # Extract model_type and model_path from model_args
            # Make a copy to avoid modifying the original dict
            model_args_copy = model_args.copy()
            model_type = model_args_copy.pop("model_type")
            model_path = model_args_copy.pop("model_path")
            
            # Create model-specific config using ModelConfig classes
            # This ensures type safety and proper parameter handling
            model_config_map = {
                "LSTM": LSTMConfig,
                "CNNLSTM": CNNLSTMConfig,
                "AELSTM": AELSTMConfig,
                "CNNAELSTM": CNNAELSTMConfig,
            }
            
            # Create model config if model type is supported
            model_config = None
            if model_type.upper() in model_config_map:
                config_class = model_config_map[model_type.upper()]
                # Extract model-specific parameters from model_args_copy
                model_config_kwargs = {}
                if 'kernel_size' in model_args_copy:
                    model_config_kwargs['kernel_size'] = model_args_copy.pop('kernel_size')
                # You can add more model-specific parameters here if needed
                # For now, defaults will be used for hidden_size, num_layers, etc.
                model_config = config_class(**model_config_kwargs)
            
            # Create TrainerConfig with all parameters
            config = TrainerConfig(
                stocks=stocks,
                time_args=[start, end],
                batch_size=8,  # Default batch size (can be customized)
                num_epochs=total_epochs,
                saved_model=model_path,
                prediction_type=prediction_type,
                k=10,  # Default portfolio parameter (can be customized)
                cost_bps_per_side=5.0,  # Default transaction cost (can be customized)
                save_every_epochs=25,  # Default save frequency (can be customized)
                model_type=model_type,
                model_config=model_config,  # Use ModelConfig class instance
                early_stop_patience=7,  # Early stopping patience (default: 7 epochs)
                early_stop_min_delta=0.001,  # Minimum improvement for early stopping (default: 0.001)
                period_type=period_type,  # Period type for feature extraction ("LS" or "full")
                **model_args_copy  # Pass remaining model args (use_nlp, nlp_method, etc.)
            )
            
            # Create Trainer with config
            train_obj = trainer.Trainer(config=config)
        except Exception as e:
            print(f"[ERROR] Failed to initialize Trainer: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            continue
    
        if type(train_obj) == int:
            print("Error getting data")
            continue
    
        try:
            for name, param in train_obj.Model.named_parameters():
                print("name: {}".format(name))
                print("param: {}".format(param.numel()))
                print()
        except Exception as e:
            print(f"[ERROR] Failed to access model parameters: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            train_obj.stop()
            continue
    
        #total_epochs = train_obj.num_epochs
    
        stop = False  # Initialize stop condition
        try:
            for epoch in range(total_epochs):
                try:
                    stop = train_obj.train_one_epoch(epoch)
                    if stop: 
                        if train_obj.is_main:
                            print("Early stop at epoch: {}".format(epoch))
                        break
                except Exception as e:
                    print(f"[ERROR] Failed during training epoch {epoch}: {e}")
                    print("\nFull traceback:")
                    traceback.print_exc()
                    break
        finally:
            # Always clean up distributed training resources
            try:
                train_obj.stop()
            except Exception as e:
                print(f"[WARNING] Error during cleanup: {e}")

if __name__ == '__main__':
    main()
