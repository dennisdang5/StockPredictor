"""
Example script showing how to integrate sentiment features with LSTM model.

This script demonstrates how to:
1. Load the sentiment data from get_nyt_sentiment.py
2. Align it with price series data
3. Create features for LSTM input
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_sentiment_data(sentiment_file: str = "nyt_sentiment_features_daily.csv") -> pd.DataFrame:
    """
    Load daily aggregated sentiment data.
    
    Args:
        sentiment_file: Path to daily sentiment CSV file
    
    Returns:
        DataFrame with columns: ticker, date, sentiment_mean, sentiment_std, headline_count
    """
    df = pd.read_csv(sentiment_file)
    df["date"] = pd.to_datetime(df["date"])
    return df


def align_sentiment_with_prices(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker: str,
    lookback_days: int = 30,
) -> np.ndarray:
    """
    Align sentiment features with price data for a specific ticker.
    
    Args:
        price_df: DataFrame with price data (index should be date, columns should include ticker)
        sentiment_df: DataFrame with sentiment data from load_sentiment_data()
        ticker: Stock ticker symbol
        lookback_days: Number of days to look back for sentiment aggregation
    
    Returns:
        numpy array of sentiment features (one per day in price_df)
        Features: [sentiment_mean, sentiment_std, headline_count] for each day
    """
    # Filter sentiment data for this ticker
    ticker_sentiment = sentiment_df[sentiment_df["ticker"] == ticker].copy()
    
    if ticker_sentiment.empty:
        # No sentiment data for this ticker, return zeros
        return np.zeros((len(price_df), 3))
    
    # Set date as index for easier alignment
    ticker_sentiment = ticker_sentiment.set_index("date")
    
    # Initialize feature array
    features = []
    
    for price_date in price_df.index:
        # Get sentiment for this date and lookback window
        lookback_start = price_date - pd.Timedelta(days=lookback_days)
        
        # Filter sentiment in lookback window
        window_sentiment = ticker_sentiment[
            (ticker_sentiment.index >= lookback_start) &
            (ticker_sentiment.index <= price_date)
        ]
        
        if len(window_sentiment) > 0:
            # Aggregate features: mean sentiment, std sentiment, total headlines
            features.append([
                window_sentiment["sentiment_mean"].mean(),
                window_sentiment["sentiment_std"].mean() if "sentiment_std" in window_sentiment.columns else 0.0,
                window_sentiment["headline_count"].sum(),
            ])
        else:
            # No sentiment data in window, use zeros
            features.append([0.0, 0.0, 0.0])
    
    return np.array(features)


def create_lstm_features_with_sentiment(
    price_data: np.ndarray,
    sentiment_features: np.ndarray,
    lookback: int = 240,
) -> np.ndarray:
    """
    Create LSTM input features combining price data and sentiment.
    
    Args:
        price_data: Price data array (shape: [time_steps, price_features])
        sentiment_features: Sentiment features array (shape: [time_steps, sentiment_features])
        lookback: Number of time steps to look back
    
    Returns:
        Combined features array (shape: [samples, lookback, price_features + sentiment_features])
    """
    # Ensure same length
    min_length = min(len(price_data), len(sentiment_features))
    price_data = price_data[:min_length]
    sentiment_features = sentiment_features[:min_length]
    
    # Combine features
    combined_features = np.concatenate([price_data, sentiment_features], axis=1)
    
    # Create sequences
    sequences = []
    for i in range(lookback, len(combined_features)):
        sequences.append(combined_features[i - lookback:i])
    
    return np.array(sequences)


def example_usage():
    """
    Example showing how to use sentiment features with your existing LSTM pipeline.
    """
    print("=" * 60)
    print("Example: Integrating Sentiment Features with LSTM")
    print("=" * 60)
    
    # 1. Load sentiment data
    print("\n1. Loading sentiment data...")
    try:
        sentiment_df = load_sentiment_data("nyt_sentiment_features_daily.csv")
        print(f"   Loaded {len(sentiment_df)} sentiment records")
        print(f"   Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
        print(f"   Tickers: {sentiment_df['ticker'].unique()}")
    except FileNotFoundError:
        print("   Error: Sentiment file not found. Run get_nyt_sentiment.py first.")
        return
    
    # 2. Example: Load price data (replace with your actual price loading)
    print("\n2. Loading price data...")
    # This is a placeholder - replace with your actual price data loading
    # price_df = your_price_loading_function(...)
    print("   (Placeholder - replace with your actual price data loading)")
    
    # 3. Align sentiment with prices
    print("\n3. Aligning sentiment with prices...")
    # Example for a specific ticker
    ticker = "AAPL"
    # sentiment_features = align_sentiment_with_prices(price_df, sentiment_df, ticker)
    print(f"   (Placeholder - would align sentiment for {ticker})")
    
    # 4. Create combined features
    print("\n4. Creating combined LSTM features...")
    # combined_features = create_lstm_features_with_sentiment(
    #     price_data=your_price_array,
    #     sentiment_features=sentiment_features_array,
    #     lookback=240
    # )
    print("   (Placeholder - would create combined features)")
    
    print("\n" + "=" * 60)
    print("Integration complete! Use the combined features in your LSTM model.")
    print("=" * 60)
    print("\nNote: Modify your model.py to accept additional input dimensions:")
    print("  - Current: input_dim=(31, 3)  # 31 stocks, 3 price features")
    print("  - With sentiment: input_dim=(31, 6)  # 31 stocks, 3 price + 3 sentiment")


if __name__ == "__main__":
    example_usage()



