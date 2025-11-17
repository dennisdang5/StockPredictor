"""
NLP Feature Extraction Module for LSTM Stock Predictor

This module implements Phase 1 of NLP integration:
1. Reads NYT article CSVs (aggregated method) OR yfinance news (individual method)
2. Runs FinBERT sentiment analysis on headlines/articles
3. Aggregates variable-length articles into fixed-size daily features
4. Handles missing days vs neutral sentiment days properly
5. Produces clean feature vectors ready for LSTM input

IMPORTANT: Dataset Path Differentiation
--------------------------------------
The cache paths in util.py differentiate between NLP methods:
- Aggregated method (NYT): Uses suffix "_nlp_agg" in cache filenames
- Individual method (yfinance per stock): Uses suffix "_nlp_ind" in cache filenames

This ensures that datasets created with different NLP methods don't conflict.

Usage:
    from nlp_features import extract_daily_nlp_features, extract_daily_nlp_features_yfinance, align_nlp_with_trading_days
    
    # Aggregated method: Extract daily NLP features from NYT CSVs (shared across all stocks)
    nlp_df = extract_daily_nlp_features(
        csv_paths=['path/to/nyt_1990.csv', 'path/to/nyt_1991.csv'],
        start_date='1989-12-01',
        end_date='2015-09-30'
    )
    
    # Individual method: Extract NLP features from yfinance per stock ticker
    stock_nlp_features = extract_daily_nlp_features_yfinance(
        stocks=['AAPL', 'MSFT', 'GOOGL'],
        start_date='1990-01-01',
        end_date='2015-09-30'
    )
    
    # Align with trading days
    aligned_features = align_nlp_with_trading_days(
        nlp_df, 
        trading_days=your_trading_day_index
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import warnings
from tqdm import tqdm
import yfinance as yf
import time

# Try to import transformers for sentiment analysis
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "transformers library not available. Install with: pip install transformers torch"
    )


# ============================================================================
# Configuration
# ============================================================================

# Default sentiment model - FinBERT for financial text
DEFAULT_SENTIMENT_MODEL = "ProsusAI/finbert"  # FinBERT for financial sentiment

# Finance-related keywords for filtering articles
FINANCE_KEYWORDS = [
    'financial', 'finance', 'business', 'economy', 'economic', 'market', 'markets',
    'stock', 'stocks', 'trading', 'trade', 'investment', 'investor', 'investors',
    'earnings', 'revenue', 'profit', 'loss', 'quarterly', 'quarter', 'fiscal',
    'bank', 'banking', 'banker', 'corporate', 'company', 'companies', 'firm',
    'dollar', 'currency', 'dow', 'nasdaq', 's&p', 'sp500', 'index', 'indices',
    'ipo', 'merger', 'acquisition', 'deal', 'deals', 'billion', 'million',
    'ceo', 'cfo', 'executive', 'board', 'shareholder', 'dividend', 'shares'
]

# Finance-related sections/desks in NYT
FINANCE_SECTIONS = [
    'Business Day', 'Business', 'Financial Desk', 'Economy', 'Markets',
    'DealBook', 'Your Money', 'Mutual Funds', 'Stocks & Bonds'
]


# ============================================================================
# Sentiment Analysis Functions
# ============================================================================

def load_sentiment_model(device: Optional[str] = None):
    """
    Load FinBERT sentiment analysis model from Hugging Face.
    
    Args:
        device: Device to load model on ('cuda', 'cpu', or None for auto)
    
    Returns:
        Tuple of (tokenizer, model, device)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers torch")
    
    # Always use FinBERT
    model_name = DEFAULT_SENTIMENT_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print(f"[nlp] Loaded {model_name} for financial sentiment analysis")
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return tokenizer, model, device


def analyze_sentiment_batch(
    texts: List[str],
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 512
) -> np.ndarray:
    """
    Analyze sentiment for a batch of texts.
    
    Args:
        texts: List of text strings to analyze
        tokenizer: Hugging Face tokenizer
        model: Hugging Face sentiment model
        device: Device to run inference on
        batch_size: Batch size for processing
        max_length: Maximum sequence length
    
    Returns:
        numpy array of shape (len(texts), 3) with [p_pos, p_neg, p_neu] probabilities
    """
    if not texts:
        return np.array([])
    
    all_probs = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert to probabilities (softmax)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        all_probs.append(probs)
    
    return np.vstack(all_probs)


# ============================================================================
# Data Loading and Filtering
# ============================================================================

def load_nyt_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load NYT article CSV file.
    
    The NYT CSV has columns: headline, source, word_count, url, print_section, 
    print_page, author, type, pub_date, news_desk, section, year
    
    Args:
        csv_path: Path to NYT CSV file
    
    Returns:
        DataFrame with NYT articles, with parsed pub_date as datetime
    """
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Parse pub_date (format: '1990-01-01T05:00:00+0000')
    if 'pub_date' in df.columns:
        df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce', utc=True)
        # Convert to date (calendar day) for grouping, but keep datetime for alignment
        df['date'] = df['pub_date'].dt.date
        df['date_dt'] = df['pub_date'].dt.tz_localize(None).dt.normalize()  # Remove timezone, keep as datetime
    else:
        raise ValueError(f"CSV file {csv_path} must have a 'pub_date' column")
    
    # Ensure we have headline column
    if 'headline' not in df.columns:
        raise ValueError(f"CSV file {csv_path} must have a 'headline' column")
    
    # Remove rows with missing dates or headlines
    df = df.dropna(subset=['date', 'headline'])
    
    return df


def filter_finance_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter articles to keep only finance/business-related ones.
    
    Uses the actual NYT CSV columns: news_desk and section.
    NYT has 'Financial Desk' and 'Business Day' section.
    
    Args:
        df: DataFrame with NYT articles
    
    Returns:
        Filtered DataFrame
    """
    # Filter by news_desk (e.g., 'Financial Desk')
    desk_mask = False
    if 'news_desk' in df.columns:
        desk_mask = df['news_desk'].str.contains('Financial|Business', case=False, na=False)
    
    # Filter by section (e.g., 'Business Day')
    section_mask = False
    if 'section' in df.columns:
        section_mask = df['section'].str.contains('|'.join(FINANCE_SECTIONS), case=False, na=False)
    
    # Filter by keywords in headline (broader catch)
    headline_mask = df['headline'].str.contains('|'.join(FINANCE_KEYWORDS), case=False, na=False)
    
    # Combine filters: keep if matches desk, section, or headline keywords
    finance_mask = desk_mask | section_mask | headline_mask
    
    filtered_df = df[finance_mask].copy()
    
    print(f"[filter] Filtered {len(df)} articles -> {len(filtered_df)} finance-related articles")
    
    return filtered_df


# ============================================================================
# yfinance News Extraction
# ============================================================================

def extract_daily_nlp_features_yfinance(
    stocks: List[str],
    start_date: Optional[Union[str, datetime, date]] = None,
    end_date: Optional[Union[str, datetime, date]] = None,
    batch_size: int = 32,
    progress: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Extract daily NLP features from yfinance news for individual stock tickers.
    
    This function:
    1. Fetches news articles from yfinance for each stock ticker
    2. Runs FinBERT sentiment analysis on headlines
    3. Aggregates to daily statistics per stock
    
    Args:
        stocks: List of stock ticker symbols
        start_date: Start date for filtering articles (inclusive). Can be string, datetime, or date.
        end_date: End date for filtering articles (inclusive). Can be string, datetime, or date.
        batch_size: Batch size for sentiment analysis
        progress: Whether to show progress bars
    
    Returns:
        Dictionary mapping stock ticker -> DataFrame with daily NLP features
        Each DataFrame has columns:
            - date: Calendar date
            - n_articles: Number of articles that day
            - has_news: 1 if articles exist, 0 otherwise
            - mean_pos, mean_neg, mean_neu: Sentiment means
            - max_pos, max_neg: Max sentiment
            - std_pos, std_neg: Std dev
            - net_sent: Net sentiment
    """
    # Parse dates
    start_date_parsed = None
    end_date_parsed = None
    
    if start_date is not None:
        if isinstance(start_date, str):
            start_date_parsed = pd.to_datetime(start_date).date()
        elif isinstance(start_date, datetime):
            start_date_parsed = start_date.date()
        elif isinstance(start_date, date):
            start_date_parsed = start_date
    
    if end_date is not None:
        if isinstance(end_date, str):
            end_date_parsed = pd.to_datetime(end_date).date()
        elif isinstance(end_date, datetime):
            end_date_parsed = end_date.date()
        elif isinstance(end_date, date):
            end_date_parsed = end_date
    
    # Load FinBERT model
    print(f"[nlp-yfinance] Loading FinBERT sentiment model...")
    tokenizer, model, device = load_sentiment_model()
    
    # Dictionary to store results per stock
    stock_nlp_features = {}
    
    print(f"[nlp-yfinance] Fetching news for {len(stocks)} stocks...")
    
    for stock_idx, stock in enumerate(stocks):
        if progress and (stock_idx % 10 == 0 or stock_idx == len(stocks) - 1):
            print(f"[nlp-yfinance] Processing stock {stock_idx + 1}/{len(stocks)}: {stock}")
        
        try:
            # Fetch news from yfinance
            ticker = yf.Ticker(stock)
            news = ticker.news
            
            if not news:
                # No news available for this stock
                stock_nlp_features[stock] = pd.DataFrame()
                continue
            
            # Extract headlines and dates
            articles_data = []
            for article in news:
                # Get publish time
                if 'providerPublishTime' in article:
                    pub_time = pd.to_datetime(article['providerPublishTime'], unit='s')
                elif 'pubDate' in article:
                    pub_time = pd.to_datetime(article['pubDate'])
                else:
                    continue
                
                pub_date = pub_time.date()
                
                # Filter by date range if specified
                if start_date_parsed is not None and pub_date < start_date_parsed:
                    continue
                if end_date_parsed is not None and pub_date > end_date_parsed:
                    continue
                
                # Get headline
                headline = article.get('title', '')
                if not headline:
                    continue
                
                articles_data.append({
                    'date': pub_date,
                    'headline': headline
                })
            
            if not articles_data:
                stock_nlp_features[stock] = pd.DataFrame()
                continue
            
            # Convert to DataFrame
            articles_df = pd.DataFrame(articles_data)
            
            # Run sentiment analysis on headlines
            texts = articles_df['headline'].astype(str).tolist()
            texts = [text.strip() for text in texts if text.strip()]
            
            if not texts:
                stock_nlp_features[stock] = pd.DataFrame()
                continue
            
            # Analyze sentiment in batches
            all_probs = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                probs = analyze_sentiment_batch(batch_texts, tokenizer, model, device, batch_size)
                all_probs.append(probs)
            probs_array = np.vstack(all_probs)
            
            # Add sentiment probabilities
            articles_df['p_pos'] = probs_array[:, 0]
            articles_df['p_neg'] = probs_array[:, 1]
            articles_df['p_neu'] = probs_array[:, 2]
            
            # Group by date, but don't aggregate - just take the first article's sentiment for each day
            # If multiple articles exist on the same day, use the first one
            daily = articles_df.groupby('date').first().reset_index()
            
            # Keep only: date, p_pos, p_neg, p_neu
            # Add has_news = 1 (since we have articles)
            daily = daily[['date', 'p_pos', 'p_neg', 'p_neu']].copy()
            daily['has_news'] = 1
            
            # Add placeholder columns to match expected format (will be ignored but kept for compatibility)
            daily['n_articles'] = 1
            daily['mean_pos'] = daily['p_pos']
            daily['mean_neg'] = daily['p_neg']
            daily['mean_neu'] = daily['p_neu']
            daily['max_pos'] = daily['p_pos']
            daily['max_neg'] = daily['p_neg']
            daily['std_pos'] = 0.0
            daily['std_neg'] = 0.0
            daily['net_sent'] = daily['p_pos'] - daily['p_neg']
            
            daily = daily.sort_values('date').reset_index(drop=True)
            stock_nlp_features[stock] = daily
            
            # Rate limiting - be nice to yfinance API
            time.sleep(0.1)
            
        except Exception as e:
            warnings.warn(f"Error fetching news for {stock}: {e}")
            stock_nlp_features[stock] = pd.DataFrame()
            continue
    
    print(f"[nlp-yfinance] Completed processing {len(stocks)} stocks")
    stocks_with_news = sum(1 for df in stock_nlp_features.values() if len(df) > 0)
    print(f"[nlp-yfinance] Stocks with news data: {stocks_with_news}/{len(stocks)}")
    
    return stock_nlp_features


# ============================================================================
# Daily Aggregation
# ============================================================================

def extract_daily_nlp_features(
    csv_paths: Union[str, Path, List[Union[str, Path]]],
    batch_size: int = 32,
    progress: bool = True,
    start_date: Optional[Union[str, datetime, date]] = None,
    end_date: Optional[Union[str, datetime, date]] = None
) -> pd.DataFrame:
    """
    Extract daily NLP features from NYT article CSVs.
    
    This function:
    1. Loads NYT article CSVs
    2. Filters to finance-related articles only
    3. Runs FinBERT sentiment analysis on HEADLINES ONLY (no other text)
    4. Aggregates to daily statistics
    
    Args:
        csv_paths: Path(s) to NYT CSV file(s)
        batch_size: Batch size for sentiment analysis
        progress: Whether to show progress bars
        start_date: Start date for filtering articles (inclusive). Can be string, datetime, or date.
            If None, uses earliest date in data.
        end_date: End date for filtering articles (inclusive). Can be string, datetime, or date.
            If None, uses latest date in data.
    
    Returns:
        DataFrame with columns:
            - date: Calendar date
            - n_articles: Number of articles that day
            - has_news: 1 if articles exist, 0 otherwise
            - mean_pos: Mean positive sentiment probability
            - mean_neg: Mean negative sentiment probability
            - mean_neu: Mean neutral sentiment probability
            - max_pos: Maximum positive sentiment
            - max_neg: Maximum negative sentiment
            - std_pos: Std dev of positive sentiment
            - std_neg: Std dev of negative sentiment
            - net_sent: Net sentiment (mean_pos - mean_neg)
    """
    # Handle single path vs list of paths
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    
    # Load all CSVs
    print(f"[nlp] Loading {len(csv_paths)} NYT CSV file(s)...")
    all_articles = []
    
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            warnings.warn(f"CSV file not found: {csv_path}")
            continue
        
        df = load_nyt_csv(csv_path)
        all_articles.append(df)
    
    if not all_articles:
        raise ValueError("No valid CSV files found")
    
    # Combine all articles
    articles_df = pd.concat(all_articles, ignore_index=True)
    print(f"[nlp] Loaded {len(articles_df)} total articles")
    
    # Filter by date range if specified
    if start_date is not None or end_date is not None:
        # Convert start_date and end_date to date objects for comparison
        start_date_parsed = None
        end_date_parsed = None
        
        if start_date is not None:
            if isinstance(start_date, str):
                start_date_parsed = pd.to_datetime(start_date).date()
            elif isinstance(start_date, datetime):
                start_date_parsed = start_date.date()
            elif isinstance(start_date, date):
                start_date_parsed = start_date
            else:
                raise TypeError(f"start_date must be str, datetime, or date, got {type(start_date)}")
        
        if end_date is not None:
            if isinstance(end_date, str):
                end_date_parsed = pd.to_datetime(end_date).date()
            elif isinstance(end_date, datetime):
                end_date_parsed = end_date.date()
            elif isinstance(end_date, date):
                end_date_parsed = end_date
            else:
                raise TypeError(f"end_date must be str, datetime, or date, got {type(end_date)}")
        
        # Filter by date range
        date_mask = True
        if start_date_parsed is not None:
            date_mask = date_mask & (articles_df['date'] >= start_date_parsed)
        if end_date_parsed is not None:
            date_mask = date_mask & (articles_df['date'] <= end_date_parsed)
        
        articles_df = articles_df[date_mask].copy()
        print(f"[nlp] Filtered to date range: {start_date_parsed or 'earliest'} to {end_date_parsed or 'latest'}")
        print(f"[nlp] Articles after date filtering: {len(articles_df)}")
    
    # Always filter to finance articles
    articles_df = filter_finance_articles(articles_df)
    
    if len(articles_df) == 0:
        warnings.warn("No articles remaining after filtering")
        return pd.DataFrame()
    
    # Load FinBERT sentiment model (always use FinBERT)
    print(f"[nlp] Loading FinBERT sentiment model...")
    tokenizer, model, device = load_sentiment_model()
    
    # Prepare texts for sentiment analysis - ONLY use headlines
    # FinBERT only analyzes headlines, nothing else
    # Filter out empty headlines BEFORE sentiment analysis to keep articles_df in sync
    articles_df = articles_df.copy()  # Work on a copy to avoid modifying original
    articles_df['headline_clean'] = articles_df['headline'].astype(str).str.strip()
    
    # Filter out rows with empty headlines - this ensures articles_df matches texts length
    valid_mask = articles_df['headline_clean'].str.len() > 0
    articles_df = articles_df[valid_mask].reset_index(drop=True)
    
    if len(articles_df) == 0:
        warnings.warn("No valid headlines found after filtering")
        return pd.DataFrame()
    
    # Extract texts from filtered dataframe
    texts = articles_df['headline_clean'].tolist()
    
    # Run sentiment analysis
    print(f"[nlp] Running sentiment analysis on {len(texts)} articles...")
    if progress:
        # Process with progress bar
        all_probs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i + batch_size]
            probs = analyze_sentiment_batch(batch_texts, tokenizer, model, device, batch_size)
            all_probs.append(probs)
        probs_array = np.vstack(all_probs)
    else:
        probs_array = analyze_sentiment_batch(texts, tokenizer, model, device, batch_size)
    
    # Verify lengths match (should always match now since we filtered articles_df)
    if len(articles_df) != len(probs_array):
        raise ValueError(f"Length mismatch: articles_df has {len(articles_df)} rows but probs_array has {len(probs_array)} rows. This should not happen!")
    
    # Add sentiment probabilities to dataframe
    # FinBERT outputs: [positive, negative, neutral] (labels: ['positive', 'negative', 'neutral'])
    articles_df['p_pos'] = probs_array[:, 0]
    articles_df['p_neg'] = probs_array[:, 1]
    articles_df['p_neu'] = probs_array[:, 2]
    
    # Drop the temporary headline_clean column
    articles_df = articles_df.drop(columns=['headline_clean'])
    
    # Aggregate by date
    print(f"[nlp] Aggregating to daily features...")
    daily = articles_df.groupby('date').agg({
        'headline': 'size',  # Count articles
        'p_pos': ['mean', 'max', 'std'],
        'p_neg': ['mean', 'max', 'std'],
        'p_neu': 'mean'
    }).reset_index()
    
    # Flatten column names
    daily.columns = [
        'date',
        'n_articles',
        'mean_pos', 'max_pos', 'std_pos',
        'mean_neg', 'max_neg', 'std_neg',
        'mean_neu'
    ]
    
    # Add derived features
    daily['has_news'] = (daily['n_articles'] > 0).astype(int)
    daily['net_sent'] = daily['mean_pos'] - daily['mean_neg']
    
    # Fill NaN std values (happens when only 1 article per day)
    daily['std_pos'] = daily['std_pos'].fillna(0.0)
    daily['std_neg'] = daily['std_neg'].fillna(0.0)
    
    # Sort by date
    daily = daily.sort_values('date').reset_index(drop=True)
    
    print(f"[nlp] Created daily features for {len(daily)} days")
    print(f"[nlp] Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"[nlp] Days with news: {daily['has_news'].sum()}")
    
    return daily


# ============================================================================
# Alignment with Trading Days
# ============================================================================

def align_nlp_with_trading_days(
    nlp_df: pd.DataFrame,
    trading_days: Union[pd.DatetimeIndex, List[datetime], List[date]],
    fill_method: str = 'zero',
    sentinel_value: float = -2.0
) -> pd.DataFrame:
    """
    Align NLP features with yfinance trading days.
    
    yfinance returns DatetimeIndex with trading days only (no weekends/holidays).
    News articles are calendar days, so we need to map:
    - News on trading day → use that day
    - News on weekend/holiday → forward-fill to next trading day
    
    This function:
    1. Takes yfinance's date_index (DatetimeIndex of trading days)
    2. Maps calendar-day news to trading days (forward-fill from previous calendar days)
    3. Fills missing trading days with appropriate values
    4. Distinguishes "no news" from "neutral sentiment"
    
    Args:
        nlp_df: DataFrame with daily NLP features (from extract_daily_nlp_features)
            Must have 'date' column (date objects) and feature columns
        trading_days: yfinance DatetimeIndex of trading days (from open_close.index)
            Can also be list of datetime/date objects
        fill_method: How to fill missing days ('zero' or 'sentinel')
        sentinel_value: Value to use for missing sentiment if fill_method='sentinel'
    
    Returns:
        DataFrame indexed by trading days with columns:
            - date: Trading day (as date object)
            - has_news: 1 if news exists, 0 otherwise
            - n_articles: Number of articles (0 if no news)
            - mean_pos, mean_neg, mean_neu: Sentiment means (0 or sentinel if no news)
            - max_pos, max_neg: Max sentiment (0 or sentinel if no news)
            - std_pos, std_neg: Std dev (0 if no news)
            - net_sent: Net sentiment (0 or sentinel if no news)
    """
    # Convert trading_days to DatetimeIndex if needed
    if isinstance(trading_days, pd.DatetimeIndex):
        trading_index = trading_days.copy()
    elif isinstance(trading_days, (list, tuple)):
        trading_index = pd.DatetimeIndex(trading_days)
    else:
        raise TypeError(f"trading_days must be DatetimeIndex or list, got {type(trading_days)}")
    
    # Remove timezone if present (yfinance returns timezone-aware, but util.py removes it)
    if trading_index.tz is not None:
        trading_index = trading_index.tz_localize(None)
    
    # Normalize to dates (remove time component)
    trading_index = trading_index.normalize()
    
    # Convert nlp_df date column to datetime for alignment
    nlp_df = nlp_df.copy()
    if 'date' not in nlp_df.columns:
        raise ValueError("nlp_df must have a 'date' column")
    
    # nlp_df['date'] is date objects, convert to datetime and set as index
    date_col = pd.to_datetime(nlp_df['date'])
    nlp_df = nlp_df.set_index(date_col)
    nlp_df.index.name = 'date_dt'  # Name the index for easier access later
    
    # Reindex to trading days, forward-filling from previous calendar days
    # This handles weekends/holidays: news from Saturday/Sunday forward-fills to Monday
    aligned = nlp_df.reindex(trading_index, method='ffill')
    
    # Reset index to get date column back
    aligned = aligned.reset_index()
    # After reset_index(), pandas uses the index name as the column name (if index.name is set)
    # But if index.name is None or in some pandas versions, it uses 'index'
    # Find the datetime column that was created from the index
    date_dt_col = None
    if 'date_dt' in aligned.columns:
        date_dt_col = 'date_dt'
    elif 'index' in aligned.columns:
        # pandas might use 'index' as the column name
        date_dt_col = 'index'
        aligned = aligned.rename(columns={'index': 'date_dt'})
    else:
        # Find the first column that looks like a datetime index
        for col in aligned.columns:
            if col not in nlp_df.columns:  # This should be the index column
                date_dt_col = col
                aligned = aligned.rename(columns={col: 'date_dt'})
                break
    
    if date_dt_col is None or 'date_dt' not in aligned.columns:
        raise ValueError(f"Could not find date_dt column after reset_index. Available columns: {list(aligned.columns)}")
    
    # Create date column from datetime column
    aligned['date'] = pd.to_datetime(aligned['date_dt']).dt.date
    
    # Drop the datetime column, keep date
    aligned = aligned.drop(columns=['date_dt'])
    
    # Fill missing values based on method
    if fill_method == 'zero':
        # Fill sentiment stats with 0, keep has_news and n_articles meaningful
        sentiment_cols = ['mean_pos', 'mean_neg', 'mean_neu', 'max_pos', 'max_neg', 
                          'net_sent', 'std_pos', 'std_neg']
        for col in sentiment_cols:
            if col in aligned.columns:
                aligned[col] = aligned[col].fillna(0.0)
        
        aligned['n_articles'] = aligned['n_articles'].fillna(0).astype(int)
        aligned['has_news'] = aligned['has_news'].fillna(0).astype(int)
    
    elif fill_method == 'sentinel':
        # Use sentinel value for missing sentiment
        sentiment_cols = ['mean_pos', 'mean_neg', 'mean_neu', 'max_pos', 'max_neg', 'net_sent']
        for col in sentiment_cols:
            if col in aligned.columns:
                aligned[col] = aligned[col].fillna(sentinel_value)
        
        # Std dev should be 0 for missing days
        if 'std_pos' in aligned.columns:
            aligned['std_pos'] = aligned['std_pos'].fillna(0.0)
        if 'std_neg' in aligned.columns:
            aligned['std_neg'] = aligned['std_neg'].fillna(0.0)
        
        aligned['n_articles'] = aligned['n_articles'].fillna(0).astype(int)
        aligned['has_news'] = aligned['has_news'].fillna(0).astype(int)
    
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}. Use 'zero' or 'sentinel'")
    
    # Sort by date
    aligned = aligned.sort_values('date').reset_index(drop=True)
    
    print(f"[align] Aligned NLP features to {len(aligned)} trading days")
    print(f"[align] Days with news: {aligned['has_news'].sum()}")
    print(f"[align] Days without news: {(aligned['has_news'] == 0).sum()}")
    
    return aligned


def get_nlp_feature_vector(nlp_row: pd.Series, nlp_method: str) -> np.ndarray:
    """
    Extract NLP feature vector from a row of aligned NLP DataFrame.
    
    Args:
        nlp_row: Single row from aligned NLP DataFrame
        use_simple: If True, return only [has_news, p_pos, p_neg, p_neu] (4 features)
            If False, return full feature vector (10 features)
    
    Returns:
        numpy array of NLP features
        If use_simple=True: [has_news, p_pos, p_neg, p_neu]
        If use_simple=False: [has_news, n_articles, mean_pos, mean_neg, mean_neu, 
                              max_pos, max_neg, std_pos, std_neg, net_sent]
    """
    if nlp_method == "aggregated":
        # Aggregated format: full feature vector with aggregations (10 features)
        features = [
            nlp_row['has_news'],
            nlp_row['n_articles'],
            nlp_row['mean_pos'],
            nlp_row['mean_neg'],
            nlp_row['mean_neu'],
            nlp_row['max_pos'],
            nlp_row['max_neg'],
            nlp_row['std_pos'],
            nlp_row['std_neg'],
            nlp_row['net_sent']
        ]
    elif nlp_method == "individual":
        # Individual format: simple format with just sentiment probabilities (4 features)
        features = [
            nlp_row['has_news'],
            nlp_row.get('p_pos', nlp_row.get('mean_pos', 0.0)),
            nlp_row.get('p_neg', nlp_row.get('mean_neg', 0.0)),
            nlp_row.get('p_neu', nlp_row.get('mean_neu', 0.0))
        ]
    else:
        raise ValueError(f"Invalid nlp_method: {nlp_method}")
    return np.array(features, dtype=np.float32)


def get_nlp_feature_dim(nlp_method: str) -> int:
    """
    Return the dimensionality of NLP features.
    
    Args:
        nlp_method: "aggregated" or "individual"
    
    Returns:
        Number of NLP features
        - Aggregated format: 10 (has_news, n_articles, mean_pos, mean_neg, mean_neu, 
                              max_pos, max_neg, std_pos, std_neg, net_sent)
        - Individual format: 4 (has_news, p_pos, p_neg, p_neu)
    """
    if nlp_method == "aggregated":
        return 10
    elif nlp_method == "individual":
        return 4
    else:
        raise ValueError(f"Invalid nlp_method: {nlp_method}")


# ============================================================================
# Main Execution / Testing
# ============================================================================

def main():
    """
    Example usage of NLP feature extraction.
    """
    import glob
    
    print("=" * 60)
    print("NLP Feature Extraction - Phase 1")
    print("=" * 60)
    
    # Find NYT CSV files (relative to logan-version directory)
    script_dir = Path(__file__).parent.absolute()
    nyt_dir = script_dir / "huggingface_nyt_articles"
    csv_files = sorted(glob.glob(str(nyt_dir / "new_york_times_stories_*.csv")))
    
    if not csv_files:
        print(f"Error: No NYT CSV files found in {nyt_dir}")
        return
    
    print(f"\nFound {len(csv_files)} NYT CSV files")
    print(f"Processing first 2 files as example: {csv_files[:2]}")
    
    # Extract daily NLP features
    nlp_df = extract_daily_nlp_features(
        csv_paths=csv_files[:2],  # Process first 2 years as example
        batch_size=32,
        progress=True
    )
    
    if len(nlp_df) == 0:
        print("No NLP features extracted. Check your CSV files.")
        return
    
    # Display sample
    print("\n" + "=" * 60)
    print("Sample Daily NLP Features:")
    print("=" * 60)
    print(nlp_df.head(10))
    
    print("\n" + "=" * 60)
    print("Feature Statistics:")
    print("=" * 60)
    print(nlp_df.describe())
    
    # Example: Align with trading days
    print("\n" + "=" * 60)
    print("Example: Aligning with Trading Days")
    print("=" * 60)
    
    # Create example trading days (weekdays in date range)
    start_date = nlp_df['date'].min()
    end_date = nlp_df['date'].max()
    trading_days = pd.bdate_range(start=start_date, end=end_date)
    
    aligned = align_nlp_with_trading_days(nlp_df, trading_days, fill_method='zero')
    
    print(f"\nAligned features shape: {aligned.shape}")
    print(f"Feature columns: {list(aligned.columns)}")
    print(f"\nSample aligned features:")
    print(aligned.head(10))
    
    # Show feature vector example
    print("\n" + "=" * 60)
    print("Example Feature Vector:")
    print("=" * 60)
    example_row = aligned.iloc[0]
    feature_vec = get_nlp_feature_vector(example_row, nlp_method="aggregated")
    print(f"Feature vector shape: {feature_vec.shape}")
    print(f"Feature vector: {feature_vec}")
    print(f"\nFeature names:")
    print("  [has_news, n_articles, mean_pos, mean_neg, mean_neu,")
    print("   max_pos, max_neg, std_pos, std_neg, net_sent]")
    
    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Integrate this into util.py data pipeline")
    print("2. Update model.py to accept variable input dimensions")
    print("3. Test with full dataset")


if __name__ == "__main__":
    main()

