"""
Historical News Headlines with Sentiment Analysis for LSTM Features

This script fetches historical headlines from the New York Times API (going back to 1989)
and runs sentiment analysis to create features for LSTM stock prediction models.

The script:
1. Maps stock tickers to company names for NYT searches
2. Fetches headlines from NYT Archive API (1989+) and Article Search API
3. Runs sentiment analysis using Hugging Face transformers
4. Stores time-indexed sentiment scores aligned with price series

Usage:
    # Set your NYT API key
    export NYT_API_KEY="your_api_key_here"
    
    # Edit the configuration variables at the top of this file:
    #   - TICKERS: List of stock tickers to search
    #   - START_YEAR / END_YEAR: Date range
    #   - OUTPUT_FILE: Output file path
    #   - Other configuration options
    
    # Run the script
    python get_nyt_sentiment.py
"""

import requests
import os
import json
import time
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURATION VARIABLES - Modify these to change script behavior
# ============================================================================

# Stock tickers to search for (will be mapped to company names)
TICKERS = [
        
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
                                    
]  # Example: ["AAPL", "MSFT", "GOOGL", "AMZN"]

# OR use company names directly (set TICKERS to None if using this)
COMPANIES = None  # Example: ["Apple Inc", "Microsoft", "International Business Machines"]

# Date range
START_YEAR = 1989
END_YEAR = 2015

# Output file path
OUTPUT_FILE = "nyt_sentiment_features.json"

# Use Archive API instead of Article Search (slower but more comprehensive)
USE_ARCHIVE_API = True

# Sentiment model from Hugging Face
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# ============================================================================

# Try to import transformers for sentiment analysis
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    SENTIMENT_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available. Install with: pip install transformers torch")
    SENTIMENT_AVAILABLE = False

# NYT API configuration
NYT_API_KEY = os.getenv("NYT_API_KEY", "")
NYT_ARCHIVE_BASE_URL = "https://api.nytimes.com/svc/archive/v1"
NYT_SEARCH_BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# Rate limiting: NYT allows 10 requests per minute, 4000 per day
NYT_RATE_LIMIT_DELAY = 6.1  # seconds between requests (slightly more than 6 to be safe)

# Cache file to track completed queries
CACHE_FILE = "nyt_query_cache.json"

# Default ticker to company name mapping (common S&P 500 companies)
TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "GOOG": "Google",
    "AMZN": "Amazon",
    "META": "Facebook",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "JPM": "JPMorgan Chase",
    "JPMorgan": "JPMorgan Chase",
    "BAC": "Bank of America",
    "V": "Visa",
    "MA": "Mastercard",
    "WFC": "Wells Fargo",
    "GS": "Goldman Sachs",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "WMT": "Walmart",
    "PG": "Procter & Gamble",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth",
    "PFE": "Pfizer",
    "HD": "Home Depot",
    "DIS": "Disney",
    "IBM": "IBM",
    "INTC": "Intel",
    "CSCO": "Cisco",
    "ORCL": "Oracle",
    "BA": "Boeing",
    "CAT": "Caterpillar",
    "GE": "General Electric",
    "GM": "General Motors",
    "F": "Ford",
    "NKE": "Nike",
    "MCD": "McDonald's",
    "KO": "Coca-Cola",
    "PEP": "Pepsi",
    "COST": "Costco",
    "TGT": "Target",
    "SBUX": "Starbucks",
    "NFLX": "Netflix",
    "ADBE": "Adobe",
    "CRM": "Salesforce",
    "AVGO": "Broadcom",
    "AMGN": "Amgen",
    "ABBV": "AbbVie",
    "MRK": "Merck",
    "TMO": "Thermo Fisher",
    "BLK": "BlackRock",
    "AXP": "American Express",
    "C": "Citigroup",
    "USB": "U.S. Bancorp",
    "PNC": "PNC",
    "HON": "Honeywell",
    "UPS": "UPS",
    "RTX": "Raytheon",
    "LMT": "Lockheed Martin",
}


def load_query_cache() -> Dict[str, bool]:
    """
    Load the query cache from disk.
    
    Returns:
        Dictionary mapping query keys to True if already processed
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_query_cache(cache: Dict[str, bool]):
    """
    Save the query cache to disk.
    
    Args:
        cache: Dictionary mapping query keys to True if already processed
    """
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save query cache: {e}")


def check_query_exists(company: str, year: int, month: Optional[int] = None, 
                       output_file: str = "nyt_sentiment_features.csv") -> bool:
    """
    Check if we already have data for this query by checking the output CSV file.
    
    Args:
        company: Company name
        year: Year
        month: Optional month (for Archive API)
        output_file: Path to output CSV file
    
    Returns:
        True if data already exists, False otherwise
    """
    # Check cache first
    cache = load_query_cache()
    if month is not None:
        cache_key = f"{company}_{year}_{month:02d}"
    else:
        cache_key = f"{company}_{year}"
    
    if cache.get(cache_key, False):
        return True
    
    # Check if output file exists and contains data for this query
    if os.path.exists(output_file):
        try:
            # Read just the first few lines to check if file has data
            with open(output_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return False
                
                # Check if company column exists
                try:
                    company_col_idx = header.index("company")
                except ValueError:
                    return False
                
                # Check a sample of rows to see if this company/year exists
                sample_size = 1000
                rows_checked = 0
                for row in reader:
                    if rows_checked >= sample_size:
                        break
                    if len(row) > company_col_idx:
                        if company.lower() in row[company_col_idx].lower():
                            # Check date if we have it
                            try:
                                date_col_idx = header.index("date")
                                if len(row) > date_col_idx:
                                    date_str = row[date_col_idx]
                                    if date_str.startswith(f"{year}"):
                                        # Found matching data, update cache
                                        cache[cache_key] = True
                                        save_query_cache(cache)
                                        return True
                            except (ValueError, IndexError):
                                pass
                    rows_checked += 1
        except Exception as e:
            # If there's an error reading, assume we don't have the data
            return False
    
    return False


def get_company_names_from_tickers(tickers: List[str]) -> Dict[str, List[str]]:
    """
    Map tickers to company names for NYT searches.
    
    Args:
        tickers: List of stock ticker symbols
    
    Returns:
        Dictionary mapping ticker to list of search terms (company names)
    """
    ticker_to_names = {}
    
    for ticker in tickers:
        # Direct lookup
        if ticker in TICKER_TO_COMPANY:
            ticker_to_names[ticker] = [TICKER_TO_COMPANY[ticker]]
        else:
            # For unmapped tickers, try the ticker itself and common patterns
            ticker_to_names[ticker] = [ticker]
    
    return ticker_to_names


def fetch_nyt_archive_month(year: int, month: int, query: Optional[str] = None) -> List[Dict]:
    """
    Fetch NYT articles for a specific month using Archive API.
    
    Args:
        year: Year (e.g., 1989)
        month: Month (1-12)
        query: Optional search query to filter articles
    
    Returns:
        List of article dictionaries with headline, date, etc.
    """
    if not NYT_API_KEY:
        print("Warning: NYT_API_KEY not set. Cannot fetch articles.")
        return []
    
    url = f"{NYT_ARCHIVE_BASE_URL}/{year}/{month}.json"
    params = {"api-key": NYT_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"  HTTP {response.status_code} error for {year}-{month:02d}")
            if response.status_code == 429:
                print("  Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
            return []
        
        data = response.json()
        
        if "response" not in data or "docs" not in data["response"]:
            return []
        
        docs = data["response"]["docs"]
        
        # Filter by query if provided (simple keyword matching)
        if query:
            query_lower = query.lower()
            docs = [
                d for d in docs
                if query_lower in d.get("headline", {}).get("main", "").lower()
                or query_lower in d.get("snippet", "").lower()
                or query_lower in " ".join([kw.get("value", "") for kw in d.get("keywords", [])]).lower()
            ]
        
        articles = []
        for doc in docs:
            try:
                # NYT API pub_date is in ISO 8601 format: "YYYY-MM-DDTHH:MM:SSZ"
                # This includes the exact day, month, year, and time of publication
                pub_date_str = doc.get("pub_date", "")
                if pub_date_str:
                    pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                else:
                    continue
                
                headline = doc.get("headline", {}).get("main", "")
                if not headline:
                    continue
                
                articles.append({
                    "headline": headline,
                    "date": pub_date.strftime("%Y-%m-%d"),  # Exact day preserved: YYYY-MM-DD
                    "datetime": pub_date.isoformat(),  # Full datetime with time: YYYY-MM-DDTHH:MM:SS
                    "snippet": doc.get("snippet", ""),
                    "url": doc.get("web_url", ""),
                    "section": doc.get("section_name", ""),
                    "keywords": [kw.get("value", "") for kw in doc.get("keywords", [])],
                })
            except Exception as e:
                continue
        
        return articles
    
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching {year}-{month:02d}: {e}")
        return []
    except Exception as e:
        print(f"  Error processing {year}-{month:02d}: {e}")
        return []


def fetch_nyt_article_search(query: str, begin_date: str, end_date: str, page: int = 0) -> Tuple[List[Dict], int]:
    """
    Fetch NYT articles using Article Search API (more targeted, can search by company name).
    
    Args:
        query: Search query (e.g., "Apple" or "IBM")
        begin_date: Start date in YYYYMMDD format (e.g., "19890101")
        end_date: End date in YYYYMMDD format (e.g., "19891231")
        page: Page number (0-indexed, 10 results per page)
    
    Returns:
        Tuple of (list of articles, total number of results)
    """
    if not NYT_API_KEY:
        return [], 0
    
    url = NYT_SEARCH_BASE_URL
    params = {
        "q": query,
        "begin_date": begin_date,
        "end_date": end_date,
        "page": page,
        "api-key": NYT_API_KEY,
        "sort": "newest",  # Sort by newest first
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            if response.status_code == 429:
                print("  Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
            return [], 0
        
        data = response.json()
        
        if "response" not in data:
            return [], 0
        
        response_data = data["response"]
        docs = response_data.get("docs", [])
        total_hits = response_data.get("meta", {}).get("hits", 0)
        
        articles = []
        for doc in docs:
            try:
                # NYT API pub_date is in ISO 8601 format: "YYYY-MM-DDTHH:MM:SSZ"
                # This includes the exact day, month, year, and time of publication
                pub_date_str = doc.get("pub_date", "")
                if pub_date_str:
                    pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                else:
                    continue
                
                headline = doc.get("headline", {}).get("main", "")
                if not headline:
                    continue
                
                articles.append({
                    "headline": headline,
                    "date": pub_date.strftime("%Y-%m-%d"),  # Exact day preserved: YYYY-MM-DD
                    "datetime": pub_date.isoformat(),  # Full datetime with time: YYYY-MM-DDTHH:MM:SS
                    "snippet": doc.get("snippet", ""),
                    "url": doc.get("web_url", ""),
                    "section": doc.get("section_name", ""),
                    "keywords": [kw.get("value", "") for kw in doc.get("keywords", [])],
                })
            except Exception as e:
                continue
        
        return articles, total_hits
    
    except requests.exceptions.RequestException as e:
        print(f"  Error in article search: {e}")
        return [], 0


def load_sentiment_model(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """
    Load a sentiment analysis model from Hugging Face.
    
    Args:
        model_name: Hugging Face model identifier
    
    Returns:
        Tuple of (tokenizer, model) or (None, None) if not available
    """
    if not SENTIMENT_AVAILABLE:
        return None, None
    
    try:
        print(f"Loading sentiment model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode
        print("✓ Sentiment model loaded successfully")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        print("Continuing without sentiment analysis...")
        return None, None


def analyze_sentiment(headlines: List[str], tokenizer, model) -> List[Dict]:
    """
    Analyze sentiment for a batch of headlines.
    
    Args:
        headlines: List of headline strings
        tokenizer: Hugging Face tokenizer
        model: Hugging Face sentiment model
    
    Returns:
        List of sentiment dictionaries with scores and labels
    """
    if tokenizer is None or model is None:
        # Return neutral scores if model not available
        return [{"label": "neutral", "score": 0.5} for _ in headlines]
    
    try:
        # Tokenize all headlines
        inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get label mappings (depends on the model)
        # For twitter-roberta-base-sentiment-latest: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        
        results = []
        for i, pred in enumerate(predictions):
            label_idx = pred.argmax().item()
            score = pred[label_idx].item()
            label = label_map.get(label_idx, "neutral")
            
            # Normalize to -1 to 1 scale (negative=-1, neutral=0, positive=1)
            if label == "negative":
                normalized_score = -score
            elif label == "positive":
                normalized_score = score
            else:
                normalized_score = 0.0
            
            results.append({
                "label": label,
                "score": score,
                "normalized_score": normalized_score,
            })
        
        return results
    
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return [{"label": "neutral", "score": 0.5, "normalized_score": 0.0} for _ in headlines]


def fetch_historical_headlines(
    company_names: List[str],
    start_year: int,
    end_year: int,
    use_article_search: bool = True,
    output_file: str = "nyt_sentiment_features.csv",
) -> Dict[str, List[Dict]]:
    """
    Fetch historical headlines for given company names from NYT.
    Checks cache before making API calls to avoid redundant requests.
    
    Args:
        company_names: List of company names to search for
        start_year: Start year (e.g., 1989)
        end_year: End year (e.g., 2000)
        use_article_search: If True, use Article Search API (more targeted). If False, use Archive API.
        output_file: Path to output CSV file (used for cache checking)
    
    Returns:
        Dictionary mapping company name to list of articles
    """
    all_articles = defaultdict(list)
    cache = load_query_cache()
    skipped_count = 0
    
    print(f"Fetching headlines for {len(company_names)} companies from {start_year} to {end_year}...")
    
    for company in company_names:
        print(f"\nSearching for: {company}")
        
        if use_article_search:
            # Use Article Search API (more targeted, better for company names)
            for year in range(start_year, end_year + 1):
                # Check if we already have this data
                if check_query_exists(company, year, output_file=output_file):
                    print(f"  {year}: Skipping (already processed)")
                    skipped_count += 1
                    # Load existing articles from CSV if available
                    try:
                        if os.path.exists(output_file):
                            df = pd.read_csv(output_file)
                            company_year_df = df[
                                (df["company"].str.lower() == company.lower()) &
                                (df["date"].str.startswith(str(year)))
                            ]
                            if len(company_year_df) > 0:
                                # Convert back to article format
                                for _, row in company_year_df.iterrows():
                                    all_articles[company].append({
                                        "headline": row["headline"],
                                        "date": row["date"],
                                        "datetime": row["datetime"],
                                        "snippet": row.get("snippet", ""),
                                        "url": row.get("url", ""),
                                        "section": row.get("section", ""),
                                    })
                                print(f"  {year}: Loaded {len(company_year_df)} existing articles from cache")
                                continue
                    except Exception as e:
                        print(f"  {year}: Error loading cache: {e}, fetching from API")
                
                begin_date = f"{year}0101"
                end_date = f"{year}1231"
                
                page = 0
                total_fetched = 0
                
                while True:
                    articles, total_hits = fetch_nyt_article_search(
                        query=company,
                        begin_date=begin_date,
                        end_date=end_date,
                        page=page
                    )
                    
                    if not articles:
                        break
                    
                    all_articles[company].extend(articles)
                    total_fetched += len(articles)
                    
                    print(f"  {year}: Found {len(articles)} articles (page {page + 1}, total: {total_fetched}/{total_hits})")
                    
                    # Check if there are more pages
                    if total_fetched >= total_hits or len(articles) < 10:
                        break
                    
                    page += 1
                    time.sleep(NYT_RATE_LIMIT_DELAY)  # Rate limiting
                
                # Mark this query as completed in cache
                cache_key = f"{company}_{year}"
                cache[cache_key] = True
                save_query_cache(cache)
                
                time.sleep(NYT_RATE_LIMIT_DELAY)  # Rate limiting between years
        
        else:
            # Use Archive API (month by month, slower but more comprehensive)
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    # Check if we already have this data
                    if check_query_exists(company, year, month, output_file=output_file):
                        print(f"  {year}-{month:02d}: Skipping (already processed)")
                        skipped_count += 1
                        # Load existing articles from CSV if available
                        try:
                            if os.path.exists(output_file):
                                df = pd.read_csv(output_file)
                                company_month_df = df[
                                    (df["company"].str.lower() == company.lower()) &
                                    (df["date"].str.startswith(f"{year}-{month:02d}"))
                                ]
                                if len(company_month_df) > 0:
                                    for _, row in company_month_df.iterrows():
                                        all_articles[company].append({
                                            "headline": row["headline"],
                                            "date": row["date"],
                                            "datetime": row["datetime"],
                                            "snippet": row.get("snippet", ""),
                                            "url": row.get("url", ""),
                                            "section": row.get("section", ""),
                                        })
                                    print(f"  {year}-{month:02d}: Loaded {len(company_month_df)} existing articles from cache")
                                    continue
                        except Exception as e:
                            print(f"  {year}-{month:02d}: Error loading cache: {e}, fetching from API")
                    
                    articles = fetch_nyt_archive_month(year, month, query=company)
                    if articles:
                        all_articles[company].extend(articles)
                        print(f"  {year}-{month:02d}: Found {len(articles)} articles")
                    
                    # Mark this query as completed in cache
                    cache_key = f"{company}_{year}_{month:02d}"
                    cache[cache_key] = True
                    save_query_cache(cache)
                    
                    time.sleep(NYT_RATE_LIMIT_DELAY)  # Rate limiting
    
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} queries that were already processed (cache hit)")
    
    return dict(all_articles)


def process_and_save_sentiment(
    articles_by_company: Dict[str, List[Dict]],
    ticker_mapping: Dict[str, str],
    output_file: str = "nyt_sentiment_features.json",
    sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
):
    """
    Process articles, add sentiment scores, and save to file incrementally.
    
    This function writes results progressively to avoid storing all data in memory.
    Uses JSONL (JSON Lines) format for JSON output and appends to CSV incrementally.
    
    Args:
        articles_by_company: Dictionary mapping company names to articles
        ticker_mapping: Dictionary mapping company names to tickers
        output_file: Output JSON file path
        sentiment_model_name: Hugging Face model name for sentiment
    """
    print("\n" + "=" * 60)
    print("Processing sentiment analysis...")
    print("=" * 60)
    
    # Load sentiment model
    tokenizer, model = load_sentiment_model(sentiment_model_name)
    
    # Prepare output files
    jsonl_file = output_file.replace(".json", ".jsonl")  # Use JSONL format for streaming
    csv_file = output_file.replace(".json", ".csv")
    
    # Open files for writing (will overwrite if they exist)
    jsonl_f = open(jsonl_file, "w", encoding="utf-8")
    csv_f = open(csv_file, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_f)
    
    # Write CSV header
    csv_writer.writerow([
        "ticker", "company", "date", "datetime", "headline", "snippet",
        "url", "section", "sentiment_label", "sentiment_score", "sentiment_normalized"
    ])
    
    # Track statistics for summary
    total_articles = 0
    unique_tickers = set()
    dates = []
    sentiment_scores = []
    
    try:
        # Process each company
        for company, articles in tqdm(articles_by_company.items(), desc="Processing companies"):
            ticker = ticker_mapping.get(company, company)
            
            # Extract headlines
            headlines = [article["headline"] for article in articles]
            
            if not headlines:
                continue
            
            # Analyze sentiment in batches
            batch_size = 32
            sentiment_results = []
            
            for i in range(0, len(headlines), batch_size):
                batch = headlines[i:i + batch_size]
                batch_sentiments = analyze_sentiment(batch, tokenizer, model)
                sentiment_results.extend(batch_sentiments)
            
            # Process and write articles incrementally
            for article, sentiment in zip(articles, sentiment_results):
                processed_item = {
                    "ticker": ticker,
                    "company": company,
                    "date": article["date"],
                    "datetime": article["datetime"],
                    "headline": article["headline"],
                    "snippet": article.get("snippet", ""),
                    "url": article.get("url", ""),
                    "section": article.get("section", ""),
                    "sentiment_label": sentiment["label"],
                    "sentiment_score": sentiment["score"],
                    "sentiment_normalized": sentiment["normalized_score"],  # -1 to 1 scale
                }
                
                # Write to JSONL (one JSON object per line)
                jsonl_f.write(json.dumps(processed_item, default=str) + "\n")
                
                # Write to CSV (csv module handles proper escaping automatically)
                csv_writer.writerow([
                    processed_item["ticker"],
                    processed_item["company"],
                    processed_item["date"],
                    processed_item["datetime"],
                    processed_item["headline"],
                    processed_item["snippet"],
                    processed_item["url"],
                    processed_item["section"],
                    processed_item["sentiment_label"],
                    processed_item["sentiment_score"],
                    processed_item["sentiment_normalized"],
                ])
                
                # Update statistics
                total_articles += 1
                unique_tickers.add(ticker)
                dates.append(article["date"])
                sentiment_scores.append(sentiment["normalized_score"])
                
                # Flush periodically to ensure data is written (every 100 articles)
                if total_articles % 100 == 0:
                    jsonl_f.flush()
                    csv_f.flush()
    
    finally:
        # Close files
        jsonl_f.close()
        csv_f.close()
    
    print(f"\nSaved {total_articles} processed articles to {jsonl_file}")
    print(f"Also saved CSV version: {csv_file}")
    
    # Create daily aggregated sentiment (useful for LSTM features)
    if total_articles > 0:
        # Read the CSV we just wrote to create daily aggregation
        print("Creating daily aggregated sentiment...")
        df = pd.read_csv(csv_file)
        df["date"] = pd.to_datetime(df["date"])
        
        # Group by ticker and date, aggregate sentiment
        daily_sentiment = df.groupby(["ticker", "date"]).agg({
            "sentiment_normalized": ["mean", "std", "count"],
        }).reset_index()
        
        daily_sentiment.columns = ["ticker", "date", "sentiment_mean", "sentiment_std", "headline_count"]
        daily_sentiment = daily_sentiment.fillna(0)  # Fill NaN std with 0
        
        # Save daily aggregated sentiment
        daily_file = output_file.replace(".json", "_daily.csv")
        daily_sentiment.to_csv(daily_file, index=False)
        print(f"Daily aggregated sentiment saved: {daily_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total articles: {total_articles}")
        print(f"Unique companies: {len(unique_tickers)}")
        if dates:
            print(f"Date range: {min(dates)} to {max(dates)}")
        if sentiment_scores:
            print(f"Average sentiment: {sum(sentiment_scores) / len(sentiment_scores):.3f}")
        print(f"Daily sentiment records: {len(daily_sentiment)}")
    
    # Also create a standard JSON array file if requested (for backwards compatibility)
    # Note: This requires reading the file, so only do it if needed
    if output_file.endswith(".json"):
        print(f"\nNote: JSONL format saved to {jsonl_file}")
        print("To convert to JSON array format, you can use: jq -s '.' {jsonl_file} > {output_file}")
    
    return total_articles


def main():
    # Check API key
    if not NYT_API_KEY:
        print("=" * 60)
        print("ERROR: NYT_API_KEY not set!")
        print("=" * 60)
        print("Please set your New York Times API key:")
        print("  export NYT_API_KEY='your_api_key_here'")
        print("\nGet a free API key at: https://developer.nytimes.com/")
        return
    
    # Determine companies to search
    if TICKERS:
        ticker_to_companies = get_company_names_from_tickers(TICKERS)
        company_names = []
        ticker_mapping = {}  # company name -> ticker
        
        for ticker, companies in ticker_to_companies.items():
            for company in companies:
                company_names.append(company)
                ticker_mapping[company] = ticker
        
        print(f"Searching for {len(TICKERS)} tickers: {TICKERS}")
        print(f"Using company names: {company_names}")
    
    elif COMPANIES:
        company_names = COMPANIES
        ticker_mapping = {name: name for name in company_names}
        print(f"Searching for companies: {company_names}")
    
    else:
        print("ERROR: Please set either TICKERS or COMPANIES in the configuration section")
        return
    
    # Fetch headlines (pass output file path for cache checking)
    csv_file = OUTPUT_FILE.replace(".json", ".csv")
    articles_by_company = fetch_historical_headlines(
        company_names=company_names,
        start_year=START_YEAR,
        end_year=END_YEAR,
        use_article_search=not USE_ARCHIVE_API,
        output_file=csv_file,
    )
    
    # Process and save with sentiment
    process_and_save_sentiment(
        articles_by_company=articles_by_company,
        ticker_mapping=ticker_mapping,
        output_file=OUTPUT_FILE,
        sentiment_model_name=SENTIMENT_MODEL,
    )
    
    print("\n" + "=" * 60)
    print("Done! Use the output files to align sentiment features with your price series.")
    print("=" * 60)


if __name__ == "__main__":
    main()

