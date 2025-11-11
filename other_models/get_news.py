"""
Script to fetch news headlines associated with stock market tickers.

This script uses the Finnhub API to retrieve company news and market headlines.
Finnhub provides a free tier with 60 API calls per minute.

Usage:
    # As a script
    python get_news.py AAPL MSFT GOOGL
    
    # As a module
    from get_news import get_news_for_ticker, get_news_for_tickers
    news = get_news_for_ticker("AAPL")
"""

import requests
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time


# Finnhub API configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# NewsAPI configuration (alternative)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_BASE_URL = "https://newsapi.org/v2"


def get_news_for_ticker_newsapi(ticker: str, days_back: int = 30) -> List[Dict]:
    """
    Alternative: Fetch news using NewsAPI (requires NEWSAPI_KEY).
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
        days_back: Number of days to look back for news (default: 30)
    
    Returns:
        List of dictionaries containing news articles
    """
    if not NEWSAPI_KEY:
        return []
    
    # Calculate date range
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")
    
    url = f"{NEWSAPI_BASE_URL}/everything"
    params = {
        "q": ticker,
        "from": from_date,
        "to": to_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWSAPI_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"  NewsAPI HTTP {response.status_code} error for {ticker}")
            return []
        
        data = response.json()
        
        if data.get("status") != "ok":
            print(f"  NewsAPI error for {ticker}: {data.get('message', 'Unknown error')}")
            return []
        
        articles = data.get("articles", [])
        # Convert NewsAPI format to similar format as Finnhub
        formatted_articles = []
        for article in articles:
            # Parse datetime safely
            published_at = article.get("publishedAt", "")
            timestamp = 0
            if published_at:
                try:
                    # Handle ISO format with Z timezone
                    dt_str = published_at.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(dt_str)
                    timestamp = int(dt.timestamp())
                except (ValueError, AttributeError):
                    timestamp = 0
            
            formatted_articles.append({
                "headline": article.get("title", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "summary": article.get("description", ""),
                "url": article.get("url", ""),
                "datetime": timestamp,
                "image": article.get("urlToImage", "")
            })
        
        if len(formatted_articles) > 0:
            print(f"  Found {len(formatted_articles)} articles for {ticker} (via NewsAPI)")
        return formatted_articles
    
    except requests.exceptions.RequestException as e:
        print(f"  NewsAPI error for {ticker}: {e}")
        return []


def get_news_for_ticker(ticker: str, days_back: int = 90) -> List[Dict]:
    """
    Fetch news headlines for a specific stock ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
        days_back: Number of days to look back for news (default: 30)
    
    Returns:
        List of dictionaries containing news articles with fields:
        - category: News category
        - datetime: Unix timestamp
        - headline: News headline
        - id: Unique article ID
        - image: Image URL
        - related: Related tickers
        - source: News source
        - summary: Article summary
        - url: Article URL
    """
    if not FINNHUB_API_KEY:
        print(f"  FINNHUB_API_KEY not set. Trying NewsAPI as fallback...")
        # Try NewsAPI as fallback
        newsapi_result = get_news_for_ticker_newsapi(ticker, days_back)
        if newsapi_result:
            return newsapi_result
        print(f"  Cannot fetch news for {ticker}. Get a free API key:")
        print(f"    - Finnhub: https://finnhub.io/register")
        print(f"    - NewsAPI: https://newsapi.org/register")
        return []
    
    # Calculate date range
    today = datetime.now()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")
    
    url = f"{FINNHUB_BASE_URL}/company-news"
    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        # Check status code
        if response.status_code != 200:
            print(f"  HTTP {response.status_code} error for {ticker}: {response.text[:200]}")
            return []
        
        news_data = response.json()
        
        # Debug: print what we got
        if not news_data:
            print(f"  No news data returned for {ticker} (empty response)")
            return []
        
        # Handle API errors
        if isinstance(news_data, dict):
            if "error" in news_data:
                error_msg = news_data.get('error', 'Unknown error')
                print(f"  API error for {ticker}: {error_msg}")
                return []
            # Sometimes API returns dict with specific structure
            if "data" in news_data:
                news_data = news_data["data"]
        
        # Limit results
        if isinstance(news_data, list):
            if len(news_data) > 0:
                print(f"  Found {len(news_data)} articles for {ticker}")
            else:
                # If Finnhub returns empty, try NewsAPI as fallback
                if not news_data and NEWSAPI_KEY:
                    print(f"  Finnhub returned no results for {ticker}, trying NewsAPI...")
                    newsapi_result = get_news_for_ticker_newsapi(ticker, days_back)
                    if newsapi_result:
                        return newsapi_result
            return news_data
        
        print(f"  Unexpected response format for {ticker}: {type(news_data)}")
        if isinstance(news_data, dict):
            print(f"  Response keys: {list(news_data.keys())}")
        
        # Try NewsAPI as fallback
        if NEWSAPI_KEY:
            print(f"  Trying NewsAPI as fallback for {ticker}...")
            newsapi_result = get_news_for_ticker_newsapi(ticker, days_back)
            if newsapi_result:
                return newsapi_result
        
        return []
    
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching news for {ticker}: {e}")
        return []


def get_market_news(category: str = "general") -> List[Dict]:
    """
    Fetch general market news headlines.
    
    Args:
        category: News category - "general", "forex", "crypto", "merger" (default: "general")
    
    Returns:
        List of dictionaries containing news articles
    """
    if not FINNHUB_API_KEY:
        print("Warning: FINNHUB_API_KEY not set. Cannot fetch market news.")
        print("Get a free API key at: https://finnhub.io/register")
        return []
    
    url = f"{FINNHUB_BASE_URL}/news"
    params = {
        "category": category,
        "token": FINNHUB_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        news_data = response.json()
        
        if isinstance(news_data, list):
            return news_data
        
        return []
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching market news: {e}")
        return []


def get_news_for_tickers(tickers: List[str], delay: float = 0.5) -> Dict[str, List[Dict]]:
    """
    Fetch news for multiple stock tickers.
    
    Args:
        tickers: List of stock ticker symbols
        delay: Delay between API calls in seconds to respect rate limits (default: 0.5)
    
    Returns:
        Dictionary mapping ticker symbols to their news articles
    """
    results = {}
    
    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        news = get_news_for_ticker(ticker)
        results[ticker] = news
        
        # Add delay to respect rate limits (60 calls/minute for free tier)
        if delay > 0:
            time.sleep(delay)
    
    return results


def format_news_article(article: Dict) -> str:
    """
    Format a news article dictionary into a readable string.
    
    Args:
        article: News article dictionary
    
    Returns:
        Formatted string representation
    """
    headline = article.get("headline", "N/A")
    source = article.get("source", "Unknown")
    
    # Convert Unix timestamp to readable date
    timestamp = article.get("datetime", 0)
    if timestamp:
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    else:
        date_str = "N/A"
    
    url = article.get("url", "")
    summary = article.get("summary", "")
    
    formatted = f"Headline: {headline}\n"
    formatted += f"Source: {source}\n"
    formatted += f"Date: {date_str}\n"
    if summary:
        formatted += f"Summary: {summary[:200]}...\n" if len(summary) > 200 else f"Summary: {summary}\n"
    if url:
        formatted += f"URL: {url}\n"
    
    return formatted


def print_news_summary(ticker: str, news: List[Dict], max_articles: int = 5):
    """
    Print a summary of news articles for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        news: List of news article dictionaries
        max_articles: Maximum number of articles to print (default: 5)
    """
    if not news:
        print(f"\nNo news found for {ticker}")
        return
    
    print(f"\n{'='*60}")
    print(f"News for {ticker} ({len(news)} articles)")
    print(f"{'='*60}")
    
    for i, article in enumerate(news[:max_articles], 1):
        print(f"\n--- Article {i} ---")
        print(format_news_article(article))


def main():
    """
    Main function for command-line usage.
    """
    # Check API keys
    print("=" * 60)
    print("Stock News Fetcher")
    print("=" * 60)
    if FINNHUB_API_KEY:
        print(f"✓ Finnhub API key found (starts with: {FINNHUB_API_KEY[:8]}...)")
    else:
        print("✗ Finnhub API key not set (set FINNHUB_API_KEY env var)")
        print("  Get free key at: https://finnhub.io/register")
    
    if NEWSAPI_KEY:
        print(f"✓ NewsAPI key found (starts with: {NEWSAPI_KEY[:8]}...)")
    else:
        print("✗ NewsAPI key not set (set NEWSAPI_KEY env var)")
        print("  Get free key at: https://newsapi.org/register")
    
    if not FINNHUB_API_KEY and not NEWSAPI_KEY:
        print("\n⚠️  WARNING: No API keys configured. Cannot fetch news.")
        print("   Please set at least one API key and try again.")
        return
    
    print("=" * 60)
    print()
    
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",
    "JPM", "BAC", "V", "MA", "WFC", "GS", "BLK", "AXP",
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO",
    "WMT", "PG", "HD", "COST", "MCD", "NKE",
    "BA", "CAT", "XOM", "CVX"]

    print(f"Fetching news for {len(tickers)} tickers...")
    print()
    
    # Fetch news for all tickers
    all_news = get_news_for_tickers(tickers, delay=0.5)
    
    # Print summaries
    for ticker, news in all_news.items():
        print_news_summary(ticker, news, max_articles=5)
    
    # Optionally save to JSON
    output_file = "news_headlines.json"
    with open(output_file, "w") as f:
        json.dump(all_news, f, indent=2, default=str)
    print(f"\n\nNews data saved to {output_file}")


if __name__ == "__main__":
    main()

