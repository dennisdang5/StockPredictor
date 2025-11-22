"""
Verify yfinance news download functionality for individual NLP method.

This script tests:
1. Direct yfinance news API access
2. News data structure and fields
3. Date parsing and filtering
4. News availability for different stocks and date ranges
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, date
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available")


def test_direct_yfinance_news():
    """Test 1: Direct yfinance news API access."""
    print("=" * 80)
    print("TEST 1: Direct yfinance News API Access")
    print("=" * 80)
    
    if not YFINANCE_AVAILABLE:
        print("✗ yfinance not available - skipping test")
        return None
    
    test_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    
    print(f"\nTesting news access for stocks: {test_stocks}")
    
    news_results = {}
    for stock in test_stocks:
        print(f"\n{stock}:")
        try:
            ticker = yf.Ticker(stock)
            news = ticker.news
            
            if news is None:
                print(f"  ✗ News is None")
                news_results[stock] = None
                continue
            
            if len(news) == 0:
                print(f"  ⚠ No news articles returned (empty list)")
                news_results[stock] = []
                continue
            
            print(f"  ✓ Retrieved {len(news)} news articles")
            
            # Show first article structure
            if len(news) > 0:
                first_article = news[0]
                print(f"  First article keys: {list(first_article.keys())}")
                print(f"  Sample article:")
                for key in ['title', 'providerPublishTime', 'pubDate', 'uuid', 'type']:
                    if key in first_article:
                        value = first_article[key]
                        if key in ['providerPublishTime', 'pubDate']:
                            try:
                                if isinstance(value, (int, float)):
                                    dt = pd.to_datetime(value, unit='s')
                                    print(f"    {key}: {value} -> {dt}")
                                else:
                                    print(f"    {key}: {value}")
                            except:
                                print(f"    {key}: {value}")
                        else:
                            print(f"    {key}: {str(value)[:100]}")
            
            news_results[stock] = news
            
            # Rate limiting
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            news_results[stock] = None
    
    return news_results


def test_news_date_parsing(news_results):
    """Test 2: News date parsing and filtering."""
    print("\n" + "=" * 80)
    print("TEST 2: News Date Parsing and Filtering")
    print("=" * 80)
    
    if news_results is None:
        print("Skipping - no news results available")
        return
    
    for stock, news in news_results.items():
        if news is None or len(news) == 0:
            continue
        
        print(f"\n{stock}:")
        print(f"  Total articles: {len(news)}")
        
        # Parse dates
        articles_with_dates = []
        for article in news:
            pub_date = None
            pub_time = None
            
            # Try providerPublishTime (Unix timestamp)
            if 'providerPublishTime' in article:
                try:
                    pub_time = pd.to_datetime(article['providerPublishTime'], unit='s')
                    pub_date = pub_time.date()
                except:
                    pass
            
            # Try pubDate (alternative format)
            if pub_date is None and 'pubDate' in article:
                try:
                    pub_time = pd.to_datetime(article['pubDate'])
                    pub_date = pub_time.date()
                except:
                    pass
            
            if pub_date:
                articles_with_dates.append({
                    'date': pub_date,
                    'datetime': pub_time,
                    'title': article.get('title', ''),
                    'article': article
                })
        
        print(f"  Articles with valid dates: {len(articles_with_dates)}")
        
        if len(articles_with_dates) > 0:
            # Show date range
            dates = [a['date'] for a in articles_with_dates]
            min_date = min(dates)
            max_date = max(dates)
            print(f"  Date range: {min_date} to {max_date}")
            
            # Show sample dates
            print(f"  Sample dates (first 5):")
            for i, article in enumerate(articles_with_dates[:5]):
                print(f"    {i+1}. {article['date']}: {article['title'][:60]}")
            
            # Test date filtering
            test_start = date(2020, 1, 1)
            test_end = date(2020, 12, 31)
            
            filtered = [
                a for a in articles_with_dates
                if test_start <= a['date'] <= test_end
            ]
            print(f"\n  Filtered to {test_start} - {test_end}: {len(filtered)} articles")
            
            if len(filtered) > 0:
                print(f"  Filtered date range: {min(a['date'] for a in filtered)} to {max(a['date'] for a in filtered)}")
            else:
                print(f"  ⚠ No articles in test date range (2020)")
                print(f"  This explains why individual NLP extraction returned no data!")


def test_news_structure():
    """Test 3: Detailed news structure analysis."""
    print("\n" + "=" * 80)
    print("TEST 3: Detailed News Structure Analysis")
    print("=" * 80)
    
    if not YFINANCE_AVAILABLE:
        print("✗ yfinance not available - skipping test")
        return
    
    # Test with a popular stock that likely has recent news
    test_stock = "AAPL"
    print(f"\nFetching recent news for {test_stock}...")
    
    try:
        ticker = yf.Ticker(test_stock)
        news = ticker.news
        
        if news is None or len(news) == 0:
            print(f"✗ No news available for {test_stock}")
            return
        
        print(f"✓ Retrieved {len(news)} articles")
        
        # Analyze structure
        print(f"\nAnalyzing article structure...")
        all_keys = set()
        for article in news:
            all_keys.update(article.keys())
        
        print(f"  All keys found: {sorted(all_keys)}")
        
        # Count articles with each key
        key_counts = {}
        for key in all_keys:
            count = sum(1 for article in news if key in article)
            key_counts[key] = count
        
        print(f"\n  Key frequency:")
        for key, count in sorted(key_counts.items()):
            print(f"    {key}: {count}/{len(news)} articles")
        
        # Show a complete sample article
        print(f"\n  Complete sample article (first one):")
        sample = news[0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"    {key}: {value[:100]}...")
            else:
                print(f"    {key}: {value}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_recent_vs_historical():
    """Test 4: Compare recent vs historical news availability."""
    print("\n" + "=" * 80)
    print("TEST 4: Recent vs Historical News Availability")
    print("=" * 80)
    
    if not YFINANCE_AVAILABLE:
        print("✗ yfinance not available - skipping test")
        return
    
    test_stock = "AAPL"
    print(f"\nTesting {test_stock}...")
    
    try:
        ticker = yf.Ticker(test_stock)
        news = ticker.news
        
        if news is None or len(news) == 0:
            print(f"✗ No news available")
            return
        
        print(f"✓ Retrieved {len(news)} articles")
        
        # Parse dates
        articles_with_dates = []
        for article in news:
            pub_date = None
            
            if 'providerPublishTime' in article:
                try:
                    pub_time = pd.to_datetime(article['providerPublishTime'], unit='s')
                    pub_date = pub_time.date()
                except:
                    pass
            
            if pub_date is None and 'pubDate' in article:
                try:
                    pub_time = pd.to_datetime(article['pubDate'])
                    pub_date = pub_time.date()
                except:
                    pass
            
            if pub_date:
                articles_with_dates.append(pub_date)
        
        if len(articles_with_dates) == 0:
            print("✗ No articles with valid dates")
            return
        
        dates = sorted(articles_with_dates)
        min_date = min(dates)
        max_date = max(dates)
        
        print(f"\nDate range: {min_date} to {max_date}")
        print(f"Total articles with dates: {len(dates)}")
        
        # Check recent vs historical
        today = date.today()
        one_year_ago = date(today.year - 1, today.month, today.day)
        two_years_ago = date(today.year - 2, today.month, today.day)
        
        recent = [d for d in dates if d >= one_year_ago]
        one_year = [d for d in dates if two_years_ago <= d < one_year_ago]
        older = [d for d in dates if d < two_years_ago]
        
        print(f"\nNews distribution:")
        print(f"  Recent (last year): {len(recent)} articles")
        print(f"  1-2 years ago: {len(one_year)} articles")
        print(f"  Older than 2 years: {len(older)} articles")
        
        # Check 2020 specifically
        year_2020 = [d for d in dates if d.year == 2020]
        print(f"\n  2020 specifically: {len(year_2020)} articles")
        
        if len(year_2020) == 0:
            print(f"\n⚠ WARNING: No news articles from 2020!")
            print(f"  This explains why individual NLP extraction returned no data for 2020.")
            print(f"  yfinance news API appears to only return recent news (typically last few months).")
            print(f"  For historical data (1990-2015), individual method may not work.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_extract_function():
    """Test 5: Test the actual extract_daily_nlp_features_yfinance function."""
    print("\n" + "=" * 80)
    print("TEST 5: extract_daily_nlp_features_yfinance Function")
    print("=" * 80)
    
    try:
        from nlp_features import extract_daily_nlp_features_yfinance
        
        # Test with recent date range (more likely to have news)
        from datetime import datetime, timedelta
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)  # Last 90 days
        
        print(f"\nTesting with recent date range:")
        print(f"  Start: {start_date}")
        print(f"  End: {end_date}")
        print(f"  Stocks: ['AAPL', 'MSFT']")
        
        stock_nlp_features = extract_daily_nlp_features_yfinance(
            stocks=['AAPL', 'MSFT'],
            start_date=start_date,
            end_date=end_date,
            batch_size=32,
            progress=True
        )
        
        print(f"\nResults:")
        for stock, nlp_df in stock_nlp_features.items():
            if len(nlp_df) > 0:
                print(f"  {stock}: {len(nlp_df)} days with news")
                print(f"    Date range: {nlp_df['date'].min()} to {nlp_df['date'].max()}")
            else:
                print(f"  {stock}: No news data")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("YFINANCE NEWS DOWNLOAD VERIFICATION")
    print("=" * 80)
    
    # Test 1: Direct API access
    news_results = test_direct_yfinance_news()
    
    # Test 2: Date parsing
    test_news_date_parsing(news_results)
    
    # Test 3: Structure analysis
    test_news_structure()
    
    # Test 4: Recent vs historical
    test_recent_vs_historical()
    
    # Test 5: Extract function with recent dates
    test_extract_function()
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. yfinance news API typically returns recent news only (last few months)")
    print("2. Historical news (e.g., 1990-2015) may not be available via yfinance")
    print("3. For historical data, use 'aggregated' method with NYT articles instead")
    print("4. Individual method works best for recent date ranges")


if __name__ == "__main__":
    main()

