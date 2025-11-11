# Other Models and Utilities

This directory contains alternative models and utility scripts for stock prediction.

## Files

### 1. ARIMA.py
ARIMA model implementation for time series forecasting.

### 2. get_news.py
Fetches recent news headlines from Finnhub and NewsAPI for stock tickers (limited to recent dates, typically last few months/years).

### 3. get_nyt_sentiment.py
**Historical news headlines with sentiment analysis for LSTM features.**

Fetches historical headlines from the New York Times API (going back to 1989) and runs sentiment analysis to create features for LSTM stock prediction models.

**Features:**
- Maps stock tickers to company names for NYT searches
- Fetches headlines from NYT Archive API (1989+) and Article Search API
- Runs sentiment analysis using Hugging Face transformers
- Stores time-indexed sentiment scores aligned with price series
- Creates daily aggregated sentiment features ready for LSTM input

**Setup:**
```bash
# Install dependencies
pip install requests transformers torch pandas tqdm

# Set your NYT API key (get free key at https://developer.nytimes.com/)
export NYT_API_KEY="your_api_key_here"
```

**Usage:**
```bash
# Search by tickers (automatically maps to company names)
python get_nyt_sentiment.py --tickers AAPL MSFT IBM --start-year 1989 --end-year 2000

# Search by company names directly
python get_nyt_sentiment.py --companies "Apple Inc" "Microsoft" --start-year 1989 --end-year 2000

# Use Archive API instead of Article Search (slower but more comprehensive)
python get_nyt_sentiment.py --tickers AAPL --start-year 1989 --end-year 1990 --use-archive

# Specify custom output file
python get_nyt_sentiment.py --tickers AAPL --start-year 1989 --end-year 2000 --output my_sentiment_data.json
```

**Output Files:**
- `nyt_sentiment_features.json`: Full article data with sentiment scores
- `nyt_sentiment_features.csv`: CSV version for easy inspection
- `nyt_sentiment_features_daily.csv`: Daily aggregated sentiment (mean, std, count) - ready for LSTM features

**Integration with LSTM:**
The daily aggregated CSV can be aligned with your price series by date and ticker, then used as an additional feature input to your LSTM model.

**Note:** NYT API has rate limits (10 requests/minute, 4000/day). The script automatically handles rate limiting with delays between requests.

### 4. BERT_tiny.py
Example script demonstrating BERT tiny model usage for text embeddings.
