import trainer
import torch

print("my code")
#start = "1989-12-01"
start = "2000-12-01"
end = "2015-09-30"

# Top 30 S&P 500 stocks by market cap (subset for testing)
stocks = [
    # Top Technology & Growth
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM"]

"""  
# Major Financial Services
"JPM", "BAC", "V", "MA", "WFC", "GS", "BLK", "AXP",

# Healthcare Leaders
"JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO",

# Consumer & Retail Giants
"WMT", "PG", "HD", "COST", "MCD", "NKE",

# Industrial & Energy Leaders
"BA", "CAT", "XOM", "CVX"
"""

trainer = trainer.Trainer(stocks=stocks, time_args=[start,end], num_epochs=5000)

if type(trainer) == int:
    print("Error getting data")
    exit()


