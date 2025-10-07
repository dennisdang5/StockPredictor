import train
import torch

print("my code")
start = "1989-12-01"
end = "2015-09-30"

# Top 30 S&P 500 stocks by market cap (subset for testing)
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
trainer = train.Trainer(stocks=stocks, time_args=[start,end])

for name, param in trainer.lstmModel.named_parameters():
    print("name: {}".format(name))
    print("param: {}".format(param.numel()))
    print()

#total_epochs = trainer.num_epochs
total_epochs = 2

for epoch in range(total_epochs):
    stop = trainer.train_one_epoch(epoch)
    if stop: 
        print("Early stop at epoch: {}".format(epoch))
        trainer.stop()
        break
