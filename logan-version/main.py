import trainer
import trainer
import torch

def pick_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"[device] CUDA: {name}")
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[device] Apple Metal Performance Shaders (mps)")
    else:
        dev = torch.device("cpu")
        print("[device] CPU")
    return dev

print("my code")
start = "1989-12-01"
end = "2015-09-30"

# Top 30 S&P 500 stocks by market cap (subset for testing)
stocks = [
    # Top Technology & Growth
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",
]

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

train_obj = trainer.Trainer(stocks=stocks, time_args=[start,end], num_epochs=5000)

if type(train_obj) == int:
    print("Error getting data")
    exit()

for name, param in train_obj.Model.named_parameters():
    print("name: {}".format(name))
    print("param: {}".format(param.numel()))
    print()

#total_epochs = train_obj.num_epochs
total_epochs = 2

for epoch in range(total_epochs):
    stop = train_obj.train_one_epoch(epoch)
    if stop: 
        print("Early stop at epoch: {}".format(epoch))
        train_obj.stop()
        break
