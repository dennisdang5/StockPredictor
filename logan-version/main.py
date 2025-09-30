import train
import torch

print("my code")
start = "1989-12-01"
end = "2015-09-30"
stocks = ["AAPL","MSFT"]
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
