import train
import torch

trainer = train.Trainer()

for epoch in range(trainer.num_epochs):
    stop = trainer.train_one_epoch(epoch)
    if stop: 
        print("Early stop at epoch: {}".format(epoch))
        trainer.writer.close()