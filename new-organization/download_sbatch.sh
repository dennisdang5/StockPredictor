#!/bin/bash

source /scratch1/lkyamamo/stock_prediction/new_train/StockPredictor/.venv/bin/activate 

echo "starting download"

python download_data.py

echo "finish download"

sbatch job.slurm
