#!/bin/bash

source /Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.venv/bin/activate

echo "run lstm base"
python main_lstm_base.py

ech "run lstm nlp"
python main_lstm_nlp.py