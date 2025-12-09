#!/bin/bash

export QUICK_TEST_EPOCHS=2

source /Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.venv/bin/activate

#echo "run lstm base"
#python main_lstm_base.py

#echo "run lstm nlp full"
#python main_lstm_full_nlp.py

#echo "run lstm base full"
#python main_lstm_full_base.py

echo "run caelstm base test set"
python main_caelstm_base.py

echo "run aelstm base test set"
python main_aelstm_base.py
