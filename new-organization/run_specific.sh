#!/bin/bash

export QUICK_TEST_EPOCHS=2

source /Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.venv/bin/activate

echo "run lstm base"
python main_lstm_base.py

echo "run aelstm base"
python main_aelstm_base.py

echo "run caelstm base"
python main_caelstm_base.py

echo "run lstm nlp"
python main_lstm_nlp.py

echo "run aelstm nlp"
python main_aelstm_nlp.py

echo "run caelstm nlp"
python main_caelstm_nlp.py
