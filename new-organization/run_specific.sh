#!/bin/bash

export QUICK_TEST_EPOCHS=2

source /Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.venv/bin/activate

python main_aelstm_base.py
python main_caelstm_base.py

python main_lstm_nlp.py
python main_aelstm_nlp.py
python main_caelstm_nlp.py

#echo "run caelstm full base"
#python main_caelstm_full_base.py

#echo "run caelstm full nlp"
#python main_caelstm_full_nlp.py

#echo "run aelstm full base"
#python main_aelstm_full_base.py

#echo "run aelstm full nlp"
#python main_aelstm_full_nlp.py
