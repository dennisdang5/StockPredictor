#!/bin/bash

#export QUICK_TEST_EPOCHS=2

source /Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.venv/bin/activate

#echo "run lstm nlp full"
#python main_lstm_full_nlp.py

#echo "run lstm base full"
#python main_lstm_full_base.py

#echo "run caelstm base"
#python main_caelstm_base.py

#echo "run aelstm base"#
#python main_aelstm_base.py

#echo "run aelstm nlp"
#python main_aelstm_nlp.py

echo "run caelstm nlp"
python main_caelstm_nlp.py

echo "run caelstm full base"
python main_caelstm_full_base.py

echo "run caelstm full nlp"
python main_caelstm_full_nlp.py

