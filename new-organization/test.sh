#!/bin/bash
source /Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.venv/bin/activate

export QUICK_TEST_EPOCHS=2

#echo "Running LSTM Base NLP"
#python main_lstm_base_nlp.py

#echo "Running CNNLSTM Base"
#python main_cnnlstm_base.py

#echo "Running AELSTM Base"
#python main_aelstm_base.py

echo "Running CNNAELSTM Base"
python main_cnnaelstm_base.py

#echo "Running TimesNet Base"
#python main_timesnet_small_base.py

echo "Running Portfolio LSTM Base"
python main_portfolio_lstm_base.py