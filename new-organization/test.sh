#!/bin/bash
source /Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.venv/bin/activate

export QUICK_TEST_EPOCHS=2

#echo "Running Portfolio LSTM Independent Base"
#python main_portfolio_lstm_independent_base.py

#echo "Running Portfolio LSTM Shared Base"
#python main_portfolio_lstm_shared_base.py

#echo "Running Portfolio TabPFN Independent Base"
#python main_portfolio_tabpfn_individual_base.py

echo "Running Portfolio TabPFN Shared Base"
python main_portfolio_tabpfn_shared_base.py
