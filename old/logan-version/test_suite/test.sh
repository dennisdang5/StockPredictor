conda activate csci566-project
python evaluate_model.py --model-name savedmodel_classification_cnn_lstm.pth --model-type cnn_lstm
python evaluation_analysis.py \
        --input evaluation_results_savedmodel_classification_cnn_lstm.json \
        --output evaluation_analysis_report_cnn_lstm.json \
        --portfolio-table-csv portfolio_table_cnn_lstm.csv