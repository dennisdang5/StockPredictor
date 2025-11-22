"""
Production S&P 500 Top 100 Stock Prediction Training and Evaluation

This script trains and evaluates the LSTM model on the top 100 S&P 500 stocks
by market cap for the specified timeframe (1989-12-01 to 2015-09-30).
This provides a good balance between comprehensive coverage and computational efficiency.
"""

import trainer
import torch
import os
import time
from datetime import datetime

def get_top100_sp500_stocks():
    """
    Return the top 100 S&P 500 stocks by market cap for the given timeframe.
    These are the most liquid and important stocks that were in the S&P 500 during 1989-2015.
    """
    return [
        # Technology & Growth (25 stocks)
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",
        "ADBE", "NFLX", "INTC", "AMD", "CSCO", "IBM", "QCOM", "TXN", "AMAT", "MU",
        "NOW", "SNOW", "PLTR", "CRWD", "ZM",
        
        # Financial Services (25 stocks)
        "JPM", "BAC", "V", "MA", "WFC", "GS", "BLK", "AXP", "C", "USB",
        "PNC", "TFC", "COF", "SCHW", "MS", "BK", "CB", "AON", "MMC", "SPGI",
        "FIS", "FISV", "ADP", "PAYX", "WLTW",
        
        # Healthcare (20 stocks)
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "DHR", "ABT", "BMY", "AMGN",
        "GILD", "BIIB", "REGN", "VRTX", "ILMN", "MRNA", "ZTS", "ISRG", "EW", "CI",
        
        # Consumer & Retail (15 stocks)
        "WMT", "PG", "HD", "COST", "MCD", "NKE", "KO", "PEP", "SBUX", "TGT",
        "LOW", "TJX", "BKNG", "CMG", "YUM",
        
        # Industrial & Energy (15 stocks)
        "BA", "CAT", "XOM", "CVX", "GE", "HON", "UPS", "FDX", "LMT", "RTX",
        "NOC", "GD", "EMR", "ETN", "ITW"
    ]

def main():
    """Main production training and evaluation pipeline for top 100 S&P 500 stocks."""
    
    print("=" * 80)
    print("S&P 500 TOP 100 PRODUCTION TRAINING & EVALUATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    start_date = "1989-12-01"
    end_date = "2015-09-30"
    prediction_type = "classification"
    model_path = f"savedmodel_top100_{prediction_type}.pth"
    
    # Get top 100 S&P 500 stocks
    stocks = get_top100_sp500_stocks()
    print(f"Training on {len(stocks)} top S&P 500 stocks")
    print(f"Timeframe: {start_date} to {end_date}")
    print(f"Prediction type: {prediction_type}")
    print(f"Model will be saved as: {model_path}")
    
    # Training phase
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        train_obj = trainer.Trainer(
            stocks=stocks, 
            time_args=[start_date, end_date], 
            num_epochs=2000,  # More epochs for better convergence
            prediction_type=prediction_type, 
            saved_model=model_path
        )
        
        if type(train_obj) == int:
            print("❌ Error getting data")
            return
        
        # Print model architecture
        print("\nModel Architecture:")
        print("-" * 30)
        total_params = 0
        for name, param in train_obj.Model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            print(f"{name:30}: {param_count:8,} parameters")
        print(f"{'Total':30}: {total_params:8,} parameters")
        
        # Training loop
        print(f"\nStarting training for up to {train_obj.num_epochs} epochs...")
        print("Early stopping enabled - training will stop if validation loss doesn't improve")
        
        stop = False
        best_epoch = 0
        
        for epoch in range(train_obj.num_epochs):
            epoch_start = time.time()
            stop = train_obj.train_one_epoch(epoch)
            epoch_time = time.time() - epoch_start
            
            if stop: 
                if train_obj.is_main:
                    print(f"🛑 Early stop at epoch {epoch}")
                break
            best_epoch = epoch
            
            # Print progress every 100 epochs
            if epoch % 100 == 0 and train_obj.is_main:
                print(f"Epoch {epoch:4d} completed in {epoch_time:.2f}s")
        
        training_time = time.time() - start_time
        
        # Clean up distributed training resources
        train_obj.stop()
        
        if train_obj.is_main:
            print(f"✅ Training completed in {training_time/60:.1f} minutes")
            print(f"📊 Best epoch: {best_epoch}")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return
    
    # Evaluation phase
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)
    
    # Check if model was saved and run evaluation
    if os.path.exists(model_path) and train_obj.is_main:
        print("🎯 Training completed! Starting comprehensive model evaluation...")
        
        try:
            from evaluator import ModelEvaluator
            
            # Initialize evaluator
            evaluator = ModelEvaluator(
                model_path=model_path,
                stocks=stocks,
                time_args=[start_date, end_date],
                log_dir=f"runs/top100_evaluation_{prediction_type}"
            )
            
            # Run comprehensive evaluation
            print("Running evaluation with all metrics...")
            results = evaluator.evaluate_all_metrics(
                batch_size=128,  # Larger batch size for production
                create_plots=True
            )
            
            # Save results
            results_file = f"top100_evaluation_results_{prediction_type}.json"
            evaluator.save_results(results_file)
            evaluator.close()
            
            print("✅ Evaluation completed successfully!")
            print(f"📁 Results saved to: {results_file}")
            print(f"📈 TensorBoard logs: runs/top100_evaluation_{prediction_type}")
            print("💡 Run 'tensorboard --logdir runs/top100_evaluation_{prediction_type}' to view visualizations")
            
            # Print key metrics summary
            print("\n" + "=" * 50)
            print("KEY METRICS SUMMARY")
            print("=" * 50)
            
            if 'real_world' in results:
                rw = results['real_world']
                print(f"Annualized Return (Predicted): {rw.get('annualized_prediction_return', 0):.2%}")
                print(f"Annualized Return (Actual):    {rw.get('annualized_actual_return', 0):.2%}")
                print(f"Total Return (Predicted):      {rw.get('total_prediction_return', 0):.2f}%")
                print(f"Total Return (Actual):         {rw.get('total_actual_return', 0):.2f}%")
                print(f"Outperformance vs Random:      {rw.get('outperformance_vs_random', 0):.2f}%")
                print(f"Share Positive Returns (Pred): {rw.get('share_positive_prediction_returns', 0):.1f}%")
                print(f"Share Positive Returns (Act):  {rw.get('share_positive_actual_returns', 0):.1f}%")
            
            if 'basic' in results:
                basic = results['basic']
                print(f"RMSE:                          {basic.get('rmse', 0):.4f}")
                print(f"MAE:                           {basic.get('mae', 0):.4f}")
                print(f"R² Score:                      {basic.get('r2', 0):.4f}")
                print(f"MAPE:                          {basic.get('mape', 0):.2f}%")
            
            if 'directional' in results:
                dir_metrics = results['directional']
                print(f"Directional Accuracy:          {dir_metrics.get('directional_accuracy', 0):.2f}%")
                print(f"Upward Accuracy:               {dir_metrics.get('upward_accuracy', 0):.2f}%")
                print(f"Downward Accuracy:             {dir_metrics.get('downward_accuracy', 0):.2f}%")
            
            if 'risk' in results:
                risk_metrics = results['risk']
                print(f"Prediction Volatility:         {risk_metrics.get('prediction_volatility', 0):.2%}")
                print(f"Actual Volatility:             {risk_metrics.get('actual_volatility', 0):.2%}")
                print(f"Max Drawdown (Predicted):      {risk_metrics.get('max_prediction_drawdown', 0):.2f}%")
                print(f"Max Drawdown (Actual):         {risk_metrics.get('max_actual_drawdown', 0):.2f}%")
            
        except ImportError:
            print("⚠️  ModelEvaluator not available. Install required dependencies.")
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            
    else:
        if not os.path.exists(model_path):
            print("⚠️  No saved model found. Model may not have been saved due to early stopping.")
        if not train_obj.is_main:
            print("⚠️  Evaluation skipped on non-main process in distributed training.")
    
    total_time = time.time() - start_time
    print(f"\n" + "=" * 80)
    print("PRODUCTION PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

