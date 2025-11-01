import trainer
import torch
import os

print("my code")
start = "1989-12-01"
end = "2015-09-30"
prediction_type = "classification"
model_path = "savedmodel_classification.pth"

# Top 30 S&P 500 stocks by market cap (subset for testing)
stocks = [
    # Top Technology & Growth
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",

    # Major Financial Services
    "JPM", "BAC", "V", "MA", "WFC", "GS", "BLK", "AXP",

    # Healthcare Leaders
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO",

    # Consumer & Retail Giants
    "WMT", "PG", "HD", "COST", "MCD", "NKE",

    # Industrial & Energy Leaders
    "BA", "CAT", "XOM", "CVX"
]

# Training phase
print("=" * 60)
print("TRAINING PHASE")
print("=" * 60)

train_obj = trainer.Trainer(stocks=stocks, time_args=[start,end], num_epochs=5000, prediction_type=prediction_type, saved_model=model_path)

if type(train_obj) == int:
    print("Error getting data")
    exit()

for name, param in train_obj.Model.named_parameters():
    print("name: {}".format(name))
    print("param: {}".format(param.numel()))
    print()

#total_epochs = train_obj.num_epochs
total_epochs = 100

stop = False  # Initialize stop condition
for epoch in range(total_epochs):
    stop = train_obj.train_one_epoch(epoch)
    if stop: 
        if train_obj.is_main:
            print("Early stop at epoch: {}".format(epoch))
        break

# Always clean up distributed training resources
train_obj.stop()

# Evaluation phase
print("\n" + "=" * 60)
print("EVALUATION PHASE")
print("=" * 60)

# Check if model was saved and run evaluation
if os.path.exists("savedmodel.pth") and train_obj.is_main:
    print("🎯 Training completed! Starting model evaluation...")
    
    try:
        from evaluator import ModelEvaluator
        
        # Initialize evaluator with same parameters as training
        evaluator = ModelEvaluator(
            model_path="savedmodel.pth",
            stocks=stocks,
            time_args=[start, end],
            log_dir="runs/full_evaluation"
        )
        
        # Run comprehensive evaluation
        results = evaluator.evaluate_all_metrics(
            batch_size=32,
            create_plots=True
        )
        
        # Save results
        evaluator.save_results("full_evaluation_results.json")
        evaluator.close()
        
        print("✅ Evaluation completed successfully!")
        print("📁 Results saved to: full_evaluation_results.json")
        print("📈 TensorBoard logs: runs/full_evaluation")
        print("💡 Run 'tensorboard --logdir runs/full_evaluation' to view visualizations")
        
    except ImportError:
        print("⚠️  ModelEvaluator not available. Install required dependencies or run evaluate_model.py separately.")
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        print("💡 You can run evaluate_model.py separately to evaluate the model.")
        
else:
    if not os.path.exists("savedmodel.pth"):
        print("⚠️  No saved model found. Model may not have been saved due to early stopping.")
    if not train_obj.is_main:
        print("⚠️  Evaluation skipped on non-main process in distributed training.")

print("\n" + "=" * 60)
print("PROGRAM COMPLETED")
print("=" * 60)
