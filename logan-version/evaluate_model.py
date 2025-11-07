"""
Example script for evaluating a trained stock prediction model.

This script demonstrates how to use the ModelEvaluator class to comprehensively
evaluate a saved model with multiple metrics and visualizations.
"""

import os
import sys
from evaluator import ModelEvaluator

def evaluate_saved_model(model_path: str = "savedmodel.pth", 
                        stocks: list = None,
                        time_args: list = None,
                        log_dir: str = "runs/evaluation"):
    """
    Evaluate a saved model using the ModelEvaluator.
    
    Args:
        model_path: Path to the saved model file
        stocks: List of stock symbols to evaluate on
        time_args: Time arguments for data loading
        log_dir: Directory for TensorBoard logs
    """
    
    # Default parameters
    if stocks is None:
        stocks = ["AAPL", "MSFT"]  # Default stocks for evaluation
        
    if time_args is None:
        time_args = ["3y"]  # Default time period
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("Please train a model first or check the model path.")
        return False
    
    print(f"🔍 Evaluating model: {model_path}")
    print(f"📊 Stocks: {stocks}")
    print(f"📅 Time period: {time_args}")
    print(f"📝 Log directory: {log_dir}")
    print("-" * 50)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=model_path,
            stocks=stocks,
            time_args=time_args,
            log_dir=log_dir
        )
        
        # Run comprehensive evaluation
        print("🚀 Starting evaluation...")
        results = evaluator.evaluate_all_metrics(
            batch_size=32,  # Adjust based on your GPU memory
            create_plots=True
        )
        
        # Save results to JSON
        results_file = f"evaluation_results_{os.path.basename(model_path).replace('.pth', '')}.json"
        evaluator.save_results(results_file)
        
        # Clean up
        evaluator.close()
        
        print(f"✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📈 TensorBoard logs: {log_dir}")
        print(f"💡 Run 'tensorboard --logdir {log_dir}' to view visualizations")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        return False

def main():
    """Main function with example usage."""
    
    # Example 1: Evaluate with default parameters
    print("=" * 60)
    print("STOCK PREDICTION MODEL EVALUATION")
    print("=" * 60)
    
    success = evaluate_saved_model()
    
    if success:
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Check the generated JSON file for detailed metrics")
        print("2. Launch TensorBoard to view visualizations:")
        print("   tensorboard --logdir runs/evaluation")
        print("3. Analyze the results to understand model performance")
    else:
        print("\n" + "=" * 60)
        print("EVALUATION FAILED!")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Ensure you have trained a model first")
        print("2. Check that savedmodel.pth exists in the current directory")
        print("3. Verify data files are available in the data/ directory")
        print("4. Check error messages above for specific issues")

if __name__ == "__main__":
    main()
