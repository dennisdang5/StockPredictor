#!/usr/bin/env python3
"""
Production Runner Script

This script provides an easy way to run different production configurations
for S&P 500 stock prediction training and evaluation.
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, prediction_type="classification"):
    """Run a production script with the specified prediction type."""
    print(f"🚀 Running {script_name} with prediction_type={prediction_type}")
    print("=" * 60)
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, script_name
        ], check=True, capture_output=False)
        
        print(f"✅ {script_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Script {script_name} not found")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run S&P 500 production training and evaluation")
    parser.add_argument(
        "--mode", 
        choices=["top100", "full_sp500", "original"], 
        default="top100",
        help="Choose which production mode to run"
    )
    parser.add_argument(
        "--prediction-type",
        choices=["classification"],
        default="classification",
        help="Choose prediction type (only classification is supported)"
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List available modes and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_modes:
        print("Available production modes:")
        print("  top100      - Train on top 100 S&P 500 stocks (recommended)")
        print("  full_sp500  - Train on comprehensive S&P 500 stock list")
        print("  original    - Run the original main.py with 30 stocks")
        print("\nPrediction types:")
        print("  classification - Predict binary classification (revenue > median)")
        return
    
    print("=" * 80)
    print("S&P 500 PRODUCTION RUNNER")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Prediction Type: {args.prediction_type}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Choose script based on mode
    if args.mode == "top100":
        script_name = "main_production_top100.py"
    elif args.mode == "full_sp500":
        script_name = "main_production_sp500.py"
    elif args.mode == "original":
        script_name = "main.py"
    else:
        print(f"❌ Unknown mode: {args.mode}")
        return
    
    # Check if script exists
    if not os.path.exists(script_name):
        print(f"❌ Script {script_name} not found in current directory")
        return
    
    # Run the selected script
    success = run_script(script_name, args.prediction_type)
    
    if success:
        print("\n🎉 Production pipeline completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated model files (*.pth)")
        print("2. Review evaluation results (*.json)")
        print("3. Launch TensorBoard to view visualizations:")
        if args.mode == "top100":
            print("   tensorboard --logdir runs/top100_evaluation_price")
        elif args.mode == "full_sp500":
            print("   tensorboard --logdir runs/sp500_evaluation_price")
        else:
            print("   tensorboard --logdir runs/full_evaluation")
    else:
        print("\n❌ Production pipeline failed!")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()

