"""
Comprehensive Model Evaluator for Stock Price Prediction

This module provides a flexible evaluation framework that can test multiple metrics
on saved models and log results to TensorBoard for analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Union
import os

from model import LSTMModelPricePredict
import util


class ModelEvaluator:
    """
    A comprehensive evaluator for stock price prediction models.
    
    This class provides:
    - Multiple evaluation metrics (MSE, MAE, RMSE, directional accuracy, etc.)
    - TensorBoard logging of evaluation results
    - Visualization capabilities
    - Support for loading saved models
    - Time-series specific analysis
    """
    
    def __init__(self, 
                 model_path: str,
                 stocks: List[str] = ["MSFT", "AAPL"], 
                 time_args: List[str] = ["3y"],
                 data_dir: str = "data",
                 log_dir: str = "runs/evaluation",
                 device: Optional[torch.device] = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model_path: Path to the saved model (.pth file)
            stocks: List of stock symbols to evaluate on
            time_args: Time arguments for data loading
            data_dir: Directory containing data files
            log_dir: Directory for TensorBoard logs
            device: Device to run evaluation on (auto-detect if None)
        """
        self.model_path = model_path
        self.stocks = stocks
        self.time_args = time_args
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        print(f"[Evaluator] Using device: {self.device}")
        
        # Load data
        self._load_data()
        
        # Initialize model
        self._load_model()
        
        # Initialize TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Metrics storage
        self.results = {}
        
    def _load_data(self):
        """Load test data for evaluation."""
        print("[Evaluator] Loading test data...")
        
        # Load data using the same method as training
        input_data = util.get_data(self.stocks, self.time_args)
        if isinstance(input_data, int):
            raise RuntimeError("Error getting data from util.get_data()")
        
        X_train, X_val, X_test, Y_train, Y_val, Y_test, D_train, D_val, D_test = input_data
        
        # Store test data
        self.X_test = X_test
        self.Y_test = Y_test
        self.test_dates = D_test
        
        print(f"[Evaluator] Loaded {len(self.X_test)} test samples")
        
    def _load_model(self):
        """Load the saved model."""
        print(f"[Evaluator] Loading model from {self.model_path}")
        
        # Initialize model
        self.model = LSTMModelPricePredict()
        self.model = self.model.to(self.device)
        
        # Load state dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        self.model.eval()
        
        print("[Evaluator] Model loaded successfully")
        
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array in a device-agnostic way."""
        return tensor.detach().cpu().numpy()
        
    def predict(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for the test set.
        
        Args:
            batch_size: Batch size for prediction
            
        Returns:
            Tuple of (predictions, targets) as numpy arrays
        """
        self.model.eval()
        predictions = []
        targets = []
        
        # Create data loader for batching
        dataset = torch.utils.data.TensorDataset(self.X_test, self.Y_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print("[Evaluator] Generating predictions...")
        with torch.no_grad():
            for X_batch, Y_batch in tqdm(dataloader, desc="Predicting"):
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                # Get predictions
                Y_pred = self.model(X_batch)
                
                # Convert to numpy
                predictions.extend(self._tensor_to_numpy(Y_pred.squeeze()))
                targets.extend(self._tensor_to_numpy(Y_batch.squeeze()))
        
        return np.array(predictions), np.array(targets)
        
    def calculate_basic_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic regression metrics.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['r2'] = r2_score(targets, predictions)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        metrics['mape'] = mape
        
        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(np.abs(targets - predictions) / ((np.abs(targets) + np.abs(predictions)) / 2)) * 100
        metrics['smape'] = smape
        
        return metrics
        
    def calculate_directional_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate directional accuracy metrics for stock price prediction.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Dictionary of directional metrics
        """
        metrics = {}
        
        # Calculate price changes (returns)
        pred_changes = np.diff(predictions)
        true_changes = np.diff(targets)
        
        # Directional accuracy (percentage of correct direction predictions)
        correct_direction = np.sum(np.sign(pred_changes) == np.sign(true_changes))
        total_predictions = len(pred_changes)
        metrics['directional_accuracy'] = (correct_direction / total_predictions) * 100
        
        # Upward movement accuracy
        up_mask = true_changes > 0
        if np.sum(up_mask) > 0:
            up_accuracy = np.sum((pred_changes > 0) & up_mask) / np.sum(up_mask) * 100
            metrics['upward_accuracy'] = up_accuracy
        
        # Downward movement accuracy
        down_mask = true_changes < 0
        if np.sum(down_mask) > 0:
            down_accuracy = np.sum((pred_changes < 0) & down_mask) / np.sum(down_mask) * 100
            metrics['downward_accuracy'] = down_accuracy
        
        return metrics
        
    def calculate_risk_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk-related metrics for financial predictions.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        # Calculate returns
        pred_returns = np.diff(predictions) / predictions[:-1]
        true_returns = np.diff(targets) / targets[:-1]
        
        # Volatility metrics
        metrics['prediction_volatility'] = np.std(pred_returns) * np.sqrt(252)  # Annualized
        metrics['actual_volatility'] = np.std(true_returns) * np.sqrt(252)
        
        # Sharpe ratio approximation (assuming zero risk-free rate)
        metrics['prediction_sharpe'] = np.mean(pred_returns) / np.std(pred_returns) * np.sqrt(252)
        metrics['actual_sharpe'] = np.mean(true_returns) / np.std(true_returns) * np.sqrt(252)
        
        # Maximum drawdown
        pred_cumulative = np.cumprod(1 + pred_returns)
        true_cumulative = np.cumprod(1 + true_returns)
        
        pred_peak = np.maximum.accumulate(pred_cumulative)
        true_peak = np.maximum.accumulate(true_cumulative)
        
        pred_drawdown = (pred_cumulative - pred_peak) / pred_peak
        true_drawdown = (true_cumulative - true_peak) / true_peak
        
        metrics['max_prediction_drawdown'] = np.min(pred_drawdown) * 100
        metrics['max_actual_drawdown'] = np.min(true_drawdown) * 100
        
        return metrics
        
    def calculate_time_series_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate time series specific metrics.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Dictionary of time series metrics
        """
        metrics = {}
        
        # Autocorrelation of residuals
        residuals = targets - predictions
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        metrics['residual_autocorrelation'] = autocorr
        
        # Trend accuracy (using linear regression slope)
        x = np.arange(len(predictions))
        pred_slope = np.polyfit(x, predictions, 1)[0]
        true_slope = np.polyfit(x, targets, 1)[0]
        metrics['trend_direction_accuracy'] = 100 if np.sign(pred_slope) == np.sign(true_slope) else 0
        metrics['slope_error'] = abs(pred_slope - true_slope) / abs(true_slope) * 100 if true_slope != 0 else 0
        
        return metrics

    def calculate_real_world_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate real-world metrics for financial predictions.
        """
        metrics = {}

        # Calculate returns for both predictions and actual values
        pred_returns = np.diff(predictions) / predictions[:-1]
        true_returns = np.diff(targets) / targets[:-1]

        #
        
        
        # Mean return
        metrics['mean_prediction_return'] = np.mean(pred_returns)
        metrics['mean_actual_return'] = np.mean(true_returns)

        # Annualized return (assuming daily data, 252 trading days per year)
        metrics['annualized_prediction_return'] = np.mean(pred_returns) * 252
        metrics['annualized_actual_return'] = np.mean(true_returns) * 252

        # Excess return (prediction return - actual return)
        excess_returns = pred_returns - true_returns
        metrics['mean_excess_return'] = np.mean(excess_returns)
        metrics['annualized_excess_return'] = np.mean(excess_returns) * 252

        # Share of positive returns
        metrics['share_positive_prediction_returns'] = np.sum(pred_returns > 0) / len(pred_returns) * 100
        metrics['share_positive_actual_returns'] = np.sum(true_returns > 0) / len(true_returns) * 100

        # Cumulative money growth (assuming starting with $1)
        pred_cumulative_growth = np.cumprod(1 + pred_returns)
        true_cumulative_growth = np.cumprod(1 + true_returns)
        
        metrics['final_prediction_growth'] = pred_cumulative_growth[-1]  # Final value
        metrics['final_actual_growth'] = true_cumulative_growth[-1]  # Final value
        metrics['total_prediction_return'] = (pred_cumulative_growth[-1] - 1) * 100  # Total return %
        metrics['total_actual_return'] = (true_cumulative_growth[-1] - 1) * 100  # Total return %

        # Random "monkey" benchmark (random walk with same volatility as actual)
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.normal(np.mean(true_returns), np.std(true_returns), len(true_returns))
        random_cumulative_growth = np.cumprod(1 + random_returns)
        
        metrics['random_benchmark_final_growth'] = random_cumulative_growth[-1]
        metrics['random_benchmark_total_return'] = (random_cumulative_growth[-1] - 1) * 100
        metrics['random_benchmark_annualized_return'] = np.mean(random_returns) * 252
        
        # Performance vs random benchmark
        metrics['outperformance_vs_random'] = (pred_cumulative_growth[-1] - random_cumulative_growth[-1]) * 100
        
        return metrics
        
    def log_metrics_to_tensorboard(self, metrics: Dict[str, Dict[str, float]]):
        """
        Log all metrics to TensorBoard.
        
        Args:
            metrics: Nested dictionary of metric categories and values
        """
        print("[Evaluator] Logging metrics to TensorBoard...")
        
        for category, metric_dict in metrics.items():
            for metric_name, value in metric_dict.items():
                self.writer.add_scalar(f'Evaluation/{category}/{metric_name}', value)
                
        # Log summary metrics
        all_metrics = {}
        for category, metric_dict in metrics.items():
            all_metrics.update({f"{category}_{k}": v for k, v in metric_dict.items()})
            
        self.writer.add_hparams(
            hparam_dict={},
            metric_dict=all_metrics
        )
        
        self.writer.flush()
        print("[Evaluator] Metrics logged to TensorBoard")
        
    def create_visualizations(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Create and log visualizations to TensorBoard.
        
        Args:
            predictions: Model predictions
            targets: True values
        """
        print("[Evaluator] Creating visualizations...")
        
        # 1. Time series plot
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(self.test_dates, targets, label='Actual', alpha=0.7, linewidth=1)
        ax.plot(self.test_dates, predictions, label='Predicted', alpha=0.7, linewidth=1)
        ax.set_title('Stock Price Predictions vs Actual Values')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        self._format_date_axis(plt, self.test_dates)
        plt.tight_layout()
        
        self.writer.add_figure('Evaluation/Time_Series_Prediction', fig)
        plt.close()
        
        # 2. Scatter plot: Predictions vs Actual
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(targets, predictions, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predictions vs Actual Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.writer.add_figure('Evaluation/Scatter_Predictions_vs_Actual', fig)
        plt.close()
        
        # 3. Residuals plot
        residuals = targets - predictions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Residuals over time
        ax1.plot(self.test_dates, residuals, alpha=0.7, linewidth=1)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax1.set_title('Residuals Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Residuals (Actual - Predicted)')
        ax1.grid(True, alpha=0.3)
        self._format_date_axis(plt, self.test_dates)
        
        # Residuals histogram
        ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Residuals')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.writer.add_figure('Evaluation/Residuals_Analysis', fig)
        plt.close()
        
        # 4. Error distribution by time periods
        if len(self.test_dates) > 100:  # Only if we have enough data
            # Split into quarters and analyze error
            dates_array = np.array(self.test_dates)
            errors = np.abs(residuals)
            
            # Create quarterly bins
            quarters = []
            quarter_errors = []
            
            for i in range(0, len(dates_array), len(dates_array)//4):
                end_idx = min(i + len(dates_array)//4, len(dates_array))
                quarter_errors.append(np.mean(errors[i:end_idx]))
                quarters.append(f"Q{len(quarters)+1}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(quarters, quarter_errors, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title('Mean Absolute Error by Time Period')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Mean Absolute Error')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, error in zip(bars, quarter_errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{error:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            self.writer.add_figure('Evaluation/Error_by_Time_Period', fig)
            plt.close()
        
        print("[Evaluator] Visualizations created and logged")
        
    def _format_date_axis(self, plt, datetime_dates):
        """Format x-axis dates with adaptive resolution based on data span."""
        if not datetime_dates:
            return
            
        # Calculate the span of dates
        min_date = min(datetime_dates)
        max_date = max(datetime_dates)
        date_span = (max_date - min_date).days
        num_points = len(datetime_dates)
        
        # Determine appropriate tick resolution based on data span
        if date_span <= 30:  # Less than 1 month
            if num_points <= 15:
                locator = mdates.DayLocator(interval=1)
                formatter = mdates.DateFormatter('%Y-%m-%d')
            else:
                locator = mdates.DayLocator(interval=2)
                formatter = mdates.DateFormatter('%m-%d')
                
        elif date_span <= 90:  # 1-3 months
            locator = mdates.WeekdayLocator(interval=1)
            formatter = mdates.DateFormatter('%m-%d')
            
        elif date_span <= 365:  # 3-12 months
            locator = mdates.MonthLocator(interval=1)
            formatter = mdates.DateFormatter('%Y-%m')
            
        elif date_span <= 730:  # 1-2 years
            locator = mdates.MonthLocator(interval=2)
            formatter = mdates.DateFormatter('%Y-%m')
            
        else:  # More than 2 years
            if date_span <= 1095:  # 3 years
                locator = mdates.MonthLocator(interval=6)
                formatter = mdates.DateFormatter('%Y-%m')
            else:
                locator = mdates.YearLocator()
                formatter = mdates.DateFormatter('%Y')
        
        # Apply the formatting
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=45)
        
        # For small datasets, show all points as ticks
        if num_points <= 20 and date_span <= 90:
            plt.xticks(datetime_dates, [d.strftime('%m-%d') for d in datetime_dates], rotation=45)
            
    def evaluate_all_metrics(self, batch_size: int = 32, create_plots: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive evaluation with all metrics.
        
        Args:
            batch_size: Batch size for prediction
            create_plots: Whether to create and log visualizations
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print("[Evaluator] Starting comprehensive evaluation...")
        
        # Generate predictions
        predictions, targets = self.predict(batch_size)
        
        # Calculate all metrics
        metrics = {}
        metrics['basic'] = self.calculate_basic_metrics(predictions, targets)
        metrics['directional'] = self.calculate_directional_metrics(predictions, targets)
        metrics['risk'] = self.calculate_risk_metrics(predictions, targets)
        metrics['time_series'] = self.calculate_time_series_metrics(predictions, targets)
        metrics['real_world'] = self.calculate_real_world_metrics(predictions, targets)
        
        # Store results
        self.results = metrics
        self.predictions = predictions
        self.targets = targets
        
        # Log to TensorBoard
        self.log_metrics_to_tensorboard(metrics)
        
        # Create visualizations
        if create_plots:
            self.create_visualizations(predictions, targets)
        
        # Print summary
        self._print_evaluation_summary(metrics)
        
        return metrics
        
    def _print_evaluation_summary(self, metrics: Dict[str, Dict[str, float]]):
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for category, metric_dict in metrics.items():
            print(f"\n{category.upper()} METRICS:")
            print("-" * 30)
            for metric_name, value in metric_dict.items():
                if isinstance(value, float):
                    if 'accuracy' in metric_name or 'mape' in metric_name or 'smape' in metric_name:
                        print(f"  {metric_name:25}: {value:8.2f}%")
                    else:
                        print(f"  {metric_name:25}: {value:8.6f}")
                else:
                    print(f"  {metric_name:25}: {value}")
        
        print("\n" + "="*60)
        
    def save_results(self, filepath: str):
        """
        Save evaluation results to a file.
        
        Args:
            filepath: Path to save results (JSON format)
        """
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Create serializable results
        serializable_results = {}
        for category, metric_dict in self.results.items():
            serializable_results[category] = {
                k: convert_numpy(v) for k, v in metric_dict.items()
            }
        
        # Add metadata
        output = {
            'model_path': self.model_path,
            'stocks': self.stocks,
            'time_args': self.time_args,
            'evaluation_date': datetime.now().isoformat(),
            'num_test_samples': len(self.X_test),
            'metrics': serializable_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[Evaluator] Results saved to {filepath}")
        
    def close(self):
        """Close TensorBoard writer and clean up resources."""
        if hasattr(self, 'writer'):
            self.writer.close()
        print("[Evaluator] Evaluation completed and resources cleaned up")


def main_example():
    """
    Example usage of the ModelEvaluator.
    """
    # Example: Evaluate a saved model
    model_path = "savedmodel.pth"  # Update with your model path
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train a model first.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        stocks=["AAPL", "MSFT"],
        time_args=["3y"],
        log_dir="runs/evaluation_example"
    )
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_all_metrics()
    
    # Save results
    evaluator.save_results("evaluation_results.json")
    
    # Clean up
    evaluator.close()
    
    print("\nEvaluation completed! Check TensorBoard at 'runs/evaluation_example' for visualizations.")


if __name__ == "__main__":
    main_example()
