"""
Comprehensive Model Evaluator for Stock Classification

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

from model import LSTMModel
import util


class ModelEvaluator:
    """
    A comprehensive evaluator for stock classification models.
    
    This class provides:
    - Multiple evaluation metrics (accuracy, trading performance, etc.)
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
        # Try cache first, then download if needed
        input_data = util.load_data_from_cache(self.stocks, self.time_args, data_dir=self.data_dir, prediction_type="classification")
        if input_data is None:
            input_data = util.get_data(self.stocks, self.time_args, data_dir=self.data_dir, prediction_type="classification")
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
        self.model = LSTMModel()
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
        Calculate basic classification metrics.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Classification accuracy
        correct = np.sum(np.sign(predictions) == np.sign(targets))
        total = len(predictions)
        metrics['accuracy'] = (correct / total) * 100 if total > 0 else 0
        
        # MSE for continuous predictions
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets, predictions)
        
        return metrics
        
    def calculate_directional_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate directional accuracy metrics for classification.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Dictionary of directional metrics
        """
        metrics = {}
        
        # Classification accuracy
        correct = np.sum(np.sign(predictions) == np.sign(targets))
        total_predictions = len(predictions)
        metrics['directional_accuracy'] = (correct / total_predictions) * 100 if total_predictions > 0 else 0
        
        # Upward movement accuracy
        up_mask = targets > 0
        if np.sum(up_mask) > 0:
            up_accuracy = np.sum((predictions > 0) & up_mask) / np.sum(up_mask) * 100
            metrics['upward_accuracy'] = up_accuracy
        else:
            metrics['upward_accuracy'] = 0
        
        # Downward movement accuracy
        down_mask = targets < 0
        if np.sum(down_mask) > 0:
            down_accuracy = np.sum((predictions < 0) & down_mask) / np.sum(down_mask) * 100
            metrics['downward_accuracy'] = down_accuracy
        else:
            metrics['downward_accuracy'] = 0
        
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
        ax.set_title('Stock Classification Predictions vs Actual Values')
        ax.set_xlabel('Date')
        ax.set_ylabel('Classification Value')
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


class ModelComparator:
    """
    Compare multiple models using their Trainer instances.
    
    This class allows you to compare multiple models side-by-side using all
    the comprehensive metrics from the Trainer class including:
    - Trading performance metrics
    - Statistical tests
    - Time series diagnostics
    """
    
    def __init__(self, 
                 trainers: Union[Dict[str, any], List[any], set],
                 log_dir: str = "runs/comparison",
                 device: Optional[torch.device] = None):
        """
        Initialize the ModelComparator.
        
        Args:
            trainers: Dictionary mapping model names to Trainer instances,
                     List of Trainer instances, or set of Trainer instances.
                     For lists/sets, model names will be auto-generated as "model_0", "model_1", etc.
                     Example: {"model1": trainer1, "model2": trainer2}
                     or [trainer1, trainer2] or {trainer1, trainer2}
            log_dir: Directory for TensorBoard logs
            device: Device to run evaluation on (auto-detect if None)
        """
        # Convert list/set to dict if needed
        if isinstance(trainers, (list, set)):
            self.trainers = {f"model_{i}": t for i, t in enumerate(trainers)}
        else:
            self.trainers = trainers
        self.model_names = list(self.trainers.keys())
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
        
        # Initialize TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Results storage
        self.results = {}
        
        print(f"[Comparator] Initialized with {len(self.trainers)} models: {', '.join(self.model_names)}")
        print(f"[Comparator] Using device: {self.device}")
    
    def compare_all_metrics(self, run_evaluation: bool = True) -> Dict[str, Dict]:
        """
        Compare all metrics across all models.
        
        Args:
            run_evaluation: If True, run evaluate() on each trainer first
        
        Returns:
            Dictionary of results for each model
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON - ALL METRICS")
        print("="*80)
        
        for model_name, trainer in self.trainers.items():
            print(f"\n{'='*80}")
            print(f"Evaluating Model: {model_name}")
            print(f"{'='*80}")
            
            # Run evaluation if requested
            if run_evaluation:
                trainer.evaluate()
            
            # Collect all metrics
            model_results = {
                'trading_performance': trainer.get_trading_performance_metrics(),
                'statistical_tests': trainer.get_statistical_tests(),
                'time_series_diagnostics': trainer.get_time_series_diagnostics()
            }
            
            self.results[model_name] = model_results
        
        return self.results
    
    def print_comparison_summary(self):
        """Print a formatted comparison summary of all models."""
        if not self.results:
            print("[Comparator] No results to compare. Run compare_all_metrics() first.")
            return
        
        print("\n" + "="*100)
        print("MODEL COMPARISON SUMMARY")
        print("="*100)
        
        # Trading Performance Metrics
        print("\n1. TRADING PERFORMANCE METRICS")
        print("-" * 100)
        trading_metrics_to_compare = [
            'accuracy', 'sharpe_ratio', 'sortino_ratio', 'annualized_return',
            'value_at_risk_1pct', 'maximum_drawdown', 'excess_return_annualized'
        ]
        
        for metric in trading_metrics_to_compare:
            print(f"\n  {metric.replace('_', ' ').title()}:")
            values = {}
            for model_name in self.model_names:
                if model_name in self.results:
                    val = self.results[model_name]['trading_performance'].get(metric)
                    if val is not None:
                        values[model_name] = val
            
            if values:
                # Find best and worst
                if 'ratio' in metric.lower() or 'accuracy' in metric.lower():
                    best_model = max(values, key=values.get)
                    worst_model = min(values, key=values.get)
                else:
                    best_model = max(values, key=values.get)
                    worst_model = min(values, key=values.get)
                
                for model_name in self.model_names:
                    if model_name in values:
                        is_best = "🏆 BEST" if model_name == best_model else ""
                        is_worst = "⚠️ WORST" if model_name == worst_model and best_model != worst_model else ""
                        marker = is_best or is_worst
                        print(f"    {model_name:20s}: {values[model_name]:12.6f} {marker}")
        
        # Statistical Tests
        print("\n2. STATISTICAL TESTS")
        print("-" * 100)
        
        for test_name in ['diebold_mariano', 'pesaran_timmermann', 'newey_west_mean_return']:
            print(f"\n  {test_name.replace('_', ' ').title()}:")
            for model_name in self.model_names:
                if model_name in self.results:
                    test_result = self.results[model_name]['statistical_tests'].get(test_name, {})
                    stat = test_result.get('statistic')
                    p_val = test_result.get('p_value')
                    if stat is not None and p_val is not None:
                        significance = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"
                        print(f"    {model_name:20s}: stat={stat:8.4f}, p={p_val:8.4f} {significance}")
        
        # Time Series Diagnostics
        print("\n3. TIME SERIES DIAGNOSTICS")
        print("-" * 100)
        diagnostics_to_compare = ['aic', 'bic', 'ljung_box', 'adf_residuals']
        
        for diag in diagnostics_to_compare:
            print(f"\n  {diag.replace('_', ' ').title()}:")
            for model_name in self.model_names:
                if model_name in self.results:
                    diag_result = self.results[model_name]['time_series_diagnostics'].get(diag, {})
                    if isinstance(diag_result, dict):
                        p_val = diag_result.get('p_value')
                        if p_val is not None:
                            significance = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"
                            print(f"    {model_name:20s}: p={p_val:8.4f} {significance}")
                    else:
                        val = diag_result
                        if val is not None:
                            print(f"    {model_name:20s}: {val:12.4f}")
        
        print("\n" + "="*100)
        print("Comparison complete!")
        print("="*100)
    
    def get_best_model(self, metric: str) -> Tuple[str, float]:
        """
        Get the model that performs best on a specific metric.
        
        Args:
            metric: Name of the metric (e.g., 'sharpe_ratio', 'accuracy')
        
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.results:
            return None, None
        
        values = {}
        for model_name in self.model_names:
            if model_name in self.results:
                # Try trading performance first
                val = self.results[model_name]['trading_performance'].get(metric)
                if val is not None:
                    values[model_name] = val
        
        if not values:
            return None, None
        
        best_model = max(values, key=values.get)
        return best_model, values[best_model]
    
    def compare_diebold_mariano(self, model1_name: str, model2_name: str) -> Dict:
        """
        Compare two models using the Diebold-Mariano test.
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model
        
        Returns:
            Dictionary with comparison results
        """
        if model1_name not in self.results or model2_name not in self.results:
            print(f"Error: One or both models not found in results")
            return None
        
        # Get predictions and actuals for both models
        trainer1 = self.trainers[model1_name]
        trainer2 = self.trainers[model2_name]
        
        pred1 = np.array(trainer1.predicted_values)
        actual1 = np.array(trainer1.actual_values)
        pred2 = np.array(trainer2.predicted_values)
        actual2 = np.array(trainer2.actual_values)
        
        if len(pred1) != len(pred2) or len(actual1) != len(actual2):
            print("Error: Models have different prediction lengths")
            return None
        
        # Calculate losses
        loss1 = (pred1 - actual1) ** 2
        loss2 = (pred2 - actual2) ** 2
        
        # Difference in losses
        d = loss1 - loss2
        d_bar = np.mean(d)
        
        # DM test from trainer1 (it should have the _newey_west_variance method)
        var_hac = trainer1._newey_west_variance(d)
        n = len(d)
        
        if var_hac > 0:
            dm_stat = d_bar / np.sqrt(var_hac / n)
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
        else:
            dm_stat = 0
            p_value = 1.0
        
        return {
            'model1': model1_name,
            'model2': model2_name,
            'mean_loss_diff': d_bar,
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'interpretation': 'Model 1 better' if d_bar < 0 and p_value < 0.05 else (
                'Model 2 better' if d_bar > 0 and p_value < 0.05 else 'Equal performance'
            )
        }
    
    def save_comparison(self, filepath: str = "model_comparison.json"):
        """
        Save comparison results to a JSON file.
        
        Args:
            filepath: Path to save the results
        """
        import json
        
        if not self.results:
            print("[Comparator] No results to save. Run compare_all_metrics() first.")
            return
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        output = {
            'comparison_date': datetime.now().isoformat(),
            'models': self.model_names,
            'results': convert_numpy(self.results)
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[Comparator] Comparison results saved to {filepath}")
    
    def close(self):
        """Close TensorBoard writer and clean up resources."""
        if hasattr(self, 'writer'):
            self.writer.close()
        print("[Comparator] Comparison completed and resources cleaned up")
    
    def generate_csv_panels(self, 
                           output_dir: str = "comparison_csvs",
                           run_evaluation: bool = True,
                           horizon: str = "1d",
                           k: Optional[int] = None,
                           costs_bps_per_side: Optional[float] = None,
                           test_period: Optional[str] = None,
                           baseline_model: Optional[str] = None) -> Dict[str, str]:
        """
        Generate CSV files for all 6 panels (A-F) as defined in the evaluation structure.
        
        Args:
            output_dir: Directory to save CSV files
            run_evaluation: If True, run evaluate() on each trainer first
            horizon: Prediction horizon (e.g., "1d", "5min") for Panel A
            k: Number of long/short positions for Panel B (if None, will try to infer)
            costs_bps_per_side: Trading costs in basis points per side for Panel B
            test_period: Test period identifier for metadata (if None, auto-generated)
            baseline_model: Model name to use as baseline for DM tests (if None, uses first model)
        
        Returns:
            Dictionary mapping panel names to CSV file paths
        """
        print("\n" + "="*80)
        print("GENERATING CSV PANELS (A-F)")
        print("="*80)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Run evaluation if needed
        if run_evaluation or not self.results:
            self.compare_all_metrics(run_evaluation=run_evaluation)
        
        # Auto-generate test_period if not provided
        if test_period is None:
            test_period = datetime.now().strftime("%Y-%m")
        
        # Set baseline model
        if baseline_model is None and len(self.model_names) > 0:
            baseline_model = self.model_names[0]
        
        csv_paths = {}
        
        # Panel A: Forecasting metrics
        csv_paths['A'] = self._generate_panel_a(output_dir, horizon, test_period, baseline_model)
        
        # Panel B: Trading performance
        csv_paths['B'] = self._generate_panel_b(output_dir, k, costs_bps_per_side, test_period)
        
        # Panel C: Distribution diagnostics
        csv_paths['C'] = self._generate_panel_c(output_dir, k, test_period)
        
        # Panel D: Significance tests
        csv_paths['D'] = self._generate_panel_d(output_dir, horizon)
        
        # Panel E: Setup metadata
        csv_paths['E'] = self._generate_panel_e(output_dir, k, costs_bps_per_side, test_period)
        
        # Panel F: Efficiency
        csv_paths['F'] = self._generate_panel_f(output_dir)
        
        print("\n" + "="*80)
        print(f"All CSV panels generated in: {output_dir}")
        print("="*80)
        for panel, path in csv_paths.items():
            print(f"  Panel {panel}: {path}")
        
        return csv_paths
    
    def _generate_panel_a(self, output_dir: str, horizon: str, test_period: str, baseline_model: str) -> str:
        """Generate Panel A: Forecasting metrics"""
        rows = []
        
        for model_name in self.model_names:
            if model_name not in self.results:
                continue
            
            result = self.results[model_name]
            trading = result.get('trading_performance', {})
            stats = result.get('statistical_tests', {})
            trainer = self.trainers[model_name]
            
            # Get number of test observations
            n = len(trainer.predicted_values) if hasattr(trainer, 'predicted_values') and len(trainer.predicted_values) > 0 else 0
            
            # Directional accuracy
            acc_dir_pct = trading.get('accuracy', 0)
            
            # PT p-value
            pt_test = stats.get('pesaran_timmermann', {})
            pt_p = pt_test.get('p_value', None) if pt_test else None
            
            # RMSE, MAE, NMSE
            if hasattr(trainer, 'residuals') and len(trainer.residuals) > 0:
                residuals = np.array(trainer.residuals)
                rmse = np.sqrt(np.mean(residuals**2))
                mae = np.mean(np.abs(residuals))
                
                # NMSE: Normalized MSE (MSE / variance of actuals)
                if hasattr(trainer, 'actual_values') and len(trainer.actual_values) > 0:
                    actuals = np.array(trainer.actual_values)
                    var_actuals = np.var(actuals)
                    nmse = (rmse**2) / var_actuals if var_actuals > 0 else 0
                else:
                    nmse = 0
            else:
                rmse = mae = nmse = None
            
            # MI (Mutual Information) - simplified approximation using correlation
            mi_bits = None
            if hasattr(trainer, 'predicted_values') and hasattr(trainer, 'actual_values'):
                preds = np.array(trainer.predicted_values)
                actuals = np.array(trainer.actual_values)
                if len(preds) > 0 and len(actuals) > 0:
                    # Use correlation as proxy for MI (0 to ~0.3 bits for strong correlation)
                    corr = np.corrcoef(preds, actuals)[0, 1] if len(preds) > 1 else 0
                    mi_bits = abs(corr) * 0.3 if not np.isnan(corr) else 0
            
            # DM p-val vs baseline
            dm_p = None
            if baseline_model and baseline_model != model_name and baseline_model in self.results:
                dm_result = self.compare_diebold_mariano(model_name, baseline_model)
                if dm_result:
                    dm_p = dm_result.get('p_value')
            elif model_name == baseline_model:
                # For baseline model, compare to naive forecast (from its own stats)
                dm_test = stats.get('diebold_mariano', {})
                dm_p = dm_test.get('p_value', None) if dm_test else None
            
            row = {
                'model_id': model_name,
                'horizon': horizon,
                'test_period': test_period,
                'n': n,
                'acc_dir_pct': acc_dir_pct if acc_dir_pct else None,
                'pt_p': pt_p if pt_p is not None else None,
                'rmse': rmse if rmse is not None else None,
                'mae': mae if mae is not None else None,
                'nmse': nmse if nmse is not None else None,
                'mi_bits': mi_bits if mi_bits is not None else None,
                'dm_p': dm_p if dm_p is not None else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "A_forecast.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel A: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _generate_panel_b(self, output_dir: str, k: Optional[int], costs_bps_per_side: Optional[float], test_period: str) -> str:
        """Generate Panel B: Trading performance"""
        rows = []
        
        for model_name in self.model_names:
            if model_name not in self.results:
                continue
            
            result = self.results[model_name]
            trading = result.get('trading_performance', {})
            trainer = self.trainers[model_name]
            
            # If k not provided, try to infer or use default
            k_val = k if k is not None else 10  # Default to 10
            
            # If costs not provided, use default or try to infer
            costs = costs_bps_per_side if costs_bps_per_side is not None else 5.0  # Default 5 bps
            
            # Mean daily return (%)
            mean_r_daily = trading.get('mean_daily_return', 0) * 100
            
            # Annualized return (%)
            ann_return = trading.get('annualized_return', 0) * 100
            
            # Annualized volatility (%)
            ann_vol = trading.get('standard_deviation_annualized', 0) * 100
            
            # Sharpe ratio
            sharpe = trading.get('sharpe_ratio', 0)
            
            # Sortino ratio
            sortino = trading.get('sortino_ratio', 0)
            
            # VaR 1% (%/day)
            var_1pct = trading.get('value_at_risk_1pct', 0) * 100
            
            # Maximum drawdown (%)
            maxdd = trading.get('maximum_drawdown', 0) * 100
            
            # % Positive days
            share_pos = trading.get('fraction_positive_returns', 0)
            
            # SE(μ_d) - Standard error of mean daily return
            if hasattr(trainer, 'real_world_returns') and len(trainer.real_world_returns) > 0:
                returns = np.array(trainer.real_world_returns)
            elif hasattr(trainer, 'actual_returns') and len(trainer.actual_returns) > 0:
                returns = np.array(trainer.actual_returns)
            else:
                returns = np.array([])
            
            if len(returns) > 0:
                se_mean_daily = np.std(returns, ddof=1) / np.sqrt(len(returns)) * 100
            else:
                se_mean_daily = None
            
            row = {
                'model_id': model_name,
                'k': k_val,
                'costs_bps_per_side': costs,
                'test_period': test_period,
                'mean_r_daily_pct': mean_r_daily if mean_r_daily else None,
                'ann_return_pct': ann_return if ann_return else None,
                'ann_vol_pct': ann_vol if ann_vol else None,
                'sharpe': sharpe if sharpe else None,
                'sortino': sortino if sortino else None,
                'var_1pct_daily_pct': var_1pct if var_1pct else None,
                'maxdd_pct': maxdd if maxdd else None,
                'share_pos_pct': share_pos if share_pos else None,
                'se_mean_daily_pct': se_mean_daily if se_mean_daily is not None else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "B_trading.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel B: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _generate_panel_c(self, output_dir: str, k: Optional[int], test_period: str) -> str:
        """Generate Panel C: Distribution diagnostics"""
        rows = []
        
        for model_name in self.model_names:
            if model_name not in self.results:
                continue
            
            result = self.results[model_name]
            trading = result.get('trading_performance', {})
            trainer = self.trainers[model_name]
            
            k_val = k if k is not None else 10
            
            # Get returns
            if hasattr(trainer, 'real_world_returns') and len(trainer.real_world_returns) > 0:
                returns = np.array(trainer.real_world_returns)
            elif hasattr(trainer, 'actual_returns') and len(trainer.actual_returns) > 0:
                returns = np.array(trainer.actual_returns)
            else:
                returns = np.array([])
            
            if len(returns) == 0:
                continue
            
            n_days = len(returns)
            
            # Convert to percentage
            returns_pct = returns * 100
            
            # Percentiles
            min_pct = np.min(returns_pct)
            q1_pct = np.percentile(returns_pct, 25)
            median_pct = np.median(returns_pct)
            q3_pct = np.percentile(returns_pct, 75)
            max_pct = np.max(returns_pct)
            
            # Skewness and kurtosis
            skew = trading.get('skewness', None)
            kurt = trading.get('kurtosis', None)
            
            row = {
                'model_id': model_name,
                'k': k_val,
                'test_period': test_period,
                'n_days': n_days,
                'min_pct': min_pct,
                'q1_pct': q1_pct,
                'median_pct': median_pct,
                'q3_pct': q3_pct,
                'max_pct': max_pct,
                'skew': skew if skew is not None else None,
                'kurt': kurt if kurt is not None else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "C_distribution.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel C: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _generate_panel_d(self, output_dir: str, horizon: str) -> str:
        """Generate Panel D: Significance tests"""
        rows = []
        
        # Generate comparisons between all pairs of models
        for i, model1_name in enumerate(self.model_names):
            for j, model2_name in enumerate(self.model_names[i+1:], start=i+1):
                # Get DM test result
                dm_result = self.compare_diebold_mariano(model1_name, model2_name)
                if not dm_result:
                    continue
                
                # Get PT stats from both models
                model1_result = self.results.get(model1_name, {})
                model2_result = self.results.get(model2_name, {})
                
                model1_stats = model1_result.get('statistical_tests', {})
                model2_stats = model2_result.get('statistical_tests', {})
                
                # Use average PT stat if both available, otherwise use one
                pt_stat = None
                pt_p = None
                
                pt1 = model1_stats.get('pesaran_timmermann', {})
                pt2 = model2_stats.get('pesaran_timmermann', {})
                
                if pt1.get('statistic') is not None and pt2.get('statistic') is not None:
                    pt_stat = (pt1['statistic'] + pt2['statistic']) / 2
                    pt_p = min(pt1.get('p_value', 1.0), pt2.get('p_value', 1.0))
                elif pt1.get('statistic') is not None:
                    pt_stat = pt1['statistic']
                    pt_p = pt1.get('p_value')
                elif pt2.get('statistic') is not None:
                    pt_stat = pt2['statistic']
                    pt_p = pt2.get('p_value')
                
                row = {
                    'comparison': f"{model1_name} vs {model2_name}",
                    'horizon': horizon,
                    'dm_stat': dm_result.get('dm_statistic'),
                    'dm_p': dm_result.get('p_value'),
                    'pt_stat': pt_stat,
                    'pt_p': pt_p
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "D_significance.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel D: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _generate_panel_e(self, output_dir: str, k: Optional[int], costs_bps_per_side: Optional[float], test_period: str) -> str:
        """Generate Panel E: Setup metadata"""
        rows = []
        
        for model_name in self.model_names:
            trainer = self.trainers[model_name]
            
            # Extract setup information from trainer
            universe = ", ".join(trainer.stocks) if hasattr(trainer, 'stocks') else "Unknown"
            
            # Train/test years from time_args
            train_test_str = "/".join(trainer.time_args) if hasattr(trainer, 'time_args') else "Unknown"
            
            # Rebalance frequency
            rebalance = "Daily"  # Default assumption
            
            # Rule
            k_val = k if k is not None else 10
            rule = f"Top{k_val}/Bottom{k_val}, equal-weight"
            
            # Costs
            costs = costs_bps_per_side if costs_bps_per_side is not None else 5.0
            
            # Neutrality (market-neutral)
            neutral = "Yes"
            
            # Exclusions
            exclusions = "Zero-volume"  # Default assumption
            
            # Avg names/day - try to infer from data
            avg_names_per_day = len(trainer.stocks) if hasattr(trainer, 'stocks') else None
            
            # Turnover - approximate from returns if available
            turnover_pct = None
            if hasattr(trainer, 'real_world_returns') and len(trainer.real_world_returns) > 0:
                # Rough approximation: assume daily rebalancing with k positions
                # This is a simplified estimate
                turnover_pct = (k_val * 2 / avg_names_per_day * 100) if avg_names_per_day and avg_names_per_day > 0 else None
            
            row = {
                'model_id': model_name,
                'universe': universe,
                'train_years': train_test_str.split('/')[0] if '/' in train_test_str else train_test_str,
                'test_years': train_test_str.split('/')[1] if '/' in train_test_str and len(train_test_str.split('/')) > 1 else test_period,
                'rebalance': rebalance,
                'rule': rule,
                'costs_bps_per_side': costs,
                'neutral': neutral,
                'exclusions': exclusions,
                'avg_names_per_day': avg_names_per_day if avg_names_per_day else None,
                'turnover_pct': turnover_pct if turnover_pct else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "E_setup.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel E: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _generate_panel_f(self, output_dir: str) -> str:
        """Generate Panel F: Efficiency"""
        rows = []
        
        for model_name in self.model_names:
            trainer = self.trainers[model_name]
            
            # Model parameters
            if hasattr(trainer, 'Model'):
                model = trainer.Model.module if hasattr(trainer.Model, 'module') else trainer.Model
                params_m = sum(p.numel() for p in model.parameters()) / 1e6
            else:
                params_m = None
            
            # Hardware
            if hasattr(trainer, 'device'):
                device_type = str(trainer.device)
                if 'cuda' in device_type:
                    if torch.cuda.is_available():
                        hardware = torch.cuda.get_device_name(0)
                    else:
                        hardware = "CUDA (unknown)"
                elif 'mps' in device_type:
                    hardware = "Apple Silicon"
                else:
                    hardware = "CPU"
            else:
                hardware = "Unknown"
            
            # Training time - try to get from trainer attributes
            train_hours = None
            if hasattr(trainer, 'training_time'):
                train_hours = trainer.training_time / 3600  # Convert seconds to hours
            
            # Epochs
            epochs = None
            if hasattr(trainer, 'num_epochs'):
                epochs = trainer.num_epochs
            elif hasattr(trainer, 'stopper') and hasattr(trainer.stopper, 'counter'):
                # Try to infer from early stopping
                epochs = trainer.stopper.counter if hasattr(trainer.stopper, 'min_validation_loss') else None
            
            # Decision latency and inference time - these would need to be measured
            # For now, provide placeholder values
            decision_latency_ms = None  # Would need actual measurement
            infer_seconds_per_day = None  # Would need actual measurement
            
            row = {
                'model_id': model_name,
                'params_m': params_m if params_m else None,
                'hardware': hardware,
                'train_hours': train_hours if train_hours else None,
                'epochs': epochs if epochs else None,
                'decision_latency_ms': decision_latency_ms,
                'infer_seconds_per_day': infer_seconds_per_day
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "F_efficiency.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel F: {filepath} ({len(rows)} rows)")
        return filepath


def generate_csv_panels_from_trainers(trainers: Union[Dict[str, any], List[any], set],
                                     output_dir: str = "comparison_csvs",
                                     run_evaluation: bool = True,
                                     horizon: str = "1d",
                                     k: Optional[int] = None,
                                     costs_bps_per_side: Optional[float] = None,
                                     test_period: Optional[str] = None,
                                     baseline_model: Optional[str] = None) -> Dict[str, str]:
    """
    Convenience function to generate CSV panels from a list, set, or dict of Trainer instances.
    
    This is a standalone function that creates a ModelComparator internally and generates
    all 6 CSV panels (A-F) as defined in the evaluation structure.
    
    Args:
        trainers: Dictionary mapping model names to Trainer instances,
                 List of Trainer instances, or set of Trainer instances.
                 For lists/sets, model names will be auto-generated as "model_0", "model_1", etc.
        output_dir: Directory to save CSV files
        run_evaluation: If True, run evaluate() on each trainer first
        horizon: Prediction horizon (e.g., "1d", "5min") for Panel A
        k: Number of long/short positions for Panel B (if None, defaults to 10)
        costs_bps_per_side: Trading costs in basis points per side for Panel B (default: 5.0)
        test_period: Test period identifier for metadata (if None, auto-generated)
        baseline_model: Model name to use as baseline for DM tests (if None, uses first model)
    
    Returns:
        Dictionary mapping panel names to CSV file paths
    
    Example:
        >>> from evaluator import generate_csv_panels_from_trainers
        >>> from trainer import Trainer
        >>> 
        >>> # Create trainers
        >>> trainer1 = Trainer(stocks=["AAPL", "MSFT"], time_args=["3y"])
        >>> trainer2 = Trainer(stocks=["AAPL", "MSFT"], time_args=["3y"])
        >>> 
        >>> # Generate CSVs from list
        >>> csv_paths = generate_csv_panels_from_trainers(
        ...     [trainer1, trainer2],
        ...     output_dir="results_csvs",
        ...     k=10,
        ...     costs_bps_per_side=5.0
        ... )
        >>> 
        >>> # Or from dict with custom names
        >>> csv_paths = generate_csv_panels_from_trainers(
        ...     {"LSTM": trainer1, "RF": trainer2},
        ...     output_dir="results_csvs"
        ... )
    """
    comparator = ModelComparator(trainers)
    return comparator.generate_csv_panels(
        output_dir=output_dir,
        run_evaluation=run_evaluation,
        horizon=horizon,
        k=k,
        costs_bps_per_side=costs_bps_per_side,
        test_period=test_period,
        baseline_model=baseline_model
    )


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
