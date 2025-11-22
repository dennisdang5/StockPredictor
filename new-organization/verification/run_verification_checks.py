#!/usr/bin/env python3
"""
Comprehensive Verification Runner

Runs all verification checks and generates a summary report.
This script executes:
1. Preprocessing verification
2. Overfitting test
3. Random label test

And generates a comprehensive report of findings.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from typing import Dict, List

# Add current directory (verification) to path for running scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configuration variables
STOCKS = ["AAPL", "MSFT"]
TIME_ARGS = ["1990-01-01", "1995-12-31"]
MODEL_TYPE = "LSTM"

# Dataset configuration
PERIOD_TYPE = "LS"  # "LS" or "full"
LOOKBACK = 240  # Days of historical data
USE_NLP = True  # Whether to use NLP features
NLP_METHOD = "aggregated"  # "aggregated" or "individual"
BATCH_SIZE = 32  # Batch size for training

# Model file configuration (not needed for verification - scripts use dummy paths)
# MODEL_FILE_NAME = None  # Verification scripts don't need pre-trained models
# USE_TRAINED_MODELS_DIR = True  # Not used when MODEL_FILE_NAME is None

# Test configuration
SKIP_PREPROCESSING = False
SKIP_OVERFIT = False
SKIP_RANDOM = False

# Preprocessing verification configuration (use subset for speed)
PREPROCESSING_MAX_STOCKS = 5  # Limit stocks for preprocessing verification (doesn't need all)

# Overfitting test configuration
OVERFIT_NUM_SAMPLES = 512
OVERFIT_NUM_EPOCHS = 50

# Random label test configuration
RANDOM_NUM_EPOCHS = 10

# Output configuration
OUTPUT_FILE = None  # None = auto-generate with timestamp


def run_check(script_name: str, args: List[str], check_name: str) -> Dict:
    """
    Run a verification check script and capture output.
    
    Args:
        script_name: Name of the script to run
        args: Command line arguments
        check_name: Name of the check for reporting
        
    Returns:
        Dictionary with check results
    """
    script_path = os.path.join(current_dir, script_name)
    
    if not os.path.exists(script_path):
        return {
            'name': check_name,
            'status': 'error',
            'error': f"Script not found: {script_path}",
            'output': ''
        }
    
    try:
        cmd = [sys.executable, script_path] + args
        print(f"\n{'=' * 80}")
        print(f"Running: {check_name}")
        print(f"Command: {' '.join(cmd[:10])}...")  # Show first 10 args to avoid huge output
        print(f"{'=' * 80}\n")
        print(f"[DEBUG] Starting subprocess at {datetime.now().strftime('%H:%M:%S')}")
        sys.stdout.flush()  # Force output
        
        # Use Popen to stream output in real-time
        # Set PYTHONUNBUFFERED=1 to ensure Python output is not buffered
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,  # Unbuffered
            env=env
        )
        
        # Stream output in real-time with timeout checking
        stdout_lines = []
        import select
        import time
        
        last_output_time = time.time()
        while True:
            # Check if process has finished
            if process.poll() is not None:
                # Read any remaining output
                remaining = process.stdout.read()
                if remaining:
                    print(remaining, end='')
                    sys.stdout.flush()
                    stdout_lines.append(remaining)
                break
            
            # Try to read a line (non-blocking check)
            if process.stdout.readable():
                try:
                    output = process.stdout.readline()
                    if output:
                        print(output, end='')
                        sys.stdout.flush()
                        stdout_lines.append(output)
                        last_output_time = time.time()
                    elif time.time() - last_output_time > 300:  # 5 minutes without output
                        print(f"\n[WARNING] No output for 5 minutes, process may be stuck. PID: {process.pid}")
                        sys.stdout.flush()
                        last_output_time = time.time()
                except:
                    pass
            
            time.sleep(0.1)  # Small delay to avoid busy waiting
        
        returncode = process.poll()
        
        print(f"[DEBUG] Subprocess completed at {datetime.now().strftime('%H:%M:%S')} with return code {returncode}")
        sys.stdout.flush()
        
        return {
            'name': check_name,
            'status': 'success' if returncode == 0 else 'failed',
            'returncode': returncode,
            'stdout': ''.join(stdout_lines),
            'stderr': '',
            'output': ''.join(stdout_lines)
        }
    
    except subprocess.TimeoutExpired:
        return {
            'name': check_name,
            'status': 'timeout',
            'error': 'Check timed out after 1 hour',
            'output': ''
        }
    except Exception as e:
        return {
            'name': check_name,
            'status': 'error',
            'error': str(e),
            'output': ''
        }


def generate_report(results: List[Dict], output_file: str = None):
    """
    Generate a comprehensive report from verification results.
    
    Args:
        results: List of check results
        output_file: Optional file path to save report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("VERIFICATION CHECKS SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary
    total_checks = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    errors = sum(1 for r in results if r['status'] == 'error')
    timeouts = sum(1 for r in results if r['status'] == 'timeout')
    
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Total checks: {total_checks}")
    report_lines.append(f"  ✓ Successful: {successful}")
    report_lines.append(f"  ✗ Failed: {failed}")
    report_lines.append(f"  ⚠️  Errors: {errors}")
    report_lines.append(f"  ⏱  Timeouts: {timeouts}")
    report_lines.append("")
    
    # Detailed results
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 80)
    
    for result in results:
        report_lines.append(f"\n{result['name']}")
        report_lines.append(f"  Status: {result['status']}")
        
        if result['status'] == 'error' and 'error' in result:
            report_lines.append(f"  Error: {result['error']}")
        elif result['status'] == 'failed':
            report_lines.append(f"  Return code: {result.get('returncode', 'N/A')}")
        
        # Include last 20 lines of output
        if result.get('output'):
            output_lines = result['output'].split('\n')
            if len(output_lines) > 20:
                report_lines.append("  Output (last 20 lines):")
                for line in output_lines[-20:]:
                    report_lines.append(f"    {line}")
            else:
                report_lines.append("  Output:")
                for line in output_lines:
                    report_lines.append(f"    {line}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("=" * 80)
    
    # Generate recommendations based on results
    recommendations = []
    
    if failed > 0:
        recommendations.append("Some checks failed. Review the detailed output above.")
    
    if errors > 0:
        recommendations.append("Some checks encountered errors. Check script paths and dependencies.")
    
    # Check for specific issues in output
    for result in results:
        output_lower = result.get('output', '').lower()
        
        if 'overlap' in output_lower or 'leakage' in output_lower:
            recommendations.append("⚠️  Data leakage detected! Check preprocessing verification output.")
        
        if 'normalization' in output_lower and 'difference' in output_lower:
            recommendations.append("⚠️  Normalization mismatch detected! Train and val may have different scaling.")
        
        if 'loss' in output_lower and 'flat' in output_lower and result['name'] == 'Random Label Test':
            recommendations.append("✓ Random label test passed - loss stayed flat as expected.")
        elif 'loss' in output_lower and 'decrease' in output_lower and result['name'] == 'Random Label Test':
            recommendations.append("⚠️  Random label test failed - loss decreased, indicating possible bug.")
    
    if not recommendations:
        recommendations.append("All checks completed. Review individual check outputs for details.")
    
    for rec in recommendations:
        report_lines.append(f"  {rec}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Print report
    report_text = '\n'.join(report_lines)
    print("\n" + report_text)
    
    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")
    
    # Also save JSON version
    json_file = output_file.replace('.txt', '.json') if output_file else 'verification_report.json'
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total_checks,
                'successful': successful,
                'failed': failed,
                'errors': errors,
                'timeouts': timeouts
            },
            'results': results,
            'recommendations': recommendations
        }, f, indent=2)
    print(f"JSON report saved to: {json_file}")


def main():
    """Main function to run all verification checks."""
    # Generate output filename if not provided
    output_file = OUTPUT_FILE
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"verification_report_{timestamp}.txt"
    
    print("=" * 80)
    print("COMPREHENSIVE VERIFICATION CHECKS")
    print("=" * 80)
    print(f"Stocks: {STOCKS}")
    print(f"Time period: {TIME_ARGS}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Period type: {PERIOD_TYPE}")
    print(f"Lookback: {LOOKBACK}")
    print(f"Use NLP: {USE_NLP}")
    print(f"NLP method: {NLP_METHOD}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Note: Verification checks use new models (no pre-trained models needed)")
    print()
    
    results = []
    
    # Common arguments (dataset config passed to all scripts)
    common_args = [
        "--stocks"] + STOCKS + [
        "--time-args"] + TIME_ARGS + [
        "--model-type", MODEL_TYPE,
        "--period-type", PERIOD_TYPE,
        "--lookback", str(LOOKBACK),
        "--use-nlp", "true" if USE_NLP else "false",
        "--nlp-method", NLP_METHOD,
        "--batch-size", str(BATCH_SIZE),
        "--use-trained-models-dir", "false"  # Not needed for verification
    ]
    
    # 1. Preprocessing verification
    if not SKIP_PREPROCESSING:
        # Use subset of stocks for preprocessing verification (faster, still validates preprocessing)
        preprocessing_stocks = STOCKS[:PREPROCESSING_MAX_STOCKS] if len(STOCKS) > PREPROCESSING_MAX_STOCKS else STOCKS
        if len(STOCKS) > PREPROCESSING_MAX_STOCKS:
            print(f"[INFO] Limiting preprocessing verification to {PREPROCESSING_MAX_STOCKS} stocks (from {len(STOCKS)}) for speed")
        preprocessing_args = [
            "--stocks"] + preprocessing_stocks + [
            "--time-args"] + TIME_ARGS + [
            "--model-type", MODEL_TYPE,
            "--period-type", PERIOD_TYPE,
            "--lookback", str(LOOKBACK),
            "--use-nlp", "true" if USE_NLP else "false",
            "--nlp-method", NLP_METHOD,
            "--batch-size", str(BATCH_SIZE),
            "--use-trained-models-dir", "false"  # Not needed for verification
        ]
        
        result = run_check(
            "verify_preprocessing.py",
            preprocessing_args,
            "Preprocessing Verification"
        )
        results.append(result)
    
    # 2. Overfitting test
    if not SKIP_OVERFIT:
        overfit_args = common_args + [
            "--num-samples", str(OVERFIT_NUM_SAMPLES),
            "--num-epochs", str(OVERFIT_NUM_EPOCHS)
        ]
        result = run_check(
            "verify_overfit.py",
            overfit_args,
            "Overfitting Test"
        )
        results.append(result)
    
    # 3. Random label test
    if not SKIP_RANDOM:
        random_args = common_args + [
            "--num-epochs", str(RANDOM_NUM_EPOCHS)
        ]
        result = run_check(
            "verify_random_labels.py",
            random_args,
            "Random Label Test"
        )
        results.append(result)
    
    # Generate report
    generate_report(results, output_file)
    
    # Return appropriate exit code
    if any(r['status'] in ['failed', 'error'] for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

