"""
Verification module for training diagnostics.

This module contains scripts to verify and diagnose training issues:
- verify_overfit.py: Tests if model can overfit tiny dataset
- verify_random_labels.py: Tests training on shuffled labels
- verify_preprocessing.py: Verifies preprocessing consistency
- run_verification_checks.py: Main runner for all checks
"""

__all__ = [
    'verify_overfit',
    'verify_random_labels',
    'verify_preprocessing',
    'run_verification_checks'
]

