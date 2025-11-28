"""
Adapter for TabPFN estimators.

This module wraps the official TabPFN implementations (`tabpfn_client` or
`tabpfn`) in a `BaseModel` interface so the rest of the codebase can treat
them like regular models. The adapter is intentionally stateless with
respect to PyTorch autograd – TabPFN performs closed-form inference and does
not participate in gradient-based training. Call `fit()` once with flattened
tabular data, then use `forward()`/`predict()` for inference.
"""

from __future__ import annotations

import os
import numpy as np
import torch

# MPS workarounds: Disable problematic SDP backends on Apple Silicon
# These can cause "Invalid buffer size" errors even when memory should be sufficient
if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_SDP_DISABLE_FLASH_ATTENTION", "1")
    os.environ.setdefault("PYTORCH_SDP_DISABLE_MEM_EFFICIENT", "1")
    # Note: We don't disable FAST_PATH as it's generally safe

from ..base import BaseModel
from ..configs import TabPFNConfig
from ..registry import ModelRegistry


class _BackendNotAvailable(RuntimeError):
    pass


def _load_backend(name: str):
    """
    Lazily import the requested TabPFN backend and return the constructor.
    """
    name = name.lower()
    if name == "client":
        try:
            from tabpfn_client import TabPFNClassifier, init as tabpfn_init
        except ImportError as exc:
            raise _BackendNotAvailable(
                "tabpfn_client is not installed. Install it or switch backend='local'."
            ) from exc

        tabpfn_init()
        return TabPFNClassifier

    if name == "local":
        try:
            from tabpfn import TabPFNClassifier  # type: ignore
        except ImportError as exc:
            raise _BackendNotAvailable(
                "tabpfn is not installed. Install it or switch backend='client'."
            ) from exc
        return TabPFNClassifier

    raise ValueError(f"Unknown TabPFN backend '{name}'. Expected 'client' or 'local'.")


class TabPFNAdapter(BaseModel):
    """
    Thin wrapper around a TabPFN classifier.

    Notes:
        * Inputs must be 2-D tabular arrays (rows, features). If a tensor with
          shape (rows, timesteps, features_per_step) is provided, it will be
          flattened automatically to (rows, timesteps * features_per_step).
        * Training is handled via the `.fit()` method, not PyTorch optimisers.
    """

    def __init__(self, model_config: TabPFNConfig):
        super().__init__(model_config)
        if model_config is None:
            raise ValueError("model_config is required for TabPFNAdapter")
        if not isinstance(model_config, TabPFNConfig):
            raise TypeError(
                f"TabPFNAdapter requires TabPFNConfig, got {type(model_config).__name__}"
            )

        self.backend = model_config.backend
        self.max_samples = model_config.max_samples
        self.random_state = model_config.random_state
        self.model_params = model_config.model_params or {}
        self.inference_batch_size = model_config.inference_batch_size

        backend_ctor = _load_backend(self.backend)
        self.estimator = backend_ctor(random_state=self.random_state, **self.model_params)
        self.is_fitted = False
        
        # Debug logging: track forward calls for shape monitoring
        self._forward_call_count = 0
        self._debug_logging = True  # Set to False to disable debug prints

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_inputs(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        if x_np.ndim == 3:
            rows, timesteps, feats = x_np.shape
            return x_np.reshape(rows, timesteps * feats)
        if x_np.ndim == 2:
            return x_np

        raise ValueError(
            f"TabPFNAdapter expects 2-D or 3-D inputs, received shape {x_np.shape}"
        )

    @staticmethod
    def _prepare_targets(y: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np.reshape(-1)
        return y_np

    # ------------------------------------------------------------------
    # TabPFN public API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """
        Fit TabPFN on flattened features.
        """
        X_flat = self._flatten_inputs(X)
        y_vec = self._prepare_targets(y)

        if len(X_flat) > self.max_samples:
            raise ValueError(
                f"TabPFN requires <= {self.max_samples} rows, received {len(X_flat)}."
            )

        self.estimator.fit(X_flat, y_vec)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("TabPFNAdapter must be fit before calling predict().")
        X_flat = self._flatten_inputs(X)
        return self.estimator.predict(X_flat)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("TabPFNAdapter must be fit before calling predict_proba().")
        X_flat = self._flatten_inputs(X)
        proba = self.estimator.predict_proba(X_flat)
        return proba

    # ------------------------------------------------------------------
    # BaseModel overrides
    # ------------------------------------------------------------------
    def forward(self, x, params=None):
        """
        Forward pass used by the Trainer. Raises if the estimator has not been fit.
        Returns probabilities (Nx1 tensor) for binary classification.
        
        Automatically chunks large batches to avoid memory issues with TabPFN's
        quadratic attention complexity.
        """
        if not self.is_fitted:
            raise RuntimeError("TabPFNAdapter must be fit before calling forward().")
        
        # Get batch size from config
        chunk_size = self.inference_batch_size
        
        # Flatten inputs once
        X_flat = self._flatten_inputs(x)
        num_samples = X_flat.shape[0]
        
        # Debug logging: print shape info for first few calls or when batch is large
        if self._debug_logging:
            self._forward_call_count += 1
            # Log first 3 calls, then every 100th call, or if batch is suspiciously large
            should_log = (
                self._forward_call_count <= 3 or
                self._forward_call_count % 100 == 0 or
                num_samples > 1000
            )
            if should_log:
                print(f"[TabPFN] Forward call #{self._forward_call_count}: "
                      f"X_flat.shape={X_flat.shape}, chunk_size={chunk_size}, "
                      f"will_chunk={num_samples > chunk_size}")
        
        # If batch is small enough, process directly
        if num_samples <= chunk_size:
            proba = self.estimator.predict_proba(X_flat)
            proba = torch.from_numpy(proba).to(x.device, dtype=x.dtype)
            return proba
        
        # Otherwise, chunk the batch
        all_proba = []
        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        if self._debug_logging and should_log:
            print(f"[TabPFN] Chunking {num_samples} samples into {num_chunks} chunks of ~{chunk_size}")
        
        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)
            X_chunk = X_flat[i:end_idx]
            proba_chunk = self.estimator.predict_proba(X_chunk)
            all_proba.append(proba_chunk)
        
        # Concatenate results
        proba = np.concatenate(all_proba, axis=0)
        proba = torch.from_numpy(proba).to(x.device, dtype=x.dtype)
        return proba

    @classmethod
    def from_config(cls, model_config):
        return cls(model_config)


# Register adapter with the registry (case-insensitive)
ModelRegistry.register("TABPFN", lambda config: TabPFNAdapter(config), TabPFNConfig)

