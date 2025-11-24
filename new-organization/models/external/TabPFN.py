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

import numpy as np
import torch

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

        backend_ctor = _load_backend(self.backend)
        self.estimator = backend_ctor(random_state=self.random_state, **self.model_params)
        self.is_fitted = False

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
        """
        proba = self.predict_proba(x)
        # Assume binary classification; take probability of the positive class.
        if proba.ndim == 2 and proba.shape[1] > 1:
            positive = proba[:, 1]
        else:
            positive = proba.reshape(-1)
        tensor = torch.from_numpy(positive).to(x.device, dtype=x.dtype)
        return tensor.view(-1, 1)

    @classmethod
    def from_config(cls, model_config):
        return cls(model_config)


# Register adapter with the registry (case-insensitive)
ModelRegistry.register("TABPFN", lambda config: TabPFNAdapter(config), TabPFNConfig)

