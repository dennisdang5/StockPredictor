"""
PortfolioTrainer: Specialized trainer for Portfolio architectures.

Handles special cases for Portfolio models, particularly:
- Independent TabPFN portfolios: fits TabPFN backbones before training
- Shared TabPFN portfolios: supports both "naive" (ensemble) and "cotraining" methods
  - Naive method: uses base Trainer's ensemble approach
  - Co-training method: iteratively re-fits TabPFN backbone using MLP predictions
- Freezes TabPFN backbones (they don't support gradients)
- Trains only the MLP head with backpropagation
"""

import numpy as np
import torch
import copy
from trainer import Trainer
from models.external.TabPFN import TabPFNAdapter
from models.configs import TabPFNConfig


class PortfolioTrainer(Trainer):
    """
    Specialized trainer for Portfolio model architectures.
    
    Inherits from Trainer and adds portfolio-specific logic, particularly
    for handling TabPFN backbones which require fitting before training.
    """
    
    def __init__(self, config):
        """
        Initialize PortfolioTrainer.
        
        Args:
            config: TrainerConfig instance with model_type="Portfolio"
        """
        # Call parent constructor first
        super().__init__(config)
        
        # Detect TabPFN portfolio configurations
        self._detect_tabpfn_portfolio()
        
        # For shared TabPFN portfolios, check training method
        if self.is_shared_tabpfn_portfolio:
            self.shared_tabpfn_training_method = getattr(config, 'shared_tabpfn_training_method', 'naive').lower()
            if self.shared_tabpfn_training_method not in ("naive", "cotraining"):
                raise ValueError(
                    f"shared_tabpfn_training_method must be 'naive' or 'cotraining', "
                    f"got '{self.shared_tabpfn_training_method}'"
                )
            
            if self.shared_tabpfn_training_method == "cotraining":
                self.cotraining_refit_interval = getattr(config, 'cotraining_refit_interval', 5)
                self.cotraining_start_epoch = getattr(config, 'cotraining_start_epoch', 1)
                self.shared_tabpfn_backbone_fitted = False
                if self.is_main:
                    print(f"[Shared TabPFN Portfolio] Using co-training method "
                          f"(refit_interval={self.cotraining_refit_interval}, "
                          f"start_epoch={self.cotraining_start_epoch})")
            else:
                if self.is_main:
                    print(f"[Shared TabPFN Portfolio] Using naive (ensemble) method")
        
        # Prepare data for TabPFN backbones if needed
        if self.is_independent_tabpfn_portfolio:
            self._prepare_tabpfn_portfolio_data()
        
        # For shared TabPFN portfolios with co-training, prepare data
        if self.is_shared_tabpfn_portfolio and self.shared_tabpfn_training_method == "cotraining":
            self._prepare_shared_tabpfn_portfolio_data_for_cotraining()
        
        # Freeze TabPFN backbones after model creation
        if self.is_independent_tabpfn_portfolio:
            self._freeze_tabpfn_backbones()
    
    def _detect_tabpfn_portfolio(self):
        """
        Detect if this is a Portfolio model with TabPFN backbones.
        Sets flags for independent and shared strategies.
        
        Note: Parent Trainer already sets is_shared_tabpfn_portfolio for shared strategy,
        so we preserve that and only set independent flag.
        """
        self.is_tabpfn_portfolio = False
        self.is_independent_tabpfn_portfolio = False
        
        # Preserve parent's is_shared_tabpfn_portfolio flag if it was set
        # (parent Trainer sets this during __init__)
        
        if self.model_type_upper != "PORTFOLIO":
            return
        
        if not (hasattr(self.model_config, 'base_model_type') and 
                hasattr(self.model_config, 'strategy')):
            return
        
        base_model_type_upper = str(self.model_config.base_model_type).upper()
        strategy_lower = str(self.model_config.strategy).lower()
        
        if base_model_type_upper == "TABPFN":
            self.is_tabpfn_portfolio = True
            if strategy_lower == "independent":
                self.is_independent_tabpfn_portfolio = True
            # For shared strategy, parent Trainer already sets is_shared_tabpfn_portfolio
            # so we don't need to set it here
    
    def _prepare_tabpfn_portfolio_data(self):
        """
        Prepare per-stock training data for fitting TabPFN backbones.
        Called during __init__ after data is loaded.
        """
        if not self.is_independent_tabpfn_portfolio:
            return
        
        if self.is_main:
            print("[Independent TabPFN Portfolio] Preparing per-stock data for TabPFN backbone fitting...")
        
        # Ensure we have stock indices
        if any(x is None for x in (self.train_stock_indices_tensor, self.val_stock_indices_tensor)):
            raise ValueError(
                "Independent TabPFN Portfolio models require stock index metadata. "
                "Please regenerate data with return_stock_indices=True."
            )
        
        # Access data from IndexedDataset instances
        # IndexedDataset stores data as self.X and self.Y
        X_train = self.train_ds.X
        Y_train = self.train_ds.Y
        X_val = self.validationLoader.dataset.X
        Y_val = self.validationLoader.dataset.Y
        
        # Convert to tensors if needed
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(Y_train, np.ndarray):
            Y_train = torch.from_numpy(Y_train).float()
        if isinstance(X_val, np.ndarray):
            X_val = torch.from_numpy(X_val).float()
        if isinstance(Y_val, np.ndarray):
            Y_val = torch.from_numpy(Y_val).float()
        
        # Combine train and val data
        X_train_val = torch.cat([X_train, X_val], dim=0)
        Y_train_val = torch.cat([Y_train, Y_val], dim=0)
        Strain_val_combined = torch.cat([
            self.train_stock_indices_tensor,
            self.val_stock_indices_tensor
        ], dim=0)
        
        # Store for later fitting
        self.tabpfn_portfolio_train_data = {
            'X': X_train_val,
            'Y': Y_train_val,
            'stock_indices': Strain_val_combined,
            'stocks': self.filtered_stock_list
        }
        self.tabpfn_backbones_fitted = False
        
        if self.is_main:
            print(f"[Independent TabPFN Portfolio] Prepared data for {len(self.filtered_stock_list)} stocks")
    
    def _freeze_tabpfn_backbones(self):
        """
        Freeze all TabPFN backbone models in a Portfolio architecture.
        TabPFN models don't support gradient-based training.
        """
        if self.Model is None:
            return
        
        # Handle DDP wrapping
        model = self.Model.module if hasattr(self.Model, 'module') else self.Model
        
        if not hasattr(model, 'stock_models') and not hasattr(model, 'shared_model'):
            return
        
        if hasattr(model, 'stock_models'):
            # Independent strategy: freeze each stock's TabPFN model
            frozen_count = 0
            for ticker, backbone in model.stock_models.items():
                if isinstance(backbone, TabPFNAdapter):
                    for param in backbone.parameters():
                        param.requires_grad = False
                    frozen_count += 1
                    if self.is_main:
                        print(f"[Independent TabPFN Portfolio] Frozen TabPFN backbone for {ticker}")
            if self.is_main and frozen_count > 0:
                print(f"[Independent TabPFN Portfolio] Frozen {frozen_count} TabPFN backbones")
        elif hasattr(model, 'shared_model'):
            # Shared strategy: freeze the shared TabPFN model
            if isinstance(model.shared_model, TabPFNAdapter):
                for param in model.shared_model.parameters():
                    param.requires_grad = False
                if self.is_main:
                    print("[Shared TabPFN Portfolio] Frozen shared TabPFN backbone")
    
    def _fit_independent_tabpfn_portfolio_backbones(self):
        """
        Fit each TabPFN backbone in an independent TabPFN portfolio model.
        Collects per-stock data from training data and fits each stock's TabPFN model.
        """
        if not hasattr(self, 'tabpfn_portfolio_train_data'):
            raise RuntimeError(
                "TabPFN portfolio training data not prepared. "
                "This should be set in _prepare_tabpfn_portfolio_data()"
            )
        
        data_dict = self.tabpfn_portfolio_train_data
        X_train_val = data_dict['X']
        Y_train_val = data_dict['Y']
        Strain_val = data_dict['stock_indices']
        stocks = data_dict['stocks']
        
        # Handle DDP wrapping
        model = self.Model.module if hasattr(self.Model, 'module') else self.Model
        
        if not hasattr(model, 'stock_models'):
            raise ValueError(
                "Model does not have stock_models attribute. "
                "Expected independent strategy Portfolio model."
            )
        
        # Get base model config for TabPFN
        base_config = self.model_config.base_model_config
        if not isinstance(base_config, TabPFNConfig):
            raise TypeError(
                f"Expected TabPFNConfig, got {type(base_config).__name__}. "
                f"Portfolio base_model_config must be TabPFNConfig for TabPFN portfolios."
            )
        
        # Flatten inputs for TabPFN (TabPFN expects 2D tabular data)
        # Convert (N, seq_len, features) to (N, seq_len * features)
        if isinstance(X_train_val, torch.Tensor):
            X_flat = X_train_val.detach().cpu().numpy()
        else:
            X_flat = np.asarray(X_train_val)
        
        if X_flat.ndim == 3:
            N, seq_len, features = X_flat.shape
            X_flat = X_flat.reshape(N, seq_len * features)
        elif X_flat.ndim != 2:
            raise ValueError(
                f"Expected 2D or 3D input array, got shape {X_flat.shape}"
            )
        
        # Prepare targets: convert from [-1, 1] to [0, 1] for binary classification
        if isinstance(Y_train_val, torch.Tensor):
            Y_flat = Y_train_val.detach().cpu().numpy()
        else:
            Y_flat = np.asarray(Y_train_val)
        
        # TabPFN expects class labels 0 and 1 for binary classification
        # Our targets are -1 and 1, so convert: -1 -> 0, 1 -> 1
        Y_encoded = np.where(Y_flat.flatten() > 0, 1, 0).astype(np.int32)
        
        # Fit each stock's TabPFN backbone
        Strain_np = Strain_val.numpy() if isinstance(Strain_val, torch.Tensor) else np.asarray(Strain_val)
        
        fitted_count = 0
        failed_count = 0
        
        for stock_idx, ticker in enumerate(stocks):
            # Get data for this stock
            stock_mask = (Strain_np == stock_idx)
            if not stock_mask.any():
                if self.is_main:
                    print(f"[Independent TabPFN Portfolio] Warning: No training data found for {ticker}")
                failed_count += 1
                continue
            
            X_stock = X_flat[stock_mask]
            Y_stock = Y_encoded[stock_mask]
            
            # Check max_samples constraint
            max_samples = getattr(base_config, 'max_samples', 50000)
            if len(X_stock) > max_samples:
                if self.is_main:
                    print(f"[Independent TabPFN Portfolio] Warning: {ticker} has {len(X_stock)} samples, "
                          f"but max_samples={max_samples}. Randomly sampling {max_samples} samples.")
                # Randomly sample max_samples
                random_state = getattr(base_config, 'random_state', 42)
                rng = np.random.RandomState(random_state)
                sample_indices = rng.choice(len(X_stock), size=max_samples, replace=False)
                X_stock = X_stock[sample_indices]
                Y_stock = Y_stock[sample_indices]
            
            # Get the TabPFN backbone for this stock
            if ticker not in model.stock_models:
                if self.is_main:
                    print(f"[Independent TabPFN Portfolio] Warning: No backbone model found for {ticker}")
                failed_count += 1
                continue
            
            backbone = model.stock_models[ticker]
            if not isinstance(backbone, TabPFNAdapter):
                if self.is_main:
                    print(f"[Independent TabPFN Portfolio] Warning: Backbone for {ticker} is not a TabPFNAdapter, skipping fit")
                failed_count += 1
                continue
            
            # Fit the TabPFN model
            try:
                backbone.fit(X_stock, Y_stock)
                fitted_count += 1
                if self.is_main:
                    print(f"[Independent TabPFN Portfolio] Fitted TabPFN backbone for {ticker} "
                          f"({len(X_stock)} samples)")
            except Exception as e:
                failed_count += 1
                if self.is_main:
                    print(f"[Independent TabPFN Portfolio] Error fitting TabPFN backbone for {ticker}: {e}")
                # Continue with other stocks even if one fails
                continue
        
        if self.is_main:
            print(f"[Independent TabPFN Portfolio] Finished fitting TabPFN backbones: "
                  f"{fitted_count} succeeded, {failed_count} failed")
    
    def _apply_weighted_sampling_for_shared_tabpfn_wrapper(self, Strain_train_val, D_train_val, max_samples):
        """
        Wrapper to call base Trainer's weighted sampling method.
        Prepares inputs and calls the inherited _apply_weighted_sampling_for_shared_tabpfn method.
        
        Returns:
            selected_indices: numpy array of selected sample indices, or None if weighted sampling cannot be applied
        """
        num_stocks = len(self.filtered_stock_list)
        
        # Convert dates to numpy array
        if isinstance(D_train_val, torch.Tensor):
            D_train_val = D_train_val.detach().cpu().numpy()
        elif isinstance(D_train_val, list):
            D_train_val = np.array(D_train_val)
        elif D_train_val is None:
            return None  # Cannot do weighted sampling without dates
        
        # Convert stock indices to numpy
        if isinstance(Strain_train_val, torch.Tensor):
            Strain_train_val = Strain_train_val.detach().cpu().numpy()
        Strain_train_val = np.asarray(Strain_train_val, dtype=int)
        
        # Get unique dates
        unique_dates = np.unique(D_train_val)
        num_dates = len(unique_dates)
        
        # Calculate parameters (same as base Trainer)
        total_datapoints = len(D_train_val)
        stocks_per_date = max(1, total_datapoints // max_samples)
        datapoints_per_stock = max_samples // num_stocks
        
        # Get random state from config
        base_config = self.model_config.base_model_config
        random_state = getattr(base_config, 'random_state', 42)
        rng = np.random.default_rng(random_state)
        
        if self.is_main:
            print(f"[Shared TabPFN Portfolio] Weighted sampling: "
                  f"{num_stocks} stocks, {num_dates} dates, max_samples={max_samples}")
        
        # Call base Trainer's weighted sampling method
        return self._apply_weighted_sampling_for_shared_tabpfn(
            Strain_train_val, D_train_val, unique_dates, num_stocks, stocks_per_date,
            datapoints_per_stock, rng, max_samples
        )
    
    def _prepare_shared_tabpfn_portfolio_data_for_cotraining(self):
        """
        Prepare combined train+val data for fitting shared TabPFN backbone during co-training.
        Stores full dataset along with date information for weighted sampling during re-fitting.
        Called during __init__ after data is loaded.
        """
        if not self.is_shared_tabpfn_portfolio:
            return
        
        if self.is_main:
            print("[Shared TabPFN Portfolio] Preparing combined train+val data for co-training...")
        
        # Access data from IndexedDataset instances
        X_train = self.train_ds.X
        Y_train = self.train_ds.Y
        X_val = self.validationLoader.dataset.X
        Y_val = self.validationLoader.dataset.Y
        
        # Get date information (IndexedDataset stores dates)
        D_train = getattr(self.train_ds, 'dates', None)
        D_val = getattr(self.validationLoader.dataset, 'dates', None)
        
        # Convert to tensors if needed
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(Y_train, np.ndarray):
            Y_train = torch.from_numpy(Y_train).float()
        if isinstance(X_val, np.ndarray):
            X_val = torch.from_numpy(X_val).float()
        if isinstance(Y_val, np.ndarray):
            Y_val = torch.from_numpy(Y_val).float()
        
        # Combine train and val data
        X_train_val = torch.cat([X_train, X_val], dim=0)
        Y_train_val = torch.cat([Y_train, Y_val], dim=0)
        
        # Combine date information
        D_train_val = None
        if D_train is not None and D_val is not None:
            if isinstance(D_train, (list, np.ndarray)):
                D_train_val = np.concatenate([np.asarray(D_train), np.asarray(D_val)], axis=0)
            elif isinstance(D_train, torch.Tensor):
                D_train_val = torch.cat([D_train, D_val], dim=0)
        
        # Combine stock indices if available
        Strain_train_val = None
        if self.train_stock_indices_tensor is not None and self.val_stock_indices_tensor is not None:
            Strain_train_val = torch.cat([
                self.train_stock_indices_tensor,
                self.val_stock_indices_tensor
            ], dim=0)
        
        # Store for later fitting (with date info for weighted sampling)
        self.shared_tabpfn_data = {
            'X': X_train_val,
            'Y': Y_train_val,
            'stock_indices': Strain_train_val,
            'dates': D_train_val,
            'stocks': self.filtered_stock_list
        }
        
        if self.is_main:
            print(f"[Shared TabPFN Portfolio] Prepared data for co-training: "
                  f"{len(X_train_val)} samples from {len(self.filtered_stock_list)} stocks")
    
    def _fit_shared_tabpfn_backbone(self, use_mlp_targets=False, mlp_predictions=None):
        """
        Fit the shared TabPFN backbone for shared TabPFN portfolio models.
        Uses weighted sampling when available to ensure balanced representation across dates and stocks.
        Reuses utility functions from base Trainer for flattening and encoding.
        
        Args:
            use_mlp_targets: If True, use MLP predictions as soft targets instead of original labels
            mlp_predictions: MLP predictions (probabilities or logits) to use as targets.
                           Should be shape (N,) or (N, 1) for N samples.
                           Only used if use_mlp_targets=True.
        """
        if not hasattr(self, 'shared_tabpfn_data'):
            raise RuntimeError(
                "Shared TabPFN training data not prepared. "
                "This should be set in _prepare_shared_tabpfn_portfolio_data_for_cotraining()"
            )
        
        data_dict = self.shared_tabpfn_data
        X_train_val = data_dict['X']
        Y_train_val = data_dict['Y']
        Strain_train_val = data_dict['stock_indices']
        D_train_val = data_dict.get('dates', None)
        
        # Get base model config
        base_config = self.model_config.base_model_config
        if not isinstance(base_config, TabPFNConfig):
            raise TypeError(
                f"Expected TabPFNConfig, got {type(base_config).__name__}."
            )
        
        # Handle DDP wrapping
        model = self.Model.module if hasattr(self.Model, 'module') else self.Model
        
        if not hasattr(model, 'shared_model'):
            raise ValueError(
                "Model does not have shared_model attribute. "
                "Expected shared strategy Portfolio model."
            )
        
        backbone = model.shared_model
        if not isinstance(backbone, TabPFNAdapter):
            raise TypeError(
                f"Expected TabPFNAdapter for shared model, got {type(backbone).__name__}."
            )
        
        max_samples = getattr(base_config, 'max_samples', 50000)
        
        # Apply weighted sampling if we have date information and exceed max_samples
        selected_indices = None
        if len(X_train_val) > max_samples and D_train_val is not None and Strain_train_val is not None:
            # Use base Trainer's weighted sampling method
            selected_indices = self._apply_weighted_sampling_for_shared_tabpfn_wrapper(
                Strain_train_val, D_train_val, max_samples
            )
        
        # If weighted sampling failed or not applicable, use simple random sampling
        if selected_indices is None:
            if len(X_train_val) > max_samples:
                # Fall back to simple random sampling
                random_state = getattr(base_config, 'random_state', 42)
                rng = np.random.RandomState(random_state)
                selected_indices = rng.choice(len(X_train_val), size=max_samples, replace=False)
                if self.is_main:
                    print(f"[Shared TabPFN Portfolio] Using simple random sampling "
                          f"({len(selected_indices)} samples from {len(X_train_val)})")
            else:
                # Use all data
                selected_indices = np.arange(len(X_train_val))
        
        # Select samples using indices
        if isinstance(X_train_val, torch.Tensor):
            X_selected = X_train_val[selected_indices]
        else:
            X_selected = X_train_val[selected_indices] if hasattr(X_train_val, '__getitem__') else np.asarray(X_train_val)[selected_indices]
        
        if isinstance(Y_train_val, torch.Tensor):
            Y_selected = Y_train_val[selected_indices]
        else:
            Y_selected = Y_train_val[selected_indices] if hasattr(Y_train_val, '__getitem__') else np.asarray(Y_train_val)[selected_indices]
        
        # Get corresponding MLP predictions if using them
        mlp_selected = None
        if use_mlp_targets and mlp_predictions is not None:
            if isinstance(mlp_predictions, torch.Tensor):
                mlp_selected = mlp_predictions[selected_indices]
            else:
                mlp_selected = mlp_predictions[selected_indices] if hasattr(mlp_predictions, '__getitem__') else np.asarray(mlp_predictions)[selected_indices]
        
        # Use base Trainer's utility function to flatten inputs
        X_flat = self._flatten_tabpfn_tensor(X_selected)
        
        # Prepare targets
        if use_mlp_targets:
            if mlp_selected is None:
                raise ValueError("mlp_predictions must be provided when use_mlp_targets=True")
            
            # Convert MLP predictions to class labels
            if isinstance(mlp_selected, torch.Tensor):
                mlp_pred = mlp_selected.detach().cpu().numpy()
            else:
                mlp_pred = np.asarray(mlp_selected)
            
            mlp_pred = mlp_pred.flatten() if mlp_pred.ndim > 1 else mlp_pred
            
            # Convert to class labels (0 or 1)
            # If values are in [0, 1], treat as probabilities; otherwise as logits
            if mlp_pred.min() >= 0 and mlp_pred.max() <= 1:
                Y_encoded = (mlp_pred > 0.5).astype(np.int32)
            else:
                Y_encoded = (mlp_pred > 0).astype(np.int32)
            
            target_source = "MLP predictions"
        else:
            # Use base Trainer's utility function for original labels
            Y_encoded, _ = self._encode_tabpfn_targets(Y_selected)
            Y_encoded = Y_encoded.astype(np.int32)  # Ensure int32 for TabPFN
            target_source = "original labels"
        
        # Fit the shared TabPFN backbone
        try:
            backbone.fit(X_flat, Y_encoded)
            if self.is_main:
                print(f"[Shared TabPFN Portfolio] Fitted shared TabPFN backbone "
                      f"({len(X_flat)} samples, using {target_source})")
        except Exception as e:
            if self.is_main:
                print(f"[Shared TabPFN Portfolio] Error fitting shared TabPFN backbone: {e}")
            raise
    
    def _get_mlp_predictions_on_train_val(self):
        """
        Run the MLP head on combined train+val data to get predictions for co-training.
        The MLP predictions are used as soft targets for re-fitting the TabPFN backbone.
        
        Returns:
            mlp_predictions: Tensor of shape (N, 1) containing MLP predictions (logits)
        """
        if not hasattr(self, 'shared_tabpfn_data'):
            raise RuntimeError(
                "Shared TabPFN training data not prepared. "
                "This should be set in _prepare_shared_tabpfn_portfolio_data_for_cotraining()"
            )
        
        data_dict = self.shared_tabpfn_data
        X_train_val = data_dict['X']
        Strain_train_val = data_dict['stock_indices']
        
        # Ensure data is on the correct device
        device = next(self.Model.parameters()).device
        X_train_val = X_train_val.to(device)
        if Strain_train_val is not None:
            Strain_train_val = Strain_train_val.to(device)
        else:
            raise ValueError(
                "Stock indices are required for Portfolio model forward pass. "
                "Please regenerate data with return_stock_indices=True."
            )
        
        # Set model to evaluation mode
        self.Model.eval()
        
        # Collect predictions in batches to avoid memory issues
        batch_size = self.batch_size
        mlp_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_train_val), batch_size):
                end_idx = min(i + batch_size, len(X_train_val))
                X_batch = X_train_val[i:end_idx]
                Strain_batch = Strain_train_val[i:end_idx]
                
                # Forward pass through Portfolio model (MLP head)
                # Portfolio model expects params dict with stock_indices
                params = {'stock_indices': Strain_batch}
                predictions = self.Model(X_batch, params=params)
                
                mlp_predictions.append(predictions.cpu())
        
        # Concatenate all predictions
        mlp_predictions = torch.cat(mlp_predictions, dim=0)
        
        # Set model back to training mode
        self.Model.train()
        
        if self.is_main:
            print(f"[Shared TabPFN Portfolio] Obtained MLP predictions for {len(mlp_predictions)} samples")
        
        return mlp_predictions
    
    def train_one_epoch(self, epoch):
        """
        Override train_one_epoch to handle TabPFN backbone fitting.
        
        For independent TabPFN portfolios:
        1. Fit TabPFN backbones on first epoch (if not already fitted)
        2. Then proceed with normal training (only MLP head trains via gradients)
        """
        # Handle shared TabPFN portfolios
        if self.is_shared_tabpfn_portfolio:
            if self.shared_tabpfn_training_method == "naive":
                # Use base Trainer's ensemble method
                return super().train_one_epoch(epoch)
            else:  # cotraining method
                # Initial fit at epoch 0 with original labels
                if epoch == 0 and not getattr(self, 'shared_tabpfn_backbone_fitted', False):
                    if self.is_main:
                        print("[Shared TabPFN Portfolio] Co-training: Initial fitting of shared TabPFN backbone...")
                    self._fit_shared_tabpfn_backbone(use_mlp_targets=False)
                    self._freeze_tabpfn_backbones()
                    self.shared_tabpfn_backbone_fitted = True
                    if self.is_main:
                        print("[Shared TabPFN Portfolio] Shared TabPFN backbone fitted and frozen. "
                              "Training MLP head with backpropagation...")
                
                # Periodic re-fitting with MLP predictions as soft targets
                elif (epoch >= self.cotraining_start_epoch and 
                      epoch > 0 and 
                      epoch % self.cotraining_refit_interval == 0):
                    if self.is_main:
                        print(f"[Shared TabPFN Portfolio] Co-training: Re-fitting TabPFN backbone at epoch {epoch + 1}...")
                    mlp_predictions = self._get_mlp_predictions_on_train_val()
                    self._fit_shared_tabpfn_backbone(use_mlp_targets=True, mlp_predictions=mlp_predictions)
                    self._freeze_tabpfn_backbones()
                    if self.is_main:
                        print("[Shared TabPFN Portfolio] TabPFN backbone re-fitted using MLP predictions. "
                              "Continuing training...")
                
                # Continue with normal training (MLP head trains via backpropagation)
                return super().train_one_epoch(epoch)
        
        # Handle independent TabPFN portfolios: fit backbones before training
        if self.is_independent_tabpfn_portfolio:
            if epoch == 0 and not getattr(self, 'tabpfn_backbones_fitted', False):
                if self.is_main:
                    print("[Independent TabPFN Portfolio] Fitting TabPFN backbones before training...")
                self._fit_independent_tabpfn_portfolio_backbones()
                self.tabpfn_backbones_fitted = True
                # Ensure backbones are frozen after fitting
                self._freeze_tabpfn_backbones()
                if self.is_main:
                    print("[Independent TabPFN Portfolio] TabPFN backbones fitted and frozen. "
                          "Training MLP head with backpropagation...")
        
        # Continue with parent's training logic
        return super().train_one_epoch(epoch)

