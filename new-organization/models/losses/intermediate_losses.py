"""
Loss functions that use intermediate layer outputs.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from .base_loss import BaseLoss


class AutoEncoderLoss(BaseLoss):
    """
    Loss function for AutoEncoder models.
    
    Combines prediction loss with reconstruction loss from decoder output.
    Optionally includes regularization on encoder output.
    """
    
    def __init__(
        self,
        prediction_weight: float = 1.0,
        reconstruction_weight: float = 0.5,
        intermediate_layers: Optional[List[str]] = None
    ):
        """
        Initialize AutoEncoderLoss.
        
        Args:
            prediction_weight: Weight for prediction loss (MSE between predictions and targets)
            reconstruction_weight: Weight for reconstruction loss (MSE between decoder output and input)
            intermediate_layers: List of layer names to hook (default: ["encoder", "decoder"])
        """
        super().__init__()
        self.prediction_loss = nn.MSELoss()
        self.reconstruction_loss = nn.MSELoss()
        self.prediction_weight = prediction_weight
        self.reconstruction_weight = reconstruction_weight
        
        if intermediate_layers is None:
            intermediate_layers = ["encoder", "decoder"]
        self._intermediate_layers = intermediate_layers
    
    def requires_intermediates(self) -> bool:
        """AutoEncoderLoss requires intermediate outputs."""
        return True
    
    def get_hook_targets(self) -> List[str]:
        """Return list of layers to hook."""
        return self._intermediate_layers
    
    def __call__(self, predictions, targets, intermediates: Optional[Dict[str, Any]] = None):
        """
        Compute combined prediction and reconstruction loss.
        
        Args:
            predictions: Model predictions tensor
            targets: Target values tensor
            intermediates: Dictionary with "encoder" and "decoder" outputs
        
        Returns:
            Combined loss tensor
        """
        pred_loss = self.prediction_loss(predictions, targets)
        
        if intermediates and "decoder" in intermediates and "input" in intermediates:
            # Reconstruction loss: decoder output should match input
        # Hook captures decoder Sequential output (flattened), need to reshape to match input
            decoder_output = intermediates["decoder"]
            input_tensor = intermediates["input"]
            
            # Reshape decoder output to match input shape
            # Input shape: (batch, seq_len, num_features)
            # Decoder output: (batch, seq_len * num_features) - flattened
            if decoder_output.dim() == 2 and input_tensor.dim() == 3:
                # Reshape flattened decoder output to match input
                batch_size, seq_len, num_features = input_tensor.shape
                decoder_output = decoder_output.unflatten(1, (seq_len, num_features))
            
            recon_loss = self.reconstruction_loss(decoder_output, input_tensor)
            return (self.prediction_weight * pred_loss +
                   self.reconstruction_weight * recon_loss)
        
        # Fallback: only prediction loss if intermediates not available
        return self.prediction_weight * pred_loss


class AELSTMLoss(BaseLoss):
    """
    Loss function for AELSTM models.
    
    Combines prediction loss with autoencoder intermediate features.
    Can include reconstruction loss from the autoencoder component.
    """
    
    def __init__(
        self,
        prediction_weight: float = 1.0,
        ae_reconstruction_weight: float = 0.3,
        intermediate_layers: Optional[List[str]] = None
    ):
        """
        Initialize AELSTMLoss.
        
        Args:
            prediction_weight: Weight for prediction loss
            ae_reconstruction_weight: Weight for autoencoder reconstruction loss
            intermediate_layers: List of layer names to hook
                                (default: ["AE.encoder", "AE.decoder"])
        """
        super().__init__()
        self.prediction_loss = nn.MSELoss()
        self.reconstruction_loss = nn.MSELoss()
        self.prediction_weight = prediction_weight
        self.ae_reconstruction_weight = ae_reconstruction_weight
        
        if intermediate_layers is None:
            intermediate_layers = ["AE.encoder", "AE.decoder"]
        self._intermediate_layers = intermediate_layers
    
    def requires_intermediates(self) -> bool:
        """AELSTMLoss requires intermediate outputs."""
        return True
    
    def get_hook_targets(self) -> List[str]:
        """Return list of layers to hook."""
        return self._intermediate_layers
    
    def __call__(self, predictions, targets, intermediates: Optional[Dict[str, Any]] = None):
        """
        Compute combined prediction and autoencoder reconstruction loss.
        
        Args:
            predictions: Model predictions tensor
            targets: Target values tensor
            intermediates: Dictionary with autoencoder intermediate outputs
        
        Returns:
            Combined loss tensor
        """
        pred_loss = self.prediction_loss(predictions, targets)
        
        if intermediates:
            total_loss = self.prediction_weight * pred_loss
            
            # Add reconstruction loss if decoder and input are available
            if "AE.decoder" in intermediates and "input" in intermediates:
                # Hook captures decoder Sequential output (flattened), need to reshape to match input
                decoder_output = intermediates["AE.decoder"]
                input_tensor = intermediates["input"]
                
                # Reshape decoder output to match input shape
                # Input shape: (batch, seq_len, num_features)
                # Decoder output: (batch, seq_len * num_features) - flattened
                if decoder_output.dim() == 2 and input_tensor.dim() == 3:
                    # Reshape flattened decoder output to match input
                    batch_size, seq_len, num_features = input_tensor.shape
                    decoder_output = decoder_output.unflatten(1, (seq_len, num_features))
                
                recon_loss = self.reconstruction_loss(decoder_output, input_tensor)
                total_loss += self.ae_reconstruction_weight * recon_loss
            
            return total_loss
        
        return self.prediction_weight * pred_loss

class CAELSTMLoss(BaseLoss):
    """
    Loss function for CAELSTM models.
    
    Combines prediction loss with CAE reconstruction loss.
    CAELSTM has decoder norm layers directly on the model (not nested under CAE).
    """
    
    def __init__(
        self,
        prediction_weight: float = 1.0,
        reconstruction_weight: float = 0.3,
        intermediate_layers: Optional[List[str]] = None
    ):
        """
        Initialize CAELSTMLoss.
        
        Args:
            prediction_weight: Weight for prediction loss
            reconstruction_weight: Weight for CAE reconstruction loss
            intermediate_layers: List of layer names to hook (default: ["short_dec_norm", "long_dec_norm"])
        """
        super().__init__()
        self.prediction_loss = nn.MSELoss()
        self.reconstruction_loss = nn.MSELoss()
        self.prediction_weight = prediction_weight
        self.reconstruction_weight = reconstruction_weight
        
        if intermediate_layers is None:
            intermediate_layers = ["short_dec_norm", "long_dec_norm"]
        self._intermediate_layers = intermediate_layers
    
    def requires_intermediates(self) -> bool:
        """CAELSTMLoss requires intermediate outputs."""
        return True
    
    def get_hook_targets(self) -> List[str]:
        """Return list of layers to hook."""
        return self._intermediate_layers
    
    def __call__(self, predictions, targets, intermediates: Optional[Dict[str, Any]] = None):
        """
        Compute combined prediction and CAE reconstruction loss.
        
        Args:
            predictions: Model predictions tensor
            targets: Target values tensor
            intermediates: Dictionary with decoder intermediate outputs
            
        Returns:
            Combined loss tensor
        """
        pred_loss = self.prediction_loss(predictions, targets)
        
        if intermediates:
            total_loss = self.prediction_weight * pred_loss
            
            # Add reconstruction loss if decoder norms and input are available
            # Hooks capture short_dec_norm and long_dec_norm separately, concatenate here
            if "short_dec_norm" in intermediates and "long_dec_norm" in intermediates and "input" in intermediates:
                # Get decoder outputs (already normalized and in correct shape [batch, time, features])
                short_dec = intermediates["short_dec_norm"]  # [batch, 20, num_features] for CAELSTM
                long_dec = intermediates["long_dec_norm"]  # [batch, 11, num_features] for CAELSTM
                
                # Concatenate in correct order: short (first 20) then long (last 11)
                decoder_output = torch.cat((short_dec, long_dec), dim=1)  # [batch, 31, num_features]
                
                recon_loss = self.reconstruction_loss(
                    decoder_output,
                    intermediates["input"]
                )
                total_loss += self.reconstruction_weight * recon_loss
            
            return total_loss
        
        return self.prediction_weight * pred_loss


class CompositeLoss(BaseLoss):
    """
    Generic composite loss that combines multiple loss terms.
    
    Useful for combining prediction loss with various regularization terms
    from intermediate layers.
    """
    
    def __init__(
        self,
        primary_loss: nn.Module,
        auxiliary_losses: Optional[Dict[str, tuple]] = None,
        intermediate_layers: Optional[List[str]] = None
    ):
        """
        Initialize CompositeLoss.
        
        Args:
            primary_loss: Primary loss function (e.g., nn.MSELoss())
            auxiliary_losses: Dict mapping layer names to (loss_fn, weight) tuples
                             e.g., {"encoder": (nn.MSELoss(), 0.1)}
            intermediate_layers: List of layer names to hook
        """
        super().__init__()
        self.primary_loss = primary_loss
        self.auxiliary_losses = auxiliary_losses or {}
        
        if intermediate_layers is None:
            intermediate_layers = list(self.auxiliary_losses.keys())
        self._intermediate_layers = intermediate_layers
    
    def requires_intermediates(self) -> bool:
        """CompositeLoss requires intermediates if auxiliary losses are specified."""
        return len(self.auxiliary_losses) > 0
    
    def get_hook_targets(self) -> List[str]:
        """Return list of layers to hook."""
        return self._intermediate_layers
    
    def __call__(self, predictions, targets, intermediates: Optional[Dict[str, Any]] = None):
        """
        Compute composite loss.
        
        Args:
            predictions: Model predictions tensor
            targets: Target values tensor
            intermediates: Dictionary of intermediate layer outputs
        
        Returns:
            Combined loss tensor
        """
        # Primary loss
        total_loss = self.primary_loss(predictions, targets)
        
        # Add auxiliary losses from intermediates
        if intermediates:
            for layer_name, (loss_fn, weight) in self.auxiliary_losses.items():
                if layer_name in intermediates:
                    # For now, use targets as reference (can be customized)
                    aux_loss = loss_fn(intermediates[layer_name], targets)
                    total_loss += weight * aux_loss
        
        return total_loss
