from ..base import BaseModel
from ..configs import AutoEncoderConfig
import torch
import torch.nn as nn

class AutoEncoder(BaseModel):
    """
    AutoEncoder for feature extraction and dimensionality reduction.
    
    Architecture: Encodes input to 2x dimensions, then decodes back to original shape.
    Used for feature extraction and denoising.
    
    Args:
        model_config: Configuration object containing model parameters.
                     Must have the following attributes (all optional with defaults):
    
    Expected model_config attributes:
        - input_shape (tuple): Shape of input data (timesteps, features).
                              Format: (sequence_length, num_features)
                              Used to calculate total input dimensions (timesteps * features).
                              Default: (31, 3)
    """
    def __init__(self, model_config):
        super(AutoEncoder,self).__init__(model_config)
        if model_config is None:
            raise ValueError("model_config is required for AutoEncoder")
        
        # Type checking: ensure model_config is AutoEncoderConfig or compatible
        if not isinstance(model_config, AutoEncoderConfig):
            # Allow SimpleNamespace for backward compatibility but warn
            raise TypeError(
                f"AutoEncoder requires AutoEncoderConfig instance, "
                f"got {type(model_config).__name__}. "
                f"Use AutoEncoderConfig(input_shape=...) to create the config."
            )
        
        self.input_shape = model_config.to_dict().get('input_shape', (31, 3))
        def _dof(x):
            output = 1
            for val in x:
                output *= val
            return output

        self.dof = _dof(self.input_shape)
        #self.embedding_dim = model_config.to_dict().get('embedding_dim', 2 * self.input_shape[1])
        self.embedding_dim = 2 * self.dof
        
        # Input normalization
        self.input_norm = nn.LayerNorm(self.dof)
        self.encoder=nn.Sequential(
            # naive concatenation
            nn.Linear(self.dof, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),  # Normalization after encoder linear
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(self.embedding_dim, self.dof),
            nn.LayerNorm(self.dof),  # Normalization after decoder linear
            nn.ReLU()
        )

        ###### other parameters ####
        # nn.MSELoss()
        # optimizer = torch.optim.Adam()

    def forward(self, x, params=None, return_encoded=False):
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_features) or (seq_len, num_features)
            params: Optional parameters (unused)
            return_encoded: If True, return encoder output (compressed representation) instead of decoder output.
                          Default: False (returns decoder output for reconstruction)
        
        Returns:
            If return_encoded=False: Decoder output (batch, seq_len, num_features) - same shape as input
            If return_encoded=True: Encoder output reshaped to (batch, seq_len, 2*num_features)
        
        Note:
            Decoder is always executed (even when return_encoded=True) to enable reconstruction loss
            for multi-task learning (training autoencoder and LSTM simultaneously).
            DDP compatibility is handled via _set_static_graph() in the trainer.
        """
        # assume x in shape (batch, 31, 3) or (31, 3)
        if x.dim() == 2:
            # Single sample, add batch dimension
            x = x.unsqueeze(0)
        original_shape = x.shape
        x = torch.flatten(x, start_dim=1)  # Flatten spatial dimensions, keep batch
        # Normalize input
        x = self.input_norm(x)
        encoded = self.encoder(x)  # Shape: (batch, 2*dof) where dof = seq_len * num_features
        
        # Always decode (even if returning encoded) so hooks can capture decoder output for reconstruction loss
        # This enables multi-task learning: training autoencoder (reconstruction loss) and LSTM (prediction loss) simultaneously
        decoded = self.decoder(encoded)
        
        if return_encoded:
            # Return encoder output reshaped to (batch, seq_len, 2*num_features)
            # This is the compressed representation that should go to LSTM
            # Note: Decoder was still executed above so hooks can capture it for reconstruction loss
            encoded_reshaped = torch.unflatten(encoded, dim=1, sizes=(self.input_shape[0], 2*self.input_shape[1]))
            # Remove batch dimension if it was added
            if original_shape[0] == 1 and len(original_shape) == 2:
                encoded_reshaped = encoded_reshaped.squeeze(0)
            return encoded_reshaped
        
        # Return decoder output (for reconstruction loss or when not using return_encoded)
        # Unflatten back to original spatial shape
        x = torch.unflatten(decoded, dim=1, sizes=(self.input_shape[0], self.input_shape[1]))
        # Remove batch dimension if it was added
        if original_shape[0] == 1 and len(original_shape) == 2:
            x = x.squeeze(0)
        return x

    @classmethod
    def from_config(cls, model_config):
        """Factory hook so the registry can instantiate the model."""
        return cls(model_config)
