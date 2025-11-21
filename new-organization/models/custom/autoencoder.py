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
        # Input normalization
        self.input_norm = nn.LayerNorm(self.dof)
        self.encoder=nn.Sequential(
            # naive concatenation
            nn.Linear(self.dof, 2*self.dof),
            nn.LayerNorm(2*self.dof),  # Normalization after encoder linear
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(2*self.dof, self.dof),
            nn.LayerNorm(self.dof),  # Normalization after decoder linear
            nn.ReLU()
        )

        ###### other parameters ####
        # nn.MSELoss()
        # optimizer = torch.optim.Adam()

    def forward(self, x):
        # assume x in shape (batch, 31, 3) or (31, 3)
        if x.dim() == 2:
            # Single sample, add batch dimension
            x = x.unsqueeze(0)
        original_shape = x.shape
        x = torch.flatten(x, start_dim=1)  # Flatten spatial dimensions, keep batch
        # Normalize input
        x = self.input_norm(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Unflatten back to original spatial shape
        x = torch.unflatten(decoded, dim=1, sizes=(self.input_shape[0], self.input_shape[1]))
        # Remove batch dimension if it was added
        if original_shape[0] == 1 and len(original_shape) == 2:
            x = x.squeeze(0)
        return x