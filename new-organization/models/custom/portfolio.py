import torch
import torch.nn as nn
from ..base import BaseModel
from ..configs import PortfolioConfig, BaseModelConfig
from ..registry import ModelRegistry


class PortfolioArchitecture(BaseModel):
    """
    Composite portfolio model that stitches per-stock backbones with a shared MLP head.

    Modes:
        - independent: one backbone instance per stock (fully separate weights).
        - shared: a single backbone shared by all stocks, conditioned via embeddings.
    """

    def __init__(self, model_config: PortfolioConfig):
        super().__init__(model_config)

        if not model_config.stocks:
            raise ValueError("PortfolioConfig.stocks must contain at least one ticker")

        self.stocks = list(model_config.stocks)
        self.num_stocks = len(self.stocks)
        self.base_model_type = model_config.base_model_type
        self.strategy = model_config.strategy.lower()
        if self.strategy not in {"independent", "shared"}:
            raise ValueError(f"Unsupported strategy '{self.strategy}'. Use 'independent' or 'shared'.")

        base_config = model_config.base_model_config
        if base_config is None or not isinstance(base_config, BaseModelConfig):
            raise ValueError("PortfolioConfig.base_model_config must be a BaseModelConfig instance.")

        # Keep a prototype config we can clone for each backbone.
        self.prototype_base_config = model_config.base_model_config
        self._sync_base_config_shapes()

        # Build backbones and infer their output dimensionality.
        self.base_output_dim = self._build_backbone_modules()

        self.use_stock_embeddings = (
            bool(model_config.use_stock_embeddings) and model_config.embedding_dim > 0
        )
        if self.use_stock_embeddings:
            self.stock_embeddings = nn.Embedding(self.num_stocks, model_config.embedding_dim)
        else:
            self.stock_embeddings = None

        head_input_dim = self.base_output_dim
        if self.use_stock_embeddings:
            head_input_dim += model_config.embedding_dim

        # Determine output dimension: 3 for LSTM-based classification models, 1 otherwise
        self.portfolio_output_dim = self._determine_portfolio_output_dim()

        self.portfolio_head = self._build_mlp(
            input_dim=head_input_dim,
            hidden_dims=model_config.mlp_hidden_dims,
            activation=model_config.activation,
            dropout=model_config.dropout,
            output_dim=self.portfolio_output_dim,
        )

        if model_config.freeze_base_models:
            for param in self._backbone_parameters():
                param.requires_grad = False

    def _sync_base_config_shapes(self):
        """
        Ensure the nested base config inherits input_shape/seq_len/lookback from the portfolio config.
        """
        proto = self.prototype_base_config
        for attr in ("input_shape", "seq_len", "lookback"):
            if hasattr(self.model_config, attr) and getattr(self.model_config, attr) is not None:
                setattr(proto, attr, getattr(self.model_config, attr))
                if hasattr(proto, "parameters"):
                    proto.parameters[attr] = getattr(self.model_config, attr)

    def _instantiate_base_model(self):
        import copy
        config_clone = copy.deepcopy(self.prototype_base_config)
        return ModelRegistry.create(self.base_model_type, config_clone)

    def _build_backbone_modules(self):
        """
        Instantiate backbones per the requested strategy and return their output dim.
        """
        first_model = self._instantiate_base_model()
        output_dim = self._infer_output_dim(first_model)

        if self.strategy == "independent":
            self.stock_models = nn.ModuleDict()
            self.stock_models[self.stocks[0]] = first_model
            for ticker in self.stocks[1:]:
                self.stock_models[ticker] = self._instantiate_base_model()
        elif self.strategy == "shared":
            # single model for all stocks
            self.shared_model = first_model
        ##### TODO: Implement industry strategy #####
        elif self.strategy == "industry":
            self.industry_models = nn.ModuleDict()
            self.industry_models[self.industries[0]] = first_model
            for industry in self.industries[1:]:
                self.industry_models[industry] = self._instantiate_base_model()
        else:
            raise ValueError(f"Unsupported strategy '{self.strategy}'. Use 'independent' or 'shared'.")

        return output_dim

    def _infer_output_dim(self, model: BaseModel) -> int:
        # TabPFN models must be fit before inference, but we know they output 1 dim for binary classification
        from ..external.TabPFN import TabPFNAdapter
        if isinstance(model, TabPFNAdapter):
            return 2
        
        # For other models, infer by doing a forward pass
        input_shape = getattr(self.model_config, 'input_shape', None)
        if not input_shape:
            input_shape = getattr(self.prototype_base_config, 'input_shape', (31, 3))
        seq_len, features = input_shape
        dummy = torch.zeros(1, seq_len, features, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        
        # Handle different output shapes: [B] or [B, D]
        if out.dim() == 1:
            inferred_dim = out.shape[0]
        elif out.dim() == 2:
            inferred_dim = out.shape[1]  # [B, D] -> D
        else:
            # Flatten and take last dimension
            inferred_dim = out.shape[-1]
        
        # Only validate if output_dim is specified in config (optional check)
        if hasattr(self.model_config, 'output_dim') and self.model_config.output_dim is not None:
            if inferred_dim != self.model_config.output_dim:
                raise ValueError(
                    f"Output dimension mismatch. Expected {self.model_config.output_dim}, got {inferred_dim}."
                )
        
        return inferred_dim

    def _determine_portfolio_output_dim(self) -> int:
        """
        Determine the output dimension for the portfolio head.
        
        For LSTM-based models (LSTM, AELSTM, CAELSTM), output 3 logits for 3-class classification.
        For other models (TabPFN, regression models), output 1.
        """
        lstm_models = {"LSTM", "AELSTM", "CAELSTM"}
        if self.base_model_type.upper() in lstm_models:
            return 3  # 3-class classification
        return 1  # Regression or binary classification

    def _backbone_parameters(self):
        if self.strategy == "independent":
            for module in self.stock_models.values():
                yield from module.parameters()
        else:
            yield from self.shared_model.parameters()

    def _build_mlp(self, input_dim, hidden_dims, activation: str, dropout: float, output_dim: int = 1):
        layers = []
        prev_dim = input_dim
        act_cls = getattr(nn, activation.capitalize(), nn.ReLU)
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(act_cls())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _forward_backbone(self, stock_idx: int, inputs):
        if self.strategy == "independent":
            ticker = self.stocks[stock_idx]
            try:
                backbone = self.stock_models[ticker]
            except KeyError:
                raise KeyError(f"No backbone found for stock '{ticker}' (idx={stock_idx})")
            return backbone(inputs)
        return self.shared_model(inputs)

    def _compose_head_input(self, backbone_out, stock_idx: int):
        head_in = backbone_out.view(backbone_out.size(0), -1)
        if self.stock_embeddings is None:
            return head_in
        device = backbone_out.device
        stock_ids = torch.full(
            (backbone_out.size(0),),
            stock_idx,
            dtype=torch.long,
            device=device
        )
        emb = self.stock_embeddings(stock_ids)
        return torch.cat([head_in, emb], dim=-1)

    def forward(self, x, params=None):
        if params is None or "stock_indices" not in params:
            raise ValueError("PortfolioArchitecture forward requires 'stock_indices' in params.")
        stock_indices = params["stock_indices"]
        if not isinstance(stock_indices, torch.Tensor):
            stock_indices = torch.tensor(stock_indices, device=x.device, dtype=torch.long)
        stock_indices = stock_indices.view(-1).to(dtype=torch.long, device=x.device)

        # Initialize outputs with correct shape: [B, output_dim]
        outputs = torch.zeros(x.size(0), self.portfolio_output_dim, device=x.device, dtype=x.dtype)
        unique_stocks = torch.unique(stock_indices)
        for stock_id in unique_stocks.tolist():
            mask = stock_indices == stock_id
            if not mask.any():
                continue
            batch_inputs = x[mask]
            base_out = self._forward_backbone(stock_id, batch_inputs)
            head_in = self._compose_head_input(base_out, stock_id)
            outputs[mask] = self.portfolio_head(head_in).to(dtype=outputs.dtype)

        return outputs

    @classmethod
    def from_config(cls, model_config):
        return cls(model_config)


# Register model with the global registry
from ..registry import ModelRegistry
ModelRegistry.register("Portfolio", lambda config: PortfolioArchitecture(config), PortfolioConfig)
