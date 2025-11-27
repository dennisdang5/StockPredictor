# Portfolio Architecture Data Shape Analysis

This document traces the data shapes through each layer of the `PortfolioArchitecture` model.

## Assumptions
- **Batch size**: `B` (varies per forward pass)
- **Sequence length**: `seq_len` (default: 31)
- **Input features**: `features` (default: 3 without NLP, 13 with NLP)
- **Base model output dim**: `base_output_dim` (typically 1, inferred from base model)
- **Stock embedding dim**: `embedding_dim` (default: 32, from PortfolioConfig)
- **MLP hidden dims**: `mlp_hidden_dims` (default: [64], from PortfolioConfig)
- **Number of stocks**: `num_stocks` (from PortfolioConfig.stocks)

---

## Forward Pass Shape Trace

### 1. Input to `forward()` method
**Location**: `forward(self, x, params=None)` - Line 162

**Input `x`**:
- Shape: `(B, seq_len, features)`
- Example: `(32, 31, 3)` for batch of 32 samples, 31 timesteps, 3 features
- Type: `torch.Tensor`

**Input `params`**:
- Must contain `"stock_indices"` key
- `stock_indices`: Shape `(B,)` or `(B, 1)` - integer indices mapping each sample to a stock
- Example: `[0, 0, 1, 1, 2, 0, ...]` for batch of 32 samples across 3 stocks

---

### 2. Stock Indices Processing
**Location**: Lines 165-168

**After processing**:
- `stock_indices`: Shape `(B,)` - flattened to 1D tensor
- Example: `torch.tensor([0, 0, 1, 1, 2, 0, ...], dtype=torch.long)`

---

### 3. Output Tensor Initialization
**Location**: Line 170

**`outputs`**:
- Shape: `(B, 1)`
- Initialized to zeros
- Example: `torch.zeros(32, 1)`
- This will be filled in per-stock during the loop

---

### 4. Per-Stock Processing Loop
**Location**: Lines 172-179

For each unique stock ID in the batch:

#### 4a. Mask Creation
**Location**: Line 173

**`mask`**:
- Shape: `(B,)` - boolean tensor
- Example: `[True, True, False, False, False, True, ...]` for stock_id=0
- Used to select samples belonging to current stock

#### 4b. Batch Input Selection
**Location**: Line 176

**`batch_inputs`**:
- Shape: `(batch_size_for_stock, seq_len, features)`
- Example: If 10 samples belong to stock 0: `(10, 31, 3)`
- This is a subset of the original `x` tensor

---

### 5. Backbone Forward Pass
**Location**: `_forward_backbone()` - Line 177

**Input to backbone**:
- Shape: `(batch_size_for_stock, seq_len, features)`
- Example: `(10, 31, 3)`

**Backbone model** (e.g., LSTM):
- Takes: `(batch_size_for_stock, seq_len, features)`
- Processes through LSTM layers, normalization, dropout
- Returns: `(batch_size_for_stock, base_output_dim)`
- Example: `(10, 1)` if base_output_dim=1

**`base_out`** (returned from `_forward_backbone`):
- Shape: `(batch_size_for_stock, base_output_dim)`
- Example: `(10, 1)`
- This is the output from the base model (LSTM, TimesNet, etc.)

---

### 6. Head Input Composition
**Location**: `_compose_head_input()` - Line 178

#### 6a. Flatten Backbone Output
**Location**: Line 149

**`head_in`** (after flatten):
- Shape: `(batch_size_for_stock, base_output_dim)`
- Example: `(10, 1)`
- Note: `.view(batch_size_for_stock, -1)` flattens any extra dimensions

#### 6b. Stock Embedding (if enabled)
**Location**: Lines 152-160

**If `use_stock_embeddings=True`**:

**`stock_ids`**:
- Shape: `(batch_size_for_stock,)`
- Example: `torch.tensor([0, 0, 0, ...], dtype=torch.long)` - all same stock ID
- Example values: `(10,)` filled with stock_id=0

**`emb`** (from embedding layer):
- Shape: `(batch_size_for_stock, embedding_dim)`
- Example: `(10, 32)` if embedding_dim=32

**Concatenated `head_in`**:
- Shape: `(batch_size_for_stock, base_output_dim + embedding_dim)`
- Example: `(10, 1 + 32) = (10, 33)`

**If `use_stock_embeddings=False`**:
- `head_in` remains: `(batch_size_for_stock, base_output_dim)`
- Example: `(10, 1)`

---

### 7. Portfolio MLP Head
**Location**: `portfolio_head()` - Line 179

**Input to MLP**:
- Shape: `(batch_size_for_stock, head_input_dim)`
- Where `head_input_dim = base_output_dim + embedding_dim` (if embeddings) or `base_output_dim` (if no embeddings)
- Example with embeddings: `(10, 33)`
- Example without embeddings: `(10, 1)`

**MLP Architecture** (from `_build_mlp()` - Lines 126-137):
- Layer 1: `Linear(head_input_dim, mlp_hidden_dims[0])`
  - Input: `(batch_size_for_stock, head_input_dim)`
  - Output: `(batch_size_for_stock, mlp_hidden_dims[0])`
  - Example: `(10, 64)` if mlp_hidden_dims=[64]
- Activation: ReLU (or specified activation)
- Dropout (if enabled): Same shape maintained
- Layer 2 (if mlp_hidden_dims has 2+ elements): `Linear(mlp_hidden_dims[0], mlp_hidden_dims[1])`
  - Input: `(batch_size_for_stock, mlp_hidden_dims[0])`
  - Output: `(batch_size_for_stock, mlp_hidden_dims[1])`
- ... (continues for each hidden dim)
- Final Layer: `Linear(prev_dim, 1)`
  - Input: `(batch_size_for_stock, last_hidden_dim)`
  - Output: `(batch_size_for_stock, 1)`

**MLP Output**:
- Shape: `(batch_size_for_stock, 1)`
- Example: `(10, 1)`

---

### 8. Output Assignment
**Location**: Line 179

**`outputs[mask]`**:
- Assigns MLP output back to the full batch tensor
- Shape: `(batch_size_for_stock, 1)` → assigned to `outputs[mask]`
- The `outputs` tensor is updated in-place for each stock

---

### 9. Final Return
**Location**: Line 181

**`outputs`** (final):
- Shape: `(B, 1)`
- Example: `(32, 1)`
- Contains predictions for all samples in the batch, organized by stock

---

## Summary Table

| Layer/Step | Input Shape | Output Shape | Notes |
|------------|-------------|--------------|-------|
| **Input `x`** | `(B, seq_len, features)` | - | Raw time series data |
| **Stock indices** | `(B,)` or `(B, 1)` | `(B,)` | Flattened to 1D |
| **Outputs init** | - | `(B, 1)` | Zeros tensor |
| **Batch inputs** (per stock) | `(B, seq_len, features)` | `(batch_size_for_stock, seq_len, features)` | Masked subset |
| **Backbone output** | `(batch_size_for_stock, seq_len, features)` | `(batch_size_for_stock, base_output_dim)` | Base model (LSTM/etc) |
| **Flattened head input** | `(batch_size_for_stock, base_output_dim)` | `(batch_size_for_stock, base_output_dim)` | After view() |
| **Stock embeddings** (if enabled) | `(batch_size_for_stock,)` | `(batch_size_for_stock, embedding_dim)` | Embedding lookup |
| **Concatenated head input** | - | `(batch_size_for_stock, base_output_dim + embedding_dim)` | If embeddings enabled |
| **MLP Layer 1** | `(batch_size_for_stock, head_input_dim)` | `(batch_size_for_stock, mlp_hidden_dims[0])` | Linear + activation |
| **MLP Layer N** | `(batch_size_for_stock, mlp_hidden_dims[N-1])` | `(batch_size_for_stock, mlp_hidden_dims[N])` | Intermediate layers |
| **MLP Final Layer** | `(batch_size_for_stock, last_hidden_dim)` | `(batch_size_for_stock, 1)` | Final prediction |
| **Final output** | - | `(B, 1)` | Aggregated predictions |

---

## Example Walkthrough

**Configuration**:
- Batch size: `B = 32`
- Sequence length: `seq_len = 31`
- Features: `features = 3`
- Base output dim: `base_output_dim = 1`
- Embedding dim: `embedding_dim = 32`
- MLP hidden dims: `[64, 32]`
- Stocks: `['AAPL', 'MSFT', 'GOOGL']` (3 stocks)
- Stock indices: `[0, 0, 1, 1, 2, 0, ...]` (32 samples)

**Shape progression for stock 0 (10 samples)**:

1. `x`: `(32, 31, 3)` → `batch_inputs`: `(10, 31, 3)`
2. Backbone (LSTM): `(10, 31, 3)` → `(10, 1)`
3. Flatten: `(10, 1)` → `(10, 1)` (no change)
4. Embeddings: `(10,)` → `(10, 32)`
5. Concatenate: `(10, 1)` + `(10, 32)` → `(10, 33)`
6. MLP Layer 1: `(10, 33)` → `(10, 64)`
7. MLP Layer 2: `(10, 64)` → `(10, 32)`
8. MLP Final: `(10, 32)` → `(10, 1)`
9. Assign to `outputs[mask]`: `(10, 1)` → fills positions in `(32, 1)`

**Final output**: `(32, 1)` - one prediction per sample in batch

---

## Notes

1. **Variable batch sizes**: Each stock may have a different number of samples in a batch, so `batch_size_for_stock` varies per iteration.

2. **Base model output**: The `_infer_output_dim()` method uses a dummy input to determine the base model's output dimension. This should match `model_config.output_dim`.

3. **Embedding concatenation**: Stock embeddings are concatenated along the feature dimension (dim=-1), increasing the head input dimension.

4. **MLP flexibility**: The MLP can have any number of hidden layers specified in `mlp_hidden_dims`. Each layer reduces/increases dimensionality as specified.

5. **Strategy independence**: The shape flow is the same for both "independent" and "shared" strategies - only the backbone model instance(s) differ, not the data flow.

