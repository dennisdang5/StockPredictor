## Portfolio Architecture Overview

The `PortfolioArchitecture` model composes multiple single-stock backbones (TabPFN, LSTM, etc.) and a portfolio-level MLP head.

- **Independent mode**: Instantiates one backbone per stock ticker. Each backbone learns idiosyncratic behaviour before feeding a shared MLP head that produces trade scores.
- **Shared mode (shared weights + stock embeddings)**: Builds a single backbone that all stocks share. A learnable embedding per stock is concatenated to the backbone output, allowing the shared weights to adapt to stock-specific nuances without duplicating parameters. This is what “shared weights with stock embeddings” refers to.

Both modes emit per-stock predictions that the existing Trainer already turns into long/short allocations. Switch between them via `PortfolioConfig.strategy`.

- **TabPFN portfolio mode**: When `model_type="TabPFN"`, the trainer now flattens each stock’s long/short window into a single tabular row, trains an independent TabPFN estimator per stock, and zero-fills any missing samples before fitting. Predictions are later recombined to drive the usual portfolio metrics.

