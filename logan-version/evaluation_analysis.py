#!/usr/bin/env python3
"""
Evaluation analysis script.

This script operationalises the high-level checklist in
`evaluation_analysis_pseudocode.txt`. It reads an evaluation results JSON
produced by `test_suite/evaluate_model.py`, computes a battery of diagnostic
statistics, and writes a consolidated JSON report.

Many of the requested diagnostics need richer inputs (e.g. full cross-sectional
scores, market regime data). Where those inputs are not available in the
evaluation results file, the script records a structured "not_available"
placeholder so the consumer can decide how to extend the pipeline.

Usage:
    python evaluation_analysis.py \
        --input /path/to/evaluation_results_savedmodel_classification.json \
        --output /path/to/evaluation_analysis_report.json

Optional arguments let you plug in additional datasets (cross-sectional scores,
market regime data) to unlock more diagnostics. See `parse_args` for details.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


JsonDict = Dict[str, Any]


@dataclass
class AnalysisConfig:
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 42
    bootstrap_alpha: float = 0.05
    rolling_window: int = 60
    min_rolling_observations: int = 30
    change_point_z_threshold: float = 2.0
    change_point_min_gap: int = 20


class EvaluationAnalysis:
    """
    Turn evaluation JSON artefacts into higher-level diagnostics.
    """

    def __init__(
        self,
        evaluation_path: Path,
        config: AnalysisConfig,
        cross_sectional_path: Optional[Path] = None,
        market_data_path: Optional[Path] = None,
    ) -> None:
        self.evaluation_path = evaluation_path
        self.cross_sectional_path = cross_sectional_path
        self.market_data_path = market_data_path
        self.config = config

        self.evaluation_payload = self._load_json(evaluation_path)
        self.paper_metrics = self.evaluation_payload["metrics"]["paper_aligned"]
        self.additional_metrics = self.evaluation_payload["metrics"].get(
            "additional_diagnostics", {}
        )

        self.daily_metrics = self._build_daily_metrics_frame()
        self.error_counts = self._build_error_counts_frame()
        self.calendar_heatmap = self.paper_metrics["cross_sectional_diagnostics"].get(
            "calendar_heatmap", []
        )
        self.portfolio_metrics = self.paper_metrics.get("portfolio_performance", {})
        self.benchmark_metrics = self.paper_metrics.get("benchmarks", {})

        # Optional datasets (None unless supplied via CLI)
        self.cross_sectional_df = self._load_cross_sectional_frame()
        self.market_data = self._load_market_data_frame()

    @staticmethod
    def _coerce_float(value: Optional[Any]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (float, int, np.floating, np.integer)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _leg_after_cost(leg_return: Optional[Any], mean_daily_cost: Optional[Any]) -> Optional[float]:
        leg = EvaluationAnalysis._coerce_float(leg_return)
        if leg is None:
            return None
        cost = EvaluationAnalysis._coerce_float(mean_daily_cost)
        if cost is None:
            return leg
        return leg - (cost / 2.0)

    def _model_label(self) -> str:
        model_name = self.evaluation_payload.get("model_name")
        if model_name:
            return str(model_name)
        model_path = self.evaluation_payload.get("model_path")
        if model_path:
            try:
                return Path(model_path).stem
            except (TypeError, ValueError):
                return str(model_path)
        return "model"

    # --------------------------------------------------------------------- #
    # Loading helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _load_json(path: Path) -> JsonDict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_daily_metrics_frame(self) -> pd.DataFrame:
        daily = self.paper_metrics["cross_sectional_diagnostics"]["daily_metrics"]
        df = pd.DataFrame(daily)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        return df.replace({None: np.nan})

    def _build_error_counts_frame(self) -> pd.DataFrame:
        errors = self.paper_metrics["cross_sectional_diagnostics"]["error_counts"]
        df = pd.DataFrame(errors)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        return df.replace({None: np.nan})

    def _load_cross_sectional_frame(self) -> Optional[pd.DataFrame]:
        if self.cross_sectional_path is None:
            return None

        if not self.cross_sectional_path.exists():
            raise FileNotFoundError(
                f"Cross-sectional file not found: {self.cross_sectional_path}"
            )

        suffix = self.cross_sectional_path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(self.cross_sectional_path)
        elif suffix in {".csv", ".txt"}:
            frame = pd.read_csv(self.cross_sectional_path)
        else:
            raise ValueError(
                "Unsupported cross-sectional file format. "
                "Provide CSV or Parquet."
            )

        if "date" not in frame.columns:
            raise ValueError(
                "Cross-sectional dataset must include a 'date' column."
            )

        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def _load_market_data_frame(self) -> Optional[pd.DataFrame]:
        if self.market_data_path is None:
            return None

        if not self.market_data_path.exists():
            raise FileNotFoundError(
                f"Market data file not found: {self.market_data_path}"
            )

        suffix = self.market_data_path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(self.market_data_path)
        elif suffix in {".csv", ".txt"}:
            df = pd.read_csv(self.market_data_path)
        else:
            raise ValueError(
                "Unsupported market data format. Provide CSV or Parquet."
            )

        if "date" not in df.columns:
            raise ValueError("Market data must include a 'date' column.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        return df

    # --------------------------------------------------------------------- #
    # Core computations
    # --------------------------------------------------------------------- #

    @staticmethod
    def _dropna_series(series: pd.Series) -> np.ndarray:
        return series.replace({np.nan: None}).dropna().to_numpy(dtype=float)

    @staticmethod
    def _safe_std(array: np.ndarray) -> float:
        if array.size == 0:
            return float("nan")
        return float(np.std(array, ddof=1)) if array.size > 1 else 0.0

    @staticmethod
    def _longest_streak(bools: Iterable[bool]) -> int:
        max_streak = 0
        current = 0
        for flag in bools:
            if flag:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @staticmethod
    def _newey_west_variance(series: np.ndarray, max_lags: Optional[int] = None) -> float:
        """
        Compute Newey-West HAC variance estimate for a series of observations.
        """
        series = np.asarray(series, dtype=float)
        series = series[~np.isnan(series)]
        n = series.size
        if n == 0:
            return float("nan")
        if n == 1:
            return 0.0

        if max_lags is None:
            max_lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        max_lags = min(max_lags, n - 1)

        series_mean = series.mean()
        centred = series - series_mean
        var = np.mean(centred ** 2)

        for lag in range(1, max_lags + 1):
            weight = 1.0 - lag / (max_lags + 1.0)
            cov = np.mean(centred[:-lag] * centred[lag:])
            var += 2.0 * weight * cov

        return max(var, 0.0)

    def _newey_west_test(self, series: np.ndarray) -> JsonDict:
        """
        One-sided test of mean(series) > 0 using Newey-West variance.
        """
        if series.size == 0:
            return {
                "t_stat": None,
                "p_value_one_sided": None,
                "mean": None,
                "std": None,
                "n": 0,
            }

        mean_val = float(series.mean())
        variance = self._newey_west_variance(series)
        n = series.size

        if not math.isfinite(variance) or variance <= 0.0 or n == 0:
            return {
                "t_stat": None,
                "p_value_one_sided": None,
                "mean": mean_val,
                "std": float(np.std(series, ddof=1)) if n > 1 else 0.0,
                "n": n,
            }

        se = math.sqrt(variance / n)
        if se == 0:
            t_stat = math.inf if mean_val > 0 else -math.inf
        else:
            t_stat = mean_val / se

        norm = NormalDist()
        p_val = 1.0 - norm.cdf(t_stat)
        return {
            "t_stat": float(t_stat),
            "p_value_one_sided": float(p_val),
            "mean": mean_val,
            "std": float(np.std(series, ddof=1)) if n > 1 else 0.0,
            "n": n,
        }

    def _fisher_z_average(self, ic_series: pd.Series, n_series: pd.Series) -> Optional[float]:
        ic = self._dropna_series(ic_series)
        if ic.size == 0:
            return None

        n_values = n_series.reindex(ic_series.index).to_numpy(dtype=float)
        n_values = np.nan_to_num(n_values, nan=0.0)

        clipped = np.clip(ic, -0.999999, 0.999999)
        z_scores = np.arctanh(clipped)

        weights = np.maximum(n_values - 3.0, 0.0)
        valid = weights > 0
        if not np.any(valid):
            return float(np.tanh(np.mean(z_scores)))

        weighted_avg = np.average(z_scores[valid], weights=weights[valid])
        return float(np.tanh(weighted_avg))

    # ------------------------------------------------------------------ #
    # Section computations
    # ------------------------------------------------------------------ #

    def compute_cross_sectional_rank_skill(self) -> JsonDict:
        df = self.daily_metrics
        ic = self._dropna_series(df["ic"])
        ic_series = df["ic"].dropna()
        stats: JsonDict = {}

        if ic.size == 0:
            return {"status": "not_available", "reason": "No IC data present."}

        stats["median_ic"] = float(np.median(ic))
        stats["iqr"] = {
            "q25": float(np.percentile(ic, 25)),
            "q75": float(np.percentile(ic, 75)),
        }
        stats["quantiles_5_95"] = {
            "q5": float(np.percentile(ic, 5)),
            "q95": float(np.percentile(ic, 95)),
        }
        stats["positive_share"] = float((ic > 0).mean())
        stats["longest_positive_streak"] = int(
            self._longest_streak(ic_series > 0)
        )
        stats["longest_negative_streak"] = int(
            self._longest_streak(ic_series < 0)
        )
        stats["newey_west_test"] = self._newey_west_test(ic)

        std_ic = self._safe_std(ic)
        if std_ic and not math.isnan(std_ic):
            stats["information_ratio"] = float(
                ic.mean() / std_ic * math.sqrt(252.0)
            )
        else:
            stats["information_ratio"] = None

        stats["fisher_z_average"] = self._fisher_z_average(
            ic_series, df["n"]
        )

        return stats

    def _bootstrap_bca_ci(
        self,
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
    ) -> Optional[Tuple[float, float]]:
        cfg = self.config
        rng = np.random.default_rng(cfg.bootstrap_seed)
        clean = data[~np.isnan(data)]
        n = clean.size
        if n == 0:
            return None

        theta_hat = statistic_fn(clean)

        boot = np.empty(cfg.bootstrap_samples, dtype=float)
        for i in range(cfg.bootstrap_samples):
            sample = rng.choice(clean, size=n, replace=True)
            boot[i] = statistic_fn(sample)
        boot.sort()

        jackknife = np.empty(n, dtype=float)
        for i in range(n):
            jackknife[i] = statistic_fn(np.delete(clean, i))
        jack_mean = jackknife.mean()
        numerator = np.sum((jack_mean - jackknife) ** 3)
        denominator = np.sum((jack_mean - jackknife) ** 2) ** 1.5
        acc = 0.0 if denominator == 0 else numerator / (6.0 * denominator)

        norm = NormalDist()
        eps = 1e-12
        prop = np.clip((boot < theta_hat).mean(), eps, 1 - eps)
        z0 = norm.inv_cdf(prop)

        def _percentile(alpha: float) -> float:
            alpha = np.clip(alpha, eps, 1 - eps)
            z = norm.inv_cdf(alpha)
            denom = 1.0 - acc * (z0 + z)
            if denom == 0:
                adj = z0 + z
            else:
                adj = z0 + (z0 + z) / denom
            prob = np.clip(norm.cdf(adj), eps, 1 - eps)
            index = prob * (cfg.bootstrap_samples - 1)
            lower_idx = int(np.floor(index))
            upper_idx = min(lower_idx + 1, cfg.bootstrap_samples - 1)
            frac = index - lower_idx
            return (1 - frac) * boot[lower_idx] + frac * boot[upper_idx]

        alpha_low = cfg.bootstrap_alpha / 2.0
        alpha_high = 1.0 - cfg.bootstrap_alpha / 2.0
        return (_percentile(alpha_low), _percentile(alpha_high))

    def compute_classification_quality(self) -> JsonDict:
        df = self.daily_metrics
        if df.empty:
            return {"status": "not_available", "reason": "No daily metrics found."}

        auc = self._dropna_series(df["auc"])
        logloss = self._dropna_series(df["logloss"])
        brier = self._dropna_series(df["brier"])

        metrics: JsonDict = {
            "median_auc": float(np.median(auc)) if auc.size else None,
            "median_logloss": float(np.median(logloss)) if logloss.size else None,
            "median_brier": float(np.median(brier)) if brier.size else None,
        }

        if auc.size:
            ci = self._bootstrap_bca_ci(auc, np.median)
            metrics["median_auc_bca_ci"] = (
                {"lower": float(ci[0]), "upper": float(ci[1])}
                if ci is not None
                else None
            )
        else:
            metrics["median_auc_bca_ci"] = None

        # Baseline-adjusted hit-rate (top-k hit rate minus daily base probability)
        if "topk_hit_rate" in df.columns and "n" in df.columns:
            k = self.portfolio_metrics.get("k")
            if k is None:
                metrics["baseline_adjusted_hit_rate"] = {
                    "status": "not_available",
                    "reason": "Portfolio configuration is missing top-k size.",
                }
            else:
                base_rate = np.clip(k / df["n"].astype(float), 0.0, 1.0)
                spread = df["topk_hit_rate"] - base_rate
                spread_series = {
                    idx.isoformat(): float(val)
                    for idx, val in spread.dropna().items()
                }
                metrics["baseline_adjusted_hit_rate"] = {
                    "mean": float(spread.mean()),
                    "median": float(spread.median()),
                    "quantiles": {
                        "q5": float(spread.quantile(0.05)),
                        "q95": float(spread.quantile(0.95)),
                    },
                    "series": spread_series,
                }
        else:
            metrics["baseline_adjusted_hit_rate"] = {
                "status": "not_available",
                "reason": "Daily metrics missing topk_hit_rate or n.",
            }

        metrics["brier_decomposition"] = {
            "status": "not_available",
            "reason": (
                "Decomposition requires forecast probability bins. "
                "Provide cross-sectional predictions via --cross-sectional to enable."
            ),
        }

        metrics["expected_calibration_error"] = {
            "status": "not_available",
            "reason": (
                "ECE requires probability bins or raw predictions; "
                "supply --cross-sectional to compute calibration drift."
            ),
        }

        return metrics

    def compute_portfolio_edge(self) -> JsonDict:
        metrics = dict(self.portfolio_metrics)
        if not metrics:
            return {"status": "not_available", "reason": "Portfolio metrics missing."}

        sharpe = metrics.get("sharpe_ratio")
        skew = metrics.get("skewness")
        kurt = metrics.get("kurtosis")
        n_days = metrics.get("n_trading_days")
        metrics["deflated_sharpe_ratio"] = self._deflated_sharpe_ratio(
            sharpe=sharpe,
            skew=skew,
            kurtosis=kurt,
            n_obs=n_days,
        )

        random_percentile = self.benchmark_metrics.get(
            "percentile_rank_in_random_distribution"
        )
        metrics["random_benchmark_context"] = {
            "percentile_rank": random_percentile,
            "mean_vs_random": self.benchmark_metrics.get("outperformance_vs_random"),
            "random_percentiles": {
                "p5": self.benchmark_metrics.get("random_percentile_5_annualized"),
                "p50": self.benchmark_metrics.get("random_percentile_50_annualized"),
                "p95": self.benchmark_metrics.get("random_percentile_95_annualized"),
            },
        }
        return metrics

    def build_portfolio_table(self) -> pd.DataFrame:
        columns = ["Group", "Metric", "Model", "Before costs", "After costs"]
        if not self.portfolio_metrics:
            return pd.DataFrame(columns=columns)

        metrics = self.portfolio_metrics
        gross_metrics = self.paper_metrics.get("before_cost_risk_metrics", {})
        portfolio_distribution = self.paper_metrics.get("portfolio_distribution", {})
        costs_payload = portfolio_distribution.get("costs", {}) if portfolio_distribution else {}
        mean_daily_cost = metrics.get("mean_daily_transaction_cost")
        if mean_daily_cost is None:
            mean_daily_cost = costs_payload.get("mean_daily_cost")

        model_label = self._model_label()
        rows: List[Dict[str, Any]] = []

        def add_row(group: str, metric_name: str, before_val: Optional[Any], after_val: Optional[Any], model: Optional[str] = None) -> None:
            rows.append(
                {
                    "Group": group,
                    "Metric": metric_name,
                    "Model": model or model_label,
                    "Before costs": self._coerce_float(before_val),
                    "After costs": self._coerce_float(after_val),
                }
            )

        # Panel A: Return distribution (before vs after costs)
        add_row(
            "Panel A",
            "Mean return (long)",
            metrics.get("mean_long_leg_return"),
            self._leg_after_cost(metrics.get("mean_long_leg_return"), mean_daily_cost),
        )
        add_row(
            "Panel A",
            "Mean return (short)",
            metrics.get("mean_short_leg_return"),
            self._leg_after_cost(metrics.get("mean_short_leg_return"), mean_daily_cost),
        )

        mean_return_before = metrics.get("mean_daily_return_before_cost") or gross_metrics.get("mean_return")
        mean_return_after = metrics.get("mean_daily_return") or metrics.get("mean_return")
        add_row("Panel A", "Mean return (portfolio)", mean_return_before, mean_return_after)

        std_before = metrics.get("std_return_before_cost") or gross_metrics.get("standard_deviation")
        std_after = metrics.get("standard_deviation")
        add_row("Panel A", "Std. dev.", std_before, std_after)

        share_before = metrics.get("share_positive_before_cost") or gross_metrics.get("share_positive")
        share_after = metrics.get("share_positive")
        add_row("Panel A", "Share > 0", share_before, share_after)

        add_row(
            "Panel A",
            "Minimum return",
            metrics.get("min_return_before_cost") or gross_metrics.get("min_return"),
            metrics.get("min_return"),
        )
        add_row(
            "Panel A",
            "First quartile",
            metrics.get("quantile_25_before_cost") or gross_metrics.get("quantile_25"),
            metrics.get("quantile_25"),
        )
        add_row(
            "Panel A",
            "Median return",
            metrics.get("median_return_before_cost") or gross_metrics.get("median"),
            metrics.get("median_return") or metrics.get("median"),
        )
        add_row(
            "Panel A",
            "Third quartile",
            metrics.get("quantile_75_before_cost") or gross_metrics.get("quantile_75"),
            metrics.get("quantile_75"),
        )
        add_row(
            "Panel A",
            "Maximum return",
            metrics.get("max_return_before_cost") or gross_metrics.get("max_return"),
            metrics.get("max_return"),
        )
        add_row(
            "Panel A",
            "Kurtosis",
            (gross_metrics or {}).get("kurtosis"),
            metrics.get("kurtosis"),
        )
        add_row(
            "Panel A",
            "Newey-West SE",
            metrics.get("newey_west_std_error_before_cost") or gross_metrics.get("newey_west_std_error"),
            metrics.get("newey_west_std_error"),
        )
        add_row(
            "Panel A",
            "Newey-West t-stat",
            metrics.get("newey_west_t_stat_before_cost") or gross_metrics.get("newey_west_t_stat"),
            metrics.get("newey_west_t_stat"),
        )
        add_row(
            "Panel A",
            "Standard error",
            metrics.get("standard_error_before_cost") or gross_metrics.get("standard_error"),
            metrics.get("standard_error"),
        )
        add_row(
            "Panel A",
            "t-stat (iid)",
            metrics.get("t_stat_before_cost") or gross_metrics.get("t_stat"),
            metrics.get("t_stat"),
        )

        # Panel B: Tail risk and drawdowns
        add_row(
            "Panel B",
            "VaR (5%)",
            (gross_metrics or {}).get("var_5pct"),
            metrics.get("var_5pct"),
        )
        add_row(
            "Panel B",
            "CVaR (5%)",
            (gross_metrics or {}).get("cvar_5pct"),
            metrics.get("cvar_5pct"),
        )
        add_row(
            "Panel B",
            "VaR (1%)",
            (gross_metrics or {}).get("var_1pct"),
            metrics.get("var_1pct"),
        )
        add_row(
            "Panel B",
            "CVaR (1%)",
            (gross_metrics or {}).get("cvar_1pct"),
            metrics.get("cvar_1pct"),
        )
        add_row(
            "Panel B",
            "Max drawdown (%)",
            (gross_metrics or {}).get("max_drawdown"),
            metrics.get("max_drawdown_pct"),
        )

        # Panel C: Annualised performance
        add_row(
            "Panel C",
            "Return p.a.",
            (gross_metrics or {}).get("annualized_return") or metrics.get("annualized_return_before_cost"),
            metrics.get("annualized_return"),
        )

        sp500_ann = self.benchmark_metrics.get("sp500_annualized_return")
        excess_before = None
        if sp500_ann is not None:
            before_val = (gross_metrics or {}).get("annualized_return") or metrics.get("annualized_return_before_cost")
            if before_val is not None:
                excess_before = float(before_val) - float(sp500_ann)
        excess_after = self.benchmark_metrics.get("excess_return_vs_sp500")
        add_row("Panel C", "Excess return p.a.", excess_before, excess_after)

        add_row(
            "Panel C",
            "Std. dev. p.a.",
            (gross_metrics or {}).get("volatility_annualized"),
            metrics.get("volatility_annualized"),
        )
        add_row(
            "Panel C",
            "Downside dev. p.a.",
            (gross_metrics or {}).get("downside_deviation_annualized"),
            metrics.get("downside_deviation_annualized"),
        )
        add_row(
            "Panel C",
            "Sharpe ratio",
            (gross_metrics or {}).get("sharpe_ratio"),
            metrics.get("sharpe_ratio"),
        )
        add_row(
            "Panel C",
            "Sortino ratio",
            (gross_metrics or {}).get("sortino_ratio"),
            metrics.get("sortino_ratio"),
        )

        # Benchmarks
        if self.benchmark_metrics:
            add_row(
                "Benchmarks",
                "Return p.a. (S&P 500)",
                self.benchmark_metrics.get("sp500_annualized_return"),
                self.benchmark_metrics.get("sp500_annualized_return"),
                model="S&P 500",
            )
            add_row(
                "Benchmarks",
                "Volatility p.a. (S&P 500)",
                self.benchmark_metrics.get("sp500_volatility"),
                self.benchmark_metrics.get("sp500_volatility"),
                model="S&P 500",
            )
            add_row(
                "Benchmarks",
                "Sharpe (S&P 500)",
                self.benchmark_metrics.get("sp500_sharpe"),
                self.benchmark_metrics.get("sp500_sharpe"),
                model="S&P 500",
            )

        df = pd.DataFrame(rows, columns=columns)
        return df

    def export_portfolio_table(self, csv_path: Path) -> None:
        table = self.build_portfolio_table()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)

    @staticmethod
    def _deflated_sharpe_ratio(
        sharpe: Optional[float],
        skew: Optional[float],
        kurtosis: Optional[float],
        n_obs: Optional[int],
        num_trials: int = 1,
    ) -> Optional[float]:
        if (
            sharpe is None
            or n_obs is None
            or n_obs <= 1
            or not math.isfinite(sharpe)
        ):
            return None

        skew = skew if skew is not None else 0.0
        kurtosis = kurtosis if kurtosis is not None else 3.0
        norm = NormalDist()

        # Expected maximum Sharpe ratio under Gaussian assumption
        if num_trials <= 1:
            sr_star = 0.0
        else:
            sr_star = norm.inv_cdf(1.0 - 1.0 / num_trials) / math.sqrt(n_obs - 1)

        numerator = (sharpe - sr_star) * math.sqrt(n_obs - 1)
        denominator = math.sqrt(
            1.0
            - skew * sharpe
            + ((kurtosis - 1.0) / 4.0) * (sharpe ** 2)
        )
        if denominator == 0:
            return None
        return float(numerator / denominator)

    def compute_temporal_analysis(self) -> JsonDict:
        df = self.daily_metrics
        if df.empty:
            return {"status": "not_available", "reason": "No daily metrics to analyse."}

        window = self.config.rolling_window
        min_periods = self.config.min_rolling_observations

        rolling = df[["ic", "auc", "topk_hit_rate"]].rolling(
            window=window, min_periods=min_periods
        )
        rolling_means = rolling.mean().dropna()
        rolling_stds = rolling.std().dropna()

        def _series_to_dict(series: pd.Series) -> Dict[str, float]:
            return {ts.isoformat(): float(val) for ts, val in series.items()}

        change_points = {
            metric: self._detect_change_points(rolling_means[metric])
            for metric in rolling_means.columns
        }

        calendar = {
            "iso_week_heatmap": self.calendar_heatmap,
        }

        return {
            "rolling_window": window,
            "rolling_mean": {
                metric: _series_to_dict(rolling_means[metric])
                for metric in rolling_means.columns
            },
            "rolling_std": {
                metric: _series_to_dict(rolling_stds[metric])
                for metric in rolling_stds.columns
            },
            "change_points": change_points,
            "calendar_heatmap": calendar,
        }

    def _detect_change_points(self, series: pd.Series) -> List[JsonDict]:
        series = series.dropna()
        if series.empty:
            return []

        threshold = self.config.change_point_z_threshold
        min_gap = self.config.change_point_min_gap
        diffs = series.diff()
        std = diffs.std(ddof=1)
        if not math.isfinite(std) or std == 0:
            return []

        events: List[JsonDict] = []
        last_index = -math.inf
        for idx, (timestamp, value) in enumerate(diffs.items()):
            if idx <= 0:
                continue
            z = value / std
            if abs(z) >= threshold and (idx - last_index) >= min_gap:
                events.append(
                    {
                        "date": timestamp.isoformat(),
                        "z_score": float(z),
                        "rolling_mean": float(series.iloc[idx]),
                    }
                )
                last_index = idx
        return events

    def compute_conditional_performance(self) -> JsonDict:
        df = self.daily_metrics
        results: JsonDict = {}

        if df.empty:
            return {"status": "not_available", "reason": "No daily metrics located."}

        # Coverage buckets as proxy for cross-sectional dispersion (if actual dispersion unavailable)
        coverage = df["n"]
        if coverage.isna().all():
            coverage_summary = {
                "status": "not_available",
                "reason": "Daily coverage counts missing.",
            }
        else:
            buckets = pd.qcut(coverage, q=3, labels=["low", "mid", "high"])
            grouped = df.groupby(buckets, observed=True)
            coverage_summary = {
                bucket: {
                    "mean_ic": float(group["ic"].mean()),
                    "mean_auc": float(group["auc"].mean()),
                    "mean_spread": float(group["long_short_spread"].mean()),
                }
                for bucket, group in grouped
                if not group.empty
            }

        results["coverage_buckets"] = coverage_summary

        if self.market_data is not None:
            merged = df.join(self.market_data, how="left")
            market_summary: JsonDict = {}

            if "spx_return" in merged.columns:
                market_summary["spx_up_vs_down"] = self._bucket_summary(
                    merged, "spx_return", bins=[-np.inf, 0, np.inf], labels=["down", "up"]
                )
            if "spx_abs_return" in merged.columns:
                terciles = np.nanpercentile(
                    merged["spx_abs_return"], [33, 66]
                )
                bins = [-np.inf, terciles[0], terciles[1], np.inf]
                market_summary["abs_spx_terciles"] = self._bucket_summary(
                    merged,
                    "spx_abs_return",
                    bins=bins,
                    labels=["low", "mid", "high"],
                )
            if "dispersion" in merged.columns:
                terciles = np.nanpercentile(merged["dispersion"], [33, 66])
                bins = [-np.inf, terciles[0], terciles[1], np.inf]
                market_summary["dispersion_buckets"] = self._bucket_summary(
                    merged,
                    "dispersion",
                    bins=bins,
                    labels=["low", "mid", "high"],
                )
            results["market_regime"] = market_summary
        else:
            results["market_regime"] = {
                "status": "not_available",
                "reason": (
                    "Supply --market-data with SPX/VIX/dispersion columns to enable "
                    "regime-conditioned metrics."
                ),
            }

        if self.cross_sectional_df is not None:
            cs = self.cross_sectional_df
            if {"ticker", "sector"}.issubset(cs.columns):
                joined = (
                    cs.merge(
                        df[["ic"]].reset_index(),
                        left_on="date",
                        right_on="date",
                        how="left",
                    )
                )
                sector_summary = (
                    joined.groupby("sector")["ret"]
                    .mean()
                    .sort_values(ascending=False)
                    .to_dict()
                )
                results["sector_buckets"] = sector_summary
            else:
                results["sector_buckets"] = {
                    "status": "not_available",
                    "reason": (
                        "Cross-sectional data missing 'ticker' or 'sector' columns."
                    ),
                }
        else:
            results["sector_buckets"] = {
                "status": "not_available",
                "reason": (
                    "Provide --cross-sectional dataset with sector attribution "
                    "to compute sector buckets."
                ),
            }

        results["k_sensitivity"] = {
            "status": "not_available",
            "reason": (
                "Need predictions across multiple k values; rerun evaluator with "
                "different k or supply custom results."
            ),
        }
        return results

    def _bucket_summary(
        self,
        df: pd.DataFrame,
        column: str,
        bins: Sequence[float],
        labels: Sequence[str],
    ) -> JsonDict:
        cats = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
        grouped = df.groupby(cats, observed=True)
        summary: JsonDict = {}
        for label, group in grouped:
            if group.empty:
                continue
            summary[str(label)] = {
                "mean_ic": float(group["ic"].mean()),
                "mean_auc": float(group["auc"].mean()),
                "mean_topk_hit": float(group["topk_hit_rate"].mean()),
                "mean_spread": float(group["long_short_spread"].mean()),
                "sample_size": int(group.shape[0]),
            }
        return summary

    def compute_stability_and_turnover(self) -> JsonDict:
        if self.cross_sectional_df is None:
            return {
                "status": "not_available",
                "reason": (
                    "Cross-sectional predictions needed for rank stability "
                    "and turnover statistics. Provide --cross-sectional."
                ),
            }

        df = self.cross_sectional_df.copy()
        required_cols = {"date", "ticker", "score"}
        if not required_cols.issubset(df.columns):
            return {
                "status": "not_available",
                "reason": (
                    f"Cross-sectional data must include {sorted(required_cols)} "
                    "columns to compute stability diagnostics."
                ),
            }

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "score"], ascending=[True, False])

        k = self.portfolio_metrics.get("k", 10)
        topk = (
            df.groupby("date")
            .head(k)
            .groupby("date")["ticker"]
            .apply(list)
            .sort_index()
        )

        overlaps = []
        for prev, curr in zip(topk.values[:-1], topk.values[1:]):
            prev_set = set(prev)
            curr_set = set(curr)
            overlaps.append(len(prev_set & curr_set) / max(len(prev_set), 1))

        overlap_series = pd.Series(
            overlaps,
            index=topk.index[1:],
            name="topk_overlap",
        )

        # Kendall tau between consecutive rankings
        tau_values = []
        dates = sorted(df["date"].unique())
        for prev_date, curr_date in zip(dates[:-1], dates[1:]):
            prev_scores = (
                df[df["date"] == prev_date]
                .set_index("ticker")["score"]
                .rank(ascending=False, method="average")
            )
            curr_scores = (
                df[df["date"] == curr_date]
                .set_index("ticker")["score"]
                .rank(ascending=False, method="average")
            )
            joined = prev_scores.to_frame("prev").join(
                curr_scores.to_frame("curr"), how="inner"
            )
            if joined.empty:
                continue
            tau = joined["prev"].corr(joined["curr"], method="kendall")
            tau_values.append(tau)

        prediction_drift = df.groupby("date")["score"].mean().pct_change()

        return {
            "topk_overlap_summary": {
                "mean": float(overlap_series.mean()),
                "median": float(overlap_series.median()),
                "quantiles": {
                    "q5": float(overlap_series.quantile(0.05)),
                    "q95": float(overlap_series.quantile(0.95)),
                },
            },
            "topk_overlap_series": {
                ts.isoformat(): float(val)
                for ts, val in overlap_series.items()
            },
            "kendall_tau_mean": float(np.nanmean(tau_values)) if tau_values else None,
            "prediction_drift_corr": float(
                df.groupby("date")["score"].mean().autocorr(lag=1)
            ),
            "prediction_drift_series": {
                ts.isoformat(): float(val)
                for ts, val in prediction_drift.dropna().items()
            },
        }

    def compute_error_anatomy(self) -> JsonDict:
        if self.error_counts.empty:
            return {
                "status": "not_available",
                "reason": "Error counts missing from evaluation payload.",
            }

        df = self.daily_metrics.join(self.error_counts, how="left")
        df["fp_rate"] = df["fp"] / df["n"]
        df["fn_rate"] = df["fn"] / df["n"]

        quantile = 0.1
        cutoff_low = df["ic"].quantile(quantile)
        cutoff_high = df["ic"].quantile(1 - quantile)

        worst = df[df["ic"] <= cutoff_low]
        best = df[df["ic"] >= cutoff_high]

        anatomy = {
            "overall": {
                "fp_total": int(df["fp"].sum()),
                "fn_total": int(df["fn"].sum()),
                "fp_rate_mean": float(df["fp_rate"].mean()),
                "fn_rate_mean": float(df["fn_rate"].mean()),
            },
            "worst_10pct_ic": {
                "fp_rate_mean": float(worst["fp_rate"].mean()),
                "fn_rate_mean": float(worst["fn_rate"].mean()),
                "top_dates": [
                    {
                        "date": idx.isoformat(),
                        "ic": float(row["ic"]),
                        "fp": int(row["fp"]),
                        "fn": int(row["fn"]),
                    }
                    for idx, row in worst.nsmallest(10, "ic").iterrows()
                ],
            },
            "best_10pct_ic": {
                "fp_rate_mean": float(best["fp_rate"].mean()),
                "fn_rate_mean": float(best["fn_rate"].mean()),
                "top_dates": [
                    {
                        "date": idx.isoformat(),
                        "ic": float(row["ic"]),
                        "fp": int(row["fp"]),
                        "fn": int(row["fn"]),
                    }
                    for idx, row in best.nlargest(10, "ic").iterrows()
                ],
            },
        }

        anatomy["threshold_scan"] = {
            "status": "not_available",
            "reason": (
                "Requires raw prediction scores to compute PR curves. "
                "Supply --cross-sectional to enable threshold sweep."
            ),
        }
        return anatomy

    def compute_regression_analysis(self) -> JsonDict:
        if self.market_data is None:
            return {
                "status": "not_available",
                "reason": (
                    "Provide --market-data with SPX/VIX/dispersion columns to "
                    "run regression diagnostics."
                ),
            }

        df = self.daily_metrics.join(self.market_data, how="left")
        required = [
            "spx_return",
            "spx_abs_return",
            "vix_change",
            "dispersion",
            "coverage",
        ]
        df["coverage"] = df["n"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            return {
                "status": "not_available",
                "reason": f"Market data missing columns: {missing}",
            }

        df = df.dropna(subset=["ic"])
        design_cols = [
            "spx_return",
            "spx_abs_return",
            "vix_change",
            "dispersion",
            "coverage",
        ]
        df = df.dropna(subset=design_cols)
        if df.empty:
            return {
                "status": "not_available",
                "reason": "Insufficient overlapping observations for regression.",
            }

        X = df[design_cols].assign(intercept=1.0)
        y = df["ic"]
        X_mat = X.to_numpy(dtype=float)
        y_vec = y.to_numpy(dtype=float)

        beta, residuals, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        fitted = X_mat @ beta
        resid = y_vec - fitted

        nw_var = self._newey_west_variance(resid)
        se = math.sqrt(nw_var / len(resid)) if len(resid) > 0 else float("nan")

        return {
            "coefficients": {
                col: float(val) for col, val in zip(design_cols + ["intercept"], beta)
            },
            "residual_newey_west_var": float(nw_var),
            "residual_se_mean": float(se),
            "r_squared": float(self._safe_r_squared(y_vec, fitted)),
            "n_obs": int(len(df)),
        }

    @staticmethod
    def _safe_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0:
            return float("nan")
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot == 0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    def prepare_visualisation_payloads(self) -> JsonDict:
        df = self.daily_metrics
        if df.empty:
            return {
                "status": "not_available",
                "reason": "No daily metrics present.",
            }

        payload = {
            "ic_distribution": df["ic"].dropna().tolist(),
            "auc_distribution": df["auc"].dropna().tolist(),
            "spread_distribution": df["long_short_spread"].dropna().tolist(),
            "rolling_metrics": self.compute_temporal_analysis(),
            "baseline_adjusted_hit_rate": self.compute_classification_quality().get(
                "baseline_adjusted_hit_rate"
            ),
        }

        if self.cross_sectional_df is not None:
            # Worst 10 days by IC with optional sector breakdown if available
            worst_dates = (
                df.nsmallest(10, "ic")
                .reset_index()[["date", "ic"]]
            )
            if {"ticker", "sector", "ret"}.issubset(self.cross_sectional_df.columns):
                cs = self.cross_sectional_df.copy()
                cs["date"] = pd.to_datetime(cs["date"])
                merged = cs.merge(worst_dates, on="date", how="inner")
                sector_heatmap = (
                    merged.groupby(["date", "sector"])["ret"]
                    .mean()
                    .unstack("sector")
                    .fillna(0.0)
                )
                payload["worst_day_sector_heatmap"] = {
                    date.isoformat(): sector_heatmap.loc[date].to_dict()
                    for date in sector_heatmap.index
                }

        return payload

    def compile_report_card(self) -> JsonDict:
        ic_stats = self.compute_cross_sectional_rank_skill()
        classification_stats = self.compute_classification_quality()
        baseline_hit = classification_stats.get("baseline_adjusted_hit_rate")
        portfolio_stats = self.compute_portfolio_edge()

        report = {
            "ic": {
                "median": ic_stats.get("median_ic"),
                "quantiles_5_95": ic_stats.get("quantiles_5_95"),
                "positive_share": ic_stats.get("positive_share"),
                "newey_west_p_value": ic_stats.get("newey_west_test", {}).get(
                    "p_value_one_sided"
                ),
            },
            "auc": {
                "median": classification_stats.get("median_auc"),
                "median_bca_ci": classification_stats.get("median_auc_bca_ci"),
            },
            "top_k": baseline_hit if isinstance(baseline_hit, dict) else baseline_hit,
            "portfolio": {
                "mean_daily_return": portfolio_stats.get("mean_daily_return")
                if isinstance(portfolio_stats, dict)
                else None,
                "sharpe_ratio": portfolio_stats.get("sharpe_ratio")
                if isinstance(portfolio_stats, dict)
                else None,
                "max_drawdown_pct": portfolio_stats.get("max_drawdown_pct")
                if isinstance(portfolio_stats, dict)
                else None,
                "percentile_vs_random": portfolio_stats.get("random_benchmark_context", {}).get(
                    "percentile_rank"
                )
                if isinstance(portfolio_stats, dict)
                else None,
            },
            "stability": None,
            "regimes": None,
        }

        stability = self.compute_stability_and_turnover()
        if isinstance(stability, dict) and "status" not in stability:
            report["stability"] = stability.get("topk_overlap_summary")

        conditional = self.compute_conditional_performance()
        if isinstance(conditional, dict) and "status" not in conditional:
            report["regimes"] = conditional.get("market_regime")

        return report

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def run(self) -> JsonDict:
        portfolio_table_records: List[JsonDict] = []
        portfolio_table_df = self.build_portfolio_table()
        if not portfolio_table_df.empty:
            portfolio_table_records = (
                portfolio_table_df.replace({np.nan: None}).to_dict(orient="records")
            )

        return {
            "metadata": {
                "evaluation_path": str(self.evaluation_path),
                "model_path": self.evaluation_payload.get("model_path"),
                "analysis_config": vars(self.config),
                "stocks_covered": len(self.evaluation_payload.get("stocks", [])),
                "analysis_notes": (
                    "Diagnostics relying on additional datasets are flagged as "
                    "'not_available'. Supply optional data sources to unlock them."
                ),
            },
            "cross_sectional_rank_skill": self.compute_cross_sectional_rank_skill(),
            "classification_quality": self.compute_classification_quality(),
            "portfolio_edge": self.compute_portfolio_edge(),
            "temporal_analysis": self.compute_temporal_analysis(),
            "conditional_performance": self.compute_conditional_performance(),
            "stability_and_turnover": self.compute_stability_and_turnover(),
            "error_anatomy": self.compute_error_anatomy(),
            "regression_analysis": self.compute_regression_analysis(),
            "visualisation_payloads": self.prepare_visualisation_payloads(),
            "report_card": self.compile_report_card(),
            "portfolio_table": portfolio_table_records,
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse evaluation outputs using the diagnostics outlined in "
            "evaluation_analysis_pseudocode.txt."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to evaluation_results_*.json file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the consolidated analysis JSON.",
    )
    parser.add_argument(
        "--cross-sectional",
        type=Path,
        default=None,
        help=(
            "Optional CSV/Parquet with per-ticker predictions. Expected columns: "
            "date, ticker, score, ret, sector (if sector diagnostics desired)."
        ),
    )
    parser.add_argument(
        "--market-data",
        type=Path,
        default=None,
        help=(
            "Optional CSV/Parquet with columns such as date, spx_return, "
            "spx_abs_return, vix_change, dispersion."
        ),
    )
    parser.add_argument(
        "--portfolio-table-csv",
        type=Path,
        default=None,
        help="Optional path to write a tidy CSV of before/after-cost portfolio metrics.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for BCa confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Significance level for BCa confidence intervals.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=60,
        help="Window size (in trading days) for rolling diagnostics.",
    )
    parser.add_argument(
        "--min-rolling-observations",
        type=int,
        default=30,
        help="Minimum observations required before emitting rolling statistics.",
    )
    parser.add_argument(
        "--change-point-z-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for marking change-points in rolling averages.",
    )
    parser.add_argument(
        "--change-point-min-gap",
        type=int,
        default=20,
        help="Minimum index gap between successive change-point flags.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    config = AnalysisConfig(
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_alpha=args.bootstrap_alpha,
        rolling_window=args.rolling_window,
        min_rolling_observations=args.min_rolling_observations,
        change_point_z_threshold=args.change_point_z_threshold,
        change_point_min_gap=args.change_point_min_gap,
    )

    analysis = EvaluationAnalysis(
        evaluation_path=args.input,
        config=config,
        cross_sectional_path=args.cross_sectional,
        market_data_path=args.market_data,
    )
    report = analysis.run()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[evaluation-analysis] Report written to {args.output}")

    if args.portfolio_table_csv is not None:
        analysis.export_portfolio_table(args.portfolio_table_csv)
        print(f"[evaluation-analysis] Portfolio table CSV written to {args.portfolio_table_csv}")


if __name__ == "__main__":
    main()

