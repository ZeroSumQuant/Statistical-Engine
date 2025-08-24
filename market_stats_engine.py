#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_stats_engine.py v3.0 (Quant-Delight Edition)

A single-file, production-ready market statistical analysis engine.

This engine discovers statistically significant support/resistance zones in
financial market data, evaluates their performance using a robust state machine,
and provides deep quantitative analysis on the results. It is designed to be
instrument-agnostic, configurable, and extensible, all while maintaining the
constraint of being a single Python file.

Key Features:
- Instrument-agnostic via YAML configuration (Futures, FX, Crypto).
- Pluggable providers for zone discovery (pivots, external levels).
- Walk-forward validation for robust out-of-sample testing.
- Advanced statistical analysis: cohort analysis, survival curves, tail risk (CVaR).
- Optional SQLite persistence for run lineage and results tracking.
- Self-contained HTML reports with embedded plots for easy sharing.
- Performance-aware with optional multiprocessing and progress bars.

Example Usage:
---------------

# 1. Basic in-sample analysis on prepared NQ data
python market_stats_engine.py analyze --data cache/nq_prepared.parquet --out results/nq_run_1

# 2. Instrument Swap: Run on Crude Oil (CL) using a custom config
# --- cl_config.yml ---
# instrument:
#   symbol: "CL"
#   tick_size: 0.01
#   point_value: 1000.0
#   session_tz: "America/New_York"
#   rth_start: "09:00"
#   rth_end: "14:30"
# ---------------------
python market_stats_engine.py analyze --data cache/cl_prepared.parquet --config cl_config.yml --out results/cl_run_1

# 3. Walk-forward validation (90-day training window, 30-day testing step, optional 30-day test size)
python market_stats_engine.py analyze --data cache/es_prepared.parquet --wf "window=90d,step=30d,test=30d" --out results/es_wf

# 4. Generate a self-contained HTML report
python market_stats_engine.py analyze --data cache/nq_prepared.parquet --report --out results/nq_report

# 5. Save results to a database for lineage tracking
python market_stats_engine.py analyze --data cache/nq_prepared.parquet --db my_results.db

# 6. Parallel sweep using 8 cores
python market_stats_engine.py sweep --data cache/nq_prepared.parquet --grid sweep_grid.yml --jobs 8 --out sweep_results/

# 7. Run the built-in self-test to verify core functionality
python market_stats_engine.py self-test

# 8. Advanced: Run analysis only for specific market regimes
# --- my_regimes.yml ---
# enabled: true
# regimes:
#   - name: "Uptrend"
#     condition: "close > ema_50 & ema_50 > ema_200"
#   - name: "HighVol"
#     condition: "atr > atr.rolling(20).mean() * 1.5"
# ----------------------
python market_stats_engine.py analyze --data cache/nq_prepared.parquet --regimes my_regimes.yml
"""

# ————————————————————————————————————————————————————————————————————————————
# TABLE OF CONTENTS
# ————————————————————————————————————————————————————————————————————————————
#
# 1. CONFIG & CONSTANTS
#    - Global constants for data quality and statistical thresholds.
#
# 2. DATAMODELS
#    - InstrumentConfig, IndicatorConfig, ZoneConfig, EpisodeConfig, etc.
#    - Enums: EpisodeOutcome, ZoneType.
#    - Zone: Dataclass for a detected support/resistance zone.
#
# 3. UTILS
#    - Statistical helpers: test_zone_significance, block_bootstrap_ci.
#    - General helpers: get_file_sha256, update_dataclass_from_dict.
#
# 4. DATA PREP
#    - DataValidator, prepare_data, annotate_sessions, add_indicators.
#
# 5. ZONE PROVIDERS
#    - Registry for pluggable zone detection logic (pivots, external levels).
#
# 6. EPISODES
#    - State machine for detecting episodes (SearchingForTouch, TrackingOutcome).
#
# 7. STATISTICS
#    - Core analysis loop, computation of cohort stats, survival, CVaR, etc.
#
# 8. PERSISTENCE
#    - SQLite writer/reader for run lineage.
#
# 9. PLOTTING
#    - Generation of PNG plots (distributions, daily maps, heatmaps).
#
# 10. REPORT
#    - Single-file HTML report generation with embedded PNGs.
#
# 11. CLI & MAIN
#    - Argparse setup and main execution logic.
#
# 12. SELF-TEST
#    - Tiny synthetic data run to verify core functionality.
#

##############################################################################
# IMPORTS
##############################################################################

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable
import datetime as dt
import json
import math
import os
import sys
import logging
import hashlib
from enum import Enum
from abc import ABC, abstractmethod

# Core dependencies
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("numpy is required") from e

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required") from e

# Optional dependencies
try:
    import polars as pl
    HAVE_POLARS = True
except Exception:
    HAVE_POLARS = False
    pl = None

try:
    import pyarrow
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    scipy_stats = None

try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

try:
    from statsmodels.stats.multitest import multipletests
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False


pytz = None
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
try:
    import pytz
except Exception:
    pytz = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

##############################################################################
# 1. CONFIG & CONSTANTS
##############################################################################

# ————————— Logging Setup —————————
LOG = logging.getLogger("market_stats_engine")
handler = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d] %(message)s"
)
handler.setFormatter(formatter)
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)

# ————————— Constants —————————
# Data quality thresholds
MAX_PRICE_CHANGE_PERCENT = 20.0  # Max % change between bars
MIN_VOLUME = 1  # Minimum valid volume
MAX_GAP_MINUTES = 120  # Max gap before considering session break

# Statistical thresholds
MIN_EPISODES_FOR_STATS = 30  # Minimum episodes for reliable statistics
ZONE_SIGNIFICANCE_ALPHA = 0.05  # Significance level for zone detection
BOOTSTRAP_BLOCK_SIZE = 10  # Block size for correlated bootstrap

##############################################################################
# 2. DATAMODELS (dataclasses & enums)
##############################################################################


# ————————— Enums —————————
class EpisodeOutcome(Enum):
    RESPECT = "RESPECT"
    PIERCE_AND_REVERT = "PIERCE_AND_REVERT"
    BREAK = "BREAK"
    TIMEOUT = "TIMEOUT"
    INVALID = "INVALID"  # For data quality issues


class ZoneType(Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


# ————————— Config Models with Validation —————————


@dataclass
class InstrumentConfig:
    """Instrument-specific parameters"""

    symbol: str = "NQ"
    tick_size: float = 0.25
    point_value: float = 20.0  # currency per point
    exchange: str = "CME"
    session_tz: str = "America/New_York"
    rth_start: str = "09:30"
    rth_end: str = "16:00"
    currency: str = "USD"
    data_tz: Optional[str] = None  # e.g., "UTC" or "America/New_York"

    def __post_init__(self):
        # Validate time format
        try:
            pd.to_datetime(self.rth_start, format="%H:%M")
            pd.to_datetime(self.rth_end, format="%H:%M")
        except:
            raise ValueError("Invalid time format for RTH window. Use HH:MM")


@dataclass
class IndicatorConfig:
    atr_n: int = 14
    rsi_n: int = 14
    stoch_n: int = 14
    stoch_d: int = 3
    rvol_lookback_sessions: int = 30
    vwap_bands_window: int = 60

    def __post_init__(self):
        if self.atr_n < 2:
            raise ValueError("ATR period must be >= 2")
        if self.stoch_d > self.stoch_n:
            raise ValueError("Stochastic D period cannot exceed K period")


@dataclass
class ZoneConfig:
    providers: List[str] = field(default_factory=lambda: ["pivots"])
    extern_levels_path: Optional[str] = None  # Path to CSV for 'extern_levels' provider
    pivot_k: int = 5
    zone_width_points: Optional[float] = 15.0
    zone_width_alpha_atr: Optional[float] = None
    cluster_width_points: Optional[float] = None
    tick_size: Optional[float] = None  # Override instrument tick_size if set
    null_width_multiplier: float = 1.5
    expire_days: int = 30
    merge_tolerance_points: float = 2.0
    max_cluster_span_days: Optional[int] = None
    min_touches_for_significance: int = 2
    significance_test: bool = True

    def __post_init__(self):
        if self.pivot_k < 1:
            raise ValueError("Pivot k must be >= 1")
        if self.zone_width_points is not None and self.zone_width_points <= 0:
            raise ValueError("Zone width must be positive")
        if self.zone_width_alpha_atr is not None and self.zone_width_alpha_atr <= 0:
            raise ValueError("Zone width ATR multiplier must be positive")
        if self.zone_width_points is None and self.zone_width_alpha_atr is None:
            self.zone_width_points = 15.0  # Default

        if self.cluster_width_points is None:
            self.cluster_width_points = self.zone_width_points
        elif self.cluster_width_points <= 0:
            raise ValueError("Cluster width must be positive")


@dataclass
class EpisodeConfig:
    R: float = 12.0  # reversal distance
    O: float = 6.0  # overshoot tolerance
    T: int = 20  # timeout bars
    treat_timeout_as_break: bool = False
    max_gap_bars: int = 5
    first_touch_only: bool = False
    min_bars_between_touches: int = 0
    max_episodes_per_zone: Optional[int] = None
    outliers_policy: str = "invalidate"  # "invalidate" | "skip"
    censor_on_session_change: bool = True

    def __post_init__(self):
        if self.R <= 0:
            raise ValueError("Reversal distance R must be positive")
        if self.O <= 0:
            raise ValueError("Overshoot tolerance O must be positive")
        if self.O >= self.R:
            raise ValueError(f"Overshoot {self.O} must be less than reversal {self.R}")
        if self.T < 1:
            raise ValueError("Timeout T must be >= 1")
        if self.T > 390:  # Typical RTH session length for NQ
            LOG.warning(f"Timeout {self.T} may exceed typical session length")
        if self.outliers_policy not in ["invalidate", "skip"]:
            raise ValueError(
                f"outliers_policy must be 'invalidate' or 'skip', not '{self.outliers_policy}'"
            )


@dataclass
class DataQualityConfig:
    """Configuration for data validation"""

    check_ohlc_integrity: bool = True
    max_price_change_pct: float = MAX_PRICE_CHANGE_PERCENT
    min_volume: int = MIN_VOLUME
    handle_gaps: bool = True
    max_gap_minutes: int = MAX_GAP_MINUTES
    remove_outliers: bool = True
    outlier_std_threshold: float = 10.0


@dataclass
class Regime:
    """Defines a single market regime via a pandas query string."""
    name: str
    condition: str


@dataclass
class RegimeConfig:
    """Configuration for market regime analysis."""
    # If false, all regime logic is skipped
    enabled: bool = True
    # List of defined regimes. If empty, a default 'all_data' regime is used.
    regimes: List[Regime] = field(default_factory=list)


@dataclass
class EngineConfig:
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    regime_config: Optional[RegimeConfig] = None
    cache_dir: str = "cache"
    out_dir: str = "runs/run_latest"
    random_seed: Optional[int] = 42
    bootstrap_block_size: Optional[int] = None  # <-- add back


@dataclass
class Zone:
    id: int
    type: ZoneType
    level: float
    width: float
    activation_time: pd.Timestamp
    touches: List[float]
    p_value: float
    is_significant: bool
    expire_days: int = 30
    regime: str = "all_data"  # <— new


##############################################################################
# 3. UTILS (time, rng, validation helpers)
##############################################################################


def test_zone_significance(
    touches: List[float],
    level: float,
    width: float,
    local_prices: pd.Series,
    alpha: float = ZONE_SIGNIFICANCE_ALPHA,
) -> Tuple[bool, float]:
    """
    Tests zone significance using a binomial test.
    Null hypothesis: touches are uniformly distributed in the local price window.
    """
    n = len(touches)
    if n < 2:
        return True, 1.0  # Not enough data to test, default to significant

    window_pts = local_prices.max() - local_prices.min()
    if window_pts <= 0:
        return True, 1.0  # Cannot determine p0, default to significant

    p0 = min(1.0, (2.0 * width) / window_pts)
    x_obs = sum(1 for t in touches if abs(t - level) <= width)

    if HAVE_SCIPY:
        pval = scipy_stats.binom.sf(k=x_obs - 1, n=n, p=p0)
    else:
        from math import comb

        try:
            pval = sum(
                comb(n, k) * (p0**k) * ((1 - p0) ** (n - k))
                for k in range(x_obs, n + 1)
            )
        except (ValueError, TypeError):
            pval = 1.0

    return pval < alpha, pval


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

# Multiprocessing guard
try:
    from multiprocessing import Pool

    HAVE_MULTIPROCESSING = True
except Exception:
    Pool = None  # type: ignore
    HAVE_MULTIPROCESSING = False


def block_bootstrap_ci(
    data: np.ndarray,
    statistic_func,
    n_boot: int = 5000,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Circular moving block bootstrap for confidence intervals."""
    n = len(data)
    if n < block_size:
        return standard_bootstrap_ci(data, statistic_func, n_boot, alpha, seed=seed)
    rng = np.random.default_rng(seed)
    bootstrap_stats = []
    extended_data = np.concatenate([data, data[: block_size - 1]])
    for _ in range(n_boot):
        num_blocks = math.ceil(n / block_size)
        start_indices = rng.integers(0, n, size=num_blocks)
        resampled_indices = np.concatenate(
            [np.arange(s, s + block_size) for s in start_indices]
        )[:n]
        resampled_data = extended_data[resampled_indices]
        bootstrap_stats.append(statistic_func(resampled_data))
    bootstrap_stats = np.array(bootstrap_stats)
    return float(np.percentile(bootstrap_stats, 100 * alpha / 2)), float(
        np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    )


def block_bootstrap_prop_ci(
    x: np.ndarray,  # 0/1 vector
    n_boot: int = 4000,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Moving-block bootstrap CI for a proportion (handles autocorrelation).
    """
    if len(x) == 0:
        return (np.nan, np.nan)
    # Fallback to Jeffreys only when too short for blocks
    if len(x) < max(8, block_size):
        k = int(x.sum()); n = int(len(x))
        if HAVE_SCIPY:
            from scipy.stats import beta
            return float(beta.ppf(alpha/2, k+0.5, n-k+0.5)), float(beta.ppf(1-alpha/2, k+0.5, n-k+0.5))
        # vanilla percentile bootstrap as last resort
        return standard_bootstrap_ci(x, np.mean, n_boot=n_boot, alpha=alpha, seed=seed)

    rng = np.random.default_rng(seed)
    n = len(x)
    z = np.concatenate([x, x[:block_size-1]])
    stats = []
    for _ in range(n_boot):
        nb = math.ceil(n / block_size)
        starts = rng.integers(0, n, size=nb)
        idx = np.concatenate([np.arange(s, s+block_size) for s in starts])[:n]
        stats.append(z[idx].mean())
    stats = np.sort(np.asarray(stats))
    lo = stats[int(np.floor((alpha/2) * len(stats)))]
    hi = stats[int(np.floor((1 - alpha/2) * len(stats)))]
    return float(lo), float(hi)


def bootstrap_risk_difference(
    x1: np.ndarray,  # 0/1 vector for cohort
    x2: np.ndarray,  # 0/1 vector for baseline
    block_size: int,
    n_boot: int = 3000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, Tuple[float, float], float]:
    """
    Computes CI and p-value for risk difference (p1 - p2) via block bootstrap.
    Handles autocorrelation by preserving the sample's block structure.
    Returns: (observed difference, (CI low, CI high), p-value)
    """
    rng = np.random.default_rng(seed)
    obs_diff = x1.mean() - x2.mean()

    n1, n2 = len(x1), len(x2)
    if n1 < block_size or n2 < block_size:
        # Fallback to standard bootstrap if data is too small for blocks
        boot_diffs = []
        for _ in range(n_boot):
            mean1 = rng.choice(x1, size=n1, replace=True).mean()
            mean2 = rng.choice(x2, size=n2, replace=True).mean()
            boot_diffs.append(mean1 - mean2)
    else:
        # Generate bootstrap samples of the *difference* using moving blocks
        boot_diffs = []
        z1 = np.concatenate([x1, x1[: block_size - 1]])
        z2 = np.concatenate([x2, x2[: block_size - 1]])

        for _ in range(n_boot):
            nb1 = math.ceil(n1 / block_size)
            starts1 = rng.integers(0, n1, size=nb1)
            idx1 = np.concatenate([np.arange(s, s + block_size) for s in starts1])[:n1]
            mean1 = z1[idx1].mean()

            nb2 = math.ceil(n2 / block_size)
            starts2 = rng.integers(0, n2, size=nb2)
            idx2 = np.concatenate([np.arange(s, s + block_size) for s in starts2])[:n2]
            mean2 = z2[idx2].mean()

            boot_diffs.append(mean1 - mean2)

    boot_diffs = np.array(boot_diffs)

    # Percentile CI
    ci_lo = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    # Two-sided p-value from bootstrap distribution shifted to satisfy H0
    boot_diffs_h0 = boot_diffs - obs_diff
    p_value = (np.abs(boot_diffs_h0) >= np.abs(obs_diff)).mean()

    return obs_diff, (float(ci_lo), float(ci_hi)), float(p_value)


def standard_bootstrap_ci(
    data: np.ndarray,
    statistic_func,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Standard bootstrap for comparison"""
    rng = np.random.default_rng(seed)
    bootstrap_stats = [
        statistic_func(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ]
    bootstrap_stats = np.array(bootstrap_stats)
    return float(np.percentile(bootstrap_stats, 100 * alpha / 2)), float(
        np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    )


def get_file_sha256(filepath: str) -> str:
    """Computes SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def update_dataclass_from_dict(dc, d: Dict):
    """Recursively update dataclass fields from a dictionary."""
    for k, v in d.items():
        if not hasattr(dc, k):
            continue
        field_value = getattr(dc, k)
        if dataclasses.is_dataclass(field_value) and isinstance(v, dict):
            update_dataclass_from_dict(field_value, v)
        else:
            setattr(dc, k, v)


def round_to_tick(x: float, tick: float) -> float:
    """Rounds a price to the nearest instrument tick size."""
    if tick == 0:
        return x
    return round(round(x / tick) * tick, 10)


def _blk(config: EngineConfig) -> int:
    """Helper to get the bootstrap block size, falling back to a dynamic default."""
    bs = getattr(config, "bootstrap_block_size", None)
    return int(bs) if (bs is not None and bs > 0) else max(5, config.episode.T // 2)


def _ensure_utc_timestamps(df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    """Ensures the timestamp column is a timezone-aware UTC timestamp."""
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column.")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df["timestamp"].dt.tz is None:
        tz = config.instrument.data_tz or "UTC"
        LOG.info(f"Timestamp column is timezone-naive, localizing to '{tz}' then converting to UTC.")
        df["timestamp"] = df["timestamp"].dt.tz_localize(tz).dt.tz_convert("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df


def _clip01(t: Tuple[float, float]) -> Tuple[float, float]:
    """Clips a confidence interval tuple to the [0, 1] range."""
    if not t or t[0] is None: return t
    return (max(0.0, min(1.0, float(t[0]))),
            max(0.0, min(1.0, float(t[1]))))


##############################################################################
# 4. DATA PREP (load, sessions, indicators, validator)
##############################################################################


class DataValidator:
    """Validates and cleans OHLCV data"""

    def __init__(self, config: DataQualityConfig):
        self.config = config

    def validate_and_clean(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate OHLCV data and return cleaned dataframe with validation report"""
        report = {
            "original_rows": len(df),
            "invalid_ohlc": 0,
            "outliers": 0,
            "gaps_detected": 0,
            "low_volume": 0,
        }

        if self.config.check_ohlc_integrity:
            invalid_mask = (
                (df["high"] < df["low"])
                | (df["high"] < df["open"])
                | (df["high"] < df["close"])
                | (df["low"] > df["open"])
                | (df["low"] > df["close"])
            )
            report["invalid_ohlc"] = invalid_mask.sum()
            if invalid_mask.any():
                LOG.warning(
                    f"Removing {invalid_mask.sum()} bars with invalid OHLC relationships"
                )
                df = df[~invalid_mask].copy()

        if self.config.max_price_change_pct > 0:
            pct_change = df["close"].pct_change().abs() * 100
            too_big_mask = pct_change > self.config.max_price_change_pct
            if too_big_mask.any():
                report["price_spike"] = int(too_big_mask.sum())
                LOG.warning(
                    f"Removing {report['price_spike']} bars with >{self.config.max_price_change_pct}% move"
                )
                df = df[~too_big_mask].copy()

        if self.config.min_volume > 0:
            invalid_vol = df["volume"] < self.config.min_volume
            report["low_volume"] = invalid_vol.sum()
            if invalid_vol.any():
                LOG.warning(
                    f"Removing {report['low_volume']} bars with volume < {self.config.min_volume}"
                )
                df = df[~invalid_vol].copy()

        df["is_outlier"] = False
        if self.config.remove_outliers:
            df["returns"] = df["close"].pct_change()
            if "session_date" in df.columns:
                mad = df.groupby("session_date")["returns"].transform(
                    lambda x: (x - x.median()).abs().median()
                )
                # floor to a tiny positive value to avoid zero-MAD blowups
                mad = mad.replace(0, np.nan)
                global_mad = (df["returns"] - df["returns"].median()).abs().median()
                mad = mad.fillna(global_mad if np.isfinite(global_mad) and global_mad > 0 else 1e-6)
                outlier_threshold = self.config.outlier_std_threshold * mad * 1.4826 + 1e-9
                outliers = df["returns"].abs() > outlier_threshold
            else:
                sigma = df["returns"].std()
                if not np.isfinite(sigma) or sigma == 0:
                    sigma = df["returns"].mad() * 1.4826 or 1e-6
                outlier_threshold = sigma * self.config.outlier_std_threshold
                outliers = df["returns"].abs() > outlier_threshold
            report["outliers"] = outliers.sum()
            if outliers.any():
                LOG.info(f"Flagging {outliers.sum()} outlier bars")
                df.loc[outliers, "is_outlier"] = True
            df = df.drop(columns=["returns"])

        if self.config.handle_gaps:
            same_session = df["session_date"] == df["session_date"].shift(1)
            time_diff = df["timestamp"].diff()
            gaps = same_session & (
                time_diff > pd.Timedelta(minutes=self.config.max_gap_minutes)
            )
            df["has_gap"] = gaps.fillna(False)
            report["gaps_detected"] = int(gaps.sum())
            if report["gaps_detected"] > 0:
                LOG.info(
                    f"Detected {report['gaps_detected']} intra-session gaps > {self.config.max_gap_minutes} minutes"
                )
        else:
            df["has_gap"] = False

        report["final_rows"] = len(df)
        report["rows_removed"] = report["original_rows"] - report["final_rows"]
        report["removal_pct"] = (
            100 * report["rows_removed"] / report["original_rows"]
            if report["original_rows"] > 0
            else 0
        )
        return df, report


def prepare_data(files: List[str], config: EngineConfig) -> pd.DataFrame:
    """Load and prepare data with validation"""
    validator = DataValidator(config.data_quality)
    all_dfs = []
    for file in files:
        LOG.info(f"Loading {file}")
        if file.endswith(".parquet"):
            if not HAVE_PARQUET:
                raise ImportError("pyarrow is required. `pip install pyarrow`")
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file)

        df.columns = [c.lower() for c in df.columns]
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if set(required) - set(df.columns):
            raise ValueError(
                f"Missing columns in {file}: {set(required) - set(df.columns)}"
            )

        df = _ensure_utc_timestamps(df, config)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        df = annotate_sessions(df, config.instrument)
        df, report = validator.validate_and_clean(df)
        LOG.info(f"Data quality report for {file}: {report}")
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = (
        combined.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )
    combined = add_indicators(combined, config.indicators)
    return combined


def annotate_sessions(df: pd.DataFrame, config: InstrumentConfig) -> pd.DataFrame:
    """Add session information based on instrument config"""
    if ZoneInfo:
        tz = ZoneInfo(config.session_tz)
    elif pytz:
        tz = pytz.timezone(config.session_tz)
    else:
        tz = None
        LOG.warning("No timezone library available, using UTC")

    df["local_time"] = df["timestamp"].dt.tz_convert(tz) if tz else df["timestamp"]
    df["session_date"] = df["local_time"].dt.date
    df["minute_of_day"] = df["local_time"].dt.hour * 60 + df["local_time"].dt.minute
    rth_start = pd.to_datetime(config.rth_start).time()
    rth_end = pd.to_datetime(config.rth_end).time()
    times = df["local_time"].dt.time
    if rth_end > rth_start:
        is_rth = (times >= rth_start) & (times < rth_end)
    else:
        # wraps midnight
        is_rth = (times >= rth_start) | (times < rth_end)
    df["is_rth"] = is_rth
    return df


def add_indicators(df: pd.DataFrame, config: IndicatorConfig) -> pd.DataFrame:
    """Add technical indicators"""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(
        axis=1
    )
    df["atr"] = tr.ewm(span=config.atr_n, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=config.rsi_n, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(span=config.rsi_n, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-12)))
    if config.rvol_lookback_sessions > 0 and "session_date" in df.columns:
        bars_per_session = int(df.groupby("session_date").size().median())
        if bars_per_session == 0: bars_per_session = 390 # Fallback for safety
        rolling_window = max(10, config.rvol_lookback_sessions * bars_per_session)
        df["avg_volume_lookback"] = (
            df["volume"]
            .rolling(window=rolling_window, min_periods=rolling_window // 10)
            .mean()
        )
        df["rvol"] = df["volume"] / (df["avg_volume_lookback"] + 1e-12)
    else:
        df["rvol"] = np.nan
    n, d = config.stoch_n, config.stoch_d
    lowest_low = df["low"].rolling(window=n, min_periods=1).min()
    highest_high = df["high"].rolling(window=n, min_periods=1).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_k"] = ((df["close"] - lowest_low) / denom * 100).clip(0, 100)
    df["stoch_d"] = df["stoch_k"].rolling(window=d, min_periods=1).mean()
    return df


def annotate_regimes(df: pd.DataFrame, regime_config: RegimeConfig) -> pd.DataFrame:
    """Adds boolean columns to the dataframe for each defined market regime."""
    if not regime_config or not regime_config.enabled or not regime_config.regimes:
        return df

    LOG.info(f"Annotating {len(regime_config.regimes)} market regimes...")
    for regime in regime_config.regimes:
        col_name = f"regime_{regime.name.lower().replace(' ', '_')}"
        try:
            # query returns the filtered frame; convert to boolean mask
            idx = df.query(regime.condition, engine='python').index
            mask = df.index.isin(idx)
            df[col_name] = mask.astype(bool)
            LOG.debug(f"Annotated regime '{regime.name}' ({mask.sum()} bars)")
        except Exception as e:
            LOG.error(
                f"Failed to evaluate regime '{regime.name}': {regime.condition}. Error: {e}"
            )
            df[col_name] = False
    return df


def ensure_prepared(df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    """Checks if data has been prepared, and if not, runs preparation steps."""
    df = _ensure_utc_timestamps(df, config)
    if any(
        c not in df.columns
        for c in ["session_date", "minute_of_day", "is_rth", "local_time"]
    ):
        LOG.info("Input data missing session columns, running annotate_sessions...")
        df = annotate_sessions(df, config.instrument)
    if any(c not in df.columns for c in ["atr", "rsi", "rvol", "stoch_k", "stoch_d"]):
        LOG.info("Input data missing indicator columns, running add_indicators...")
        df = add_indicators(df, config.indicators)

    # Annotate regimes after indicators are available
    if config.regime_config:
        df = annotate_regimes(df, config.regime_config)

    return df


##############################################################################
# 5. ZONE PROVIDERS (registry: pivots, extern_levels)
##############################################################################

# Provider registry
ZONE_PROVIDERS: Dict[
    str, Callable[[pd.DataFrame, ZoneConfig, InstrumentConfig], List[Zone]]
] = {}


def register_zone_provider(name: str) -> Callable:
    """Decorator to register a new zone provider."""

    def decorator(
        func: Callable[[pd.DataFrame, ZoneConfig, InstrumentConfig], List[Zone]],
    ):
        ZONE_PROVIDERS[name] = func
        return func

    return decorator


def merge_close_zones(zones: List["Zone"], tolerance: float) -> List["Zone"]:
    """Merges zones that are closer than the given tolerance."""
    if not zones:
        return []
    sorted_zones = sorted(zones, key=lambda z: z.level)
    merged_zones = [sorted_zones[0]]
    for current_zone in sorted_zones[1:]:
        prev_zone = merged_zones[-1]
        if abs(current_zone.level - prev_zone.level) <= tolerance:
            total_touches = len(prev_zone.touches) + len(current_zone.touches)
            if total_touches == 0:
                continue
            new_level = (
                (prev_zone.level * len(prev_zone.touches))
                + (current_zone.level * len(current_zone.touches))
            ) / total_touches
            merged_zones[-1] = dataclasses.replace(
                prev_zone,
                level=new_level,
                touches=prev_zone.touches + current_zone.touches,
                p_value=min(prev_zone.p_value, current_zone.p_value),
                activation_time=min(
                    prev_zone.activation_time, current_zone.activation_time
                ),
                width=(prev_zone.width + current_zone.width) / 2.0,
            )
        else:
            merged_zones.append(current_zone)
    return merged_zones


@register_zone_provider("pivots")
def detect_zones_from_pivots(
    df: pd.DataFrame, config: ZoneConfig, instrument: InstrumentConfig
) -> List[Zone]:
    """Detects candidate zones from price pivots."""
    pivots = detect_pivots(df, config.pivot_k, config, instrument)
    candidate_zones = []
    tick_size = instrument.tick_size

    high_pivots = [p for p in pivots if p["type"] == "HIGH"]
    low_pivots = [p for p in pivots if p["type"] == "LOW"]

    for pivot_list, zone_type in [
        (high_pivots, ZoneType.RESISTANCE),
        (low_pivots, ZoneType.SUPPORT),
    ]:
        clusters = cluster_pivots(pivot_list, config)
        for cluster in clusters:
            if len(cluster) < config.min_touches_for_significance:
                continue
            touches = [p["price"] for p in cluster]
            level = round_to_tick(np.mean(touches), tick_size)

            if config.zone_width_alpha_atr is not None:
                t0, t1 = min(p["center_time"] for p in cluster), max(
                    p["center_time"] for p in cluster
                )
                local_df = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]
                atr_ref = float(
                    local_df["atr"].median()
                    if not local_df.empty and not local_df["atr"].isnull().all()
                    else float(df["atr"].median())
                )
                width = max(
                    tick_size,
                    round_to_tick(config.zone_width_alpha_atr * atr_ref, tick_size),
                )
            else:
                width = max(
                    tick_size,
                    round_to_tick(config.zone_width_points or 15.0, tick_size),
                )

            p_value = 1.0
            if config.significance_test:
                min_time, max_time = min(p["center_time"] for p in cluster), max(
                    p["center_time"] for p in cluster
                )
                sessions_spanned = df[
                    (df["timestamp"] >= min_time) & (df["timestamp"] <= max_time)
                ]["session_date"].unique()
                local_prices_df = df[df["session_date"].isin(sessions_spanned)]
                if not local_prices_df.empty:
                    width_for_test = max(
                        width,
                        (config.cluster_width_points or width)
                        * config.null_width_multiplier,
                    )
                    _, p_value = test_zone_significance(
                        touches,
                        level,
                        width_for_test,
                        pd.concat([local_prices_df["high"], local_prices_df["low"]]),
                    )

            activation_time = (
                sorted(cluster, key=lambda p: p["confirm_time"])[1]["confirm_time"]
                if len(cluster) > 1
                else sorted(cluster, key=lambda p: p["confirm_time"])[0]["confirm_time"]
            )

            candidate_zones.append(
                Zone(
                    id=0,
                    type=zone_type,
                    level=level,
                    width=width,
                    activation_time=activation_time,
                    touches=touches,
                    p_value=p_value,
                    is_significant=False,
                    expire_days=config.expire_days,
                )
            )

    return candidate_zones


@register_zone_provider("extern_levels")
def detect_zones_from_csv(
    df: pd.DataFrame, config: ZoneConfig, instrument: InstrumentConfig
) -> List[Zone]:
    """Detects candidate zones from an external CSV file."""
    path = config.extern_levels_path
    if not path or not os.path.exists(path):
        if path:
            LOG.warning(f"External levels CSV not found at {path}")
        return []

    LOG.info(f"Loading external levels from {path}")
    try:
        levels_df = pd.read_csv(path, parse_dates=["timestamp"])
        levels_df.columns = [c.lower() for c in levels_df.columns]
        required = ["timestamp", "level", "type", "width"]
        if any(c not in levels_df.columns for c in required):
            raise ValueError(f"External levels CSV must contain columns: {required}")
    except Exception as e:
        LOG.error(f"Failed to load external levels CSV: {e}")
        return []

    candidate_zones = []
    tick_size = instrument.tick_size
    for _, row in levels_df.iterrows():
        level = round_to_tick(row["level"], tick_size)
        width = round_to_tick(row["width"], tick_size)

        type_str = str(row.get("type", "")).lower()
        if "supp" in type_str or type_str.startswith("s"):
            zone_type = ZoneType.SUPPORT
        elif "res" in type_str or type_str.startswith("r"):
            zone_type = ZoneType.RESISTANCE
        else:
            LOG.warning(f"Skipping external level with unknown type: {row.get('type')}")
            continue

        activation_time = pd.Timestamp(row["timestamp"], tz="UTC")

        touch_df = df[df["timestamp"] >= activation_time]
        touches = []
        if zone_type == ZoneType.SUPPORT:
            touches.extend(
                touch_df["low"][
                    (touch_df["low"] >= level - width)
                    & (touch_df["low"] <= level + width)
                ].tolist()
            )
        else:
            touches.extend(
                touch_df["high"][
                    (touch_df["high"] >= level - width)
                    & (touch_df["high"] <= level + width)
                ].tolist()
            )

        if len(touches) < config.min_touches_for_significance:
            continue

        p_value = 1.0
        if config.significance_test:
            sessions_spanned = touch_df["session_date"].unique()
            local_prices_df = df[df["session_date"].isin(sessions_spanned)]
            if not local_prices_df.empty:
                width_for_test = max(
                    width,
                    (config.cluster_width_points or width)
                    * config.null_width_multiplier,
                )
                _, p_value = test_zone_significance(
                    touches,
                    level,
                    width_for_test,
                    pd.concat([local_prices_df["high"], local_prices_df["low"]]),
                )

        candidate_zones.append(
            Zone(
                id=0,
                type=zone_type,
                level=level,
                width=width,
                activation_time=activation_time,
                touches=touches,
                p_value=p_value,
                is_significant=False,
                expire_days=config.expire_days,
            )
        )

    return candidate_zones


def detect_pivots(
    df: pd.DataFrame, k: int, config: ZoneConfig, instrument: InstrumentConfig
) -> List[Dict[str, Any]]:
    """Detect pivots with look-ahead safety and tolerance."""
    pivots, tick = [], (
        config.tick_size if config.tick_size is not None else instrument.tick_size
    )
    hi, lo, ts = df["high"].values, df["low"].values, df["timestamp"]
    for i in range(k, len(df) - k):
        window_hi, window_lo = hi[i - k : i + k + 1], lo[i - k : i + k + 1]
        if (
            (hi[i] >= window_hi.max() - tick)
            and (hi[i] > hi[i - 1])
            and (hi[i] >= hi[i + 1])
        ):
            pivots.append(
                {
                    "type": "HIGH",
                    "price": hi[i],
                    "center_time": ts.iat[i],
                    "confirm_time": ts.iat[i + k],
                    "index": i,
                }
            )
        if (
            (lo[i] <= window_lo.min() + tick)
            and (lo[i] < lo[i - 1])
            and (lo[i] <= lo[i + 1])
        ):
            pivots.append(
                {
                    "type": "LOW",
                    "price": lo[i],
                    "center_time": ts.iat[i],
                    "confirm_time": ts.iat[i + k],
                    "index": i,
                }
            )
    return pivots


def cluster_pivots(pivots: List[Dict], config: ZoneConfig) -> List[List[Dict]]:
    """Cluster nearby pivots"""
    if not pivots:
        return []
    sorted_pivots = sorted(pivots, key=lambda x: x["price"])
    clusters, current_cluster = [], [sorted_pivots[0]]
    for pivot in sorted_pivots[1:]:
        price_is_close = abs(
            pivot["price"] - np.mean([p["price"] for p in current_cluster])
        ) <= (config.cluster_width_points or 15.0)
        time_is_close = (
            config.max_cluster_span_days is None
            or (
                pivot["center_time"] - min(p["center_time"] for p in current_cluster)
            ).days
            <= config.max_cluster_span_days
        )
        if price_is_close and time_is_close:
            current_cluster.append(pivot)
        else:
            if len(current_cluster) >= config.min_touches_for_significance:
                clusters.append(current_cluster)
            current_cluster = [pivot]
    if len(current_cluster) >= config.min_touches_for_significance:
        clusters.append(current_cluster)
    return clusters


##############################################################################
# 6. EPISODES (FSM & policies)
##############################################################################


class EpisodeState(ABC):
    @abstractmethod
    def process_bar(
        self, context: Dict[str, Any]
    ) -> Tuple[Optional[EpisodeOutcome], Optional["EpisodeState"]]:
        pass


class SearchingForTouch(EpisodeState):
    def process_bar(
        self, context: Dict[str, Any]
    ) -> Tuple[Optional[EpisodeOutcome], Optional[EpisodeState]]:
        bar, zone = context["bar"], context["zone"]
        band_low, band_high = (
            zone["level"] - zone["width"],
            zone["level"] + zone["width"],
        )
        if bar["low"] <= band_high and bar["high"] >= band_low:
            context["touch_idx"] = context["current_idx"]
            context["touch_time"] = bar["timestamp"]
            touch_bar = context["df"].iloc[context["touch_idx"]]
            for ind in [
                "rsi",
                "atr",
                "rvol",
                "stoch_k",
                "stoch_d",
                "is_rth",
                "minute_of_day",
            ]:
                context[f"{ind}_at_touch"] = touch_bar.get(ind)
            if context["current_idx"] > 0:
                prev_close = context["prev_close"]
                if prev_close > band_high:
                    context["from_above"] = True
                elif prev_close < band_low:
                    context["from_above"] = False
                else:
                    context["from_above"] = None
            return None, TrackingOutcome()
        return None, self


class TrackingOutcome(EpisodeState):
    def process_bar(
        self, context: Dict[str, Any]
    ) -> Tuple[Optional[EpisodeOutcome], Optional[EpisodeState]]:
        config = context["config"]
        bar = context["bar"]

        # Handle outliers first based on policy
        if bar.get("is_outlier", False):
            if config.get("outliers_policy") == "skip":
                LOG.debug(f"Skipping outlier bar at {bar['timestamp']}")
                return None, self  # Remain in this state, effectively ignoring the bar
            else:  # Default policy is "invalidate"
                LOG.debug(
                    f"Invalidating episode due to outlier bar at {bar['timestamp']}"
                )
                return EpisodeOutcome.INVALID, None

        if (
            config["max_gap_bars"] > 0
            and "prev_ts" in context
            and (context["bar"]["timestamp"] - context["prev_ts"])
            > (context["bar_dt"] * config["max_gap_bars"])
        ):
            return EpisodeOutcome.INVALID, None
        context["prev_ts"] = context["bar"]["timestamp"]

        zone = context["zone"]
        band_low, band_high = (
            zone["level"] - zone["width"],
            zone["level"] + zone["width"],
        )
        band_break_high, band_break_low = (
            band_high + config["O"],
            band_low - config["O"],
        )
        direction, ref = (-1 if zone["type"] == ZoneType.RESISTANCE else 1), zone[
            "level"
        ]
        fav = (
            max(0.0, ref - bar["low"])
            if direction == -1
            else max(0.0, bar["high"] - ref)
        )
        adv = (
            max(0.0, bar["high"] - ref)
            if direction == -1
            else max(0.0, ref - bar["low"])
        )
        context["max_favorable"], context["max_adverse"] = max(
            context.get("max_favorable", 0), fav
        ), max(context.get("max_adverse", 0), adv)

        if bar.get("has_gap", False):
            return EpisodeOutcome.INVALID, None  # Gap check after outlier policy

        if (bar["high"] > band_high and bar["high"] <= band_break_high) or (
            bar["low"] < band_low and bar["low"] >= band_break_low
        ):
            context["pierced_before"] = True
        if bar["close"] > band_break_high or bar["close"] < band_break_low:
            return EpisodeOutcome.BREAK, None
        if (
            context.get("pierced_before", False)
            and context["max_favorable"] >= config["R"]
        ):
            return EpisodeOutcome.PIERCE_AND_REVERT, None
        if not context.get("exited_favorably", False):
            if (direction == 1 and bar["high"] > band_high) or (
                direction == -1 and bar["low"] < band_low
            ):
                context["exited_favorably"] = True
        if (
            context.get("exited_favorably", False)
            and context["max_favorable"] >= config["R"]
        ):
            return EpisodeOutcome.RESPECT, None
        # Censor when the bar crosses the touch session boundary (optional)
        if config.get("censor_on_session_change", True):
            if context.get("touch_idx") is not None:
                touch_sess = context["df"].iloc[context["touch_idx"]]["session_date"]
                if bar.get("session_date", touch_sess) != touch_sess:
                    return (EpisodeOutcome.TIMEOUT if not config.get("treat_timeout_as_break", False)
                            else EpisodeOutcome.BREAK), None

        if (context["current_idx"] - context["touch_idx"]) >= config["T"]:
            return (
                EpisodeOutcome.BREAK
                if config.get("treat_timeout_as_break", False)
                else EpisodeOutcome.TIMEOUT
            ), None
        return None, self


class EpisodeDetector:
    def __init__(self, config: EpisodeConfig):
        self.config = config

    def detect_episode(
        self, df: pd.DataFrame, zone: Dict[str, Any], start_idx: int
    ) -> Optional[Dict[str, Any]]:
        state, bar_dt = SearchingForTouch(), df["timestamp"].diff().median()
        if pd.isna(bar_dt) or bar_dt <= pd.Timedelta(0):
            bar_dt = pd.Timedelta(minutes=1)
        context = {
            "df": df,
            "zone": zone,
            "config": dataclasses.asdict(self.config),
            "bar_dt": bar_dt,
            "touch_idx": None,
            "touch_time": None,
            "from_above": None,
            "max_favorable": 0.0,
            "max_adverse": 0.0,
            "pierced_before": False,
        }
        expiry_time = zone.get("expiry_time")
        for i in range(start_idx, len(df)):
            bar = df.iloc[i]
            if expiry_time and bar["timestamp"] > expiry_time:
                break
            context.update(
                {
                    "current_idx": i,
                    "bar": bar.to_dict(),
                    "prev_close": df.iloc[i - 1]["close"] if i > 0 else None,
                }
            )
            outcome, next_state = state.process_bar(context)
            if next_state is not state and next_state is not None:
                state = next_state
                outcome, next_state = state.process_bar(context)
            if outcome is not None:
                episode_data = {
                    "zone_id": zone["id"],
                    "zone_type": zone["type"],
                    "outcome": outcome,
                    "touch_time": context["touch_time"],
                    "outcome_time": bar["timestamp"],
                    "bars_to_outcome": (
                        i - context["touch_idx"]
                        if context["touch_idx"] is not None
                        else -1
                    ),
                    "max_favorable": context["max_favorable"],
                    "max_adverse": context["max_adverse"],
                    "from_above": context["from_above"],
                    "end_idx": i,
                }
                for ind in [
                    "rsi",
                    "atr",
                    "rvol",
                    "stoch_k",
                    "stoch_d",
                    "is_rth",
                    "minute_of_day",
                ]:
                    episode_data[f"{ind}_at_touch"] = context.get(f"{ind}_at_touch")
                for ind in ["rsi", "atr", "rvol", "stoch_k", "stoch_d"]:
                    episode_data[f"{ind}_at_outcome"] = bar.get(ind)
                return episode_data
            if next_state is None:
                break
            state = next_state
        return None


##############################################################################
# 7. STATISTICS (rates, CIs, survival, tails, calibration)
##############################################################################


def discover_zones(df: pd.DataFrame, config: EngineConfig, regime_name: str) -> List[Zone]:
    """Runs all zone discovery providers and returns a final list of merged zones."""
    all_candidate_zones = []
    for provider_name in config.zones.providers:
        if provider_name not in ZONE_PROVIDERS:
            LOG.warning(f"Zone provider '{provider_name}' not found. Skipping.")
            continue
        LOG.info(f"Running zone provider: '{provider_name}'")
        provider_func = ZONE_PROVIDERS[provider_name]
        try:
            zones_from_provider = provider_func(df, config.zones, config.instrument)
            all_candidate_zones.extend(zones_from_provider)
            LOG.info(
                f"Provider '{provider_name}' found {len(zones_from_provider)} candidate zones."
            )
        except Exception as e:
            LOG.error(f"Error in zone provider '{provider_name}': {e}", exc_info=True)

    significant_zones = []
    if config.zones.significance_test and all_candidate_zones:
        p_values = [z.p_value for z in all_candidate_zones]
        if HAVE_STATSMODELS:
            reject, _, _, _ = multipletests(
                p_values, alpha=ZONE_SIGNIFICANCE_ALPHA, method="fdr_bh"
            )
            significant_zones = [
                zone for zone, is_sig in zip(all_candidate_zones, reject) if is_sig
            ]
        else:
            LOG.warning(
                "statsmodels not found. Falling back to simple p-value threshold."
            )
            significant_zones = [
                zone
                for zone in all_candidate_zones
                if zone.p_value < ZONE_SIGNIFICANCE_ALPHA
            ]
    else:
        significant_zones = all_candidate_zones
    LOG.info(
        f"Found {len(significant_zones)} significant zones from {len(all_candidate_zones)} candidates before merging."
    )

    if config.zones.merge_tolerance_points > 0 and significant_zones:
        support_zones = [z for z in significant_zones if z.type == ZoneType.SUPPORT]
        resistance_zones = [
            z for z in significant_zones if z.type == ZoneType.RESISTANCE
        ]
        merged_support = merge_close_zones(
            support_zones, config.zones.merge_tolerance_points
        )
        merged_resistance = merge_close_zones(
            resistance_zones, config.zones.merge_tolerance_points
        )
        zones = sorted(
            merged_support + merged_resistance, key=lambda z: z.activation_time
        )
    else:
        zones = sorted(significant_zones, key=lambda z: z.activation_time)

    final_zones = [
        dataclasses.replace(zone, id=i + 1, is_significant=True, regime=regime_name)
        for i, zone in enumerate(zones)
    ]
    LOG.info(f"Detected {len(final_zones)} final zones.")
    return final_zones


def evaluate_episodes(
    df: pd.DataFrame, zones: List[Zone], config: EngineConfig
) -> List[Dict]:
    """Evaluates all episodes for a given set of zones on a dataframe."""
    detector, episodes = EpisodeDetector(config.episode), []
    for zone in zones:
        zone_dict = {
            "id": zone.id,
            "type": zone.type,
            "level": zone.level,
            "width": zone.width,
            "expiry_time": zone.activation_time + pd.Timedelta(days=zone.expire_days),
        }
        zone_episodes, touch_number = [], 0
        potential_indices = df.index[df["timestamp"] > zone.activation_time]
        if potential_indices.size == 0:
            continue
        current_scan_idx = int(potential_indices.min())

        while (
            current_scan_idx < len(df)
            and df.iloc[current_scan_idx]["timestamp"] <= zone_dict["expiry_time"]
        ):
            if (
                config.episode.max_episodes_per_zone is not None
                and len(zone_episodes) >= config.episode.max_episodes_per_zone
            ):
                break
            if config.episode.first_touch_only and len(zone_episodes) > 0:
                break
            episode = detector.detect_episode(df, zone_dict, current_scan_idx)
            if episode:
                touch_number += 1
                episode["touch_number"] = touch_number
                zone_episodes.append(episode)
                next_scan_idx_candidate = (
                    episode["end_idx"] + 1 + config.episode.min_bars_between_touches
                )
                band_low, band_high = (
                    zone_dict["level"] - zone_dict["width"],
                    zone_dict["level"] + zone_dict["width"],
                )
                next_scan_idx = -1
                if next_scan_idx_candidate >= len(df):
                    break
                for idx in range(next_scan_idx_candidate, len(df)):
                    bar = df.iloc[idx]
                    if bar["timestamp"] > zone_dict["expiry_time"]:
                        break
                    if bar["high"] < band_low or bar["low"] > band_high:
                        next_scan_idx = idx
                        break
                if next_scan_idx != -1:
                    current_scan_idx = next_scan_idx
                else:
                    break
            else:
                break
        episodes.extend(zone_episodes)
    return episodes


def _run_single_analysis(
    df: pd.DataFrame, config: EngineConfig, regime_name: str = "all_data"
) -> Dict[str, Any]:
    """Run complete in-sample analysis pipeline for a single data slice."""
    LOG.info(f"--- Running analysis for regime: {regime_name} ---")
    zones = discover_zones(df, config, regime_name)
    episodes = evaluate_episodes(df, zones, config)
    LOG.info(f"[{regime_name}] Detected {len(episodes)} episodes across {len(zones)} zones")

    episodes_df = pd.DataFrame(episodes)
    if not episodes_df.empty:
        # --- Create Cohort Columns ---
        episodes_df["hour_at_touch"] = episodes_df["minute_of_day_at_touch"] // 60
        for col in ["atr_at_touch", "rvol_at_touch", "stoch_k_at_touch"]:
            try:
                episodes_df[f'{col.split("_")[0]}_tercile'] = pd.qcut(
                    episodes_df[col].rank(method="first"),
                    3,
                    labels=["low", "mid", "high"],
                )
            except (ValueError, TypeError):
                episodes_df[f'{col.split("_")[0]}_tercile'] = "n/a"

        if zones:
            zones_df = pd.DataFrame([dataclasses.asdict(z) for z in zones])
            episodes_df = episodes_df.merge(
                zones_df[["id", "activation_time"]],
                left_on="zone_id",
                right_on="id",
                how="left",
            )
            episodes_df["touch_time"] = pd.to_datetime(episodes_df["touch_time"])
            episodes_df["activation_time"] = pd.to_datetime(
                episodes_df["activation_time"]
            )
            episodes_df["zone_age_days"] = (
                episodes_df["touch_time"] - episodes_df["activation_time"]
            ).dt.days
        else:
            episodes_df["zone_age_days"] = np.nan

        episodes_df["approach_direction"] = episodes_df["from_above"].apply(
            lambda x: (
                "from_above" if x is True else "from_below" if x is False else "inside"
            )
        )

        # --- Build Strata for Analysis ---
        strata = {"all": episodes_df}
        simple_strata_cols = {
            "rth": episodes_df["is_rth_at_touch"] == True,
            "eth": episodes_df["is_rth_at_touch"] == False,
            "from_above": episodes_df["approach_direction"] == "from_above",
            "from_below": episodes_df["approach_direction"] == "from_below",
        }
        for name, mask in simple_strata_cols.items():
            strata[name] = episodes_df[mask]
        grouped_strata_cols = [
            "hour_at_touch",
            "atr_tercile",
            "rvol_tercile",
            "stoch_tercile",
            "touch_number",
        ]
        for col in grouped_strata_cols:
            for name, group in episodes_df.groupby(col):
                if not pd.isna(name):
                    strata[
                        f"{col.replace('_at_touch','').replace('_tercile','').lower()}_{name}"
                    ] = group
        stratified_stats = {
            name: compute_statistics(data.to_dict("records"), config)
            for name, data in strata.items()
            if not data.empty
        }

        # --- Add cohort comparisons vs baseline ---
        baseline_episodes_df = strata.get("all")
        if baseline_episodes_df is not None and not baseline_episodes_df.empty:
            baseline_outcomes = (
                baseline_episodes_df["outcome"]
                .apply(
                    lambda o: (o.value if isinstance(o, Enum) else o)
                    == EpisodeOutcome.RESPECT.value
                )
                .values
            )

            p_values = []
            cohort_names_for_correction = []
            skipped_cohort_count = 0

            # Helper for effect sizes
            eps = 1e-6

            def cohens_h(p, q):
                p = min(max(p, eps), 1 - eps)
                q = min(max(q, eps), 1 - eps)
                return 2 * (math.asin(math.sqrt(p)) - math.asin(math.sqrt(q)))

            for name, cohort_df in strata.items():
                if name == "all" or cohort_df.empty:
                    continue

                if len(cohort_df) < MIN_EPISODES_FOR_STATS:
                    LOG.debug(f"Skipping cohort '{name}' (n={len(cohort_df)} < {MIN_EPISODES_FOR_STATS}) for comparative stats.")
                    skipped_cohort_count += 1
                    continue

                cohort_outcomes = (
                    cohort_df["outcome"]
                    .apply(
                        lambda o: (o.value if isinstance(o, Enum) else o)
                        == EpisodeOutcome.RESPECT.value
                    )
                    .values
                )

                try:
                    risk_diff, ci, p_val = bootstrap_risk_difference(
                        cohort_outcomes,
                        baseline_outcomes,
                        block_size=_blk(config),
                        seed=config.random_seed,
                    )

                    p1, p2 = cohort_outcomes.mean(), baseline_outcomes.mean()
                    # Haldane–Anscombe add 0.5 to each cell for OR
                    a, b = (
                        cohort_outcomes.sum(),
                        len(cohort_outcomes) - cohort_outcomes.sum(),
                    )
                    c, d = (
                        baseline_outcomes.sum(),
                        len(baseline_outcomes) - baseline_outcomes.sum(),
                    )
                    or_est = ((a + 0.5) / (b + 0.5)) / ((c + 0.5) / (d + 0.5))

                    stratified_stats[name]["comparative_respect_rate"] = {
                        "risk_difference": risk_diff,
                        "risk_difference_ci": ci,
                        "p_value": p_val,
                        "cohens_h": cohens_h(p1, p2),
                        "odds_ratio_cc": float(or_est),
                    }
                    p_values.append(p_val)
                    cohort_names_for_correction.append(name)
                except Exception as e:
                    LOG.warning(f"Could not run cohort comparison for '{name}': {e}")

            if HAVE_STATSMODELS and p_values:
                try:
                    reject, q_values, _, _ = multipletests(
                        p_values, alpha=ZONE_SIGNIFICANCE_ALPHA, method="fdr_bh"
                    )
                    for i, name in enumerate(cohort_names_for_correction):
                        if "comparative_respect_rate" in stratified_stats[name]:
                            stratified_stats[name]["comparative_respect_rate"][
                                "q_value"
                            ] = q_values[i]
                            stratified_stats[name]["comparative_respect_rate"][
                                "reject_h0"
                            ] = bool(reject[i])
                except Exception as e:
                    LOG.error(f"Failed to run multiple comparisons correction: {e}")
    else:
        stratified_stats = {"all": compute_statistics([], config)}

    if 'skipped_cohort_count' in locals() and skipped_cohort_count > 0:
        LOG.info(f"Skipped {skipped_cohort_count} cohorts with n < {MIN_EPISODES_FOR_STATS} for comparative stats.")

    return {
        "zones": zones,
        "episodes": episodes,
        "statistics": stratified_stats,
        "data_shape": df.shape,
    }


def run_analysis(df: pd.DataFrame, config: EngineConfig) -> Dict[str, Any]:
    """
    Main analysis orchestrator. Runs the pipeline for each defined market regime.
    """
    regime_conf = config.regime_config
    if not regime_conf or not regime_conf.enabled or not regime_conf.regimes:
        # No regimes defined, run on all data
        return {"all_data": _run_single_analysis(df, config)}

    all_results = {}
    for regime in regime_conf.regimes:
        regime_col = f"regime_{regime.name.lower().replace(' ', '_')}"
        if regime_col not in df.columns:
            LOG.warning(f"Regime column '{regime_col}' not found in data. Skipping regime '{regime.name}'.")
            continue

        regime_df = df[df[regime_col]].copy()
        if regime_df.empty:
            LOG.warning(f"No data available for regime '{regime.name}'. Skipping.")
            continue

        all_results[regime.name] = _run_single_analysis(regime_df, config, regime_name=regime.name)

    return all_results


def km_survival(times: np.ndarray, event_observed: np.ndarray) -> Dict[str, Any]:
    """
    Basic Kaplan–Meier estimator.
    times: integer bars to outcome or censor time
    event_observed: 1 if outcome occurred (non-timeout break/respect/p&r), 0 if right-censored
    """
    if len(times) == 0:
        return {"curve": {"t": [], "s": []}, "median": np.nan}
    # Sort by time
    order = np.argsort(times)
    t = times[order]
    e = event_observed[order]

    uniq, idx = np.unique(t, return_index=True)
    n = len(t)
    at_risk = n
    surv = 1.0
    s_vals = []
    t_vals = []

    # Iterate unique times
    for j, start in enumerate(idx):
        tj = uniq[j]
        end = idx[j+1] if j+1 < len(idx) else len(t)
        # events and censored at time tj
        d = int(e[start:end].sum())
        c = int((1 - e[start:end]).sum())
        if at_risk > 0:
            if d > 0:
                surv *= (1 - d / at_risk)
            s_vals.append(surv)
            t_vals.append(int(tj))
            at_risk -= (d + c)
        else:
            break

    # median time to event (first time S<=0.5)
    median = np.nan
    for tt, ss in zip(t_vals, s_vals):
        if ss <= 0.5:
            median = tt
            break

    return {"curve": {"t": t_vals, "s": s_vals}, "median": median}


def hazard_table(times: np.ndarray, event: np.ndarray) -> List[Dict[str, float]]:
    """Calculates the hazard rate at each unique event time."""
    if len(times) == 0:
        return []
    t, e = np.asarray(times, int), np.asarray(event, int)
    order = np.argsort(t)
    t, e = t[order], e[order]
    uniq, idx = np.unique(t, return_index=True)
    n = len(t)
    at_risk = n
    out = []
    for j, start in enumerate(idx):
        end = idx[j + 1] if j + 1 < len(idx) else len(t)
        d = int(e[start:end].sum())
        c = int((1 - e[start:end]).sum())
        if at_risk > 0:
            out.append(
                {
                    "time": int(uniq[j]),
                    "at_risk": at_risk,
                    "events": d,
                    "censored": c,
                    "hazard": d / at_risk,
                }
            )
            at_risk -= d + c
    return out


def compute_cvar(data: np.ndarray, alpha: float = 0.95) -> Optional[float]:
    """Computes Conditional Value at Risk (CVaR) at a given alpha level."""
    if len(data) == 0:
        return None
    a = np.asarray(data)
    if len(a) < 30:
        # For small samples, this is a pragmatic tail-mean rather than a true CVaR.
        a = np.sort(a)
        k = max(0, int(np.floor(alpha * len(a))))
        # ensure at least one element in the tail
        return float(a[k:].mean() if k < len(a) else a[-1])
    var = np.percentile(a, alpha * 100)
    tail = a[a >= var]
    return float(tail.mean() if tail.size else var)


def compute_calibration_table(episodes: List[Dict]) -> List[Dict[str, Any]]:
    """Computes a calibration table for a given scoring rule."""
    if not episodes or len(episodes) < 20:
        return []

    df = pd.DataFrame(episodes)
    df["zone_type_val"] = df["zone_type"].apply(
        lambda zt: zt.value if isinstance(zt, Enum) else zt
    )
    is_support = df["zone_type_val"] == ZoneType.SUPPORT.value
    df["calibration_score"] = np.where(
        is_support, 100 - df["stoch_k_at_touch"], df["stoch_k_at_touch"]
    )

    df = df.dropna(subset=["calibration_score"])
    if len(df) < 20:
        return []

    try:
        df["score_decile"] = pd.qcut(
            df["calibration_score"].rank(method="first"),
            10,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return []

    table = []
    for decile, group in df.groupby("score_decile"):
        respect_rate = (
            group["outcome"].apply(
                lambda o: (o.value if isinstance(o, Enum) else o)
                == EpisodeOutcome.RESPECT.value
            )
        ).mean()
        table.append(
            {
                "decile": int(decile) + 1,
                "n_episodes": len(group),
                "mean_score": float(group["calibration_score"].mean()),
                "respect_rate": float(respect_rate),
            }
        )
    return table


def compute_statistics(
    episodes: List[Dict], config: EngineConfig, alpha: float = 0.05
) -> Dict[str, Any]:
    """Compute statistics with proper confidence intervals"""
    valid_episodes = [e for e in episodes if e["outcome"] != EpisodeOutcome.INVALID]
    valid_episodes = sorted(
        valid_episodes, key=lambda e: e.get("touch_time") or e.get("outcome_time")
    )
    n_invalid, n_episodes = len(episodes) - len(valid_episodes), len(valid_episodes)

    block_size = _blk(config)
    stats = {
        "n_episodes": n_episodes,
        "n_invalid_episodes": n_invalid,
        "bootstrap_block_size": block_size,
        "outcome_distribution": {},
        "outcome_rates": {},
        "outcome_rates_ci": {},
        "bootstrapped_metrics": {},
        "survival_analysis": {},
        "tail_risk": {},
        "calibration": [],
    }

    if n_episodes == 0:
        return stats

    outcomes = np.array(
        [
            (e["outcome"].value if isinstance(e["outcome"], Enum) else e["outcome"])
            for e in valid_episodes
        ]
    )
    for outcome in EpisodeOutcome:
        if outcome == EpisodeOutcome.INVALID:
            continue
        count = np.sum(outcomes == outcome.value)
        stats["outcome_distribution"][outcome.value] = int(count)
        stats["outcome_rates"][outcome.value] = (
            count / n_episodes if n_episodes > 0 else 0
        )

    # Dependence-aware CIs for outcome rates via block bootstrap
    for outcome_str, rate in stats["outcome_rates"].items():
        y = np.array([(1 if (e["outcome"].value if isinstance(e["outcome"], Enum) else e["outcome"]) == outcome_str else 0)
                      for e in valid_episodes], dtype=float)
        ci = block_bootstrap_prop_ci(
            y, n_boot=3000, block_size=_blk(config), alpha=alpha, seed=config.random_seed
        )
        stats["outcome_rates_ci"][outcome_str] = _clip01(ci)

    for metric_name in ["bars_to_outcome", "max_favorable", "max_adverse"]:
        data = np.array(
            [e[metric_name] for e in valid_episodes if e.get(metric_name) is not None]
        )
        if len(data) > 1:
            stats["bootstrapped_metrics"][metric_name] = {
                "mean": np.mean(data),
                "median": np.median(data),
                "p25": np.percentile(data, 25),
                "p75": np.percentile(data, 75),
                "mean_ci": block_bootstrap_ci(
                    data,
                    np.mean,
                    n_boot=1000,
                    block_size=_blk(config),
                    seed=config.random_seed,
                ),
                "median_ci": block_bootstrap_ci(
                    data,
                    np.median,
                    n_boot=1000,
                    block_size=_blk(config),
                    seed=config.random_seed,
                ),
            }

    bars = []; observed = []
    for e in valid_episodes:
        if "bars_to_outcome" not in e: continue
        bars.append(int(e["bars_to_outcome"]))
        # censor if timeout
        o = e["outcome"].value if isinstance(e["outcome"], Enum) else e["outcome"]
        observed.append(0 if o == EpisodeOutcome.TIMEOUT.value else 1)
    bars = np.asarray(bars, dtype=int)
    observed = np.asarray(observed, dtype=int)

    if len(bars) > 0:
        km = km_survival(bars, observed)
        stats["survival_analysis"] = {
            "median_bars_to_outcome": (
                float(km["median"]) if km["median"] == km["median"] else None
            ),
            "curve": km["curve"],
            "hazard": hazard_table(bars, observed),
        }

    adverse_excursions = np.array(
        [e["max_adverse"] for e in valid_episodes
         if ("max_adverse" in e and e["max_adverse"] is not None)],
        dtype=float,
    )
    adverse_excursions = adverse_excursions[np.isfinite(adverse_excursions)]
    if adverse_excursions.size > 0:
        stats["tail_risk"]["cvar_95_adverse_excursion"] = compute_cvar(
            adverse_excursions, alpha=0.95
        )

    stats["calibration"] = compute_calibration_table(valid_episodes)
    return stats


import sqlite3
import uuid

##############################################################################
# 8. PERSISTENCE (SQLite writer/reader)
##############################################################################


class SQLitePersistence:
    """Handles persistence of run data to a SQLite database."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = 1;")
        self.create_tables()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS runs (
            run_pk INTEGER PRIMARY KEY AUTOINCREMENT,
            run_uuid TEXT NOT NULL UNIQUE,
            run_timestamp TEXT NOT NULL,
            command TEXT,
            args_json TEXT,
            config_json TEXT,
            data_hashes_json TEXT
        )"""
        )
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS zones (
            zone_pk INTEGER PRIMARY KEY AUTOINCREMENT,
            run_fk INTEGER NOT NULL,
            regime TEXT NOT NULL,
            zone_id_in_run INTEGER NOT NULL,
            zone_type TEXT, level REAL, width REAL,
            activation_time TEXT, p_value REAL,
            FOREIGN KEY(run_fk) REFERENCES runs(run_pk) ON DELETE CASCADE
        )"""
        )
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS episodes (
            episode_pk INTEGER PRIMARY KEY AUTOINCREMENT,
            run_fk INTEGER NOT NULL,
            regime TEXT NOT NULL,
            zone_id_in_run INTEGER NOT NULL,
            zone_type TEXT,
            outcome TEXT, touch_time TEXT, outcome_time TEXT,
            bars_to_outcome INTEGER, max_favorable REAL, max_adverse REAL, touch_number INTEGER,
            FOREIGN KEY(run_fk) REFERENCES runs(run_pk) ON DELETE CASCADE
        )"""
        )
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS statistics (
            stat_pk INTEGER PRIMARY KEY AUTOINCREMENT,
            run_fk INTEGER NOT NULL,
            cohort TEXT NOT NULL,
            stats_json TEXT NOT NULL,
            FOREIGN KEY(run_fk) REFERENCES runs(run_pk) ON DELETE CASCADE
        )"""
        )
        self.conn.commit()

    def insert_run(self, metadata: Dict) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO runs (run_uuid, run_timestamp, command, args_json, config_json, data_hashes_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                metadata["run_uuid"],
                metadata["run_timestamp_utc"],
                metadata["command"],
                json.dumps(metadata["args"]),
                json.dumps(metadata["config"]),
                json.dumps(metadata["data_sha256"]),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_zones(self, run_pk: int, zones: List[Zone]):
        if not zones:
            return
        z_data = [
            (
                run_pk,
                getattr(z, 'regime', 'all_data'),
                z.id,
                z.type.value,
                z.level,
                z.width,
                z.activation_time.isoformat(),
                z.p_value,
            )
            for z in zones
        ]
        self.conn.cursor().executemany(
            "INSERT INTO zones (run_fk, regime, zone_id_in_run, zone_type, level, width, activation_time, p_value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            z_data,
        )
        self.conn.commit()

    def insert_episodes(self, run_pk: int, episodes: List[Dict]):
        if not episodes:
            return
        e_data = [
            (
                run_pk,
                e.get("regime", "all_data"),
                e["zone_id"],
                e["zone_type"].value if isinstance(e["zone_type"], Enum) else e["zone_type"],
                e["outcome"].value if isinstance(e["outcome"], Enum) else e["outcome"],
                (e["touch_time"].isoformat() if hasattr(e["touch_time"], "isoformat") else str(e["touch_time"])),
                (e["outcome_time"].isoformat() if hasattr(e["outcome_time"], "isoformat") else str(e["outcome_time"])),
                e["bars_to_outcome"],
                e["max_favorable"],
                e["max_adverse"],
                e["touch_number"],
            )
            for e in episodes
        ]
        self.conn.cursor().executemany(
            "INSERT INTO episodes (run_fk, regime, zone_id_in_run, zone_type, outcome, touch_time, outcome_time, bars_to_outcome, max_favorable, max_adverse, touch_number) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            e_data,
        )
        self.conn.commit()

    def insert_statistics(self, run_pk: int, stats: Dict[str, Any]):
        if not stats:
            return
        s_data = [
            (run_pk, cohort, json.dumps(cohort_stats, default=str))
            for cohort, cohort_stats in stats.items()
        ]
        self.conn.cursor().executemany(
            "INSERT INTO statistics (run_fk, cohort, stats_json) VALUES (?, ?, ?)",
            s_data,
        )
        self.conn.commit()


##############################################################################
# 9. PLOTTING (pngs only)
##############################################################################


def save_distribution_plots(episodes_df: pd.DataFrame, out_dir: str, regime_name: str):
    """Generates and saves distribution plots for key episode metrics."""
    if episodes_df.empty:
        return
    LOG.info(f"[{regime_name}] Generating distribution plots in {out_dir}...")
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        try:
            plt.style.use("seaborn-darkgrid")
        except Exception:
            pass  # fall back to default

    outcome_labels = {
        "RESPECT": "Respect",
        "PIERCE_AND_REVERT": "Pierce -> Revert",
        "BREAK": "Break",
        "TIMEOUT": "Timeout",
        "INVALID": "Invalid",
    }
    episodes_df["outcome_str"] = episodes_df["outcome"].apply(
        lambda x: outcome_labels.get(x.value if isinstance(x, Enum) else x, x)
    )
    plt.figure(figsize=(10, 6))
    episodes_df["outcome_str"].value_counts().plot(kind="bar")
    plt.title(f"Episode Outcome Distribution (Regime: {regime_name})")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"dist_outcomes_{regime_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    q = episodes_df["bars_to_outcome"].quantile(0.99)
    q = float(q) if np.isfinite(q) and q > 0 else episodes_df["bars_to_outcome"].max()
    episodes_df["bars_to_outcome"].hist(bins=50, range=(0, q))
    plt.title(f"Distribution of Bars to Outcome (Regime: {regime_name})")
    plt.xlabel("Number of Bars to Outcome (bars)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"dist_bars_to_outcome_{regime_name}.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    q_fav = episodes_df["max_favorable"].quantile(0.99)
    q_fav = float(q_fav) if np.isfinite(q_fav) and q_fav > 0 else episodes_df["max_favorable"].max()
    episodes_df["max_favorable"].hist(bins=50, color="C2", range=(0, q_fav))
    plt.title("Max Favorable Excursion")
    plt.xlabel("Excursion (points)")
    plt.subplot(1, 2, 2)
    q_adv = episodes_df["max_adverse"].quantile(0.99)
    q_adv = float(q_adv) if np.isfinite(q_adv) and q_adv > 0 else episodes_df["max_adverse"].max()
    episodes_df["max_adverse"].hist(bins=50, color="C3", range=(0, q_adv))
    plt.title("Max Adverse Excursion")
    plt.xlabel("Excursion (points)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"dist_excursions_{regime_name}.png"))
    plt.close()
    LOG.info(f"[{regime_name}] Distribution plots saved.")


def save_daily_maps(
    df: pd.DataFrame,
    zones: List[Zone],
    episodes: List[Dict],
    out_dir: str,
    regime_name: str,
    max_maps: int = 100,
):
    """Generates and saves daily price charts with zones and episodes."""
    if df.empty or not zones:
        return
    maps_dir = os.path.join(out_dir, "daily_maps", regime_name)
    Path(maps_dir).mkdir(parents=True, exist_ok=True)
    LOG.info(f"[{regime_name}] Generating daily maps in {maps_dir} (limit: {max_maps})...")
    episodes_df = pd.DataFrame(episodes)
    if not episodes_df.empty:
        episodes_df["touch_time"], episodes_df["outcome_time"] = pd.to_datetime(
            episodes_df["touch_time"]
        ), pd.to_datetime(episodes_df["outcome_time"])
    local_tz = df["local_time"].dt.tz or "UTC"

    def to_local_date(ts):
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.tz_convert(local_tz).date()

    map_count = 0
    # Process sessions in reverse to get the most recent maps first
    for session, day_df in sorted(df.groupby("session_date"), key=lambda x: x[0], reverse=True):
        if map_count >= max_maps:
            LOG.info(f"Generated {map_count} of {max_maps} maps. Use --max-maps to generate more.")
            break
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(
            day_df["timestamp"],
            day_df["close"],
            label="Close",
            color="black",
            linewidth=0.5,
        )
        for zone in zones:
            if (
                to_local_date(zone.activation_time)
                <= session
                <= to_local_date(
                    zone.activation_time + pd.Timedelta(days=zone.expire_days)
                )
            ):
                color = "C0" if zone.type == ZoneType.SUPPORT else "C1"
                linestyle = "--" if zone.type == ZoneType.SUPPORT else "-."
                ax.axhspan(
                    zone.level - zone.width,
                    zone.level + zone.width,
                    alpha=0.1,
                    color=color,
                )
                ax.axhline(zone.level, color=color, linestyle=linestyle, linewidth=0.7)
        if not episodes_df.empty:
            day_episodes = episodes_df[episodes_df["touch_time"].dt.date == session]
            for _, episode in day_episodes.iterrows():
                outcome_str = (
                    episode["outcome"].value
                    if isinstance(episode["outcome"], Enum)
                    else episode["outcome"]
                )
                outcome_color = {
                    "RESPECT": "blue",
                    "BREAK": "orange",
                    "PIERCE_AND_REVERT": "purple",
                }.get(outcome_str, "grey")
                ix = df["timestamp"].searchsorted(episode["touch_time"])
                ix = int(np.clip(ix, 1, len(df) - 1))
                cand = df.iloc[[ix - 1, ix]]
                row = cand.iloc[
                    (cand["timestamp"] - episode["touch_time"]).abs().values.argmin()
                ]
                touch_price = row["close"]
                ax.scatter(
                    episode["touch_time"],
                    touch_price,
                    color=outcome_color,
                    s=50,
                    zorder=5,
                    marker="o",
                )
                ax.text(
                    episode["outcome_time"],
                    touch_price,
                    f"{outcome_str} (T{episode['touch_number']})",
                    color=outcome_color,
                    fontsize=9,
                )
        ax.set_title(f"Market Map for {session.strftime('%Y-%m-%d')}")
        ax.set_ylabel("Price")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(maps_dir, f"map_{session.strftime('%Y-%m-%d')}.png"))
        plt.close(fig)
        map_count += 1
    LOG.info("Daily maps saved.")


def save_respect_rate_heatmap(*args, **kwargs):
    """Placeholder for R/O heatmap generation."""
    LOG.debug("Heatmap generation is not implemented for single runs in this version.")
    pass


##############################################################################
# 10. REPORT (single-file HTML with embedded PNGs)
##############################################################################
# To be added in a future step.

##############################################################################
# 11. CLI & MAIN
##############################################################################


def load_engine_config(path: Optional[str]) -> EngineConfig:
    """Loads engine config from YAML/JSON, overriding defaults."""
    cfg = EngineConfig()
    if not path or not os.path.exists(path):
        if path:
            LOG.warning(f"Config file not found at {path}, using defaults.")
        return cfg
    LOG.info(f"Loading config from {path}")
    with open(path, "r") as f:
        if path.lower().endswith((".yml", ".yaml")):
            if not HAVE_YAML:
                raise ImportError(
                    "pyyaml is required for YAML configs. `pip install pyyaml`"
                )
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    update_dataclass_from_dict(cfg, raw)
    return cfg


def walk_forward_slices(
    df: pd.DataFrame, window: str, step: str, test_size: Optional[str] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generates walk-forward training and testing slices."""
    if df.empty:
        return []
    window_td, step_td = pd.to_timedelta(window), pd.to_timedelta(step)
    test_td = pd.to_timedelta(test_size) if test_size else step_td
    slices, start_date, end_date = [], df["timestamp"].min(), df["timestamp"].max()
    train_start = start_date
    while train_start + window_td + test_td <= end_date:
        train_end, test_end = train_start + window_td, train_start + window_td + test_td
        train_df = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)]
        test_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < test_end)]
        if not train_df.empty and not test_df.empty:
            slices.append((train_df.copy(), test_df.copy()))
        train_start += step_td
    return slices


def run_walk_forward_analysis(
    df: pd.DataFrame, config: EngineConfig, wf_params: Dict[str, str]
):
    """Runs the analysis pipeline using walk-forward validation."""
    slices = walk_forward_slices(df, window=wf_params["window"], step=wf_params["step"],
                             test_size=wf_params.get("test"))
    if not slices:
        LOG.error("Could not generate any walk-forward slices from the data.")
        return None
    LOG.info(f"Starting walk-forward analysis with {len(slices)} slices.")

    all_oos_episodes, per_slice_stats = [], []
    for i, (train_df, test_df) in enumerate(slices):
        LOG.info(
            f"--- Processing slice {i+1}/{len(slices)}: Train {train_df['timestamp'].min().date()}->{train_df['timestamp'].max().date()}, Test {test_df['timestamp'].min().date()}->{test_df['timestamp'].max().date()} ---"
        )
        zones = discover_zones(train_df, config, "training")
        if not zones:
            LOG.warning("No zones discovered in training period. Skipping slice.")
            continue

        # Evaluate episodes on the full test_df (for 'all_data' regime)
        oos_episodes_all = evaluate_episodes(test_df, zones, config)
        if oos_episodes_all:
            all_oos_episodes.extend(oos_episodes_all)
            slice_stats = compute_statistics(oos_episodes_all, config)
            per_slice_stats.append(
                {
                    "slice_num": i + 1,
                    "train_start": str(train_df["timestamp"].min()),
                    "train_end": str(train_df["timestamp"].max()),
                    "test_start": str(test_df["timestamp"].min()),
                    "test_end": str(test_df["timestamp"].max()),
                    "stats": slice_stats,
                }
            )

    LOG.info("--- Walk-Forward Analysis Complete ---")
    # This part remains tricky with regimes. For now, we aggregate all OOS episodes
    # across all regimes for a single, final statistic. A more advanced version
    # might aggregate them per-regime.
    if config.regime_config and config.regime_config.enabled:
        LOG.warning("Regime analysis in walk-forward mode is experimental.")
        # This is a simplification. A full implementation would track OOS episodes per regime.
        # For now, we just run on all data for the final report.
        aggregated_stats = compute_statistics(all_oos_episodes, config)
        return {
            "all_data": {
                 "aggregated_statistics": aggregated_stats,
                 "per_slice_statistics": per_slice_stats,
                 "all_oos_episodes": all_oos_episodes,
            }
        }
    else:
        aggregated_stats = compute_statistics(all_oos_episodes, config)
        return {
            "all_data": {
                 "aggregated_statistics": aggregated_stats,
                 "per_slice_statistics": per_slice_stats,
                 "all_oos_episodes": all_oos_episodes,
            }
        }


import base64


def save_cohort_lift_plot(statistics: Dict[str, Any], out_dir: str, regime_name: str):
    """
    Saves a plot showing the lift in respect rate for different cohorts
    compared to the baseline for that regime.
    """
    all_cohort_stats = statistics.get("all")
    if not all_cohort_stats: return

    base_respect_rate = all_cohort_stats.get("outcome_rates", {}).get("RESPECT", 0.0)
    if base_respect_rate == 0.0:
        return

    cohort_lifts = []
    for name, stats in statistics.items():
        if name == "all" or stats["n_episodes"] < MIN_EPISODES_FOR_STATS:
            continue
        cohort_rate = stats["outcome_rates"].get("RESPECT", 0.0)
        lift = cohort_rate - base_respect_rate
        cohort_lifts.append({"name": name, "lift": lift})

    if not cohort_lifts:
        return

    top_n = sorted(cohort_lifts, key=lambda x: x["lift"], reverse=True)[:10]
    df = pd.DataFrame(top_n).set_index("name")

    plt.figure(figsize=(10, 8))
    df["lift"].plot(
        kind="barh", color=df["lift"].apply(lambda x: "C0" if x > 0 else "C1")
    )
    plt.title(f"Top 10 Cohorts by Respect Rate Lift (Regime: {regime_name})")
    plt.xlabel("Lift over Baseline Respect Rate (pp)")
    plt.axvline(0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_cohort_lift_{regime_name}.png"))
    plt.close()


def save_respect_rate_ci_plot(stats: Dict[str, Any], out_dir: str, regime: str):
    s = stats.get("all", {})
    rr = s.get("outcome_rates", {}).get("RESPECT")
    ci = s.get("outcome_rates_ci", {}).get("RESPECT")
    if rr is None or not ci or ci[0] is None: return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.errorbar([0], [rr], yerr=[[max(0, rr - ci[0])], [max(0, ci[1] - rr)]], fmt="o", capsize=6)
    plt.xticks([0], ["Respect rate"])
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title(f"Respect Rate ±95% CI (Regime: {regime})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"respect_rate_ci_{regime}.png"))
    plt.close()


def generate_html_report(all_results: Dict[str, Any], out_dir: str, config: EngineConfig):
    """Generates a single-file HTML report with embedded images."""

    def embed_img(path: str) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f'<img src="data:image/png;base64,{encoded}" alt="{os.path.basename(path)}" style="width:100%; max-width:600px;">'

    def _fmt_ci_html(ci_tuple):
        return "N/A" if not ci_tuple or ci_tuple[0] is None else f"{ci_tuple[0]:.2%} – {ci_tuple[1]:.2%}"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Market Stats Report: {config.instrument.symbol}</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            h1, h2, h3 {{ color: #333; }}
            hr {{ margin: 2em 0; }}
            table {{ border-collapse: collapse; width: 100%; max-width: 900px; margin-bottom: 2em; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .summary-table td {{ font-size: 1.1em; text-align: center; }}
            .summary-table th {{ font-size: 0.9em; font-weight: normal; color: #666; }}
            .footer {{ margin-top: 2em; font-size: 0.8em; color: #666; }}
        </style>
    </head>
    <body>
        <h1>Market Statistics Report: {config.instrument.symbol}</h1>
        <p class="footer">Generated on {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    # --- Add Method Notes ---
    num_tests = 0
    # Find the number of tests from the first regime that has them
    for res in all_results.values():
        stats = res.get("statistics", {})
        if stats:
            num_tests = sum(
                1
                for cohort in stats.values()
                if "comparative_respect_rate" in cohort
                and "q_value" in cohort.get("comparative_respect_rate", {})
            )
            if num_tests > 0:
                break

    block_size_str = (
        str(config.bootstrap_block_size)
        if config.bootstrap_block_size is not None
        else f"dynamic (T//2, default: {_blk(config)})"
    )
    timeout_treatment = "BREAK" if config.episode.treat_timeout_as_break else "TIMEOUT"

    html += f"""
        <h3>Method Notes</h3>
        <ul>
            <li><b>Censoring Policy:</b> Episodes are censored at session boundaries. Timeouts are treated as <b>{timeout_treatment}</b>.</li>
            <li><b>Bootstrap Block Size:</b> {block_size_str} bars used for CIs.</li>
            <li><b>Multiple Comparisons:</b> For each regime, {num_tests} cohort respect rates were compared against the baseline. p-values were adjusted for False Discovery Rate (FDR) using the Benjamini/Hochberg method (q-values reported).</li>
        </ul>
    """

    for regime_name, results in all_results.items():
        all_stats = results.get("statistics", {}).get("all", {})
        if not all_stats:
            continue

        rr = all_stats.get("outcome_rates", {}).get("RESPECT", float('nan'))
        ci = all_stats.get("outcome_rates_ci", {}).get("RESPECT")
        med = all_stats.get("survival_analysis", {}).get("median_bars_to_outcome", "N/A")
        cvar = all_stats.get("tail_risk", {}).get("cvar_95_adverse_excursion", float('nan'))

        html += f"""
            <hr>
            <h2>Regime: {regime_name}</h2>
            <p><i><b>Method Notes for this regime:</b> Bootstrap block size: {all_stats.get('bootstrap_block_size', 'N/A')} bars.</i></p>

            <h3>Summary Metrics</h3>
            <table class="summary-table">
                <tr>
                    <th>Total Episodes</th>
                    <th>Respect Rate (95% CI)</th>
                    <th>Median Bars to Outcome</th>
                    <th>CVaR95 Adverse Excursion</th>
                </tr>
                <tr>
                    <td>{all_stats.get('n_episodes', 'N/A')}</td>
                    <td>{rr:.2%} ({_fmt_ci_html(ci)})</td>
                    <td>{med}</td>
                    <td>{cvar:.2f} points</td>
                </tr>
            </table>

            <h3>Comparative Cohort Analysis</h3>
            <p>The following cohorts showed a statistically significant difference in respect rate compared to the baseline after FDR correction (q < {ZONE_SIGNIFICANCE_ALPHA}).</p>
            <table>
                <tr>
                    <th>Cohort</th>
                    <th>N</th>
                    <th>Respect Rate (95% CI)</th>
                    <th>Risk Diff (95% CI)</th>
                    <th>q-value</th>
                </tr>
        """

        significant_cohorts = []
        for name, stats in results["statistics"].items():
            comp = stats.get("comparative_respect_rate")
            if comp and comp.get("reject_h0"):
                rr_cohort = stats.get("outcome_rates", {}).get("RESPECT", float('nan'))
                rr_ci = stats.get("outcome_rates_ci", {}).get("RESPECT")
                rd_ci = comp.get("risk_difference_ci")
                significant_cohorts.append({
                    "name": name,
                    "n": stats.get("n_episodes"),
                    "rr_str": f"{rr_cohort:.2%} ({_fmt_ci_html(rr_ci)})",
                    "rd_str": f"{comp.get('risk_difference', 0):+.2%} ({_fmt_ci_html(rd_ci)})",
                    "q_value": f"{comp.get('q_value', 0):.3f}",
                })

        if significant_cohorts:
            for cohort in sorted(significant_cohorts, key=lambda x: x["q_value"]):
                html += f"""
                    <tr>
                        <td>{cohort['name']}</td>
                        <td>{cohort['n']}</td>
                        <td>{cohort['rr_str']}</td>
                        <td>{cohort['rd_str']}</td>
                        <td>{cohort['q_value']}</td>
                    </tr>
                """
        else:
            html += "<tr><td colspan='5' style='text-align:center;'>No cohorts showed a statistically significant difference.</td></tr>"

        html += f"""
            </table>

            <h3>Outcome & Excursion Distributions</h3>
            {embed_img(os.path.join(out_dir, "charts", f'dist_outcomes_{regime_name}.png'))}
            {embed_img(os.path.join(out_dir, "charts", f'dist_excursions_{regime_name}.png'))}
            {embed_img(os.path.join(out_dir, "charts", f'respect_rate_ci_{regime_name}.png'))}

            <h3>Top Cohorts by Lift</h3>
            {embed_img(os.path.join(out_dir, "charts", f'plot_cohort_lift_{regime_name}.png'))}
        """

    # --- Add Glossary and Data Links ---
    html += f"""
        <hr>
        <h3>Glossary of Terms</h3>
        <dl>
            <dt><strong>Respect</strong></dt>
            <dd>Price touches the zone, exits favorably, and travels at least R points away without first breaking the zone.</dd>
            <dt><strong>Pierce → Revert</strong></dt>
            <dd>Price touches the zone, pierces into it (but not beyond the overshoot O), then reverts to travel R points in the favorable direction.</dd>
            <dt><strong>Break</strong></dt>
            <dd>Price closes beyond the zone plus the overshoot tolerance O.</dd>
            <dt><strong>Timeout</strong></dt>
            <dd>The episode is censored, either by reaching the maximum T bars or by a session boundary change, before a clear outcome is observed.</dd>
        </dl>

        <h3>Data Artifacts</h3>
        <p>The following data files were generated alongside this report:</p>
        <ul>
            <li><a href="episodes.csv">episodes.csv</a></li>
            <li><a href="zones.csv">zones.csv</a></li>
            <li><a href="statistics.json">statistics.json</a></li>
            <li><a href="summary.csv">summary.csv</a></li>
            <li><a href="episodes_schema.json">episodes_schema.json</a></li>
        </ul>
    """

    html += """
    </body>
    </html>
    """

    report_path = os.path.join(out_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html)
    LOG.info(f"HTML report saved to {report_path}")


def parse_wf_params(wf_string: str) -> Optional[Dict[str, str]]:
    """Parses walk-forward parameters from a string."""
    params = {}
    if not wf_string:
        return None
    try:
        for part in wf_string.replace(" ", "").split(","):
            key, value = part.split("=")
            params[key.strip()] = value.strip()
        if "window" not in params or "step" not in params:
            raise ValueError("window and step are required parameters.")
        return params
    except Exception as e:
        LOG.error(f"Invalid --wf format: '{wf_string}'. Error: {e}")
        LOG.error("Example of valid format: --wf 'window=90d,step=30d,test=30d'")
        LOG.info(f"Parsed values: {params}")
        return None


def get_episodes_schema() -> Dict[str, str]:
    """Returns a dictionary describing the columns of the episodes output file."""
    return {
        "zone_id": "Unique identifier for the zone that this episode belongs to.",
        "zone_type": "Type of the zone: SUPPORT or RESISTANCE.",
        "outcome": "The result of the episode: RESPECT, PIERCE_AND_REVERT, BREAK, TIMEOUT, or INVALID.",
        "touch_time": "Timestamp of the first bar that touched the zone (ISO-8601 format).",
        "outcome_time": "Timestamp of the bar where the outcome was decided (ISO-8601 format).",
        "bars_to_outcome": "The number of bars from the touch to the outcome.",
        "max_favorable": "Maximum price excursion in points in the favorable direction after the touch.",
        "max_adverse": "Maximum price excursion in points in the adverse direction after the touch.",
        "from_above": "Whether the price approached the zone from above (True), below (False), or from within (None).",
        "end_idx": "The index of the outcome bar in the source dataframe.",
        "touch_number": "The sequential number of this touch for the given zone.",
        "regime": "The market regime under which this episode occurred.",
        "rsi_at_touch": "RSI value at the time of the touch.",
        "atr_at_touch": "ATR value at the time of the touch.",
        "rvol_at_touch": "Relative volume at the time of the touch.",
        "stoch_k_at_touch": "Stochastic %K value at the time of the touch.",
        "stoch_d_at_touch": "Stochastic %D value at the time of the touch.",
        "is_rth_at_touch": "Whether the touch occurred during Regular Trading Hours (RTH).",
        "minute_of_day_at_touch": "The minute of the day (in session-local time) at the time of the touch.",
        "hour_at_touch": "The hour of the day (in session-local time) at the time of the touch.",
        "atr_tercile": "The tercile (low, mid, high) of the ATR at touch, relative to other episodes.",
        "rvol_tercile": "The tercile (low, mid, high) of the relative volume at touch.",
        "stoch_tercile": "The tercile (low, mid, high) of the stochastic %K at touch.",
        "zone_age_days": "The age of the zone in days at the time of the touch.",
        "approach_direction": "A more descriptive version of 'from_above': from_above, from_below, or inside.",
        "rsi_at_outcome": "RSI value at the time of the outcome.",
        "atr_at_outcome": "ATR value at the time of the outcome.",
        "rvol_at_outcome": "Relative volume at the time of the outcome.",
        "stoch_k_at_outcome": "Stochastic %K value at the time of the outcome.",
        "stoch_d_at_outcome": "Stochastic %D value at the time of the outcome.",
    }

def save_episodes_schema(out_dir: str):
    """Saves the episodes schema to a JSON file."""
    schema = get_episodes_schema()
    path = os.path.join(out_dir, "episodes_schema.json")
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    LOG.info(f"Saved episode schema to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market_stats_engine",
        description="Production-ready market statistics engine",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity (-v or -vv)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only show warnings and errors")

    subparsers = parser.add_subparsers(dest="command", help="Commands", required=True)

    # --- Prepare Command ---
    prep_parser = subparsers.add_parser("prepare", help="Prepare and validate data")
    prep_parser.add_argument(
        "--files", nargs="+", required=True, help="Input files (CSV or Parquet)"
    )
    prep_parser.add_argument(
        "--out", default="cache/prepared.parquet", help="Output file"
    )
    prep_parser.add_argument("--config", help="Engine config file (YAML/JSON)")

    # --- Analyze Command ---
    analyze_parser = subparsers.add_parser("analyze", help="Run statistical analysis")
    analyze_parser.add_argument("--data", required=True, help="Prepared data file")
    analyze_parser.add_argument("--config", help="Engine config file (YAML/JSON)")
    analyze_parser.add_argument("--out", help="Output directory. Defaults to a unique path in 'runs/'.")
    analyze_parser.add_argument(
        "--wf",
        help='Enable walk-forward validation. e.g., "window=90d,step=30d[,test=30d]"'
    )
    analyze_parser.add_argument(
        "--db", help="Path to SQLite database for results lineage."
    )
    analyze_parser.add_argument(
        "--report", action="store_true", help="Generate a single-file HTML report."
    )
    analyze_parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible results."
    )
    analyze_parser.add_argument(
        "--max-maps",
        type=int,
        default=30,
        help="Max number of daily maps to generate (most recent).",
    )
    analyze_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and plan without executing.",
    )
    analyze_parser.add_argument(
        "--engine",
        default="pandas",
        choices=["pandas", "fast"],
        help="Execution engine ('fast' enables polars).",
    )
    analyze_parser.add_argument(
        "--regimes", help="Path to YAML file defining market regimes."
    )
    analyze_parser.add_argument(
        "--block-size", type=int, help="Override for bootstrap block size."
    )
    analyze_parser.add_argument("--no-plots", action="store_true", help="Suppress generation of all plot PNGs.")
    analyze_parser.add_argument("--no-maps", action="store_true", help="Suppress generation of daily map charts.")

    # --- Sweep Command ---
    sweep_parser = subparsers.add_parser("sweep", help="Parameter sweep")
    sweep_parser.add_argument("--data", required=True, help="Prepared data file")
    sweep_parser.add_argument(
        "--grid", required=True, help="Parameter grid file (JSON or YAML)"
    )
    sweep_parser.add_argument(
        "--out", help="Output directory. Defaults to a unique path in 'sweep_results/'."
    )
    sweep_parser.add_argument(
        "--db", help="Path to SQLite database for results lineage."
    )
    sweep_parser.add_argument(
        "--jobs", type=int, default=1, help="Number of parallel jobs for sweep."
    )
    sweep_parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible results."
    )

    # --- Self-test Command ---
    selftest_parser = subparsers.add_parser(
        "self-test", help="Run a quick synthetic data test."
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.quiet:
        LOG.setLevel(logging.WARNING)
    elif args.verbose == 1:
        LOG.setLevel(logging.INFO)
    elif args.verbose >= 2:
        LOG.setLevel(logging.DEBUG)

    if args.command == "self-test":
        run_self_test()
        return

    config = load_engine_config(getattr(args, "config", None))
    if hasattr(args, "seed") and args.seed is not None:
        config.random_seed = args.seed
        LOG.info(f"Using random seed: {config.random_seed}")

    if hasattr(args, "block_size") and args.block_size is not None:
        config.bootstrap_block_size = args.block_size
        LOG.info(f"Using manual bootstrap block size: {config.bootstrap_block_size}")

    # Deterministic RNG
    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    if hasattr(args, "regimes") and args.regimes and os.path.exists(args.regimes):
        if not HAVE_YAML:
            raise ImportError("pyyaml is required for --regimes. `pip install pyyaml`")
        LOG.info(f"Loading regime definition from {args.regimes}")
        try:
            with open(args.regimes, 'r') as f:
                regime_data = yaml.safe_load(f) or {}

            regime_config = RegimeConfig()
            if 'enabled' in regime_data:
                regime_config.enabled = bool(regime_data['enabled'])

            if 'regimes' in regime_data and isinstance(regime_data['regimes'], list):
                regime_config.regimes = [Regime(**r) for r in regime_data['regimes']]

            config.regime_config = regime_config
        except Exception as e:
            LOG.error(f"Failed to load or parse regime config: {e}", exc_info=True)
            # Decide if we should exit or just continue without regimes
            LOG.warning("Continuing analysis without market regimes due to config error.")

    if hasattr(args, "engine") and args.engine == "fast": # Not used yet
        if HAVE_POLARS:
            LOG.info("Using 'fast' engine (Polars where available).")
        else:
            LOG.warning(
                "Engine 'fast' selected but Polars not found. Falling back to pandas."
            )

    run_uuid = str(uuid.uuid4())
    if hasattr(args, "out") and not args.out:
        ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.command == 'analyze':
            symbol = config.instrument.symbol
            args.out = f"runs/{symbol}_{ts}"
        elif args.command == 'sweep':
            args.out = f"sweep_results/sweep_{ts}"
        LOG.info(f"No output directory specified. Using default: {args.out}")

    if hasattr(args, "dry_run") and args.dry_run:
        LOG.info("--- Dry Run Mode ---")
        LOG.info("Configuration:")
        LOG.info(json.dumps(dataclasses.asdict(config), indent=2, default=str))
        LOG.info(f"Command to be executed: {args.command}")
        return

    persistence = None
    try:
        if getattr(args, "db", None):
            persistence = SQLitePersistence(args.db)
            LOG.info(f"Initialized SQLite persistence at {args.db}")

        out_dir = args.out if hasattr(args, "out") else "results"
        if args.command == "prepare":
            out_dir = str(Path(args.out).parent)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # File logging
        for h in list(LOG.handlers):
            if isinstance(h, logging.FileHandler):
                LOG.removeHandler(h)
        fh = logging.FileHandler(os.path.join(out_dir, "log.txt"))
        fh.setFormatter(formatter)
        LOG.addHandler(fh)

        base_metadata = {
            "run_uuid": run_uuid,
            "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": args.command,
            "args": vars(args),
            "code_sha256": get_file_sha256(__file__),
            "data_sha256": {
                f: get_file_sha256(f)
                for f in (
                    args.files
                    if hasattr(args, "files") and args.files
                    else (getattr(args, "data", None) and [args.data])
                )
                if f
            },
            "config": dataclasses.asdict(config),
        }
        with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
            json.dump(base_metadata, f, indent=2, default=str)
        LOG.info(f"Saved run metadata to {os.path.join(out_dir, 'run_metadata.json')}")

        manifest = {
            "code_sha256": base_metadata["code_sha256"],
            "config_sha256": hashlib.sha256(json.dumps(base_metadata["config"], sort_keys=True, default=str).encode()).hexdigest(),
            "data_sha256": base_metadata["data_sha256"],
            "run_uuid": base_metadata["run_uuid"],
            "timestamp_utc": base_metadata["run_timestamp_utc"],
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        if args.command == "prepare":
            df_iter = (
                tqdm(args.files, desc="Preparing files")
                if tqdm and args.files
                else args.files
            )
            df = prepare_data(df_iter, config)
            out_path = args.out
            if out_path.endswith(".parquet"):
                if HAVE_PARQUET:
                    df.to_parquet(out_path, index=False)
                else:
                    out_path = out_path.replace(".parquet", ".csv")
                    df.to_csv(out_path, index=False)
                    LOG.warning("pyarrow not found.")
            else:
                df.to_csv(out_path, index=False)
            LOG.info(f"Saved prepared data to {out_path}, shape: {df.shape}")

        elif args.command == "analyze":
            df = (
                pd.read_parquet(args.data)
                if args.data.endswith(".parquet")
                else pd.read_csv(args.data)
            )
            df = ensure_prepared(df, config)
            run_pk = persistence.insert_run(base_metadata) if persistence else None

            if args.wf:
                wf_params = parse_wf_params(args.wf)
                if not wf_params:
                    return
                wf_results = run_walk_forward_analysis(df, config, wf_params)
                if wf_results and "all_data" in wf_results:
                    # Simplified handling for walk-forward with regimes
                    # We report the aggregate of all OOS episodes regardless of regime
                    agg_results = wf_results["all_data"]
                    with open(f"{args.out}/statistics_walkforward.json", "w") as f:
                        json.dump(
                            agg_results["aggregated_statistics"], f, indent=2, default=str
                        )
                    with open(f"{args.out}/statistics_walkforward_slices.json", "w") as f:
                        json.dump(
                            agg_results["per_slice_statistics"], f, indent=2, default=str
                        )
                    episodes_df = pd.DataFrame(agg_results["all_oos_episodes"]).copy()
                    for col in ("outcome", "zone_type"):
                        if col in episodes_df.columns:
                            episodes_df[col] = episodes_df[col].apply(lambda x: x.value if isinstance(x, Enum) else x)
                    for col in ("touch_time", "outcome_time"):
                        if col in episodes_df.columns:
                            episodes_df[col] = pd.to_datetime(episodes_df[col])
                    episodes_df.to_csv(
                        f"{args.out}/episodes_walkforward.csv", index=False
                    )
                    if persistence and run_pk:
                        persistence.insert_statistics(
                            run_pk,
                            {
                                "aggregated_walkforward": agg_results[
                                     "aggregated_statistics"
                                 ]
                            },
                        )
                        persistence.insert_episodes(
                            run_pk, agg_results["all_oos_episodes"]
                        )
                    agg_stats = agg_results["aggregated_statistics"]
                    if agg_stats["n_episodes"] > 0:
                        respect_rate = agg_stats["outcome_rates"].get("RESPECT", 0.0)
                        ci = agg_stats["outcome_rates_ci"].get("RESPECT", (None, None))
                        LOG.info("--- OOS Walk-Forward Results ---")
                        LOG.info(
                            f"Overall OOS Respect rate: {respect_rate:.2%} (CI: {f'({ci[0]:.2%}, {ci[1]:.2%})' if ci[0] is not None else 'N/A'})"
                        )
            else:
                all_results = run_analysis(df, config)

                # Save combined results
                all_zones, all_episodes, all_stats_flat = [], [], {}
                for regime_name, results in all_results.items():
                    # Tag episodes and zones with their regime
                    for z in results["zones"]: z.regime = regime_name
                    for e in results["episodes"]: e["regime"] = regime_name
                    all_zones.extend(results["zones"])
                    all_episodes.extend(results["episodes"])
                    for cohort, stats in results["statistics"].items():
                        all_stats_flat[f"{regime_name}_{cohort}"] = stats

                zones_df = pd.DataFrame([dataclasses.asdict(z) for z in all_zones])
                if not zones_df.empty:
                    zones_df["type"] = zones_df["type"].apply(lambda x: x.value if isinstance(x, Enum) else x)
                    zones_df["activation_time"] = pd.to_datetime(zones_df["activation_time"])
                zones_df.to_csv(f"{args.out}/zones.csv", index=False)

                episodes_df = pd.DataFrame(all_episodes).copy()
                if not episodes_df.empty:
                    for col in ("outcome", "zone_type"):
                        if col in episodes_df.columns:
                            episodes_df[col] = episodes_df[col].apply(lambda x: x.value if isinstance(x, Enum) else x)
                    for col in ("touch_time", "outcome_time"):
                        if col in episodes_df.columns:
                            episodes_df[col] = pd.to_datetime(episodes_df[col])

                episodes_df.to_csv(f"{args.out}/episodes.csv", index=False)
                if HAVE_PARQUET:
                    if not episodes_df.empty:
                        episodes_df.to_parquet(f"{args.out}/episodes.parquet", index=False)
                    if not zones_df.empty:
                        zones_df.to_parquet(f"{args.out}/zones.parquet", index=False)
                with open(f"{args.out}/statistics.json", "w") as f:
                    json.dump(all_stats_flat, f, indent=2, default=str)

                LOG.info(f"Results saved to {args.out}. Total Zones: {len(all_zones)}, Total Episodes: {len(all_episodes)}")

                if persistence and run_pk:
                    persistence.insert_zones(run_pk, all_zones)
                    persistence.insert_episodes(run_pk, all_episodes)
                    persistence.insert_statistics(run_pk, all_stats_flat)

                for regime_name, results in all_results.items():
                    if "all" in results["statistics"] and results["statistics"]["all"]["n_episodes"] > 0:
                        all_stats = results["statistics"]["all"]
                        respect_rate = all_stats["outcome_rates"].get("RESPECT", 0.0)
                        ci = all_stats["outcome_rates_ci"].get("RESPECT")
                        LOG.info(
                            f"[{regime_name}] Respect rate: {respect_rate:.2%} (CI: {f'({ci[0]:.2%}, {ci[1]:.2%})' if ci and ci[0] is not None else 'N/A'})"
                        )

                # Plotting for each regime
                if not getattr(args, "no_plots", False):
                    charts_dir = os.path.join(args.out, "charts")
                    Path(charts_dir).mkdir(parents=True, exist_ok=True)
                    for regime_name, results in all_results.items():
                        plot_episodes_df = pd.DataFrame(results["episodes"])
                        if not plot_episodes_df.empty:
                            save_distribution_plots(plot_episodes_df, charts_dir, regime_name)
                            save_cohort_lift_plot(results["statistics"], charts_dir, regime_name)
                            save_respect_rate_ci_plot(results["statistics"], charts_dir, regime_name)

                        # Daily maps are expensive, so let's check for zones too
                        if not getattr(args, "no_maps", False) and not df.empty and results["zones"]:
                            save_daily_maps(
                                df,
                                results["zones"],
                                results["episodes"],
                                charts_dir,
                                regime_name,
                                max_maps=args.max_maps,
                            )

                if args.report:
                    LOG.info("Generating HTML report...")
                    generate_html_report(all_results, args.out, config)

                # --- Save schema and summary ---
                save_episodes_schema(args.out)
                summary_data = []
                for regime_name, results in all_results.items():
                    s = results["statistics"].get("all", {})
                    rr = s.get("outcome_rates", {}).get("RESPECT", float('nan'))
                    ci = s.get("outcome_rates_ci", {}).get("RESPECT", (None, None))
                    med = s.get("survival_analysis", {}).get("median_bars_to_outcome", "N/A")
                    cvar = s.get("tail_risk", {}).get("cvar_95_adverse_excursion", float('nan'))
                    summary_data.append({
                        "regime": regime_name,
                        "n_episodes": s.get("n_episodes", 0),
                        "respect_rate": rr,
                        "respect_rate_ci_low": ci[0] if ci else None,
                        "respect_rate_ci_high": ci[1] if ci else None,
                        "median_bars_to_outcome": med,
                        "cvar_95_adverse_excursion": cvar,
                    })

                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_csv(os.path.join(args.out, "summary.csv"), index=False, float_format="%.4f")
                    LOG.info(f"Saved summary metrics to {os.path.join(args.out, 'summary.csv')}")

                # --- Console Summary ---
                def _fmt_ci(ci_tuple):
                    return "N/A" if not ci_tuple or ci_tuple[0] is None else f"{ci_tuple[0]:.2%} – {ci_tuple[1]:.2%}"

                LOG.info("--- Run Summary ---")
                for item in summary_data:
                    LOG.info(f"[{item['regime']:>12}] N={item['n_episodes']:5d} "
                             f"Respect={item['respect_rate']:.2%} ({_fmt_ci((item['respect_rate_ci_low'], item['respect_rate_ci_high']))})  "
                             f"Median bars={item['median_bars_to_outcome']}  CVaR95 adverse={item['cvar_95_adverse_excursion']:.2f} pts")


        elif args.command == "sweep":
            # Note: Regime analysis is not supported in sweep mode for simplicity.
            if config.regime_config and config.regime_config.enabled:
                LOG.warning("Regime analysis is disabled for parameter sweeps. Running sweep on all data.")
                config.regime_config.enabled = False

            df = (
                pd.read_parquet(args.data)
                if args.data.endswith(".parquet")
                else pd.read_csv(args.data)
            )
            df = ensure_prepared(df, config)
            with open(args.grid, "r") as f:
                param_grid = (
                    yaml.safe_load(f)
                    if args.grid.lower().endswith((".yml", ".yaml"))
                    else json.load(f)
                )

            config_dict = dataclasses.asdict(config)
            db_path = args.db if args.db else None

            # --- Prepare for sweep comparisons vs baseline ---
            LOG.info("Running baseline configuration for sweep comparison...")
            baseline_params = param_grid[0]
            baseline_config = EngineConfig()
            update_dataclass_from_dict(baseline_config, config_dict)
            update_dataclass_from_dict(baseline_config, baseline_params)

            baseline_results = _run_single_analysis(
                df.copy(), baseline_config, regime_name="all_data"
            )
            base_y = np.array(
                [
                    1
                    if (
                        e["outcome"].value
                        if isinstance(e["outcome"], Enum)
                        else e["outcome"]
                    )
                    == "RESPECT"
                    else 0
                    for e in baseline_results["episodes"]
                ],
                dtype=float,
            )
            LOG.info(
                f"Baseline run complete. Found {len(base_y)} episodes for comparison."
            )

            task_args = [
                (
                    params,
                    config_dict,
                    df,
                    base_metadata,
                    db_path,
                    base_y.tolist(),
                )
                for params in param_grid
            ]

            summary_results = []
            if args.jobs > 1 and HAVE_MULTIPROCESSING:
                LOG.info(f"Starting parallel sweep with {args.jobs} jobs.")
                with Pool(args.jobs) as pool:
                    results_iterator = pool.imap_unordered(run_sweep_item, task_args)
                    if tqdm:
                        results_iterator = tqdm(
                            results_iterator,
                            total=len(param_grid),
                            desc="Sweeping parameters",
                        )
                    summary_results = list(results_iterator)
            else:
                LOG.info("Starting serial sweep.")
                iterator = (
                    tqdm(task_args, desc="Sweeping parameters") if tqdm else task_args
                )
                summary_results = [run_sweep_item(arg_tuple) for arg_tuple in iterator]

            df_summary = pd.DataFrame(summary_results).sort_values(
                "n_valid_episodes", ascending=False
            )
            if HAVE_STATSMODELS and "p_value" in df_summary.columns:
                p_values = df_summary["p_value"].to_numpy(na_value=np.nan, dtype=float)
                mask = np.isfinite(p_values)
                if mask.any():
                    reject, q, _, _ = multipletests(
                        p_values[mask], alpha=0.05, method="fdr_bh"
                    )
                    df_summary.loc[mask, "q_value"] = q
                    df_summary.loc[mask, "reject_h0"] = reject

            if "respect_rate" in df_summary.columns and len(df_summary) > 0:
                # Use the actual baseline row (param_grid[0]) instead of the first row after sorting
                baseline_key = json.dumps(baseline_params)  # matches how we stored "params"
                mask = df_summary["params"] == baseline_key
                if mask.any():
                    baseline_rate = float(df_summary.loc[mask, "respect_rate"].iloc[0])
                else:
                    baseline_rate = float(df_summary["respect_rate"].iloc[0])

                def _h(p, q=baseline_rate):
                    p = min(max(p, 1e-12), 1 - 1e-12)
                    q = min(max(q, 1e-12), 1 - 1e-12)
                    return 2 * (math.asin(math.sqrt(p)) - math.asin(math.sqrt(q)))

                if "respect_lift" not in df_summary.columns:
                     df_summary["respect_lift"] = df_summary["respect_rate"] - baseline_rate
                df_summary["cohens_h_vs_baseline"] = df_summary["respect_rate"].apply(_h)

            df_summary.to_csv(
                os.path.join(out_dir, "sweep_summary.csv"),
                index=False,
                float_format="%.4f",
            )
            LOG.info(
                f"--- Sweep complete. Summary at {os.path.join(out_dir, 'sweep_summary.csv')} ---"
            )

    finally:
        if persistence:
            persistence.close()
            LOG.info("Database connection closed.")


##############################################################################
# 12. SELF-TEST (tiny synthetic run; optional)
##############################################################################
# To be added in a future step.


# Top-level function for multiprocessing sweep
def run_sweep_item(args_tuple):
    (
        params,
        config_dict,
        data_df,
        base_metadata,
        db_path,
        base_y_list,
    ) = args_tuple

    run_config = EngineConfig()
    update_dataclass_from_dict(run_config, config_dict)
    update_dataclass_from_dict(run_config, params)

    # Derive a deterministic, unique seed for this sweep item to ensure reproducibility
    param_bytes = json.dumps(params, sort_keys=True).encode()
    param_hash = int(hashlib.sha256(param_bytes).hexdigest()[:8], 16)  # 32-bit-ish
    item_seed = (int(run_config.random_seed or 0) ^ param_hash) & 0xFFFFFFFF
    run_config.random_seed = item_seed
    np.random.seed(run_config.random_seed)

    persistence = SQLitePersistence(db_path) if db_path else None
    try:
        if persistence:
            sweep_metadata = base_metadata.copy()
            sweep_metadata.update(
                {
                    "run_uuid": str(uuid.uuid4()),
                    "command": "sweep_item",
                    "config": dataclasses.asdict(run_config),
                }
            )
            run_pk = persistence.insert_run(sweep_metadata)
        else:
            run_pk = None

        results = _run_single_analysis(data_df.copy(), run_config, "all_data")

        if persistence and run_pk:
            persistence.insert_zones(run_pk, results["zones"])
            persistence.insert_episodes(run_pk, results["episodes"])
            persistence.insert_statistics(run_pk, results["statistics"])

        all_stats = results["statistics"].get("all", {})
        summary_row = {
            "params": json.dumps(params),
            "n_valid_episodes": all_stats.get("n_episodes", 0),
        }
        if all_stats.get("n_episodes", 0) > 0:
            summary_row.update(
                {
                    f"{k.lower()}_rate": v
                    for k, v in all_stats.get("outcome_rates", {}).items()
                }
            )
            # Add comparison to baseline
            y = np.array(
                [
                    1
                    if (
                        e["outcome"].value
                        if isinstance(e["outcome"], Enum)
                        else e["outcome"]
                    )
                    == "RESPECT"
                    else 0
                    for e in results["episodes"]
                ],
                dtype=float,
            )
            if len(y) > 0 and len(base_y_list) > 0:
                base_y = np.asarray(base_y_list, float)
                diff, ci, p = bootstrap_risk_difference(
                    y, base_y, block_size=_blk(run_config), seed=run_config.random_seed
                )
                summary_row.update(
                    {
                        "respect_lift": float(y.mean() - base_y.mean()),
                        "risk_diff": diff,
                        "risk_diff_ci_low": ci[0],
                        "risk_diff_ci_high": ci[1],
                        "p_value": p,
                    }
                )

        return summary_row
    finally:
        if persistence:
            persistence.close()


##############################################################################


def run_self_test():
    """Generates a tiny synthetic OHLC series and runs the full pipeline as a demo."""
    LOG.info("--- Running Self-Test (Demo Run) ---")

    # --- Setup ---
    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"runs/self_test_{ts}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    LOG.info(f"Self-test artifacts will be saved to: {out_dir}")

    # --- Generate Data ---
    timestamps = pd.to_datetime(pd.date_range(start="2023-01-01 09:30", periods=200, freq="1min", tz="America/New_York"))
    price = 102.0
    prices = []
    rng = np.random.default_rng(42)
    for i, ts in enumerate(timestamps):
        if 50 < i < 100 and rng.random() > 0.5:
            price = max(100.0, price - rng.uniform(0, 1) * 0.25)
        else:
            price += rng.uniform(-1, 1) * 0.25
        prices.append(price)

    df = pd.DataFrame({
        "timestamp": timestamps, "open": prices, "high": [p + 0.1 for p in prices],
        "low": [p - 0.1 for p in prices], "close": prices,
        "volume": rng.integers(100, 1000, size=len(prices)),
    })

    # --- Run Analysis ---
    config = EngineConfig()
    config.zones.min_touches_for_significance = 2
    config.zones.significance_test = False
    config.regime_config = RegimeConfig(enabled=True, regimes=[
        Regime(name="first_half", condition="index < 100"),
        Regime(name="second_half", condition="index >= 100")
    ])

    df = ensure_prepared(df, config)
    all_results = run_analysis(df, config)

    # --- Save Artifacts ---
    class MockArgs:
        def __init__(self, out_dir):
            self.out = out_dir
            self.no_plots = False
            self.no_maps = False
            self.report = True
            self.max_maps = 10

    args = MockArgs(out_dir)

    all_zones, all_episodes, all_stats_flat = [], [], {}
    for regime_name, results in all_results.items():
        for z in results["zones"]: z.regime = regime_name
        for e in results["episodes"]: e["regime"] = regime_name
        all_zones.extend(results["zones"])
        all_episodes.extend(results["episodes"])
        for cohort, stats in results["statistics"].items():
            all_stats_flat[f"{regime_name}_{cohort}"] = stats

    zones_df = pd.DataFrame([dataclasses.asdict(z) for z in all_zones])
    if not zones_df.empty:
        zones_df["type"] = zones_df["type"].apply(lambda x: x.value if isinstance(x, Enum) else x)
        zones_df["activation_time"] = pd.to_datetime(zones_df["activation_time"])
    zones_df.to_csv(os.path.join(out_dir, "zones.csv"), index=False)

    episodes_df = pd.DataFrame(all_episodes)
    if not episodes_df.empty:
        for col in ("outcome", "zone_type"):
            if col in episodes_df.columns:
                episodes_df[col] = episodes_df[col].apply(lambda x: x.value if isinstance(x, Enum) else x)
        for col in ("touch_time", "outcome_time"):
            if col in episodes_df.columns:
                episodes_df[col] = pd.to_datetime(episodes_df[col])
    episodes_df.to_csv(os.path.join(out_dir, "episodes.csv"), index=False)

    if not getattr(args, "no_plots", False):
        charts_dir = os.path.join(out_dir, "charts")
        Path(charts_dir).mkdir(parents=True, exist_ok=True)
        for regime_name, results in all_results.items():
            plot_episodes_df = pd.DataFrame(results["episodes"])
            if not plot_episodes_df.empty:
                save_distribution_plots(plot_episodes_df, charts_dir, regime_name)
                save_cohort_lift_plot(results["statistics"], charts_dir, regime_name)
                save_respect_rate_ci_plot(results["statistics"], charts_dir, regime_name)
            if not getattr(args, "no_maps", False) and not df.empty and results["zones"]:
                save_daily_maps(df, results["zones"], results["episodes"], charts_dir, regime_name, max_maps=args.max_maps)

    if args.report:
        generate_html_report(all_results, out_dir, config)

    save_episodes_schema(out_dir)

    # --- Assert basic invariants ---
    try:
        assert "first_half" in all_results, "Self-test failed: 'first_half' regime missing."
        assert len(all_results["first_half"]["zones"]) > 0, "Self-test failed: No zones in first_half."
        assert len(all_results["first_half"]["episodes"]) > 0, "Self-test failed: No episodes in first_half."
        LOG.info(f"--- SELF-TEST OK --- (Artifacts in {out_dir})")
    except AssertionError as e:
        LOG.error(f"--- SELF-TEST FAILED: {e} ---", exc_info=True)
        raise


if __name__ == "__main__":
    main()
