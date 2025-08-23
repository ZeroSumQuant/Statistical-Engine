#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_stats_engine.py v2.0

Production-ready market statistical analysis engine for NQ futures that:

1. Validates data integrity and handles gaps/invalid bars
1. Detects statistically significant support/resistance zones with significance testing
1. Tracks episodes with refactored, testable state management
1. Computes robust statistics with proper confidence intervals
1. Validates all parameters and their relationships
1. Provides comprehensive logging and error handling

Major improvements:

- Data validation pipeline with OHLC integrity checks
- Refactored episode detection with clear state transitions
- Statistical significance testing for zones
- Block bootstrap for correlated episodes
- Vectorized operations where possible
- Comprehensive parameter validation
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
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
except:
    HAVE_POLARS = False
    pl = None

try:
    import pyarrow
    HAVE_PARQUET = True
except:
    HAVE_PARQUET = False

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except:
    HAVE_SCIPY = False
    scipy_stats = None

try:
    import yaml
    HAVE_YAML = True
except:
    HAVE_YAML = False

try:
    from statsmodels.stats.multitest import multipletests
    HAVE_STATSMODELS = True
except:
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

# ————————— Logging Setup —————————

LOG = logging.getLogger("market_stats_engine")
handler = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d] %(message)s")
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

# ————————— Enums —————————

class EpisodeOutcome(Enum):
    RESPECT = "RESPECT"
    PIERCE_AND_REVERT = "PIERCE_AND_REVERT"
    BREAK = "BREAK"
    TIMEOUT = "TIMEOUT"
    INVALID = "INVALID"  # New: for data quality issues

class ZoneType(Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"

# ————————— Config Models with Validation —————————

@dataclass
class SessionConfig:
    tz_rth: str = "America/New_York"
    rth_start: str = "09:30"
    rth_end: str = "16:00"

    def __post_init__(self):
        # Validate time format
        try:
            pd.to_datetime(self.rth_start, format="%H:%M")
            pd.to_datetime(self.rth_end, format="%H:%M")
        except:
            raise ValueError("Invalid time format. Use HH:MM")

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
    pivot_k: int = 5
    zone_width_points: Optional[float] = 15.0
    zone_width_alpha_atr: Optional[float] = None
    cluster_width_points: Optional[float] = None  # New: for decoupling clustering from trade width
    tick_size: float = 0.25  # Instrument-specific tick size
    null_width_multiplier: float = 1.5 # Multiplier for significance test width
    expire_days: int = 30
    merge_tolerance_points: float = 2.0
    max_cluster_span_days: Optional[int] = None # Max time span for a single cluster
    min_touches_for_significance: int = 2
    significance_test: bool = True  # New: enable statistical testing

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
    O: float = 6.0   # overshoot tolerance
    T: int = 20      # timeout bars
    treat_timeout_as_break: bool = False
    max_gap_bars: int = 5  # New: max gap within episode

    def __post_init__(self):
        if self.R <= 0:
            raise ValueError("Reversal distance R must be positive")
        if self.O <= 0:
            raise ValueError("Overshoot tolerance O must be positive")
        if self.O >= self.R:
            raise ValueError(f"Overshoot {self.O} must be less than reversal {self.R}")
        if self.T < 1:
            raise ValueError("Timeout T must be >= 1")
        if self.T > 390:  # Typical RTH session length
            LOG.warning(f"Timeout {self.T} exceeds typical session length")

@dataclass
class DataQualityConfig:
    """New: Configuration for data validation"""
    check_ohlc_integrity: bool = True
    max_price_change_pct: float = MAX_PRICE_CHANGE_PERCENT
    min_volume: int = MIN_VOLUME
    handle_gaps: bool = True
    max_gap_minutes: int = MAX_GAP_MINUTES
    remove_outliers: bool = True
    outlier_std_threshold: float = 10.0

@dataclass
class EngineConfig:
    session: SessionConfig = field(default_factory=SessionConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    cache_dir: str = "cache"
    out_dir: str = "runs/run_latest"

# ————————— Data Validation —————————

class DataValidator:
    """Validates and cleans OHLCV data"""
    def __init__(self, config: DataQualityConfig):
        self.config = config

    def validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate OHLCV data and return cleaned dataframe with validation report
        """
        report = {
            "original_rows": len(df),
            "invalid_ohlc": 0,
            "outliers": 0,
            "gaps_detected": 0,
            "low_volume": 0
        }

        if self.config.check_ohlc_integrity:
            # OHLC relationship validation
            invalid_mask = (
                (df["high"] < df["low"]) |
                (df["high"] < df["open"]) |
                (df["high"] < df["close"]) |
                (df["low"] > df["open"]) |
                (df["low"] > df["close"])
            )
            report["invalid_ohlc"] = invalid_mask.sum()
            if invalid_mask.any():
                LOG.warning(f"Removing {invalid_mask.sum()} bars with invalid OHLC relationships")
                df = df[~invalid_mask].copy()

        # Price change validation
        if self.config.max_price_change_pct > 0:
            pct_change = df["close"].pct_change().abs() * 100
            too_big_mask = pct_change > self.config.max_price_change_pct
            if too_big_mask.any():
                report["price_spike"] = int(too_big_mask.sum())
                LOG.warning(f"Removing {report['price_spike']} bars with >{self.config.max_price_change_pct}% move")
                df = df[~too_big_mask].copy()

        # Volume validation
        if self.config.min_volume > 0:
            invalid_vol = df["volume"] < self.config.min_volume
            report["low_volume"] = invalid_vol.sum()
            if invalid_vol.any():
                LOG.warning(f"Removing {report['low_volume']} bars with volume < {self.config.min_volume}")
                df = df[~invalid_vol].copy()

        # Outlier detection
        df["is_outlier"] = False
        if self.config.remove_outliers:
            df["returns"] = df["close"].pct_change()

            if "session_date" in df.columns:
                # Compute MAD of returns per session, which is more robust to outliers
                mad = df.groupby("session_date")["returns"].transform(lambda x: (x - x.median()).abs().median())
                # 1.4826 scales MAD to be like STD for a normal distribution
                # Add a small epsilon to avoid division by zero or issues with flat series
                outlier_threshold = self.config.outlier_std_threshold * mad * 1.4826 + 1e-9
                outliers = df["returns"].abs() > outlier_threshold
            else:
                # Fallback if session_date is not available
                outlier_threshold = df["returns"].std() * self.config.outlier_std_threshold
                outliers = df["returns"].abs() > outlier_threshold

            report["outliers"] = outliers.sum()
            if outliers.any():
                LOG.info(f"Flagging {outliers.sum()} outlier bars")
                df.loc[outliers, "is_outlier"] = True

            # Drop the temporary returns column
            df = df.drop(columns=["returns"])

        # Gap detection
        if self.config.handle_gaps:
            same_session = df["session_date"] == df["session_date"].shift(1)
            time_diff = df["timestamp"].diff()

            gaps = same_session & (time_diff > pd.Timedelta(minutes=self.config.max_gap_minutes))
            df["has_gap"] = gaps.fillna(False)
            report["gaps_detected"] = int(gaps.sum())

            if report["gaps_detected"] > 0:
                LOG.info(f"Detected {report['gaps_detected']} intra-session gaps > {self.config.max_gap_minutes} minutes")
        else:
            df["has_gap"] = False

        report["final_rows"] = len(df)
        report["rows_removed"] = report["original_rows"] - report["final_rows"]
        report["removal_pct"] = 100 * report["rows_removed"] / report["original_rows"] if report["original_rows"] > 0 else 0

        return df, report

# ————————— Statistical Utilities —————————

def test_zone_significance(touches: List[float], level: float, width: float,
                           local_prices: pd.Series, alpha: float = ZONE_SIGNIFICANCE_ALPHA) -> Tuple[bool, float]:
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

    # Probability of a random price falling in the zone
    p0 = min(1.0, (2.0 * width) / window_pts)

    # Number of observed touches inside the zone band
    x_obs = sum(1 for t in touches if abs(t - level) <= width)

    if HAVE_SCIPY:
        # Use scipy for exact binomial test (survival function)
        # sf(k, n, p) is 1 - cdf(k, n, p). We want P(X >= x_obs), which is sf(x_obs - 1).
        pval = scipy_stats.binom.sf(k=x_obs - 1, n=n, p=p0)
    else:
        # Fallback to math.comb
        from math import comb
        try:
            pval = sum(comb(n, k) * (p0**k) * ((1-p0)**(n-k)) for k in range(x_obs, n + 1))
        except (ValueError, TypeError): # math.comb can fail on non-integers
            pval = 1.0 # Default to non-significant if calculation fails

    return pval < alpha, pval

def block_bootstrap_ci(data: np.ndarray, statistic_func, n_boot: int = 5000,
                       block_size: int = BOOTSTRAP_BLOCK_SIZE, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Circular moving block bootstrap for confidence intervals.
    This method is more robust for time series data as it preserves dependencies
    and handles edge effects by wrapping the data in a circle.
    """
    n = len(data)
    if n < block_size:
        # Fall back to standard bootstrap for small samples
        return standard_bootstrap_ci(data, statistic_func, n_boot, alpha)

    rng = np.random.default_rng(42)
    bootstrap_stats = []

    # Extended series for circular wrapping
    extended_data = np.concatenate([data, data[:block_size - 1]])

    for _ in range(n_boot):
        # Number of blocks needed to create a series of length n
        num_blocks = math.ceil(n / block_size)

        # Sample random block start indices
        start_indices = rng.integers(0, n, size=num_blocks)

        # Create resampled data by taking blocks and trimming to original length
        resampled_indices = np.concatenate([np.arange(s, s + block_size) for s in start_indices])[:n]
        resampled_data = extended_data[resampled_indices]

        bootstrap_stats.append(statistic_func(resampled_data))

    bootstrap_stats = np.array(bootstrap_stats)
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(ci_lower), float(ci_upper)

def standard_bootstrap_ci(data: np.ndarray, statistic_func, n_boot: int = 5000,
                          alpha: float = 0.05) -> Tuple[float, float]:
    """Standard bootstrap for comparison"""
    rng = np.random.default_rng(42)
    bootstrap_stats = []

    for _ in range(n_boot):
        resampled = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(resampled))

    bootstrap_stats = np.array(bootstrap_stats)
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(ci_lower), float(ci_upper)

# ————————— Refactored Episode Detection —————————

class EpisodeState(ABC):
    """Abstract base class for episode states"""
    @abstractmethod
    def process_bar(self, context: Dict[str, Any]) -> Tuple[Optional[EpisodeOutcome], Optional['EpisodeState']]:
        """Process a bar and return outcome (if terminal) and next state"""
        pass

class SearchingForTouch(EpisodeState):
    """Initial state: searching for first touch of zone"""
    def process_bar(self, context: Dict[str, Any]) -> Tuple[Optional[EpisodeOutcome], Optional[EpisodeState]]:
        bar = context["bar"]
        zone = context["zone"]
        df = context["df"]

        band_low = zone["level"] - zone["width"]
        band_high = zone["level"] + zone["width"]

        # Check if bar touches zone
        if bar["low"] <= band_high and bar["high"] >= band_low:
            touch_idx = context["current_idx"]
            context["touch_idx"] = touch_idx
            context["touch_time"] = bar["timestamp"]

            # Enrich context at touch
            touch_bar = df.iloc[touch_idx]
            context["rsi_at_touch"] = touch_bar.get("rsi")
            context["atr_at_touch"] = touch_bar.get("atr")
            context["rvol_at_touch"] = touch_bar.get("rvol")
            context["stoch_k_at_touch"] = touch_bar.get("stoch_k")
            context["stoch_d_at_touch"] = touch_bar.get("stoch_d")
            context["is_rth_at_touch"] = touch_bar.get("is_rth")
            context["minute_of_day_at_touch"] = touch_bar.get("minute_of_day")

            # Determine approach direction
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
    """State: tracking price after touch to determine outcome"""
    def process_bar(self, context: Dict[str, Any]) -> Tuple[Optional[EpisodeOutcome], Optional[EpisodeState]]:
        config = context["config"]
        # Check for stale data within an episode using timestamps
        if config["max_gap_bars"] > 0 and "prev_ts" in context:
            if (context["bar"]["timestamp"] - context["prev_ts"]) > (context["bar_dt"] * config["max_gap_bars"]):
                LOG.debug("Invalidating episode due to timestamp gap")
                return EpisodeOutcome.INVALID, None
        context["prev_ts"] = context["bar"]["timestamp"]

        bar = context["bar"]
        zone = context["zone"]

        band_low = zone["level"] - zone["width"]
        band_high = zone["level"] + zone["width"]
        band_break_high = band_high + config["O"]
        band_break_low = band_low - config["O"]

        # Update metrics
        direction = -1 if zone["type"] == ZoneType.RESISTANCE else 1
        ref = zone["level"]

        if direction == -1:
            fav = max(0.0, ref - bar["low"])
            adv = max(0.0, bar["high"] - ref)
        else:
            fav = max(0.0, bar["high"] - ref)
            adv = max(0.0, ref - bar["low"])

        context["max_favorable"] = max(context.get("max_favorable", 0), fav)
        context["max_adverse"] = max(context.get("max_adverse", 0), adv)

        bars_since_touch = context["current_idx"] - context["touch_idx"]

        # Check for invalidating conditions first
        if bar.get("is_outlier", False):
            LOG.debug(f"Invalidating episode due to outlier bar at {bar['timestamp']}")
            return EpisodeOutcome.INVALID, None

        if bar.get("has_gap", False):
            LOG.debug(f"Invalidating episode due to data gap before {bar['timestamp']}")
            return EpisodeOutcome.INVALID, None

        # Check outcomes in priority order
        pierced_now = (
            (bar["high"] > band_high and bar["high"] <= band_break_high) or
            (bar["low"] < band_low and bar["low"] >= band_break_low)
        )
        if pierced_now:
            context["pierced_before"] = True

        respected_now = context["max_favorable"] >= config["R"]

        # 1. Break: Close beyond band by more than O
        if bar["close"] > band_break_high or bar["close"] < band_break_low:
            return EpisodeOutcome.BREAK, None

        # 2. Pierce and Revert: Must have pierced *before* respecting
        if context.get("pierced_before", False) and respected_now:
            return EpisodeOutcome.PIERCE_AND_REVERT, None

        # Check for favorable exit before respect
        if not context.get("exited_favorably", False):
            if direction == 1:  # SUPPORT: favorable is up
                if bar["high"] > band_high:
                    context["exited_favorably"] = True
            else:               # RESISTANCE: favorable is down
                if bar["low"] < band_low:
                    context["exited_favorably"] = True

        # 3. Respect: Reversed by R without breaking, after a favorable exit
        if context.get("exited_favorably", False) and respected_now:
            return EpisodeOutcome.RESPECT, None

        # 4. Timeout
        if bars_since_touch >= config["T"]:
            if config.get("treat_timeout_as_break", False):
                return EpisodeOutcome.BREAK, None
            return EpisodeOutcome.TIMEOUT, None

        return None, self

class EpisodeDetector:
    """Refactored episode detection with clear state management"""
    def __init__(self, config: EpisodeConfig):
        self.config = config

    def detect_episode(self, df: pd.DataFrame, zone: Dict[str, Any], start_idx: int) -> Optional[Dict[str, Any]]:
        """Detect episode for a zone starting from start_idx"""
        state = SearchingForTouch()
        bar_dt = df["timestamp"].diff().median()
        if pd.isna(bar_dt) or bar_dt <= pd.Timedelta(0):
            bar_dt = pd.Timedelta(minutes=1)

        context = {
            "df": df, # Pass full dataframe to context
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
                LOG.debug(f"Zone {zone['id']} expired during episode search. Stopping.")
                break

            context["current_idx"] = i
            context["bar"] = bar.to_dict()
            if i > 0:
                context["prev_close"] = df.iloc[i-1]["close"]

            outcome, next_state = state.process_bar(context)

            # If state changed (e.g., from Searching to Tracking), re-process the same bar
            if next_state is not state and next_state is not None:
                state = next_state
                outcome, next_state = state.process_bar(context)

            if outcome is not None:
                # Terminal state reached
                return {
                    "zone_id": zone["id"],
                    "zone_type": zone["type"],
                    "outcome": outcome,
                    "touch_time": context["touch_time"],
                    "outcome_time": bar["timestamp"],
                    "bars_to_outcome": i - context["touch_idx"] if context["touch_idx"] is not None else -1,
                    "max_favorable": context["max_favorable"],
                    "max_adverse": context["max_adverse"],
                    "from_above": context["from_above"],
                    "end_idx": i,
                    # Add enriched context at touch
                    "rsi_at_touch": context.get("rsi_at_touch"),
                    "atr_at_touch": context.get("atr_at_touch"),
                    "rvol_at_touch": context.get("rvol_at_touch"),
                    "stoch_k_at_touch": context.get("stoch_k_at_touch"),
                    "stoch_d_at_touch": context.get("stoch_d_at_touch"),
                    "is_rth_at_touch": context.get("is_rth_at_touch"),
                    "minute_of_day_at_touch": context.get("minute_of_day_at_touch"),
                    # Add enriched context at outcome
                    "rsi_at_outcome": bar.get("rsi"),
                    "atr_at_outcome": bar.get("atr"),
                    "rvol_at_outcome": bar.get("rvol"),
                    "stoch_k_at_outcome": bar.get("stoch_k"),
                    "stoch_d_at_outcome": bar.get("stoch_d"),
                }

            if next_state is None:
                break

            state = next_state

        return None

# ————————— Zone Detection with Significance Testing —————————

def merge_close_zones(zones: List['Zone'], tolerance: float, config: 'ZoneConfig') -> List['Zone']:
    """Merges zones that are closer than the given tolerance."""
    if not zones:
        return []

    # Sort zones by level to easily find adjacent ones
    sorted_zones = sorted(zones, key=lambda z: z.level)

    merged_zones = [sorted_zones[0]]

    for current_zone in sorted_zones[1:]:
        prev_zone = merged_zones[-1]

        if abs(current_zone.level - prev_zone.level) <= tolerance:
            # Merge the current zone into the previous one
            total_touches = len(prev_zone.touches) + len(current_zone.touches)
            if total_touches == 0: continue

            # Weighted average for the new level
            new_level = ((prev_zone.level * len(prev_zone.touches)) +
                         (current_zone.level * len(current_zone.touches))) / total_touches

            new_touches = prev_zone.touches + current_zone.touches
            new_p_value = min(prev_zone.p_value, current_zone.p_value)
            new_activation_time = min(prev_zone.activation_time, current_zone.activation_time)
            new_width = (prev_zone.width + current_zone.width) / 2.0

            # Update the last zone in the merged list
            merged_zones[-1] = dataclasses.replace(
                prev_zone,
                level=new_level,
                touches=new_touches,
                p_value=new_p_value,
                activation_time=new_activation_time,
                width=new_width
            )
        else:
            # No merge, just add the new zone
            merged_zones.append(current_zone)

    return merged_zones

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

def detect_zones_with_significance(df: pd.DataFrame, config: ZoneConfig) -> List[Zone]:
    """Detect zones with statistical significance testing"""
    # Detect pivots first
    pivots = detect_pivots(df, config.pivot_k, config)

    final_zones = []
    zone_id = 1

    # Group pivots by type
    high_pivots = [p for p in pivots if p["type"] == "HIGH"]
    low_pivots = [p for p in pivots if p["type"] == "LOW"]

    for pivot_list, zone_type in [(high_pivots, ZoneType.RESISTANCE),
                               (low_pivots, ZoneType.SUPPORT)]:

        candidate_zones = []
        clusters = cluster_pivots(pivot_list, config)

        for cluster in clusters:
            if len(cluster) < config.min_touches_for_significance:
                continue

            touches = [p["price"] for p in cluster]
            level = np.mean(touches)

            if config.zone_width_alpha_atr is not None:
                # ... (width calculation logic is the same)
                t0, t1 = min(p["center_time"] for p in cluster), max(p["center_time"] for p in cluster)
                local_df = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]
                atr_ref = float(local_df["atr"].median()) if not local_df.empty and not local_df["atr"].isnull().all() else float(df["atr"].median())
                width = max(1e-9, config.zone_width_alpha_atr * atr_ref)
            else:
                width = config.zone_width_points or 15.0

            p_value = 0.0
            if config.significance_test:
                min_time, max_time = min(p['center_time'] for p in cluster), max(p['center_time'] for p in cluster)
                sessions_spanned = df[(df['timestamp'] >= min_time) & (df['timestamp'] <= max_time)]['session_date'].unique()
                local_prices_df = df[df['session_date'].isin(sessions_spanned)]
                if not local_prices_df.empty:
                    width_for_test = max(width, (config.cluster_width_points or width) * config.null_width_multiplier)
                    local_prices = pd.concat([local_prices_df['high'], local_prices_df['low']])
                    _, p_value = test_zone_significance(touches, level, width_for_test, local_prices)

            cluster_sorted_by_time = sorted(cluster, key=lambda p: p["confirm_time"])
            activation_time = cluster_sorted_by_time[1]["confirm_time"] if len(cluster) > 1 else cluster_sorted_by_time[0]["confirm_time"]

            candidate_zones.append(Zone(
                id=0, type=zone_type, level=level, width=width,
                activation_time=activation_time, touches=touches, p_value=p_value,
                is_significant=False, expire_days=config.expire_days
            ))

        # Multiple testing correction
        if config.significance_test and candidate_zones:
            p_values = [z.p_value for z in candidate_zones]
            if HAVE_STATSMODELS:
                reject, _, _, _ = multipletests(p_values, alpha=ZONE_SIGNIFICANCE_ALPHA, method='fdr_bh')
                significant_zones = [zone for zone, is_sig in zip(candidate_zones, reject) if is_sig]
            else:
                LOG.warning("statsmodels not found. Skipping multiple testing correction. Please `pip install statsmodels`.")
                significant_zones = [zone for zone in candidate_zones if zone.p_value < ZONE_SIGNIFICANCE_ALPHA]
        else:
            significant_zones = candidate_zones

        # Merge close zones of the same type
        if config.merge_tolerance_points > 0 and significant_zones:
            num_before_merge = len(significant_zones)
            merged_zones = merge_close_zones(significant_zones, config.merge_tolerance_points, config)
            LOG.info(f"Merged {num_before_merge - len(merged_zones)} {zone_type.value} zones.")
        else:
            merged_zones = significant_zones

        # Finalize zones with correct, contiguous IDs
        for zone in merged_zones:
            final_zones.append(dataclasses.replace(zone, id=zone_id, is_significant=True))
            zone_id += 1

    LOG.info(f"Detected {len(final_zones)} statistically significant zones after merging.")
    return final_zones

def detect_pivots(df: pd.DataFrame, k: int, config: ZoneConfig) -> List[Dict[str, Any]]:
    """Detect pivots with look-ahead safety and tolerance."""
    pivots = []
    tick = config.tick_size

    hi = df["high"].values
    lo = df["low"].values
    ts = df["timestamp"]  # Keep as a Series to preserve Timestamp objects

    for i in range(k, len(df) - k):
        window_hi = hi[i-k:i+k+1]
        window_lo = lo[i-k:i+k+1]

        # Check for high pivot with tolerance
        is_high_pivot = (hi[i] >= window_hi.max() - tick) and \
                        (hi[i] > hi[i-1]) and \
                        (hi[i] >= hi[i+1])
        if is_high_pivot:
            pivots.append({
                "type": "HIGH",
                "price": hi[i],
                "center_time": ts.iat[i],
                "confirm_time": ts.iat[i+k],
                "index": i
            })

        # Check for low pivot with tolerance
        is_low_pivot = (lo[i] <= window_lo.min() + tick) and \
                       (lo[i] < lo[i-1]) and \
                       (lo[i] <= lo[i+1])
        if is_low_pivot:
            pivots.append({
                "type": "LOW",
                "price": lo[i],
                "center_time": ts.iat[i],
                "confirm_time": ts.iat[i+k],
                "index": i
            })

    return pivots

def cluster_pivots(pivots: List[Dict], config: ZoneConfig) -> List[List[Dict]]:
    """Cluster nearby pivots"""
    if not pivots:
        return []

    sorted_pivots = sorted(pivots, key=lambda x: x["price"])
    clusters = []
    current_cluster = [sorted_pivots[0]]

    for pivot in sorted_pivots[1:]:
        # Check if pivot belongs to current cluster based on price
        cluster_center = np.mean([p["price"] for p in current_cluster])
        width = config.cluster_width_points or 15.0 # Use dedicated clustering width
        price_is_close = abs(pivot["price"] - cluster_center) <= width

        # Check time span if configured
        time_is_close = True
        if config.max_cluster_span_days is not None:
            min_time = min(p['center_time'] for p in current_cluster)
            if (pivot['center_time'] - min_time).days > config.max_cluster_span_days:
                time_is_close = False

        if price_is_close and time_is_close:
            current_cluster.append(pivot)
        else:
            if len(current_cluster) >= config.min_touches_for_significance:
                clusters.append(current_cluster)
            current_cluster = [pivot]

    # Don't forget last cluster
    if len(current_cluster) >= config.min_touches_for_significance:
        clusters.append(current_cluster)

    return clusters

# ————————— Main Pipeline Functions —————————

def prepare_data(files: List[str], config: EngineConfig) -> pd.DataFrame:
    """Load and prepare data with validation"""
    validator = DataValidator(config.data_quality)
    all_dfs = []
    for file in files:
        LOG.info(f"Loading {file}")

        # Load file
        if file.endswith(".parquet"):
            if not HAVE_PARQUET:
                raise ImportError("pyarrow is required to read Parquet files. Please `pip install pyarrow`.")
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file)

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        required = ["timestamp", "open", "high", "low", "close", "volume"]

        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {file}: {missing}")

        # Ensure UTC timestamps and sort before any processing
        df = _ensure_utc_timestamps(df)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        # Annotate sessions first to enable session-based validation
        df = annotate_sessions(df, config.session)

        # Validate and clean data
        df, report = validator.validate_and_clean(df)

        LOG.info(f"Data quality report for {file}:")
        LOG.info(f"  - Original rows: {report['original_rows']:,}")
        LOG.info(f"  - Invalid OHLC: {report['invalid_ohlc']:,}")
        LOG.info(f"  - Outliers flagged: {report['outliers']:,}")
        LOG.info(f"  - Final rows: {report['final_rows']:,}")
        LOG.info(f"  - Removed: {report['removal_pct']:.1f}%")

        all_dfs.append(df)

    # Combine all files
    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Add indicators on the full, sorted series
    combined = add_indicators(combined, config.indicators)

    return combined

def annotate_sessions(df: pd.DataFrame, config: SessionConfig) -> pd.DataFrame:
    """Add session information"""
    # Convert to session timezone
    if ZoneInfo:
        tz = ZoneInfo(config.tz_rth)
    elif pytz:
        tz = pytz.timezone(config.tz_rth)
    else:
        tz = None
        LOG.warning("No timezone library available, using UTC")

    if tz:
        df["local_time"] = df["timestamp"].dt.tz_convert(tz)
    else:
        df["local_time"] = df["timestamp"]

    # Add session markers
    df["session_date"] = df["local_time"].dt.date
    df["minute_of_day"] = df["local_time"].dt.hour * 60 + df["local_time"].dt.minute

    # RTH flag
    rth_start = pd.to_datetime(config.rth_start).time()
    rth_end = pd.to_datetime(config.rth_end).time()
    df["is_rth"] = (
        (df["local_time"].dt.time >= rth_start) &
        (df["local_time"].dt.time < rth_end)
    )

    return df

def add_indicators(df: pd.DataFrame, config: IndicatorConfig) -> pd.DataFrame:
    """Add technical indicators"""
    # ATR
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=config.atr_n, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=config.rsi_n, adjust=False).mean()
    avg_loss = loss.ewm(span=config.rsi_n, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-12)))

    # RVOL (simple implementation)
    # A more robust implementation would use average volume per minute-of-day from a lookback window.
    if config.rvol_lookback_sessions > 0:
        # Approximate rolling window based on 390 RTH minutes per session
        rolling_window = config.rvol_lookback_sessions * 390
        df["avg_volume_lookback"] = df["volume"].rolling(window=rolling_window, min_periods=rolling_window // 10).mean()
        df["rvol"] = df["volume"] / (df["avg_volume_lookback"] + 1e-12)
    else:
        df["rvol"] = np.nan

    # --- Stochastic %K and %D ---
    n = config.stoch_n
    d = config.stoch_d
    lowest_low = df["low"].rolling(window=n, min_periods=1).min()
    highest_high = df["high"].rolling(window=n, min_periods=1).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)  # avoid /0
    df["stoch_k"] = ((df["close"] - lowest_low) / denom * 100).clip(0, 100)
    df["stoch_d"] = df["stoch_k"].rolling(window=d, min_periods=1).mean()

    # Add more indicators as needed...
    return df

def run_analysis(df: pd.DataFrame, config: EngineConfig) -> Dict[str, Any]:
    """Run complete analysis pipeline"""
    # Detect zones
    zones = detect_zones_with_significance(df, config.zones)

    # Detect episodes for each zone, allowing for multiple non-overlapping episodes
    detector = EpisodeDetector(config.episode)
    episodes = []

    for zone in zones:
        zone_dict = {
            "id": zone.id,
            "type": zone.type,
            "level": zone.level,
            "width": zone.width,
            "expiry_time": zone.activation_time + pd.Timedelta(days=zone.expire_days)
        }

        # Find first potential bar after zone activation
        potential_indices = df.index[df["timestamp"] > zone.activation_time]
        if not potential_indices.any():
            continue

        current_scan_idx = int(potential_indices.min())

        while current_scan_idx < len(df) and df.iloc[current_scan_idx]["timestamp"] <= zone_dict["expiry_time"]:
            episode = detector.detect_episode(df, zone_dict, current_scan_idx)

            if episode:
                episodes.append(episode)

                # Start searching for the next episode after the current one ends
                # and after price has left the zone to ensure non-overlapping episodes.
                last_episode_end_idx = episode["end_idx"]

                # Find first bar *after* the episode that is *outside* the zone
                band_low = zone_dict["level"] - zone_dict["width"]
                band_high = zone_dict["level"] + zone_dict["width"]

                next_scan_idx = -1

                # Search from the bar *after* the episode ended
                search_from_idx = last_episode_end_idx + 1
                if search_from_idx >= len(df):
                    break

                for idx in range(search_from_idx, len(df)):
                    bar = df.iloc[idx]
                    if bar["timestamp"] > zone_dict["expiry_time"]:
                        break  # Zone expired

                    if bar["high"] < band_low or bar["low"] > band_high:
                        # Price is outside the zone, we can start searching for the next touch
                        next_scan_idx = idx
                        break

                if next_scan_idx != -1:
                    current_scan_idx = next_scan_idx
                else:
                    # No re-approach found before end of data or expiry
                    break
            else:
                # detect_episode returned None, meaning no more episodes can be found for this zone
                break

    LOG.info(f"Detected {len(episodes)} episodes across {len(zones)} zones")

    # --- Stratified Statistics ---
    episodes_df = pd.DataFrame(episodes)

    if not episodes_df.empty:
        # Add stratification columns
        episodes_df['hour_at_touch'] = episodes_df['minute_of_day_at_touch'] // 60

        # Use qcut for terciles, handle cases with not enough unique values
        try:
            episodes_df['atr_tercile'] = pd.qcut(episodes_df['atr_at_touch'].rank(method='first'), 3, labels=["low", "mid", "high"])
        except (ValueError, TypeError):
            episodes_df['atr_tercile'] = "n/a"

        try:
            episodes_df['rvol_tercile'] = pd.qcut(episodes_df['rvol_at_touch'].rank(method='first'), 3, labels=["low", "mid", "high"])
        except (ValueError, TypeError):
            episodes_df['rvol_tercile'] = "n/a"

        # Stochastic buckets at touch
        episodes_df["stoch_k_at_touch"] = pd.to_numeric(episodes_df["stoch_k_at_touch"], errors="coerce")
        episodes_df["stoch_d_at_touch"] = pd.to_numeric(episodes_df["stoch_d_at_touch"], errors="coerce")
        try:
            episodes_df["stoch_tercile"] = pd.qcut(
                episodes_df["stoch_k_at_touch"].rank(method="first"), 3, labels=["low", "mid", "high"]
            )
        except (ValueError, TypeError):
            episodes_df["stoch_tercile"] = "n/a"

        episodes_df["stoch_overbought_touch"] = episodes_df["stoch_k_at_touch"] >= 80
        episodes_df["stoch_oversold_touch"]   = episodes_df["stoch_k_at_touch"] <= 20

        strata = {
            "all": episodes_df,
            "rth": episodes_df[episodes_df["is_rth_at_touch"] == True],
            "eth": episodes_df[episodes_df["is_rth_at_touch"] == False],
            "stoch_ge80": episodes_df[episodes_df["stoch_overbought_touch"]],
            "stoch_le20": episodes_df[episodes_df["stoch_oversold_touch"]],
        }

        for hour, group in episodes_df.groupby('hour_at_touch'):
            if not pd.isna(hour):
                strata[f"hour_{int(hour)}"] = group

        for tercile, group in episodes_df.groupby('atr_tercile'):
            strata[f"atr_{tercile}"] = group

        for tercile, group in episodes_df.groupby('rvol_tercile'):
            strata[f"rvol_{tercile}"] = group

        for tercile, group in episodes_df.groupby('stoch_tercile'):
            strata[f"stoch_{tercile}"] = group

        stratified_stats = {
            name: compute_statistics(data.to_dict('records'), config)
            for name, data in strata.items() if not data.empty
        }
    else:
        stratified_stats = {"all": compute_statistics([], config)}

    return {
        "zones": zones,
        "episodes": episodes,
        "statistics": stratified_stats,
        "data_shape": df.shape
    }

def compute_statistics(episodes: List[Dict], config: EngineConfig, alpha: float = 0.05) -> Dict[str, Any]:
    """Compute statistics with proper confidence intervals"""
    # Separate valid from invalid episodes for cleaner statistics
    valid_episodes = [e for e in episodes if e["outcome"] != EpisodeOutcome.INVALID]
    n_invalid = len(episodes) - len(valid_episodes)
    n_episodes = len(valid_episodes)

    stats = {
        "n_episodes": n_episodes,
        "n_invalid_episodes": n_invalid,
        "outcome_distribution": {},
        "outcome_rates": {},
        "outcome_rates_ci": {},
        "bootstrapped_metrics": {}
    }

    if n_episodes == 0:
        return stats

    if n_episodes < MIN_EPISODES_FOR_STATS:
        LOG.debug(f"Too few valid episodes ({n_episodes}) for full stats, computing basics.")

    outcomes = np.array([e["outcome"].value for e in valid_episodes])

    # --- Outcome rates and CIs ---
    for outcome in EpisodeOutcome:
        if outcome == EpisodeOutcome.INVALID:
            continue

        count = np.sum(outcomes == outcome.value)
        stats["outcome_distribution"][outcome.value] = int(count)

        rate = count / n_episodes if n_episodes > 0 else 0
        stats["outcome_rates"][outcome.value] = rate

        if HAVE_SCIPY and n_episodes > 0:
            from scipy.stats import beta
            k = int(count)
            if k == 0:
                ci_low, ci_high = 0.0, beta.ppf(1 - alpha / 2, 1, n_episodes)
            elif k == n_episodes:
                ci_low, ci_high = beta.ppf(alpha / 2, n_episodes, 1), 1.0
            else:
                ci_low = beta.ppf(alpha / 2, k, n_episodes - k + 1)
                ci_high = beta.ppf(1 - alpha / 2, k + 1, n_episodes - k)
            stats["outcome_rates_ci"][outcome.value] = (ci_low, ci_high)
        else:
            stats["outcome_rates_ci"][outcome.value] = (None, None)

    # --- Bootstrap CIs for metrics ---
    metrics_to_bootstrap = ["bars_to_outcome", "max_favorable", "max_adverse"]
    if n_episodes >= 2:  # Need at least 2 data points for bootstrap
        for metric_name in metrics_to_bootstrap:
            data = np.array([e[metric_name] for e in valid_episodes if e.get(metric_name) is not None])
            if len(data) > 1:
                block_size = config.episode.T // 2 or 5
                stats["bootstrapped_metrics"][metric_name] = {
                    "mean": np.mean(data),
                    "mean_ci": block_bootstrap_ci(data, np.mean, n_boot=1000, block_size=block_size),
                    "median": np.median(data),
                    "median_ci": block_bootstrap_ci(data, np.median, n_boot=1000, block_size=block_size),
                    "p25": np.percentile(data, 25),
                    "p75": np.percentile(data, 75)
                }
    return stats

# ————————— CLI Interface —————————

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
                raise ImportError("pyyaml is required to load YAML configs. Please `pip install pyyaml`")
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)

    update_dataclass_from_dict(cfg, raw)
    return cfg


def save_distribution_plots(episodes_df: pd.DataFrame, out_dir: str):
    """Generates and saves distribution plots for key episode metrics."""
    if episodes_df.empty:
        LOG.info("No episodes found, skipping distribution plots.")
        return

    LOG.info(f"Generating distribution plots in {out_dir}...")

    # Plot 1: Outcome Distribution
    plt.figure(figsize=(10, 6))
    episodes_df["outcome"].value_counts().plot(kind='bar')
    plt.title("Episode Outcome Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dist_outcomes.png"))
    plt.close()

    # Plot 2: Bars to Outcome
    plt.figure(figsize=(10, 6))
    episodes_df["bars_to_outcome"].hist(bins=50, range=(0, episodes_df["bars_to_outcome"].quantile(0.99)))
    plt.title("Distribution of Bars to Outcome")
    plt.xlabel("Number of Bars")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dist_bars_to_outcome.png"))
    plt.close()

    # Plot 3: Max Favorable/Adverse Excursion
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    episodes_df["max_favorable"].hist(bins=50, color='g', range=(0, episodes_df["max_favorable"].quantile(0.99)))
    plt.title("Max Favorable Excursion")
    plt.xlabel("Points")
    plt.subplot(1, 2, 2)
    episodes_df["max_adverse"].hist(bins=50, color='r', range=(0, episodes_df["max_adverse"].quantile(0.99)))
    plt.title("Max Adverse Excursion")
    plt.xlabel("Points")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dist_excursions.png"))
    plt.close()

    LOG.info("Distribution plots saved.")

def save_daily_maps(df: pd.DataFrame, zones: List[Zone], episodes: List[Dict], out_dir: str):
    """Generates and saves daily price charts with zones and episodes."""
    if df.empty or not zones:
        LOG.info("Not enough data to generate daily maps.")
        return

    maps_dir = os.path.join(out_dir, "daily_maps")
    Path(maps_dir).mkdir(parents=True, exist_ok=True)
    LOG.info(f"Generating daily maps in {maps_dir}...")

    episodes_df = pd.DataFrame(episodes)
    if not episodes_df.empty:
        episodes_df["touch_time"] = pd.to_datetime(episodes_df["touch_time"])
        episodes_df["outcome_time"] = pd.to_datetime(episodes_df["outcome_time"])

    local_tz = df["local_time"].dt.tz
    if local_tz is None:
        LOG.warning("local_time has no tz; assuming UTC for plotting.")
        local_tz = "UTC"

    def to_local_date(ts):
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(local_tz).date()

    for session, day_df in df.groupby("session_date"):
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot price
        ax.plot(day_df["timestamp"], day_df["close"], label="Close", color='black', linewidth=0.5)

        # Plot active zones
        for zone in zones:
            zone_act_local = to_local_date(zone.activation_time)
            zone_exp_local = to_local_date(zone.activation_time + pd.Timedelta(days=zone.expire_days))
            if zone_act_local <= session <= zone_exp_local:
                color = 'green' if zone.type == ZoneType.SUPPORT else 'red'
                ax.axhspan(zone.level - zone.width, zone.level + zone.width, alpha=0.1, color=color)
                ax.axhline(zone.level, color=color, linestyle='--', linewidth=0.7)

        # Plot episodes
        if not episodes_df.empty:
            day_episodes = episodes_df[episodes_df["touch_time"].dt.date == session]
            for _, episode in day_episodes.iterrows():
                outcome = episode['outcome']
                label = outcome.value if hasattr(outcome, "value") else str(outcome)
                outcome_color = {'RESPECT': 'blue', 'BREAK': 'orange', 'PIERCE_AND_REVERT': 'purple'}.get(label, 'grey')

                # Use nearest-bar lookup for robust plotting
                ix = df['timestamp'].searchsorted(episode['touch_time'])
                ix = int(np.clip(ix, 1, len(df)-1))
                cand = df.iloc[[ix-1, ix]]
                row = cand.iloc[(cand["timestamp"] - episode['touch_time']).abs().values.argmin()]
                touch_price = row["close"]
                ax.scatter(episode['touch_time'], touch_price, color=outcome_color, s=50, zorder=5, marker='o')
                ax.text(episode['outcome_time'], touch_price, label, color=outcome_color)

        ax.set_title(f"Market Map for {session.strftime('%Y-%m-%d')}")
        ax.set_ylabel("Price")
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(maps_dir, f"map_{session.strftime('%Y-%m-%d')}.png"))
        plt.close(fig)
    LOG.info("Daily maps saved.")


def _ensure_utc_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the timestamp column is a timezone-aware UTC timestamp."""
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column.")

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    if df['timestamp'].dt.tz is None:
        LOG.info("Timestamp column is timezone-naive, localizing to UTC.")
        df['timestamp'] = df['timestamp'].dt.tz_localize("UTC")
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert("UTC")

    return df

def ensure_prepared(df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    """Checks if data has been prepared, and if not, runs preparation steps."""
    df = _ensure_utc_timestamps(df) # Make this utility self-contained and robust
    need_sessions = any(c not in df.columns for c in ["session_date","minute_of_day","is_rth","local_time"])
    if need_sessions:
        LOG.info("Input data is missing session columns, running annotate_sessions...")
        df = annotate_sessions(df, config.session)

    need_ind = any(c not in df.columns for c in ["atr", "rsi", "rvol", "stoch_k", "stoch_d"])
    if need_ind:
        LOG.info("Input data is missing indicator columns, running add_indicators...")
        df = add_indicators(df, config.indicators)

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market_stats_engine",
        description="Production-ready market statistics engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Prepare command
    prep_parser = subparsers.add_parser("prepare", help="Prepare and validate data")
    prep_parser.add_argument("--files", nargs="+", required=True, help="Input files")
    prep_parser.add_argument("--out", default="cache/prepared.parquet", help="Output file")
    prep_parser.add_argument("--config", help="Config file (YAML/JSON)")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run statistical analysis")
    analyze_parser.add_argument("--data", required=True, help="Prepared data file")
    analyze_parser.add_argument("--config", help="Config file (YAML/JSON)")
    analyze_parser.add_argument("--out", default="results/", help="Output directory")

    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Parameter sweep")
    sweep_parser.add_argument("--data", required=True, help="Prepared data file")
    sweep_parser.add_argument("--grid", required=True, help="Parameter grid file")
    sweep_parser.add_argument("--out", default="sweep_results/", help="Output directory")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load config from file if provided
    config_path = getattr(args, 'config', None)
    config = load_engine_config(config_path)

    # Create output directory and save run metadata
    if args.command in ["prepare", "analyze"]:
        if args.command == "analyze":
            out_dir = args.out
        else:  # prepare
            out_dir = str(Path(args.out).parent)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Gather metadata
        run_time = dt.datetime.now(dt.timezone.utc).isoformat()

        try:
            code_hash = get_file_sha256(__file__)
        except FileNotFoundError:
            code_hash = "n/a"

        if args.command == "prepare":
            data_hashes = {f: get_file_sha256(f) for f in args.files}
        else:  # analyze
            data_hashes = {args.data: get_file_sha256(args.data)}

        metadata = {
            "run_timestamp_utc": run_time,
            "command": args.command,
            "args": vars(args),
            "code_sha256": code_hash,
            "data_sha256": data_hashes,
            "config": dataclasses.asdict(config)
        }

        # Save metadata
        metadata_path = os.path.join(out_dir, "run_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        LOG.info(f"Saved run metadata to {metadata_path}")

    if args.command == "prepare":
        # Prepare and validate data
        df = prepare_data(args.files, config)

        # Save prepared data
        # Output directory is already created by metadata logic
        out_path = args.out
        if out_path.endswith(".parquet"):
            if HAVE_PARQUET:
                df.to_parquet(out_path, index=False)
            else:
                out_path = out_path.replace(".parquet", ".csv")
                df.to_csv(out_path, index=False)
                LOG.warning("pyarrow not found, saving as CSV instead.")
        else:
            df.to_csv(out_path, index=False)

        LOG.info(f"Saved prepared data to {out_path}")
        LOG.info(f"Shape: {df.shape}")
        LOG.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    elif args.command == "analyze":
        # Load prepared data
        if args.data.endswith(".parquet"):
            if not HAVE_PARQUET:
                raise ImportError("pyarrow is required to read Parquet files. Please `pip install pyarrow`.")
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data)

        df = _ensure_utc_timestamps(df)
        df = ensure_prepared(df, config)

        # Run analysis
        results = run_analysis(df, config)

        # Save results
        # Output directory is already created by metadata logic

        # Save zones
        zones_data = [{
            "id": z.id,
            "type": z.type.value,
            "level": z.level,
            "width": z.width,
            "p_value": z.p_value,
            "is_significant": z.is_significant,
            "n_touches": len(z.touches),
            "activation_time": z.activation_time.isoformat(),
            "expiry_time": (z.activation_time + pd.Timedelta(days=z.expire_days)).isoformat(),
        } for z in results["zones"]]
        pd.DataFrame(zones_data).to_csv(f"{args.out}/zones.csv", index=False)

        # Save episodes
        episodes_df = pd.DataFrame(results["episodes"])
        if not episodes_df.empty:
            # Convert Enums to string values for clean CSV export
            episodes_df["outcome"] = episodes_df["outcome"].apply(lambda x: x.value if hasattr(x, "value") else x)
            episodes_df["zone_type"] = episodes_df["zone_type"].apply(lambda x: x.value if hasattr(x, "value") else x)
        episodes_df.to_csv(f"{args.out}/episodes.csv", index=False)

        # Save statistics
        with open(f"{args.out}/statistics.json", "w") as f:
            json.dump(results["statistics"], f, indent=2, default=str)

        LOG.info(f"Results saved to {args.out}")
        LOG.info(f"Zones: {len(results['zones'])}")
        LOG.info(f"Episodes: {len(results['episodes'])}")

        if "all" in results["statistics"] and results["statistics"]["all"]["n_episodes"] > 0:
            all_stats = results["statistics"]["all"]
            respect_rate = all_stats["outcome_rates"].get("RESPECT", 0.0)
            ci = all_stats["outcome_rates_ci"].get("RESPECT")
            ci_str = f"({ci[0]:.2%}, {ci[1]:.2%})" if ci and ci[0] is not None else "N/A"
            LOG.info(f"Overall Respect rate: {respect_rate:.2%} (CI: {ci_str})")

        # Generate and save plots
        # Re-create DF with enum values for plotting function
        plot_episodes_df = pd.DataFrame(results["episodes"])
        if not plot_episodes_df.empty:
            plot_episodes_df["outcome"] = plot_episodes_df["outcome"].apply(lambda x: x.value)
        save_distribution_plots(plot_episodes_df, args.out)
        save_daily_maps(df, results["zones"], results["episodes"], args.out)

    elif args.command == "sweep":
        LOG.info("--- Starting Parameter Sweep ---")

        # Load data
        LOG.info(f"Loading data from {args.data}")
        if args.data.endswith(".parquet"):
            if not HAVE_PARQUET:
                raise ImportError("pyarrow is required to read Parquet files. Please `pip install pyarrow`.")
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data)

        df = _ensure_utc_timestamps(df)
        df = ensure_prepared(df, config)

        # Load parameter grid
        LOG.info(f"Loading parameter grid from {args.grid}")
        with open(args.grid, "r") as f:
            if args.grid.lower().endswith((".yml", ".yaml")):
                if not HAVE_YAML: raise ImportError("pyyaml is required for YAML grid file.")
                param_grid = yaml.safe_load(f)
            else:
                param_grid = json.load(f)

        if not isinstance(param_grid, list):
            raise ValueError("Parameter grid file must contain a JSON list of configurations.")

        # Prepare for sweep
        Path(args.out).mkdir(parents=True, exist_ok=True)
        summary_results = []

        LOG.info(f"Found {len(param_grid)} parameter sets to sweep.")

        for i, params in enumerate(param_grid):
            # Create config for this run
            run_config = EngineConfig()
            update_dataclass_from_dict(run_config, params)

            run_id = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
            LOG.info(f"--- Running sweep {i+1}/{len(param_grid)} (ID: {run_id}) ---")

            # Run analysis
            results = run_analysis(df.copy(), run_config) # Use copy of df to be safe

            # Extract summary stats
            all_stats = results["statistics"].get("all", {})
            n_episodes = all_stats.get("n_episodes", 0)

            summary_row = {"run_id": run_id, "params": json.dumps(params)}
            summary_row["n_valid_episodes"] = n_episodes
            summary_row["n_invalid_episodes"] = all_stats.get("n_invalid_episodes", 0)
            if n_episodes > 0:
                summary_row.update({
                    "respect_rate": all_stats.get("outcome_rates", {}).get("RESPECT"),
                    "pierce_revert_rate": all_stats.get("outcome_rates", {}).get("PIERCE_AND_REVERT"),
                    "break_rate": all_stats.get("outcome_rates", {}).get("BREAK"),
                    "timeout_rate": all_stats.get("outcome_rates", {}).get("TIMEOUT"),
                })
            summary_results.append(summary_row)

        # Save summary
        summary_df = pd.DataFrame(summary_results)
        summary_path = os.path.join(args.out, "sweep_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        LOG.info(f"--- Parameter sweep complete. Summary saved to {summary_path} ---")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
