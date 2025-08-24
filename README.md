# Market Stats Engine

A single-file, production-ready market statistical analysis engine.

This engine discovers statistically significant support/resistance zones in
financial market data, evaluates their performance using a robust state machine,
and provides deep quantitative analysis on the results. It is designed to be
instrument-agnostic, configurable, and extensible.

## Features

- Instrument-agnostic via YAML configuration (Futures, FX, Crypto).
- Pluggable providers for zone discovery (pivots, external levels).
- Walk-forward validation for robust out-of-sample testing.
- Advanced statistical analysis: cohort analysis, survival curves, tail risk (CVaR).
- Optional SQLite persistence for run lineage and results tracking.
- Self-contained HTML reports with embedded plots for easy sharing.
- Performance-aware with optional multiprocessing and progress bars.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  For full functionality, including Parquet support, faster analytics, and advanced statistical tests, install the optional dependencies:
    ```bash
    pip install -r optional-requirements.txt
    ```
4.  To install the project as a package with a console script entry point:
    ```bash
    pip install .
    ```
    Or for development:
    ```bash
    pip install -e .
    ```

## Usage

The engine is run from the command line.

```bash
# Basic in-sample analysis on prepared NQ data
market-stats-engine analyze --data cache/nq_prepared.parquet --out results/nq_run_1

# Generate a self-contained HTML report
market-stats-engine analyze --data cache/nq_prepared.parquet --report --out results/nq_report

# Walk-forward validation
market-stats-engine analyze --data cache/es_prepared.parquet --wf "window=90d,step=30d" --out results/es_wf

# Run a parameter sweep
market-stats-engine sweep --data cache/nq_prepared.parquet --grid sweep_grid.yml --jobs 8 --out sweep_results/

# Run the built-in self-test
market-stats-engine self-test
```

## Methodology

### Zone Discovery

Support and resistance zones are discovered using pluggable "providers". The default provider uses price pivots.
-   **Pivots**: Highs and lows that are surrounded by `k` lower highs or higher lows.
-   **Clustering**: Nearby pivots are clustered together to form candidate zones.
-   **Significance Testing**: A binomial test is used to determine if the number of touches in a candidate zone is statistically significant compared to what would be expected by chance in the surrounding price window. P-values are corrected for multiple comparisons using the Benjamini-Hochberg FDR method (if `statsmodels` is installed).

### Episode Evaluation

Once zones are discovered, the engine scans the price data for "episodes", which are interactions with the zones. Each episode is evaluated by a state machine and results in one of the following outcomes:

-   **RESPECT**: Price touches the zone, exits favorably, and travels a configurable distance `R` away.
-   **PIERCE_AND_REVERT**: Price pierces into the zone (but not beyond an overshoot tolerance `O`), then reverts and travels `R` points in the favorable direction.
-   **BREAK**: Price closes beyond the zone plus the overshoot tolerance `O`.
-   **TIMEOUT**: The episode is censored, either by reaching the maximum `T` bars or by a session boundary change, before a clear outcome is observed.

### Statistical Analysis

-   **Confidence Intervals**: Confidence intervals for outcome rates and other metrics are calculated using a moving-block bootstrap to handle autocorrelation in financial time series.
-   **Cohort Analysis**: The engine can segment episodes into cohorts (e.g., by time of day, volatility regime) and compare their performance against the baseline using risk difference with bootstrapped p-values.

## Interpreting the Report

The HTML report provides a comprehensive overview of the analysis results.

-   **Summary Metrics**: Key metrics like the overall "Respect Rate" with its 95% confidence interval.
-   **Method Notes**: Important parameters and policies used in the analysis, such as the outlier policy and censoring rules.
-   **Comparative Cohort Analysis**: A table of cohorts that showed a statistically significant difference in respect rate compared to the baseline.
-   **Distributions**: Plots showing the distribution of episode outcomes, bars to outcome, and favorable/adverse excursions.

---
*This is a fictional project for demonstration purposes.*
