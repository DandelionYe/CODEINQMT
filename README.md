# CODEINQMT

A-share quantitative strategy research framework with disciplined walk-forward validation.

**This is a research framework only. It does not provide trading signals, investment advice, or live trading capabilities.**

## Overview

CODEINQMT implements **D's_Flow** — a stage-based research pipeline for A-share (Chinese stock market) strategies built on locally-prepared CSV/Parquet market data. The workflow enforces hypothesis registration, walk-forward out-of-sample validation, diagnosis, and robustness gates before any strategy can be promoted.

### Key Principles

- Batch screening results are candidates, not trade pools.
- Walk-forward validation is the primary out-of-sample test.
- Diagnose before iterating.
- Every strategy version must be registered in `configs/research_experiments.json`.
- Strategy promotion and infrastructure improvement are separate decisions.
- Portfolio backtest validates the module under constraints, not strategy viability.

## Data Statement

**This repository does NOT contain:**

- Real market data (CSV, Parquet)
- Broker documents or client software
- API documentation from QMT/XtQuant
- Complete backtest output
- Factor data or company information
- Tokens, credentials, or broker configurations

Users must prepare their own data from legitimate sources. See `data/README.md` for expected data format.

## Security Statement

**This repository does NOT contain:**

- API tokens or authentication credentials
- Broker-specific configurations
- Live trading setup
- Personal account information

The `.gitignore` is configured to prevent accidental commit of sensitive files.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Local Data

Prepare QMT-exported CSV data in the following structure:

```
data/
  qmt_export/
    SH/          # Shanghai Stock Exchange
      price_600000.csv
      ...
    SZ/          # Shenzhen Stock Exchange
      price_000001.csv
      ...
```

See `data/README.md` for detailed format requirements.

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

### 4. Run Smoke Test

```bash
python strategies/alpha_v4_research_strategy_csv.py \
  --stock 000001.SZ --benchmark 000300.SH \
  --alpha-variant pure_momentum \
  --momentum-window 120 --trend-ma 250 --vol-window 60 \
  --breakout-window 120 --benchmark-ma 120 --start 20150101
```

### 5. Run Pipeline (Dry-Run)

```bash
python scripts/run_research_pipeline.py --list-experiments
python scripts/run_research_pipeline.py --experiment-id exp_004_next_alpha_research --dry-run
```

## Directory Structure

```
CODEINQMT/
├── configs/                    # Experiment configuration
│   ├── base/                   # Base configs (inheritable)
│   ├── research_experiments.json       # Experiment registry
│   ├── research_experiments.schema.json # JSON Schema
│   └── factor_registry.json            # Factor definitions
│
├── strategies/                 # Strategy implementations (CLI scripts)
│   ├── ma_demo_strategy_csv.py
│   ├── ma_market_filter_strategy_csv.py
│   ├── ma_v3_momentum_strategy_csv.py
│   ├── alpha_v4_research_strategy_csv.py
│   ├── alpha_v5_research_strategy_csv.py
│   ├── alpha_v6_research_strategy_csv.py
│   └── alpha_v7_research_strategy_csv.py
│
├── scripts/                    # Pipeline scripts
│   ├── run_research_pipeline.py        # Main entry point
│   ├── batch_*_backtest_csv.py         # Batch backtest
│   ├── validate_*_candidates.py        # Walk-forward validation
│   ├── analyze_*_walk_forward_results.py # Analysis
│   ├── diagnose_*_strategy_results.py   # Diagnosis
│   ├── validate_alpha_*_robustness.py   # Robustness checks
│   ├── portfolio_backtest_csv.py        # Portfolio constraints
│   ├── evaluate_alpha_signals.py        # Signal evaluation
│   ├── train_alpha_model.py             # ML model training
│   ├── build_feature_matrix.py          # Feature matrix builder
│   └── common/                          # Shared modules
│       ├── backtest/                    # Backtest engine
│       ├── models/                      # Model interfaces
│       ├── config_loader.py
│       ├── data_handler.py
│       ├── feature_expression.py
│       ├── feature_matrix.py
│       ├── signal_evaluation.py
│       ├── processors.py
│       └── metrics.py
│
├── tests/                      # Test suite
├── factors/                    # Factor data tools (see factors/README.md)
│   └── concept_industry/       # THS concept/industry scraping tools
│
├── data/                       # Market data (user-provided, see data/README.md)
├── backtests/                  # Backtest output (see backtests/README.md)
├── requirements.txt
├── conftest.py
└── pytest.ini
```

## D's_Flow Pipeline

```
0. Register experiment
1. Environment & data check
2. Single-symbol smoke test
3. Full-market batch backtest
4. Walk-forward out-of-sample validation
5. Result analysis & benchmark comparison
6. Strategy diagnosis
7. Robustness validation
8. Record conclusions & decision
9. Portfolio constraint backtest
10. Strategy iteration or factor interface extension
```

### Supported Stages

```
single_symbol_check → batch_backtest → walk_forward →
  analysis → diagnosis → robustness → portfolio_backtest
```

### Stage-Output Mapping

```
single_symbol_check -> single_symbol_dir
batch_backtest      -> batch_dir
walk_forward        -> walk_forward_dir
analysis            -> analysis_dir
diagnosis           -> diagnosis_dir
robustness          -> robustness_dir
portfolio_backtest  -> portfolio_backtest_dir
```

## QMT / XtQuant Compatibility

This framework is designed to work with QMT-exported CSV data. The strategies and pipeline scripts read standard OHLCV CSV files.

**Note:** QMT (QMT-compatible trading terminal) is a third-party trading platform. This repository does not include QMT software, API documentation, or broker-specific configurations. Users should obtain QMT from their broker through legitimate channels.

The `strategies/ma_demo_strategy.py` file demonstrates direct xtquant API usage for data download, but the main research pipeline (`scripts/`) works purely with local CSV/Parquet files.

## Experiment Registration

New experiments must be registered in `configs/research_experiments.json` with:

- `experiment_id`: unique identifier (format: `exp_NNN_short_description`)
- `hypothesis`: what is being tested
- `signal_definition`: entry/exit/filter/ranking logic
- `parameters`: strategy parameters and training filters
- `cost_model`: commission, tax, slippage assumptions
- `commands`: reproducible commands per stage
- `outputs`: output directories per stage
- `decision`: initial decision (continue/revise/abandon/promote)

See `configs/README.md` for full documentation.

## License

MIT License

## Disclaimer

This repository is for educational and research purposes only. It does not constitute investment advice. Past performance of any strategy described herein does not guarantee future results. Users are responsible for their own investment decisions and should consult qualified financial advisors.
