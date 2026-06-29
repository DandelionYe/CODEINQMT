# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CODEINQMT is a quantitative research pipeline for A-share (Chinese stock market) strategies, built on QMT-exported CSV data. The main workflow is called **D's_Flow** — a disciplined, stage-based research loop that enforces hypothesis registration, walk-forward out-of-sample validation, diagnosis, and robustness gates before any strategy can be promoted.

This is **not** a packaged Python library. It's a collection of standalone CLI scripts orchestrated by a config-driven pipeline runner.

## Environment

```bash
conda activate research-env    # required for all formal runs; env at <PYTHON_ENV>
```

Dependencies are declared in `requirements.txt` (numpy, pandas, matplotlib, pyarrow, jsonschema). `xtquant` is only needed for live-data features, not the CSV-based workflow, and is not listed (not on PyPI).

Environment check:
```bash
python scripts/check_qmt_env.py
python scripts/check_xtdata.py
```

## Running the Pipeline

The primary entry point is `scripts/run_research_pipeline.py`. It reads experiment definitions from `configs/research_experiments.json`, validates against the JSON Schema, and dispatches stage commands via subprocess.

```bash
# List registered experiments
python scripts/run_research_pipeline.py --list-experiments

# View experiment details
python scripts/run_research_pipeline.py --experiment-id exp_004_next_alpha_research --show-experiment

# Dry-run (preview commands without executing)
python scripts/run_research_pipeline.py --experiment-id exp_004_next_alpha_research --dry-run

# Execute specific stages
python scripts/run_research_pipeline.py --experiment-id exp_004_next_alpha_research --stages analysis,diagnosis,robustness --execute

# Skip stages that already have output
python scripts/run_research_pipeline.py --experiment-id exp_004_next_alpha_research --skip-existing --execute

# Debug without conda env (not for formal results)
python scripts/run_research_pipeline.py --experiment-id exp_004_next_alpha_research --dry-run --allow-env-mismatch
```

**Stage execution order:** `single_symbol_check` → `batch_backtest` → `walk_forward` → `analysis` → `diagnosis` → `robustness` → `portfolio_backtest`

## Running a Smoke Test (Single Symbol)

```bash
python strategies/alpha_v4_research_strategy_csv.py --stock 000001.SZ --benchmark 000300.SH --alpha-variant pure_momentum --momentum-window 120 --trend-ma 250 --vol-window 60 --breakout-window 120 --benchmark-ma 120 --start 20150101
```

## Architecture

### D's_Flow Stage Pipeline

```
Experiment Registration → Env Check → Smoke Test → Batch Backtest
  → Walk-Forward Validation → Analysis → Diagnosis → Robustness Gate
    → Portfolio Backtest → Decision & Iteration
```

Each experiment progresses through statuses: `planned` → `running` → `completed` → `diagnosed` → then either `abandoned`, `revise` (back to planned), or `promoted`.

### Key Directories

- **`strategies/`** — Strategy implementations as CLI scripts (not classes). `ma_demo_strategy_csv.py` doubles as a shared utility module imported by other scripts (`from strategies import ma_demo_strategy_csv as ma`). Current active strategy: `alpha_v4_research_strategy_csv.py` with 4 alpha variants.
- **`scripts/`** — Pipeline orchestrator (`run_research_pipeline.py`), batch backtest scripts, walk-forward validators, analysis/diagnosis/robustness scripts, and `portfolio_backtest_csv.py` for constrained portfolio simulation.
- **`configs/`** — `research_experiments.json` is the central experiment registry. `research_experiments.schema.json` is the JSON Schema (draft-07) that validates it. See `configs/README.md` for maintenance rules.
- **`backtests/`** — All output artifacts organized by experiment ID and stage.
- **`data/qmt_export/`** and **`data/qmt_parquet/`** — Market data (CSV source and converted parquet).
- **`Tools/qlib/`** — Vendored Microsoft Qlib source for reference only; not integrated into the main workflow.
- **`factors/`** — Early-stage factor data, not yet a unified factor registry.

### How Scripts Connect

Most scripts in `strategies/` and `scripts/` are standalone CLI tools that read CSV/parquet data from `data/` and write results to `backtests/`. They are orchestrated by `run_research_pipeline.py` which dispatches them as subprocesses based on the commands registered in `configs/research_experiments.json`.

The pipeline reads each experiment's `commands` field (keyed by stage name) and `outputs` field (keyed by stage name, mapping to output directories). Stage-to-output mapping: `single_symbol_check` → `single_symbol_dir`, `batch_backtest` → `batch_dir`, `walk_forward` → `walk_forward_dir`, `analysis` → `analysis_dir`, `diagnosis` → `diagnosis_dir`, `robustness` → `robustness_dir`, `portfolio_backtest` → `portfolio_backtest_dir`.

### Experiment Registration

New experiments must be registered in `configs/research_experiments.json` with a unique `experiment_id` (format: `exp_NNN_short_description`). Required fields include `hypothesis`, `signal_definition`, `parameters`, `cost_model`, `commands`, and `outputs`. The `configs/README.md` has full field documentation and the workflow for creating new experiment entries.

## D's_Flow Core Principles

- Batch screening results are candidates, not trade pools.
- Walk-forward is the primary validation line.
- Diagnose before iterating.
- Every strategy version must be registered.
- Strategy promotion and infrastructure improvement are separate decisions.
- Portfolio backtest validates the module under constraints, not strategy viability.
