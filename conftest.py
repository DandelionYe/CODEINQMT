import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_config():
    """Minimal valid config dict matching the schema shape."""
    return {
        "schema_version": "1.0",
        "updated_at": "2026-05-31",
        "experiments": [
            {
                "experiment_id": "exp_test_001",
                "status": "planned",
                "strategy_family": "test",
                "strategy_version": "v1",
                "strategy_name": "Test Strategy",
                "hypothesis": "Test hypothesis",
                "signal_definition": {
                    "entry_signal": "x > 0",
                    "exit_signal": "x <= 0",
                    "market_filter": "none",
                    "ranking_metric": "score",
                    "portfolio_construction": "equal_weight",
                },
                "universe": {
                    "market": "ALL",
                    "security_type": "stock",
                    "data_source": "QMT CSV export",
                    "export_root": "data/qmt_export",
                },
                "date_range": {
                    "train_start": "20150101",
                    "first_test_year": 2021,
                    "last_test_year": 2025,
                    "incomplete_year": None,
                },
                "parameters": {
                    "benchmark_list": ["000300.SH"],
                    "portfolio_size": 10,
                    "min_train_rows": 500,
                    "min_train_trades": 2,
                    "max_train_drawdown": -0.5,
                    "min_train_sharpe": 0.1,
                    "min_train_annual_return": 0.01,
                    "train_excess_mode": "stock_only",
                    "allow_negative_train_excess": False,
                },
                "cost_model": {
                    "cash": 1000000,
                    "commission": 0.0001,
                    "sell_tax": 0.0005,
                    "slippage": 0.0,
                },
                "commands": {
                    "single_symbol_check": "echo single",
                    "batch_backtest": "echo batch",
                    "walk_forward": "echo wf",
                    "analysis": "echo analysis",
                    "diagnosis": "echo diagnosis",
                },
                "outputs": {
                    "single_symbol_dir": "backtests/test/single",
                    "batch_dir": "backtests/test/batch",
                    "walk_forward_dir": "backtests/test/wf",
                    "analysis_dir": "backtests/test/analysis",
                    "diagnosis_dir": "backtests/test/diagnosis",
                },
                "decision": {
                    "decision": "continue",
                    "reason": "test",
                    "decided_at": "2026-05-31",
                    "owner": "tester",
                },
            }
        ],
    }


@pytest.fixture
def sample_experiment(sample_config):
    """The single experiment from sample_config."""
    return sample_config["experiments"][0]


@pytest.fixture
def schema_path():
    """Absolute path to the real schema file."""
    return str(Path(__file__).resolve().parent / "configs" / "research_experiments.schema.json")
