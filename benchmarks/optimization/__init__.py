"""VAD Parameter Optimization module.

Provides Bayesian optimization for VAD parameters using Optuna.

Usage:
    # Basic optimization
    python -m benchmarks.optimization --vad silero --language ja

    # With report generation
    python -m benchmarks.optimization --vad silero --language ja --report

Example:
    from benchmarks.optimization import VADOptimizer

    optimizer = VADOptimizer(vad_type="silero", language="ja")
    result = optimizer.optimize(n_trials=50)
    print(f"Best score: {result.best_score}")
    print(f"Best params: {result.best_params}")

    # Generate reports (HTML, JSON, Step Summary)
    report_paths = result.generate_reports()
    print(f"HTML report: {report_paths.html}")
    print(f"JSON export: {report_paths.json}")

Report Generation:
    Reports are generated in the following formats:
    - HTML: Interactive Plotly charts (optimization history, param importance)
    - JSON: Best parameters and metrics
    - Step Summary: GitHub Actions summary (if GITHUB_STEP_SUMMARY is set)

    For real-time monitoring, use Optuna Dashboard:
        pip install optuna-dashboard
        optuna-dashboard sqlite:///benchmark_results/optimization/studies.db
"""

from __future__ import annotations

from .param_spaces import PARAM_SPACES, suggest_params
from .objective import VADObjective
from .vad_optimizer import VADOptimizer, OptimizationResult
from .visualization import (
    OptimizationReport,
    ComparisonCSVExporter,
    ReportPaths,
    generate_report,
)

__all__ = [
    "PARAM_SPACES",
    "suggest_params",
    "VADObjective",
    "VADOptimizer",
    "OptimizationResult",
    "OptimizationReport",
    "ComparisonCSVExporter",
    "ReportPaths",
    "generate_report",
]
