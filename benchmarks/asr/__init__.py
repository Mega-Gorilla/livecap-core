"""ASR Benchmark module.

Provides tools for benchmarking ASR engine performance.

Usage:
    python -m benchmarks.asr --mode quick
"""

from __future__ import annotations

from .runner import ASRBenchmarkConfig, ASRBenchmarkRunner

__all__ = ["ASRBenchmarkConfig", "ASRBenchmarkRunner"]
