"""VAD ベンチマークモジュール

VAD + ASR パイプラインのベンチマーク機能を提供します。

Usage:
    python -m benchmarks.vad --mode quick
    python -m benchmarks.vad --engine parakeet_ja --vad silero --language ja
"""

from __future__ import annotations

from .runner import VADBenchmarkRunner, VADBenchmarkConfig
from .factory import create_vad, get_all_vad_ids, get_vad_config, VAD_REGISTRY
from .backends import VADBenchmarkBackend, VADProcessorWrapper
from .preset_integration import (
    OPTIMIZABLE_VADS,
    create_vad_with_preset,
    get_preset_vad_ids,
    get_preset_config,
    is_preset_available,
)

__all__ = [
    # Runner
    "VADBenchmarkRunner",
    "VADBenchmarkConfig",
    # Factory
    "create_vad",
    "get_all_vad_ids",
    "get_vad_config",
    "VAD_REGISTRY",
    # Backends
    "VADBenchmarkBackend",
    "VADProcessorWrapper",
    # Preset integration
    "OPTIMIZABLE_VADS",
    "create_vad_with_preset",
    "get_preset_vad_ids",
    "get_preset_config",
    "is_preset_available",
]
