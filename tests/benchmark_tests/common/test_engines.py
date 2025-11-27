"""Tests for BenchmarkEngineManager."""

from __future__ import annotations

from unittest.mock import MagicMock

from benchmarks.common.engines import BenchmarkEngineManager


class TestBenchmarkEngineManager:
    """Tests for BenchmarkEngineManager."""

    def test_init(self) -> None:
        """Test manager initialization."""
        manager = BenchmarkEngineManager()
        assert manager._cache == {}
        assert manager._model_memory == {}

    def test_unload_engine_not_found(self) -> None:
        """Test unload_engine returns False for non-existent engine."""
        manager = BenchmarkEngineManager()
        result = manager.unload_engine("nonexistent", "cuda", "ja")
        assert result is False

    def test_unload_engine_success(self) -> None:
        """Test unload_engine successfully unloads cached engine."""
        manager = BenchmarkEngineManager()

        # Manually add a mock engine to cache
        mock_engine = MagicMock()
        cache_key = "test_engine_cuda_ja"
        manager._cache[cache_key] = mock_engine
        manager._model_memory[cache_key] = 1000.0

        # Unload the engine
        result = manager.unload_engine("test_engine", "cuda", "ja")

        assert result is True
        assert cache_key not in manager._cache
        # Model memory should be preserved for reporting
        assert cache_key in manager._model_memory
        mock_engine.cleanup.assert_called_once()

    def test_unload_engine_handles_cleanup_error(self) -> None:
        """Test unload_engine handles cleanup errors gracefully."""
        manager = BenchmarkEngineManager()

        # Add engine with failing cleanup
        mock_engine = MagicMock()
        mock_engine.cleanup.side_effect = RuntimeError("Cleanup failed")
        cache_key = "test_engine_cuda_ja"
        manager._cache[cache_key] = mock_engine

        # Should not raise, should return True
        result = manager.unload_engine("test_engine", "cuda", "ja")
        assert result is True
        assert cache_key not in manager._cache

    def test_clear_cache_keep_memory(self) -> None:
        """Test _clear_cache_keep_memory clears cache but keeps memory."""
        manager = BenchmarkEngineManager()

        # Add multiple engines
        for i in range(3):
            mock_engine = MagicMock()
            cache_key = f"engine_{i}_cuda_ja"
            manager._cache[cache_key] = mock_engine
            manager._model_memory[cache_key] = float(i * 1000)

        # Clear cache
        manager._clear_cache_keep_memory()

        # Cache should be empty
        assert len(manager._cache) == 0
        # Memory should be preserved
        assert len(manager._model_memory) == 3
        assert manager._model_memory["engine_0_cuda_ja"] == 0.0
        assert manager._model_memory["engine_1_cuda_ja"] == 1000.0
        assert manager._model_memory["engine_2_cuda_ja"] == 2000.0

    def test_clear_cache_clears_both(self) -> None:
        """Test clear_cache clears both cache and memory."""
        manager = BenchmarkEngineManager()

        # Add engines
        for i in range(3):
            mock_engine = MagicMock()
            cache_key = f"engine_{i}_cuda_ja"
            manager._cache[cache_key] = mock_engine
            manager._model_memory[cache_key] = float(i * 1000)

        # Clear all
        manager.clear_cache()

        assert len(manager._cache) == 0
        assert len(manager._model_memory) == 0

    def test_get_model_memory(self) -> None:
        """Test get_model_memory returns correct value."""
        manager = BenchmarkEngineManager()
        manager._model_memory["test_engine_cuda_ja"] = 2048.0

        result = manager.get_model_memory("test_engine", "cuda", "ja")
        assert result == 2048.0

    def test_get_model_memory_not_found(self) -> None:
        """Test get_model_memory returns None for non-existent engine."""
        manager = BenchmarkEngineManager()
        result = manager.get_model_memory("nonexistent", "cuda", "ja")
        assert result is None

    def test_get_cached_engines(self) -> None:
        """Test get_cached_engines returns correct keys."""
        manager = BenchmarkEngineManager()
        manager._cache["engine_a_cuda_ja"] = MagicMock()
        manager._cache["engine_b_cuda_en"] = MagicMock()

        result = manager.get_cached_engines()
        assert set(result) == {"engine_a_cuda_ja", "engine_b_cuda_en"}


class TestClearCacheKeepMemory:
    """Tests for _clear_cache_keep_memory behavior."""

    def test_clear_cache_keep_memory_calls_cleanup(self) -> None:
        """Test that _clear_cache_keep_memory calls cleanup on all engines."""
        manager = BenchmarkEngineManager()

        # Add multiple mock engines
        mock_engines = []
        for i in range(3):
            mock_engine = MagicMock()
            cache_key = f"engine_{i}_cuda_ja"
            manager._cache[cache_key] = mock_engine
            manager._model_memory[cache_key] = float(i * 1000)
            mock_engines.append(mock_engine)

        # Clear cache
        manager._clear_cache_keep_memory()

        # All engines should have cleanup called
        for engine in mock_engines:
            engine.cleanup.assert_called_once()

        # Cache empty, memory preserved
        assert len(manager._cache) == 0
        assert len(manager._model_memory) == 3

    def test_clear_cache_keep_memory_handles_missing_cleanup(self) -> None:
        """Test _clear_cache_keep_memory handles engines without cleanup."""
        manager = BenchmarkEngineManager()

        # Add engine without cleanup method
        mock_engine = MagicMock(spec=[])  # No methods
        manager._cache["engine_cuda_ja"] = mock_engine
        manager._model_memory["engine_cuda_ja"] = 1000.0

        # Should not raise
        manager._clear_cache_keep_memory()

        assert len(manager._cache) == 0
        assert len(manager._model_memory) == 1
