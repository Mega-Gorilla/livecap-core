from typing import Any, Dict, Optional

import pytest

from livecap_core.engines.engine_factory import EngineFactory
from livecap_core.engines.metadata import EngineInfo, EngineMetadata


class DummyEngine:
    def __init__(
        self,
        device: Optional[str] = None,
        language: str = "ja",
        model_size: str = "base",
        **kwargs,
    ) -> None:
        self.device = device
        self.language = language
        self.model_size = model_size
        self.extra_kwargs = kwargs


@pytest.fixture(autouse=True)
def reset_engine_factory_and_metadata(monkeypatch):
    """Ensure tests do not leak state into EngineFactory/EngineMetadata caches."""
    # Reset EngineFactory cache
    monkeypatch.setattr(EngineFactory, "_ENGINES", None, raising=False)
    # Store original metadata
    original_engines = EngineMetadata._ENGINES.copy()
    yield
    # Restore
    monkeypatch.setattr(EngineFactory, "_ENGINES", None, raising=False)
    EngineMetadata._ENGINES = original_engines


def _add_stub_engine_to_metadata():
    """Add a stub engine to EngineMetadata for testing."""
    EngineMetadata._ENGINES["stub"] = EngineInfo(
        id="stub",
        display_name="Stub Engine",
        description="Stub description",
        supported_languages=["ja"],
        default_params={
            "language": "ja",
            "model_size": "base",
        }
    )


def test_create_engine_uses_defaults(monkeypatch):
    """Test that create_engine uses default_params from EngineMetadata."""
    _add_stub_engine_to_metadata()
    monkeypatch.setattr(EngineFactory, "_get_engine_class", lambda *_: DummyEngine)

    engine = EngineFactory.create_engine(
        engine_type="stub",
        device="cpu",
    )

    assert isinstance(engine, DummyEngine)
    assert engine.device == "cpu"
    # Defaults from EngineMetadata.default_params should be applied
    assert engine.language == "ja"
    assert engine.model_size == "base"


def test_create_engine_allows_overriding_defaults(monkeypatch):
    """Test that engine_options can override default_params."""
    _add_stub_engine_to_metadata()
    monkeypatch.setattr(EngineFactory, "_get_engine_class", lambda *_: DummyEngine)

    engine = EngineFactory.create_engine(
        engine_type="stub",
        device="cuda",
        language="en",
        model_size="large",
    )

    assert isinstance(engine, DummyEngine)
    assert engine.device == "cuda"
    # Overridden values
    assert engine.language == "en"
    assert engine.model_size == "large"


def test_create_engine_rejects_auto():
    """Test that engine_type='auto' raises ValueError."""
    with pytest.raises(ValueError, match="engine_type='auto' is deprecated"):
        EngineFactory.create_engine(engine_type="auto", device="cpu")


def test_create_engine_rejects_unknown_engine():
    """Test that unknown engine types raise ValueError."""
    with pytest.raises(ValueError, match="Unknown engine type: nonexistent"):
        EngineFactory.create_engine(engine_type="nonexistent", device="cpu")


def test_get_engine_info_returns_metadata():
    """Test that get_engine_info returns correct metadata."""
    info = EngineFactory.get_engine_info("whispers2t")
    assert info is not None
    assert info["supported_languages"]
    assert "batch_size" in info["default_params"]
    # Verify 100 languages are supported (including yue/Cantonese)
    assert len(info["supported_languages"]) == 100
    assert "yue" in info["supported_languages"]
    # Verify available_model_sizes is present
    assert "available_model_sizes" in info
    assert "large-v3-turbo" in info["available_model_sizes"]


def test_get_available_engines():
    """Test that get_available_engines returns all engines."""
    engines = EngineFactory.get_available_engines()
    assert "whispers2t" in engines
    assert "reazonspeech" in engines
    assert engines["whispers2t"]["name"]
    assert engines["whispers2t"]["description"]


def test_get_engines_for_language():
    """Test that get_engines_for_language filters correctly."""
    ja_engines = EngineFactory.get_engines_for_language("ja")
    assert "reazonspeech" in ja_engines
    assert "whispers2t" in ja_engines

    en_engines = EngineFactory.get_engines_for_language("en")
    assert "parakeet" in en_engines
    assert "whispers2t" in en_engines

    # Test new language support (100 languages including yue/Cantonese)
    vi_engines = EngineFactory.get_engines_for_language("vi")
    assert "whispers2t" in vi_engines

    yue_engines = EngineFactory.get_engines_for_language("yue")
    assert "whispers2t" in yue_engines


class TestWhisperS2TValidation:
    """Test WhisperS2T engine parameter validation."""

    def test_invalid_model_size_raises_valueerror(self):
        """Test that invalid model_size raises ValueError."""
        from livecap_core.engines.whispers2t_engine import WhisperS2TEngine

        with pytest.raises(ValueError, match="Unsupported model_size"):
            WhisperS2TEngine(device="cpu", model_size="invalid-model")

    def test_invalid_compute_type_raises_valueerror(self):
        """Test that invalid compute_type raises ValueError."""
        from livecap_core.engines.whispers2t_engine import WhisperS2TEngine

        with pytest.raises(ValueError, match="Unsupported compute_type"):
            WhisperS2TEngine(device="cpu", compute_type="invalid-type")

    def test_invalid_language_raises_valueerror(self):
        """Test that invalid language raises ValueError."""
        from livecap_core.engines.whispers2t_engine import WhisperS2TEngine

        with pytest.raises(ValueError, match="Unsupported language"):
            WhisperS2TEngine(device="cpu", language="invalid-lang")

    def test_valid_model_sizes_accepted(self):
        """Test that all documented model sizes are accepted."""
        from livecap_core.engines.whispers2t_engine import VALID_MODEL_SIZES

        expected_sizes = {
            "tiny", "base", "small", "medium",
            "large-v1", "large-v2", "large-v3",
            "large-v3-turbo", "distil-large-v3",
        }
        assert VALID_MODEL_SIZES == expected_sizes

    def test_valid_compute_types_accepted(self):
        """Test that all documented compute types are accepted."""
        from livecap_core.engines.whispers2t_engine import VALID_COMPUTE_TYPES

        expected_types = {"auto", "int8", "int8_float16", "float16", "float32"}
        assert VALID_COMPUTE_TYPES == expected_types

    def test_language_count_is_100(self):
        """Test that WhisperS2T supports exactly 100 languages."""
        from livecap_core.engines.whisper_languages import WHISPER_LANGUAGES

        assert len(WHISPER_LANGUAGES) == 100
        assert "yue" in WHISPER_LANGUAGES  # Cantonese is the 100th language
