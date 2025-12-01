# Phase 2: API 統一と Config 簡素化 実装計画

> **Status**: 📋 PLANNING
> **作成日:** 2025-12-01
> **関連 Issue:** #70
> **依存:** #69 (Phase 1: リアルタイム文字起こし実装) ✅ 完了

---

## 1. 背景と目的

### 1.1 現状の課題

Phase 1 で `StreamTranscriber` + `VADProcessor` + `VADConfig` を実装したが、既存の Config システムとの間に以下の不整合が存在する：

| 課題 | 詳細 | 影響度 |
|------|------|--------|
| VAD 設定の二重定義 | `silence_detection` と `VADConfig` でパラメータ名が異なる | 高 |
| GUI 専用セクションの残存 | `multi_source`, `vad_state_machine`, `queue` 等 | 中 |
| config/ ディレクトリの分散 | `config/` と `livecap_core/config/` が分離 | 中 |
| セクション名の不一致 | `transcription` vs 目標の `engine` | 低 |

### 1.2 目標

1. **VADConfig との整合性確保**: `silence_detection` を `vad` にリネームし、VADConfig と同じパラメータ名に統一
2. **GUI 専用セクションの削除**: クリーンな CLI 向け Config スキーマ
3. **config/ ディレクトリの統合**: 単一の `livecap_core/config/` に集約
4. **既存機能の動作維持**: FileTranscriptionPipeline, EngineFactory 等が動作すること

---

## 2. 現状分析

### 2.1 現在の Config 構造（DEFAULT_CONFIG）

```python
# livecap_core/config/defaults.py
DEFAULT_CONFIG = {
    "audio": {                          # → 削除予定
        "sample_rate": 16000,
        "chunk_duration": 0.25,
        "processing": {...},            # GUI専用
    },
    "multi_source": {...},              # → 削除予定（GUI専用）
    "silence_detection": {              # → "vad" にリネーム
        "vad_threshold": 0.5,           # → threshold
        "vad_min_speech_duration_ms": 250,  # → min_speech_ms
        "vad_speech_pad_ms": 400,       # → speech_pad_ms
        "vad_min_silence_duration_ms": 100, # → min_silence_ms
        "vad_state_machine": {...},     # → 削除（GUI専用）
    },
    "transcription": {                  # → "engine" にリネーム
        "device": None,
        "engine": "auto",
        "input_language": "ja",
        "language_engines": {...},
    },
    "translation": {...},               # → 維持
    "engines": {...},                   # → 維持
    "logging": {...},                   # → 維持
    "queue": {...},                     # → 削除予定（GUI専用）
    "debug": {...},                     # → 削除予定
    "file_mode": {...},                 # → 維持
}
```

### 2.2 VADConfig（Phase 1 で作成）

```python
# livecap_core/vad/config.py
@dataclass(frozen=True, slots=True)
class VADConfig:
    threshold: float = 0.5
    neg_threshold: Optional[float] = None
    min_speech_ms: int = 250
    min_silence_ms: int = 100
    speech_pad_ms: int = 100
    max_speech_ms: int = 0
    interim_min_duration_ms: int = 2000
    interim_interval_ms: int = 1000
```

### 2.3 既存コードの Config 使用状況

| コンポーネント | 使用ファイル | 使用セクション | 備考 |
|---------------|-------------|---------------|------|
| EngineFactory | `engines/engine_factory.py` | `transcription.*` | `build_core_config()` 経由 |
| StreamTranscriber | `livecap_core/transcription/stream.py` | なし | `VADConfig` を直接使用 |
| FileTranscriptionPipeline | `livecap_core/transcription/file_pipeline.py` | なし | config 受け取るが未使用 |
| Examples | `examples/realtime/*.py` | `transcription.*` | `get_default_config()` 使用 |

---

## 3. 目標スキーマ

### 3.1 新しい Config 構造

```python
CORE_CONFIG = {
    "engine": {
        "type": "auto",
        "device": None,
        "language": "ja",
        "language_engines": {
            "ja": "reazonspeech",
            "en": "parakeet",
            "default": "whispers2t_base",
        },
    },
    "vad": {
        "enabled": True,
        "threshold": 0.5,
        "neg_threshold": None,          # VADConfig と同名
        "min_speech_ms": 250,           # VADConfig と同名
        "min_silence_ms": 100,          # VADConfig と同名
        "speech_pad_ms": 100,           # VADConfig と同名
        "max_speech_ms": 0,             # VADConfig と同名
    },
    "translation": {
        "enabled": False,
        "service": "google",
        "target_language": "en",
    },
    "engines": {
        "reazonspeech": {},
        "parakeet": {"model_name": "nvidia/parakeet-tdt-0.6b-v3"},
        "whispers2t_base": {"model_size": "base"},
        # ...
    },
    "logging": {
        "log_dir": "logs",
        "file_log_level": "INFO",
        "console_log_level": "INFO",
    },
    "file_mode": {
        "use_vad": True,
        "min_speech_duration_ms": 200,
        "max_silence_duration_ms": 300,
    },
}
```

### 3.2 変更サマリー

| セクション | 変更前 | 変更後 | 理由 |
|-----------|--------|--------|------|
| `audio` | 存在 | **削除** | AudioSource で直接指定 |
| `multi_source` | 存在 | **削除** | GUI 専用 |
| `silence_detection` | 存在 | **`vad` にリネーム** | VADConfig と整合 |
| `silence_detection.vad_state_machine` | 存在 | **削除** | GUI 専用 |
| `transcription` | 存在 | **`engine` にリネーム** | 明確化 |
| `translation` | 存在 | 維持 | - |
| `engines` | 存在 | 維持 | - |
| `logging` | 存在 | 維持 | - |
| `queue` | 存在 | **削除** | GUI 専用 |
| `debug` | 存在 | **削除** | logging に統合 |
| `file_mode` | 存在 | 維持 | - |

---

## 4. 実装タスク

### 4.1 Config スキーマの簡素化

#### Task 1.1: 新スキーマの定義

**ファイル:** `livecap_core/config/schema.py`

```python
# 変更内容
# 1. AudioConfig, AudioProcessingConfig を削除
# 2. SilenceDetectionConfig を VADConfig 互換の VADConfigSchema に変更
# 3. TranscriptionConfig を EngineConfig にリネーム
# 4. MultiSourceConfig, QueueConfig, DebugConfig を削除
# 5. CoreConfig を更新
```

#### Task 1.2: defaults.py の更新

**ファイル:** `livecap_core/config/defaults.py`

- GUI 専用セクションを削除
- `silence_detection` → `vad` にリネーム
- `transcription` → `engine` にリネーム
- パラメータ名を VADConfig と一致させる

#### Task 1.3: validator.py の更新

**ファイル:** `livecap_core/config/validator.py`

- 新しいスキーマに対応

#### Task 1.4: VADConfig.from_config() の追加

**ファイル:** `livecap_core/vad/config.py`

```python
@classmethod
def from_config(cls, config: dict) -> VADConfig:
    """Config の vad セクションから VADConfig を作成"""
    vad_section = config.get("vad", {})
    return cls(
        threshold=vad_section.get("threshold", 0.5),
        neg_threshold=vad_section.get("neg_threshold"),
        min_speech_ms=vad_section.get("min_speech_ms", 250),
        min_silence_ms=vad_section.get("min_silence_ms", 100),
        speech_pad_ms=vad_section.get("speech_pad_ms", 100),
        max_speech_ms=vad_section.get("max_speech_ms", 0),
    )
```

### 4.2 config/ ディレクトリの統合

#### Task 2.1: core_config_builder.py の移動

**変更内容:**
- `config/core_config_builder.py` → `livecap_core/config/builder.py`
- GUI 変換ロジックを削除（または分離）
- 新スキーマに対応

#### Task 2.2: インポートパスの更新

**影響ファイル:**
- `engines/engine_factory.py`: `from config.core_config_builder import build_core_config` を更新

#### Task 2.3: 旧 config/ ディレクトリの削除

- `config/__init__.py` と `config/core_config_builder.py` を削除

### 4.3 既存コードとの互換性確保

#### Task 3.1: EngineFactory の更新

**ファイル:** `engines/engine_factory.py`

- `transcription` → `engine` への参照変更
- `input_language` → `language` への参照変更
- 新しいインポートパス

#### Task 3.2: StreamTranscriber の更新（オプション）

**ファイル:** `livecap_core/transcription/stream.py`

- Config から VADConfig を作成する便利メソッドの追加

```python
@classmethod
def from_config(cls, engine: TranscriptionEngine, config: dict) -> StreamTranscriber:
    """Config から StreamTranscriber を作成"""
    vad_config = VADConfig.from_config(config)
    return cls(engine=engine, vad_config=vad_config)
```

#### Task 3.3: Examples の更新

**影響ファイル:**
- `examples/realtime/basic_file_transcription.py`
- `examples/realtime/async_microphone.py`
- `examples/realtime/callback_api.py`
- `examples/realtime/custom_vad_config.py`

#### Task 3.4: テストの更新

**影響ファイル:**
- `tests/core/config/test_config_defaults.py`
- `tests/core/config/test_core_config_builder.py`
- `tests/integration/engines/test_smoke_engines.py`

---

## 5. 移行戦略

### 5.1 互換性の扱い

**方針:** 破壊的変更を行う（互換性維持不要）

理由:
- 本リポジトリは外部で利用されていない
- クリーンな API 設計を優先

### 5.2 移行手順

```
Step 1: 新スキーマ定義（schema.py）
    ↓
Step 2: defaults.py 更新
    ↓
Step 3: validator.py 更新
    ↓
Step 4: VADConfig.from_config() 追加
    ↓
Step 5: builder.py 移動・更新
    ↓
Step 6: EngineFactory 更新
    ↓
Step 7: Examples 更新
    ↓
Step 8: テスト更新・実行
    ↓
Step 9: 旧 config/ 削除
```

---

## 6. 検証項目

### 6.1 単体テスト

- [ ] `test_config_defaults.py` が新スキーマでパス
- [ ] `test_core_config_builder.py` が新スキーマでパス
- [ ] VADConfig.from_config() のテスト追加

### 6.2 統合テスト

- [ ] `test_smoke_engines.py` がパス
- [ ] `test_file_transcription_pipeline.py` がパス
- [ ] `test_e2e_realtime_flow.py` がパス（LIVECAP_ENABLE_REALTIME_E2E=1）

### 6.3 Examples 動作確認

- [ ] `basic_file_transcription.py` が動作
- [ ] `async_microphone.py` が動作
- [ ] `callback_api.py` が動作
- [ ] `custom_vad_config.py` が動作

---

## 7. リスクと対策

| リスク | 対策 |
|--------|------|
| テスト失敗 | 段階的に更新、各ステップで確認 |
| EngineFactory の挙動変化 | 慎重に参照パスを更新 |
| 見落としたコード | Grep で `silence_detection`, `transcription` を検索 |

---

## 8. 完了条件

- [ ] Config が新スキーマに簡素化
- [ ] `config/` ディレクトリが `livecap_core/config/` に統合
- [ ] VADConfig と Config スキーマが整合
- [ ] 全テストがパス
- [ ] Examples が新スキーマで動作

---

## 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-12-01 | 初版作成 |
