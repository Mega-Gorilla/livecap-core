# WhisperS2T エンジン統合 実装計画

> **Status**: PLANNING
> **作成日:** 2025-12-04
> **関連 Issue:** #165
> **依存:** #71 (Phase 3: パッケージ構造整理) ✅ 完了

---

## 1. 背景と目的

### 1.1 現状の課題

現在 `metadata.py` に5つの別エンジンとして WhisperS2T が定義されている。
しかし `WhisperS2TEngine` は既に `model_size` パラメータで任意のモデルを指定可能な実装になっている。

| 課題 | 詳細 | 影響度 |
|------|------|--------|
| **冗長なメタデータ定義** | 同じクラスが5つのエントリとして定義 | 中 |
| **モデル追加の手間** | 新モデル追加時に新エントリが必要 | 中 |
| **一貫性の欠如** | 他のエンジンはパラメータで切り替え可能 | 低 |
| **compute_type の最適化不足** | CPU で `float32` 使用（`int8` が1.5倍高速） | 中 |

### 1.2 目標

1. **5つのエントリを1つに統合**: `whispers2t` + `model_size` パラメータ
2. **新モデルの追加**: large-v1, large-v2, large-v3-turbo, distil-large-v3
3. **compute_type パラメータ追加**: デフォルト `auto` でデバイス最適化

---

## 2. 現状分析

### 2.1 現在の metadata.py 定義

```python
"whispers2t_tiny": EngineInfo(id="whispers2t_tiny", default_params={"model_size": "tiny", ...}),
"whispers2t_base": EngineInfo(id="whispers2t_base", default_params={"model_size": "base", ...}),
"whispers2t_small": EngineInfo(id="whispers2t_small", default_params={"model_size": "small", ...}),
"whispers2t_medium": EngineInfo(id="whispers2t_medium", default_params={"model_size": "medium", ...}),
"whispers2t_large_v3": EngineInfo(id="whispers2t_large_v3", default_params={"model_size": "large-v3", ...}),
```

### 2.2 現在の whispers2t_engine.py

```python
class WhisperS2TEngine(BaseEngine):
    def __init__(
        self,
        device: Optional[str] = None,
        language: str = "ja",
        model_size: str = "base",  # ← 既にパラメータ化済み
        batch_size: int = 24,
        use_vad: bool = True,
        **kwargs,
    ):
        self.device, self.compute_type = detect_device(device, "WhisperS2T")
        # ...
```

### 2.3 使用箇所の調査結果

| カテゴリ | ファイル数 | 主なパターン |
|----------|-----------|--------------|
| **tests/** | 4 | `whispers2t_base`, `whispers2t_large_v3`, `startswith("whispers2t_")` |
| **examples/** | 4 | `whispers2t_base`, `startswith("whispers2t_")` |
| **benchmarks/** | 3 | `whispers2t_large_v3`, `startswith("whispers2t_")` |
| **CI** | 1 | `whispers2t_base`, `startswith("whispers2t_")` |
| **docs/** | 15 | 各種言及（アーカイブ含む） |
| **livecap_core/** | 2 | `library_preloader.py`, `languages.py` |

---

## 3. 変更概要

### 3.1 エンジンID の変更

| Before | After |
|--------|-------|
| `whispers2t_tiny` | `whispers2t` + `model_size="tiny"` |
| `whispers2t_base` | `whispers2t` + `model_size="base"` (デフォルト) |
| `whispers2t_small` | `whispers2t` + `model_size="small"` |
| `whispers2t_medium` | `whispers2t` + `model_size="medium"` |
| `whispers2t_large_v3` | `whispers2t` + `model_size="large-v3"` |

### 3.2 新規追加モデル

| モデル | サイズ | 特徴 | CT2モデル |
|--------|--------|------|-----------|
| `large-v1` | 1.55GB | 初代大型モデル | 要確認 |
| `large-v2` | 1.55GB | v1の改良版 | 要確認 |
| `large-v3-turbo` | ~1.6GB | v3ベース、8倍高速 (2024年10月リリース) | [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) |
| `distil-large-v3` | ~756MB | v3比1%以内のWERで6倍高速 | [Systran/faster-distil-whisper-large-v3](https://huggingface.co/Systran/faster-distil-whisper-large-v3) |

> **注意**: 新規追加モデルについては、CTranslate2変換済みモデルの存在を実装前に確認すること。

### 3.3 新規追加パラメータ: `compute_type`

CTranslate2 の量子化タイプを制御するパラメータ。

| 値 | 説明 |
|----|------|
| `auto` (デフォルト) | デバイスに応じて最適値を自動選択 |
| `int8` | 整数8bit (CPU推奨、1.5倍高速) |
| `int8_float16` | 混合精度 (GPU高速) |
| `float16` | 半精度浮動小数点 (GPU標準) |
| `float32` | 単精度浮動小数点 (精度重視) |

**自動選択ロジック:**
- CPU: `int8` (float32比で1.5倍高速、メモリ35%削減)
- GPU: `float16` (標準的な精度と速度のバランス)

### 3.4 入力バリデーション方針

無効な `model_size` または `compute_type` が指定された場合の挙動：

| パラメータ | 無効値の挙動 | 理由 |
|-----------|-------------|------|
| `model_size` | `ValueError` を送出 | ダウンロード失敗を防ぐため早期検出 |
| `compute_type` | `ValueError` を送出 | ランタイムエラーを防ぐため早期検出 |

```python
VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3"}
VALID_COMPUTE_TYPES = {"auto", "int8", "int8_float16", "float16", "float32"}

def __init__(self, ..., model_size: str = "base", compute_type: str = "auto", ...):
    if model_size not in VALID_MODEL_SIZES:
        raise ValueError(f"Invalid model_size: {model_size}. Valid: {VALID_MODEL_SIZES}")
    if compute_type not in VALID_COMPUTE_TYPES:
        raise ValueError(f"Invalid compute_type: {compute_type}. Valid: {VALID_COMPUTE_TYPES}")
```

---

## 4. 実装タスク

### 4.1 Task 1: `EngineInfo` dataclass 拡張

**ファイル:** `livecap_core/engines/metadata.py`

```python
@dataclass
class EngineInfo:
    """エンジン情報"""
    id: str
    display_name: str
    description: str
    supported_languages: List[str]
    requires_download: bool = False
    model_size: Optional[str] = None
    device_support: List[str] = field(default_factory=lambda: ["cpu"])
    streaming: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)
    module: Optional[str] = None
    class_name: Optional[str] = None
    available_model_sizes: Optional[List[str]] = None  # 追加
```

### 4.2 Task 2: `metadata.py` の WhisperS2T エントリ統合

5つのエントリを1つに統合し、`compute_type` を追加:

```python
"whispers2t": EngineInfo(
    id="whispers2t",
    display_name="WhisperS2T",
    description="Multilingual ASR model with selectable model sizes (tiny to large-v3-turbo)",
    supported_languages=["ja", "en", "zh-CN", "zh-TW", "ko", "de", "fr", "es", "ru", "ar", "pt", "it", "hi"],
    requires_download=True,
    model_size=None,  # 複数サイズ対応のため None
    device_support=["cpu", "cuda"],
    streaming=True,
    module=".whispers2t_engine",
    class_name="WhisperS2TEngine",
    available_model_sizes=[
        # 標準モデル
        "tiny", "base", "small", "medium",
        # 大型モデル
        "large-v1", "large-v2", "large-v3",
        # 高速モデル
        "large-v3-turbo", "distil-large-v3",
    ],
    default_params={
        "model_size": "base",
        "compute_type": "auto",  # NEW: デフォルトは自動最適化
        "batch_size": 24,
        "use_vad": True,
    },
),
```

### 4.3 Task 3: `whispers2t_engine.py` に `compute_type` パラメータ追加

```python
VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3"}
VALID_COMPUTE_TYPES = {"auto", "int8", "int8_float16", "float16", "float32"}

class WhisperS2TEngine(BaseEngine):
    def __init__(
        self,
        device: Optional[str] = None,
        language: str = "ja",
        model_size: str = "base",
        compute_type: str = "auto",  # NEW
        batch_size: int = 24,
        use_vad: bool = True,
        **kwargs,
    ):
        # 入力バリデーション
        if model_size not in VALID_MODEL_SIZES:
            raise ValueError(f"Invalid model_size: {model_size}. Valid: {VALID_MODEL_SIZES}")
        if compute_type not in VALID_COMPUTE_TYPES:
            raise ValueError(f"Invalid compute_type: {compute_type}. Valid: {VALID_COMPUTE_TYPES}")

        # detect_device() は Tuple[str, str] を返すため、最初の要素のみ使用
        # 注: #166 完了後は戻り値が str になる
        device_result = detect_device(device, "WhisperS2T")
        self.device = device_result[0] if isinstance(device_result, tuple) else device_result

        self.compute_type = self._resolve_compute_type(compute_type)  # NEW
        # ...

    def _resolve_compute_type(self, compute_type: str) -> str:
        """compute_typeを解決（autoの場合はデバイスに応じて最適化）"""
        if compute_type != "auto":
            return compute_type  # ユーザー指定を尊重

        # auto: デバイスに応じた最適値
        # CPU: int8 (1.5x faster than float32, 35% less memory)
        # GPU: float16 (standard precision/speed balance)
        return "int8" if self.device == "cpu" else "float16"
```

### 4.4 Task 4: `LibraryPreloader` 更新

**ファイル:** `livecap_core/engines/library_preloader.py`

`SharedEngineManager` は `engine_type.split('_')[0]` を渡すため、統合後は `start_preloading("whispers2t")` になる。
現在の `_get_required_libraries` は `whispers2t_base` / `whispers2t_large_v3` 固定のため更新が必要。

```python
# Before (86-96行目)
library_map = {
    'parakeet': {'matplotlib', 'nemo'},
    'parakeet_ja': {'matplotlib', 'nemo'},
    'canary': {'matplotlib', 'nemo'},
    'voxtral': {'transformers'},
    'whispers2t_base': {'whisper_s2t'},      # ← 旧エンジンID
    'whispers2t_large_v3': {'whisper_s2t'},  # ← 旧エンジンID
    'reazonspeech': {'sherpa_onnx'},
}

# After
library_map = {
    'parakeet': {'matplotlib', 'nemo'},
    'parakeet_ja': {'matplotlib', 'nemo'},
    'canary': {'matplotlib', 'nemo'},
    'voxtral': {'transformers'},
    'whispers2t': {'whisper_s2t'},  # ← 統合エンジンID
    'reazonspeech': {'sherpa_onnx'},
}
```

### 4.5 Task 5: `languages.py` 更新

**ファイル:** `livecap_core/languages.py`

全16言語の `supported_engines` リストを更新:

```python
# Before (各言語で)
supported_engines=["reazonspeech", "whispers2t_base", "whispers2t_tiny",
                   "whispers2t_small", "whispers2t_medium", "whispers2t_large_v3",
                   "canary", "voxtral"],

# After
supported_engines=["reazonspeech", "whispers2t", "canary", "voxtral"],
```

**対象言語:** ja, en, zh-CN, zh-TW, ko, de, fr, es, ru, ar, pt, it, hi, nl（計14言語）

> **注意:** `Languages.get_engines_for_language()` の結果に影響するため、UI/CLIの言語→エンジン対応が正しく動作することを確認すること。

### 4.6 Task 6: 使用箇所の更新

#### 4.6.1 tests/

| ファイル | 変更内容 |
|----------|----------|
| `core/engines/test_engine_factory.py` | `whispers2t_base` → `whispers2t` |
| `core/cli/test_cli.py` | CLI出力の期待値確認・更新 |
| `integration/engines/test_smoke_engines.py` | 各バリエーションを `model_size` パラメータで指定、`startswith` 削除 |
| `integration/realtime/test_e2e_realtime_flow.py` | `whispers2t_base` → `whispers2t`、`startswith` 削除 |
| `asr/test_runner.py` | `whispers2t_large_v3` → `whispers2t` + `model_size="large-v3"` |
| `vad/test_runner.py` | 同上 |

#### 4.6.2 examples/

| ファイル | 変更内容 |
|----------|----------|
| `realtime/basic_file_transcription.py` | `whispers2t_base` → `whispers2t`、`startswith` → `==` |
| `realtime/async_microphone.py` | 同上 |
| `realtime/callback_api.py` | 同上 |
| `realtime/custom_vad_config.py` | 同上 |

#### 4.6.3 benchmarks/

| ファイル | 変更内容 |
|----------|----------|
| `asr/runner.py` | `whispers2t_large_v3` → `whispers2t` (+ model_size) |
| `vad/runner.py` | 同上 |
| `common/engines.py` | `startswith("whispers2t_")` → `== "whispers2t"` (160行目) |

#### 4.6.4 CI

| ファイル | 変更内容 |
|----------|----------|
| `.github/workflows/integration-tests.yml` | `whispers2t_base` → `whispers2t`、`startswith` 削除 (158, 355行目) |

#### 4.6.5 core

| ファイル | 変更内容 |
|----------|----------|
| `livecap_core/engines/engine_factory.py` | docstring 更新 |
| `livecap_core/engines/shared_engine_manager.py` | 必要に応じて更新 |

### 4.7 Task 7: ドキュメント更新

**全文検索で置換・確認が必要なファイル:**

```bash
grep -r "whispers2t_" docs/ --include="*.md" | grep -v "archive/"
```

| ファイル | 変更内容 |
|----------|----------|
| `README.md` | 新しい使用方法に更新 |
| `CLAUDE.md` | エンジン一覧更新 |
| `docs/guides/realtime-transcription.md` | 使用例更新 |
| `docs/guides/benchmark/asr-benchmark.md` | ベンチマーク使用例更新 |
| `docs/guides/benchmark/vad-benchmark.md` | 同上 |
| `docs/guides/benchmark/vad-optimization.md` | 同上 |
| `docs/architecture/core-api-spec.md` | API 仕様更新 |
| `docs/reference/feature-inventory.md` | エンジン一覧更新 |
| `docs/reference/vad/config.md` | 使用例更新 |
| `docs/reference/vad/comparison.md` | エンジン言及更新 |
| `docs/planning/refactoring-plan.md` | 必要に応じて更新 |
| `docs/planning/phase3-package-restructure.md` | 必要に応じて更新 |

> **アーカイブ (`docs/planning/archive/*`)** は更新不要。

---

## 5. 実装順序

```
Step 1: ブランチ作成
    git checkout -b feat/whispers2t-consolidation
    ↓
Step 2: CT2モデル存在確認
    新モデル (large-v1, large-v2, large-v3-turbo, distil-large-v3) の
    CTranslate2変換済みモデルが利用可能か確認
    ↓
Step 3: EngineInfo dataclass に available_model_sizes 追加
    livecap_core/engines/metadata.py
    ↓
Step 4: WhisperS2T エントリ統合 (5→1)
    - 5つのエントリを削除
    - 統合エントリを追加
    ↓
Step 5: whispers2t_engine.py 更新
    - compute_type パラメータ追加
    - _resolve_compute_type() メソッド追加
    - 入力バリデーション追加
    - detect_device() 戻り値のタプル対応
    ↓
Step 6: LibraryPreloader 更新
    library_map に 'whispers2t' エントリ追加
    ↓
Step 7: languages.py 更新
    全16言語の supported_engines を更新
    ↓
Step 8: テストコード更新
    - test_engine_factory.py
    - test_smoke_engines.py
    - test_e2e_realtime_flow.py
    - test_cli.py (CLI出力確認)
    ↓
Step 9: examples 更新 (4ファイル)
    ↓
Step 10: benchmarks 更新 (3ファイル)
    ↓
Step 11: CI ワークフロー更新
    ↓
Step 12: テスト実行
    uv run pytest tests/ -v
    ↓
Step 13: pip install -e . で確認
    ↓
Step 14: ドキュメント更新 (全文検索で漏れなく)
    ↓
Step 15: CLI出力確認
    livecap-core --info で whispers2t が正しく表示されることを確認
    ↓
Step 16: PR 作成・レビュー・マージ
```

---

## 6. 新しい使用方法

```python
from livecap_core import EngineFactory, EngineMetadata

# 基本使用（デフォルト: base, compute_type=auto）
engine = EngineFactory.create_engine("whispers2t", device="cuda")

# モデルサイズ指定
engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="large-v3")
engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="large-v3-turbo")

# compute_type 明示指定（上級ユーザー向け）
engine = EngineFactory.create_engine("whispers2t", device="cpu", compute_type="int8")
engine = EngineFactory.create_engine("whispers2t", device="cuda", compute_type="int8_float16")
engine = EngineFactory.create_engine("whispers2t", device="cuda", compute_type="float32")  # 精度重視

# 利用可能なモデルサイズの確認
info = EngineMetadata.get("whispers2t")
print(info.available_model_sizes)
# ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3"]
```

---

## 7. 多言語エンジン判定の変更

```python
# Before
if engine_type.startswith("whispers2t_") or engine_type in ("canary", "voxtral"):
    engine_options["language"] = lang

# After
if engine_type in ("whispers2t", "canary", "voxtral"):
    engine_options["language"] = lang
```

---

## 8. 検証項目

### 8.1 事前確認

- [ ] 新モデル (`large-v1`, `large-v2`) の CT2 変換済みモデルが利用可能
- [ ] 新モデル (`large-v3-turbo`, `distil-large-v3`) の CT2 変換済みモデルが利用可能

### 8.2 単体テスト

- [ ] `tests/core/engines/test_engine_factory.py` がパス
- [ ] 全 `tests/core/` テストがパス
- [ ] 入力バリデーションテスト（無効な model_size, compute_type）

### 8.3 統合テスト

- [ ] `tests/integration/engines/test_smoke_engines.py` がパス
- [ ] `tests/integration/realtime/test_e2e_realtime_flow.py` がパス
- [ ] 全 `tests/integration/` テストがパス

### 8.4 機能テスト

- [ ] `EngineFactory.create_engine("whispers2t")` が動作
- [ ] `model_size` パラメータで各サイズが指定可能
- [ ] `compute_type="auto"` がデバイスに応じて正しく解決される（CPU→int8, GPU→float16）
- [ ] `compute_type` 明示指定が正しく反映される
- [ ] 無効な `model_size` で `ValueError` が発生
- [ ] 無効な `compute_type` で `ValueError` が発生
- [ ] 新モデル (`large-v3-turbo`, `distil-large-v3`) の動作確認
- [ ] `LibraryPreloader.start_preloading("whispers2t")` が正しく動作

### 8.5 言語マスター

- [ ] `Languages.get_engines_for_language("ja")` に `whispers2t` が含まれる
- [ ] `Languages.get_engines_for_language("en")` に `whispers2t` が含まれる
- [ ] 旧エンジンID (`whispers2t_base` 等) が結果に含まれない

### 8.6 CLI

- [ ] `livecap-core --info` で `whispers2t` が表示される
- [ ] `livecap-core --info` で旧エンジンID が表示されない
- [ ] `tests/core/cli/test_cli.py` がパス

### 8.7 Examples

- [ ] 全 examples が正常に動作

### 8.8 CI

- [ ] 全ワークフローがグリーン

---

## 9. 完了条件

- [ ] `metadata.py` の WhisperS2T エントリが1つに統合されている
- [ ] `EngineInfo` に `available_model_sizes` フィールドが追加されている
- [ ] `whispers2t_engine.py` に `compute_type` パラメータが追加されている
- [ ] `_resolve_compute_type()` で自動最適化が実装されている
- [ ] `model_size` と `compute_type` の入力バリデーションが実装されている
- [ ] `detect_device()` のタプル戻り値が正しく処理されている
- [ ] `LibraryPreloader` が `whispers2t` に対応している
- [ ] `languages.py` の全言語で `supported_engines` が更新されている
- [ ] 全使用箇所が更新されている（全文検索で確認）
- [ ] 全テストがパス
- [ ] ドキュメントが更新されている
- [ ] CI が全てグリーン

---

## 10. リスクと対策

| リスク | レベル | 対策 |
|--------|--------|------|
| 使用箇所の更新漏れ | 中 | `grep -r "whispers2t_"` で網羅的に検索、テストで検出 |
| 新モデルのCT2モデル不在 | 中 | 実装前にHugging Faceで存在確認 |
| detect_device() タプル処理ミス | 中 | 型チェックで安全に処理 |
| LibraryPreloader 事前ロード失敗 | 中 | 統合テストで確認 |
| languages.py 更新漏れ | 中 | 全言語をリストアップして確認 |
| compute_type の誤設定 | 低 | バリデーションとユニットテストでカバー |
| CI 失敗 | 中 | ローカルで全テスト実行後に PR 作成 |

---

## 11. 関連 Issue

- **#166**: `detect_device()` リファクタリング（本 Issue 完了後に実施）
  - 戻り値を `Tuple[str, str]` → `str` に変更
  - `compute_type` は WhisperS2T 内部で解決するため不要に
  - 本 Issue では暫定対処（タプルの最初の要素を使用）

---

## 12. 参考資料

- [WhisperS2T GitHub](https://github.com/shashikg/WhisperS2T)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2 Quantization](https://opennmt.net/CTranslate2/quantization.html)
- [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
- [Systran/faster-distil-whisper-large-v3](https://huggingface.co/Systran/faster-distil-whisper-large-v3)

---

## 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-12-04 | 初版作成 |
| 2025-12-04 | レビュー対応: LibraryPreloader更新追加、languages.py更新追加、detect_device()タプル処理明記、入力バリデーション追加、CT2モデル確認手順追加、ドキュメント網羅性向上、CLIテスト確認追加 |
