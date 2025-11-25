# 統合ベンチマークフレームワーク実装計画

> **作成日:** 2025-11-25
> **関連 Issue:** #86
> **ステータス:** 計画中

---

## 1. 概要

### 1.1 目的

livecap-cli の音声認識パイプライン全体を評価するための**統合ベンチマークフレームワーク**を構築する。

**Phase 1: VAD ベンチマーク**
- 複数の VAD バックエンドを比較評価
- 全 ASR エンジンとの組み合わせで評価

**Phase 2: ASR ベンチマーク（将来）**
- ASR エンジン単体の精度・速度を評価
- 言語別の最適エンジン選定

### 1.2 背景

- Phase 1 で Silero VAD をデフォルトとして採用
- `docs/reference/vad-comparison.md` の調査により、他の VAD（JaVAD, TenVAD）が優れている可能性
- 本リポジトリには **10種類の ASR エンジン**が実装済み
- VAD × ASR の最適な組み合わせを発見する必要がある

### 1.3 スコープ

| Phase | 含む | 含まない |
|-------|------|----------|
| **Phase 1 (VAD)** | VAD × 全ASR評価、11 VAD構成 | VAD の本番切り替え |
| **Phase 2 (ASR)** | ASR 単体評価、言語別最適化 | 新 ASR エンジン実装 |

---

## 2. アーキテクチャ

### 2.1 モジュラー設計

将来の ASR ベンチマークを見据えた**共通基盤 + 個別ベンチマーク**の設計。

```
benchmarks/
├── common/                    # 共通モジュール
│   ├── __init__.py
│   ├── metrics.py             # WER/CER/RTF 計算
│   ├── datasets.py            # データセット管理
│   ├── engines.py             # ASR エンジン管理
│   └── reports.py             # レポート生成
├── vad/                       # VAD ベンチマーク (Phase 1)
│   ├── __init__.py
│   ├── runner.py              # VAD ベンチマーク実行
│   ├── cli.py                 # CLI エントリポイント
│   └── backends/              # VAD バックエンド
│       ├── __init__.py
│       ├── base.py            # VADBackend Protocol
│       ├── silero.py
│       ├── tenvad.py
│       ├── javad.py
│       └── webrtc.py
└── asr/                       # ASR ベンチマーク (Phase 2)
    ├── __init__.py
    ├── runner.py              # ASR ベンチマーク実行
    └── cli.py                 # CLI エントリポイント
```

### 2.2 共通コンポーネント

#### ASR エンジン管理 (`common/engines.py`)

既存の `engines/` モジュールを活用し、ベンチマーク用に統一インターフェースを提供。

```python
from engines.metadata import EngineMetadata
from engines.engine_factory import EngineFactory

class BenchmarkEngineManager:
    """ベンチマーク用 ASR エンジン管理"""

    @staticmethod
    def get_engines_for_language(language: str) -> list[str]:
        """言語に対応するエンジン一覧を取得"""
        return EngineMetadata.get_engines_for_language(language)

    @staticmethod
    def create_engine(engine_id: str, device: str = "cuda"):
        """ベンチマーク用エンジンを作成（VAD無効化）"""
        config = {}

        # WhisperS2T のみ内蔵 VAD を無効化
        if engine_id.startswith("whispers2t_"):
            config["whispers2t"] = {"use_vad": False}

        return EngineFactory.create_engine(
            engine_id,
            device=device,
            config=config
        )

    @staticmethod
    def get_all_engines() -> dict[str, EngineInfo]:
        """全エンジン情報を取得"""
        return EngineMetadata.get_all()
```

---

## 3. 評価マトリクス

### 3.1 VAD × ASR × 言語

```
┌─────────────────────────────────────────────────────────────────┐
│                    評価マトリクス                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  VAD (11構成)          ASR (10エンジン)         言語 (2+)       │
│  ┌─────────────┐      ┌─────────────────┐      ┌──────────┐    │
│  │ Silero v5   │      │ reazonspeech    │──ja──│ Japanese │    │
│  │ Silero v6   │      │ parakeet_ja     │──ja──│          │    │
│  │ TenVAD      │  ×   │ parakeet        │──en──│ English  │    │
│  │ JaVAD tiny  │      │ canary          │──en──│          │    │
│  │ JaVAD bal.  │      │ voxtral         │──en──│ (Future) │    │
│  │ JaVAD prec. │      │ whispers2t_*    │──all─│ de,fr,es │    │
│  │ WebRTC 0-3  │      └─────────────────┘      └──────────┘    │
│  └─────────────┘                                                 │
│                                                                  │
│  Full Matrix: 11 VAD × 10 ASR × 2 Lang = 220 combinations       │
│  Practical:   11 VAD × 3-4 ASR/lang × 2 Lang ≈ 66-88 tests     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 言語別推奨エンジン

| 言語 | 推奨エンジン | 代替エンジン |
|------|-------------|-------------|
| **Japanese (ja)** | reazonspeech, parakeet_ja | whispers2t_base |
| **English (en)** | parakeet, canary | whispers2t_base, voxtral |
| **German (de)** | canary | whispers2t_base, voxtral |
| **French (fr)** | canary | whispers2t_base, voxtral |
| **Spanish (es)** | canary | whispers2t_base, voxtral |

### 3.3 実行戦略

全組み合わせ（220件）は非現実的なため、段階的に実行：

**Quick Mode** (CI デフォルト):
- VAD: Silero v6, JaVAD precise, WebRTC mode 3
- ASR: 言語別デフォルト 1-2 エンジン
- 約 12 テスト

**Standard Mode**:
- VAD: 全 11 構成
- ASR: 言語別 2-3 エンジン
- 約 44-66 テスト

**Full Mode** (手動実行):
- VAD: 全 11 構成
- ASR: 全対応エンジン
- 約 88+ テスト

---

## 4. ASR エンジン一覧

### 4.1 実装済みエンジン

`engines/metadata.py` から取得した全 10 エンジン：

| ID | 表示名 | 対応言語 | サイズ | VAD内蔵 |
|----|--------|---------|--------|---------|
| `reazonspeech` | ReazonSpeech K2 v2 | ja | 159MB | ❌ |
| `parakeet` | NVIDIA Parakeet TDT 0.6B | en | 1.2GB | ❌ |
| `parakeet_ja` | NVIDIA Parakeet TDT CTC JA | ja | 600MB | ❌ |
| `canary` | NVIDIA Canary 1B Flash | en,de,fr,es | 1.5GB | ❌ |
| `voxtral` | MistralAI Voxtral Mini 3B | en,es,fr,pt,hi,de,nl,it | 3GB | ❌ |
| `whispers2t_tiny` | WhisperS2T Tiny | 多言語 | 39MB | ✅ |
| `whispers2t_base` | WhisperS2T Base | 多言語 | 74MB | ✅ |
| `whispers2t_small` | WhisperS2T Small | 多言語 | 244MB | ✅ |
| `whispers2t_medium` | WhisperS2T Medium | 多言語 | 769MB | ✅ |
| `whispers2t_large_v3` | WhisperS2T Large-v3 | 多言語 | 1.55GB | ✅ |

### 4.2 VAD 無効化

WhisperS2T のみ内蔵 VAD を持つ。ベンチマーク時は `use_vad=False` で無効化：

```python
# engines/whispers2t_engine.py:52
self.use_vad = self.engine_config.get('use_vad', True)

# engines/whispers2t_engine.py:290-304
if self.use_vad:
    outputs = self.model.transcribe_with_vad(...)
else:
    outputs = self.model.transcribe(...)  # VAD なし
```

他のエンジン（Parakeet, ReazonSpeech, Canary, Voxtral）は VAD 関連コードがないため、そのまま使用可能。

---

## 5. VAD 構成一覧

### 5.1 ベンチマーク対象

**合計 11 構成**:

| VAD | モデル/設定 | ライセンス | 特徴 |
|-----|------------|-----------|------|
| Silero VAD v5 | ONNX | MIT | 旧バージョン、比較用 |
| Silero VAD v6 | ONNX | MIT | 現在のデフォルト |
| TenVAD | - | 独自 | 最軽量・最高速（評価のみ） |
| JaVAD | tiny | MIT | 0.64s window、即時検出向け |
| JaVAD | balanced | MIT | 1.92s window、バランス型 |
| JaVAD | precise | MIT | 3.84s window、最高精度 |
| WebRTC VAD | mode 0 | BSD | 最も寛容、誤検出少 |
| WebRTC VAD | mode 1 | BSD | やや厳格 |
| WebRTC VAD | mode 2 | BSD | 厳格 |
| WebRTC VAD | mode 3 | BSD | 最も厳格、見逃し多 |

### 5.2 VAD バックエンド実装

```python
# benchmarks/vad/backends/base.py
from typing import Protocol, List, Tuple
import numpy as np

class VADBackend(Protocol):
    """VAD バックエンドの共通インターフェース"""

    @property
    def name(self) -> str:
        """バックエンド名"""
        ...

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[Tuple[float, float]]:
        """
        音声から発話区間を検出

        Returns:
            [(start_sec, end_sec), ...] のリスト
        """
        ...

    def reset(self) -> None:
        """状態をリセット"""
        ...
```

---

## 6. 評価方法

### 6.1 End-to-End 評価（メイン）

```
┌─────────────────────────────────────────────────────────────────┐
│ End-to-End VAD 評価フロー（全エンジン対応）                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  テスト音声 (.wav) + 言語情報                                     │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ VAD Backend (11 configurations)                         │    │
│  │ → 音声セグメント検出                                      │    │
│  │ → [(start, end, audio), ...]                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ASR Engine (言語に応じて選択)                            │    │
│  │ ┌─────────────┬─────────────┬─────────────┐            │    │
│  │ │ ja:         │ en:         │ multi:      │            │    │
│  │ │ reazonspeech│ parakeet    │ whispers2t_*│            │    │
│  │ │ parakeet_ja │ canary      │ voxtral     │            │    │
│  │ └─────────────┴─────────────┴─────────────┘            │    │
│  │ → 各セグメントを文字起こし                                │    │
│  │ → 結果を結合                                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 評価                                                     │    │
│  │ → WER/CER 計算 (vs Ground Truth)                        │    │
│  │ → RTF 計測 (VAD + ASR)                                  │    │
│  │ → メモリ使用量                                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 評価指標

| カテゴリ | 指標 | 説明 |
|---------|------|------|
| **ASR 精度** | WER | Word Error Rate |
| | CER | Character Error Rate（日本語向け） |
| **VAD 性能** | RTF | Real-Time Factor |
| | Latency | 入力→出力の遅延 (ms) |
| | Segments | 検出セグメント数 |
| **リソース** | Memory | ピークメモリ使用量 (MB) |
| | GPU Util | GPU 使用率（CUDA 時） |

---

## 7. データセット

### 7.1 既存テストアセット

```
tests/assets/audio/
├── jsut_basic5000_0001_ja.wav      # 日本語（約3秒）
├── jsut_basic5000_0001_ja.txt      # トランスクリプト
├── librispeech_test-clean_1089-134686-0001_en.wav  # 英語（約4秒）
└── librispeech_test-clean_1089-134686-0001_en.txt  # トランスクリプト
```

### 7.2 拡張データセット（オプション）

```bash
export LIVECAP_JSUT_DIR=/path/to/jsut/jsut_ver1.1
export LIVECAP_LIBRISPEECH_DIR=/path/to/librispeech/test-clean
```

---

## 8. 実装ステップ

### Phase 1: VAD ベンチマーク

#### Step 1: 共通基盤構築
1. `benchmarks/common/` ディレクトリ作成
2. `metrics.py` - WER/CER/RTF 計算
3. `datasets.py` - データセットローダー
4. `engines.py` - ASR エンジン管理（EngineFactory 活用）
5. `reports.py` - レポート生成

#### Step 2: VAD バックエンド実装
1. `VADBackend` Protocol 定義
2. Silero VAD ラッパー（v5, v6）
3. JaVAD ラッパー（tiny, balanced, precise）
4. WebRTC VAD ラッパー（mode 0-3）
5. TenVAD ラッパー

#### Step 3: ベンチマークランナー
1. `VADBenchmarkRunner` 実装
2. 全 ASR エンジン対応
3. 言語別エンジン自動選択

#### Step 4: CLI & レポート
1. CLI エントリポイント
2. JSON/Markdown 出力
3. 比較レポート生成

#### Step 5: CI 統合
1. GitHub Actions ワークフロー
2. Windows self-hosted runner (RTX 4090)

### Phase 2: ASR ベンチマーク（将来）

#### Step 1: ASR 単体評価
1. VAD なしでの ASR 評価
2. 言語別精度比較
3. 速度・メモリ比較

#### Step 2: エンジン推奨システム
1. 言語 × ユースケース別推奨
2. 自動エンジン選択の改善

---

## 9. CLI インターフェース

### 9.1 VAD ベンチマーク

```bash
# クイック実行（デフォルト VAD + デフォルト ASR）
python -m benchmarks.vad

# 全 VAD × 言語別推奨 ASR
python -m benchmarks.vad --all-vad

# 特定 VAD + 全対応 ASR
python -m benchmarks.vad --vad silero_v6 javad_precise --all-asr

# 全 VAD × 全 ASR（フルモード）
python -m benchmarks.vad --full

# 言語指定
python -m benchmarks.vad --language ja --all-vad

# 特定エンジン指定
python -m benchmarks.vad --asr reazonspeech parakeet_ja whispers2t_base

# 出力形式
python -m benchmarks.vad --output results.json --format json
python -m benchmarks.vad --output report.md --format markdown
```

### 9.2 ASR ベンチマーク（Phase 2）

```bash
# 全エンジン比較
python -m benchmarks.asr --all

# 言語別比較
python -m benchmarks.asr --language ja

# 特定エンジン
python -m benchmarks.asr --engine reazonspeech parakeet_ja
```

---

## 10. 出力フォーマット

### 10.1 コンソール出力

```
=== VAD Benchmark Results ===

Dataset: tests/assets/audio (2 files)
Mode: Standard (11 VAD × 3 ASR/lang)

┌─────────────────────────────────────────────────────────────────────────┐
│ Japanese Results (reazonspeech)                                          │
├─────────────┬────────┬────────┬────────┬──────────┬──────────┬─────────┤
│ VAD         │ WER    │ CER    │ RTF    │ Segments │ Memory   │ Status  │
├─────────────┼────────┼────────┼────────┼──────────┼──────────┼─────────┤
│ Silero v6   │ 5.2%   │ 2.1%   │ 0.012  │ 3        │ 245 MB   │ ✓       │
│ JaVAD prec. │ 4.1%   │ 1.5%   │ 0.015  │ 3        │ 312 MB   │ ✓       │
│ WebRTC m3   │ 8.3%   │ 4.2%   │ 0.003  │ 6        │ 128 MB   │ ✓       │
└─────────────┴────────┴────────┴────────┴──────────┴──────────┴─────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ English Results (parakeet)                                               │
├─────────────┬────────┬────────┬────────┬──────────┬──────────┬─────────┤
│ VAD         │ WER    │ CER    │ RTF    │ Segments │ Memory   │ Status  │
├─────────────┼────────┼────────┼────────┼──────────┼──────────┼─────────┤
│ Silero v6   │ 4.8%   │ 3.2%   │ 0.018  │ 2        │ 1.4 GB   │ ✓       │
│ JaVAD prec. │ 4.2%   │ 2.7%   │ 0.021  │ 2        │ 1.5 GB   │ ✓       │
│ WebRTC m3   │ 7.1%   │ 5.1%   │ 0.008  │ 5        │ 1.3 GB   │ ✓       │
└─────────────┴────────┴────────┴────────┴──────────┴──────────┴─────────┘

=== Summary ===
Best VAD for Japanese: JaVAD precise (CER: 1.5%)
Best VAD for English:  JaVAD precise (WER: 4.2%)
Fastest VAD:           WebRTC mode 3 (RTF: 0.003)
```

### 10.2 JSON 出力

```json
{
  "metadata": {
    "timestamp": "2025-11-25T12:00:00Z",
    "device": "cuda (RTX 4090)",
    "mode": "standard"
  },
  "results": [
    {
      "vad": "silero_v6",
      "asr": "reazonspeech",
      "language": "ja",
      "audio_file": "jsut_basic5000_0001_ja.wav",
      "metrics": {
        "wer": 0.052,
        "cer": 0.021,
        "rtf": 0.012,
        "segments": 3,
        "memory_mb": 245
      },
      "transcript": "水をマレーシアから買わなくてはならないのです",
      "reference": "水をマレーシアから買わなくてはならないのです"
    }
  ],
  "summary": {
    "best_vad_by_language": {
      "ja": {"vad": "javad_precise", "cer": 0.015},
      "en": {"vad": "javad_precise", "wer": 0.042}
    },
    "fastest_vad": {"vad": "webrtc_mode3", "rtf": 0.003}
  }
}
```

---

## 11. CI ワークフロー

### 11.1 ワークフロー設計

```yaml
# .github/workflows/benchmark.yml
name: Benchmark

on:
  workflow_dispatch:
    inputs:
      benchmark_type:
        description: 'Benchmark type'
        required: true
        default: 'vad'
        type: choice
        options:
          - vad
          - asr
          - both
      mode:
        description: 'Execution mode'
        required: true
        default: 'quick'
        type: choice
        options:
          - quick
          - standard
          - full
      language:
        description: 'Target language (empty for all)'
        required: false
        default: ''

jobs:
  benchmark-gpu:
    name: GPU Benchmark (Windows RTX 4090)
    runs-on: [self-hosted, windows]
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup FFmpeg
        run: |
          $ffmpegBinDir = Join-Path $env:GITHUB_WORKSPACE "ffmpeg-bin"
          # ... (既存の FFmpeg セットアップ)

      - name: Setup Python environment
        run: |
          uv sync --extra vad --extra engines-torch --extra engines-nemo --extra benchmark

      - name: Run Benchmark
        env:
          LIVECAP_FFMPEG_BIN: ${{ github.workspace }}\ffmpeg-bin
          LIVECAP_DEVICE: cuda
        run: |
          $type = "${{ github.event.inputs.benchmark_type }}"
          $mode = "${{ github.event.inputs.mode }}"
          $lang = "${{ github.event.inputs.language }}"

          $args = @("--mode", $mode, "--output", "results.json", "--format", "json")
          if ($lang) { $args += @("--language", $lang) }

          if ($type -eq "vad" -or $type -eq "both") {
            uv run python -m benchmarks.vad @args
          }
          if ($type -eq "asr" -or $type -eq "both") {
            uv run python -m benchmarks.asr @args
          }

      - name: Generate Report
        run: |
          uv run python -m benchmarks.common.reports --input results.json --output report.md

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results-${{ github.run_id }}
          path: |
            results.json
            report.md

      - name: Post Summary
        run: |
          Get-Content report.md | Add-Content $env:GITHUB_STEP_SUMMARY
```

### 11.2 実行モード

| モード | VAD 構成 | ASR エンジン | 推定時間 |
|--------|---------|-------------|---------|
| `quick` | 3 (Silero v6, JaVAD, WebRTC) | 言語別 1 | ~5分 |
| `standard` | 11 (全構成) | 言語別 2-3 | ~20分 |
| `full` | 11 (全構成) | 全対応 | ~60分 |

---

## 12. 依存関係

### 12.1 pyproject.toml 追加

```toml
[project.optional-dependencies]
benchmark = [
    # VAD backends
    "silero-vad>=5.1",
    "webrtcvad>=2.0.10",
    "javad",
    # Metrics
    "jiwer>=3.0",
    # Reporting
    "matplotlib",
    "pandas",
    "tabulate",
    # Profiling
    "memory_profiler",
]

benchmark-full = [
    "livecap-cli[benchmark]",
    "ten-vad",  # ライセンス注意
]
```

### 12.2 既存依存の活用

- `engines/` - ASR エンジン実装
- `engines/metadata.py` - エンジンメタデータ
- `engines/engine_factory.py` - エンジン生成

---

## 13. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| TenVAD ライセンス問題 | 商用利用不可 | 評価のみに使用、`benchmark-full` で分離 |
| 大規模モデルのメモリ不足 | テスト失敗 | エンジンごとにメモリ解放、順次実行 |
| 全組み合わせの実行時間 | CI タイムアウト | モード分離（quick/standard/full） |
| エンジン依存関係の競合 | インストール失敗 | `engines-nemo` と `engines-torch` を分離 |

---

## 14. 将来の拡張

### 14.1 ASR ベンチマーク (Phase 2)

- VAD なしでの ASR 単体評価
- エンジン間の精度・速度比較
- 言語別最適エンジンの自動選択

### 14.2 ノイズ耐性評価

- DEMAND ノイズデータセットとの混合
- SNR 別の精度評価

### 14.3 リアルタイム性能評価

- ストリーミング処理のレイテンシ測定
- メモリ使用量の時系列分析

---

## 15. 参考資料

- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [Silero VAD Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)
- [JaVAD GitHub](https://github.com/skrbnv/javad)
- [TenVAD GitHub](https://github.com/TEN-framework/ten-vad)
- [Anwarvic/VAD_Benchmark](https://github.com/Anwarvic/VAD_Benchmark)
- [jiwer (WER calculation)](https://github.com/jitsi/jiwer)
- `engines/metadata.py` - エンジンメタデータ定義
- `docs/reference/vad-comparison.md` - VAD 比較調査
