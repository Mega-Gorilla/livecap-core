# 統合ベンチマークフレームワーク実装計画

> **作成日:** 2025-11-25
> **関連 Issue:** #86
> **ステータス:** Phase C 実装準備中
> **最終更新:** 2025-11-28 (Phase C-2 設計決定: speech_ratio, vad_config, レポート拡張)

---

## 1. 概要

### 1.1 目的

livecap-cli の音声認識パイプライン全体を評価するための**統合ベンチマークフレームワーク**を構築する。

**VAD ベンチマーク + ASR ベンチマークを同時実装**し、以下を実現：

- 複数の VAD バックエンド（9構成）を比較評価
- 全 ASR エンジン（10種類）の単体性能を評価
- VAD × ASR の最適な組み合わせを発見

### 1.2 背景

- Phase 1 で Silero VAD をデフォルトとして採用
- `docs/reference/vad-comparison.md` の調査により、他の VAD（JaVAD, TenVAD）が優れている可能性
- 本リポジトリには **10種類の ASR エンジン**が実装済み
- VAD × ASR の最適な組み合わせを発見する必要がある

### 1.3 同時実装の理由

```
┌─────────────────────────────────────────────────────────────────┐
│ コード共有率分析                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  共通部分 (80%):                                                 │
│  ├── metrics.py      # WER/CER/RTF計算                          │
│  ├── datasets.py     # 音声ファイル + Ground Truth管理           │
│  ├── engines.py      # 10エンジンの統一管理                      │
│  └── reports.py      # JSON/Markdown出力                        │
│                                                                  │
│  固有部分 (20%):                                                 │
│  ├── vad/runner.py   # VAD処理 + ASR呼び出し                    │
│  └── asr/runner.py   # ASR呼び出しのみ（VADスキップ）           │
│                                                                  │
│  → 共通基盤を作る時点で、ASRベンチマークは実質完成               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**工数比較:**

| アプローチ | 共通基盤 | VAD固有 | ASR固有 | 合計 |
|-----------|---------|---------|---------|------|
| VADのみ先行 | 5日 | 2日 | - | 7日 |
| 後からASR追加 | - | - | 3日 | **10日** |
| **同時実装** | 5日 | 2日 | 1日 | **8日** |

### 1.4 スコープ

| 含む | 含まない |
|------|----------|
| VAD × 全ASR評価（10 VAD × 10 ASR） | VAD/ASR の本番切り替え |
| ASR 単体評価（10エンジン） | 新エンジン実装 |
| 日本語・英語での評価 | 全言語での評価 |
| CLI ベンチマークツール | GUI |
| Windows self-hosted runner (RTX 4090) | GitHub-hosted GPU |

---

## 2. アーキテクチャ

### 2.1 モジュラー設計

**共通基盤 + 個別ベンチマーク**の設計で、コード再利用を最大化。

```
benchmarks/
├── common/                    # 共通モジュール (80%)
│   ├── __init__.py
│   ├── metrics.py             # WER/CER/RTF 計算
│   ├── datasets.py            # データセット管理
│   ├── engines.py             # ASR エンジン管理
│   └── reports.py             # レポート生成
├── asr/                       # ASR ベンチマーク
│   ├── __init__.py
│   ├── runner.py              # ASR ベンチマーク実行
│   └── cli.py                 # CLI エントリポイント
└── vad/                       # VAD ベンチマーク
    ├── __init__.py
    ├── runner.py              # VAD ベンチマーク実行
    ├── cli.py                 # CLI エントリポイント
    ├── factory.py             # VADFactory (engines/ パターン踏襲)
    └── backends/              # VAD バックエンド
        ├── __init__.py
        ├── base.py            # VADBackend Protocol
        ├── silero.py          # SileroVADBackend
        ├── javad.py           # JaVADBackend
        ├── webrtc.py          # WebRTCVADBackend
        └── tenvad.py          # TenVADBackend
```

### 2.2 評価フローの関係

```
┌─────────────────────────────────────────────────────────────────┐
│ ASR ベンチマーク（基礎）                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  テスト音声 (.wav)                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ASR Engine (10 engines)                                 │    │
│  │ → 音声全体を文字起こし                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 評価 (common/metrics.py)                                │    │
│  │ → WER/CER 計算                                          │    │
│  │ → RTF 計測                                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
              VAD ベンチマークは「VAD処理を前に挿入」するだけ
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ VAD ベンチマーク（ASRの上に構築）                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  テスト音声 (.wav)                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ VAD Backend (9 configurations)   ← 追加部分             │    │
│  │ → 音声セグメント検出                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ASR Engine (共通部分を再利用)                            │    │
│  │ → 各セグメントを文字起こし                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 評価 (common/metrics.py を再利用)                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 共通コンポーネント (✅ 実装済み)

> 実装: `benchmarks/common/`

| モジュール | 機能 |
|-----------|------|
| `metrics.py` | WER/CER/RTF計算、GPUMemoryTracker |
| `text_normalization.py` | 言語別テキスト正規化 |
| `datasets.py` | AudioFile (lazy load)、Dataset、DatasetManager |
| `engines.py` | BenchmarkEngineManager (キャッシュ、VAD無効化) |
| `reports.py` | BenchmarkReporter (JSON/Markdown/Console出力) |

---

## 3. 評価マトリクス

### 3.1 ASR ベンチマーク

```
┌─────────────────────────────────────────────────────────────────┐
│ ASR 評価マトリクス                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ASR (10エンジン)              言語 (2+)                         │
│  ┌─────────────────┐          ┌──────────┐                      │
│  │ reazonspeech    │──────ja──│ Japanese │                      │
│  │ parakeet_ja     │──────ja──│          │                      │
│  │ parakeet        │──────en──│ English  │                      │
│  │ canary          │──────en──│          │                      │
│  │ voxtral         │──────en──│ (Future) │                      │
│  │ whispers2t_*    │──────all─│ de,fr,es │                      │
│  └─────────────────┘          └──────────┘                      │
│                                                                  │
│  Total: 10 ASR × 2 Lang = 20 tests (言語対応分のみ)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 VAD × ASR ベンチマーク

```
┌─────────────────────────────────────────────────────────────────┐
│ VAD × ASR 評価マトリクス                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  VAD (9構成)           ASR (10エンジン)         言語 (2+)       │
│  ┌─────────────┐      ┌─────────────────┐      ┌──────────┐    │
│  │ Silero v6   │      │ reazonspeech    │──ja──│ Japanese │    │
│  │ TenVAD      │      │ parakeet_ja     │──ja──│          │    │
│  │ JaVAD tiny  │  ×   │ parakeet        │──en──│ English  │    │
│  │ JaVAD bal.  │      │ canary          │──en──│          │    │
│  │ JaVAD prec. │      │ voxtral         │──en──│          │    │
│  │ WebRTC 0-3  │      │ whispers2t_*    │──all─│          │    │
│  └─────────────┘      └─────────────────┘      └──────────┘    │
│                                                                  │
│  Full Matrix: 9 VAD × 10 ASR × 2 Lang = 180 combinations        │
│  Practical:   9 VAD × 3-4 ASR/lang × 2 Lang ≈ 54-72 tests      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 言語別推奨エンジン

| 言語 | 推奨エンジン | 代替エンジン |
|------|-------------|-------------|
| **Japanese (ja)** | reazonspeech, parakeet_ja | whispers2t_base |
| **English (en)** | parakeet, canary | whispers2t_base, voxtral |
| **German (de)** | canary | whispers2t_base, voxtral |
| **French (fr)** | canary | whispers2t_base, voxtral |
| **Spanish (es)** | canary | whispers2t_base, voxtral |

### 3.4 実行戦略

**Quick Mode** (CI デフォルト):
- ASR: 言語別デフォルト 2 エンジン（約 4 テスト）
  - **ja**: `parakeet_ja`, `whispers2t_large_v3`（日本語特化 + 汎用高精度）
  - **en**: `parakeet`, `whispers2t_large_v3`（英語特化 + 汎用高精度）
  - ※ Linux/Windows 両対応を考慮し、`reazonspeech` は Quick Mode から除外
- VAD: Silero v6, JaVAD precise, WebRTC mode 3（約 12 テスト）
- 推定時間: ~5分

**Standard Mode**:
- ASR: 言語別全エンジン（約 10 テスト）
- VAD: 全 9 構成 × 言語別 2-3 エンジン（約 36-54 テスト）
- 推定時間: ~20分

**Full Mode** (手動実行):
- ASR: 全エンジン × 全対応言語（約 20 テスト）
- VAD: 全 9 構成 × 全対応エンジン（約 72+ テスト）
- 推定時間: ~60分

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

**合計 9 構成**（Silero v5 は v6 の上位互換のため除外）:

| VAD | モデル/設定 | ライセンス | 特徴 |
|-----|------------|-----------|------|
| Silero VAD v6 | ONNX | MIT | 現在のデフォルト、高精度 |
| TenVAD | - | 独自 | 最軽量・最高速（評価のみ） |
| JaVAD | tiny | MIT | 0.64s window、即時検出向け |
| JaVAD | balanced | MIT | 1.92s window、バランス型 |
| JaVAD | precise | MIT | 3.84s window、最高精度 |
| WebRTC VAD | mode 0 | BSD | 最も寛容、誤検出少 |
| WebRTC VAD | mode 1 | BSD | やや厳格 |
| WebRTC VAD | mode 2 | BSD | 厳格 |
| WebRTC VAD | mode 3 | BSD | 最も厳格、見逃し多 |

### 5.2 VAD バックエンド設計 (Phase C で実装)

**設計決定:** 既存 `livecap_core/vad/` を拡張し、複数バックエンドをサポート

**実装先:**
- 本番用: `livecap_core/vad/backends/`
- ベンチマーク専用: `benchmarks/vad/backends/`

**設計方針:**
- `VADBackend` Protocol を拡張（`frame_size`, `name` プロパティ追加）
- VADProcessor を動的フレームサイズ対応に変更
- JaVAD はベンチマーク専用（レイテンシ 640ms+ のため）

**バックエンド一覧:**

| バックエンド | Protocol 準拠 | 実装先 | 備考 |
|-------------|--------------|--------|------|
| Silero | ✅ | `livecap_core/vad/backends/silero.py` | 既存、暫定デフォルト |
| WebRTC | ✅ | `livecap_core/vad/backends/webrtc.py` | 新規 |
| TenVAD | ✅ | `livecap_core/vad/backends/tenvad.py` | 新規、使用時警告 |
| JaVAD | ❌ | `benchmarks/vad/backends/javad.py` | **ベンチマーク専用** |

**JaVAD がベンチマーク専用の理由:**
- 最小 640ms のウィンドウサイズ（リアルタイム不適）
- per-frame probability ではなく segments を返す（Protocol 不適合）
- ベンチマーク後、価値が明確になれば本番対応を検討

**TenVAD の扱い:**
- 他の VAD と同様に `livecap_core/vad/backends/` に配置
- 使用時にライセンス警告を表示（ライセンス条件限定的）

詳細な実装方針は **Phase C** (Section 8) を参照。

---

## 6. 評価指標

### 6.1 ASR 精度指標

| 指標 | 説明 | 計算方法 |
|------|------|----------|
| **WER** | Word Error Rate | `(S + D + I) / N` |
| **CER** | Character Error Rate | 文字単位の WER（日本語向け） |

- `S`: 置換数、`D`: 削除数、`I`: 挿入数、`N`: 参照単語数
- ライブラリ: [jiwer](https://github.com/jitsi/jiwer)

### 6.2 性能指標

| 指標 | 説明 | 単位 |
|------|------|------|
| **RTF** | Real-Time Factor | 比率（低いほど高速） |
| **Latency** | 入力→出力の遅延 | ms |
| **Memory (RAM)** | ピークRAM使用量 | MB |
| **GPU VRAM (Model)** | モデルロード後のVRAM使用量 | MB |
| **GPU VRAM (Peak)** | 推論中のピークVRAM使用量 | MB |

#### RAM メモリ測定方法

Python 標準ライブラリの `tracemalloc` を使用:

```python
import tracemalloc

def measure_ram_usage(func):
    """RAM 使用量を測定"""
    tracemalloc.start()
    result = func()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "memory_current_mb": current / 1024**2,
        "memory_peak_mb": peak / 1024**2,
    }
```

**注意:**
- `tracemalloc` は Python ヒープのみを測定（ネイティブメモリは含まない）
- 相対比較には十分な精度
- 必要に応じて `psutil` への切り替えを検討（Phase B で検証）

#### GPU メモリ測定方法

```python
import torch

def measure_gpu_memory(func):
    """GPU メモリ使用量を測定するデコレータ"""
    if not torch.cuda.is_available():
        return {"gpu_memory_model_mb": None, "gpu_memory_peak_mb": None}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # モデルロード後の使用量
    result = func()
    torch.cuda.synchronize()
    model_memory = torch.cuda.memory_allocated() / 1024**2

    # 推論後のピーク使用量
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "gpu_memory_model_mb": model_memory,
        "gpu_memory_peak_mb": peak_memory,
    }
```

**注意:**
- CPU モードの場合は `None` を記録
- ONNX Runtime (GPU) の場合は `nvidia-smi` 経由で測定が必要な場合がある
- 測定の再現性のため、warm-up 実行後に測定する

### 6.3 VAD 固有指標

| 指標 | 説明 |
|------|------|
| **Segments** | 検出セグメント数 |
| **Avg Duration** | 平均セグメント長 |
| **Speech Ratio** | 音声区間の割合 |
| **VAD RTF** | VAD処理の Real-Time Factor |

#### VAD RTF の定義

```
vad_rtf = VAD処理の壁時計時間 / 音声の長さ
```

**計測区間:**
- **含む**: モデル推論時間、フレーム分割処理、セグメント検出ロジック
- **含まない**: 音声ファイル I/O、リサンプリング（前処理として別途実行）

**実装例:**
```python
# リサンプリングは計測外
audio_16k = resample_to_16khz(audio, original_sr)

# VAD処理時間のみ計測
vad_start = time.perf_counter()
segments = vad.process_audio(audio_16k, 16000)
vad_time = time.perf_counter() - vad_start

vad_rtf = vad_time / audio_duration
```

#### VAD 精度指標について

**本フェーズ (C-2) では VAD 単体の精度（F1, 誤検出率等）は未計測。**

**理由:**
- ラベル付き VAD Ground Truth データが存在しない
- JSUT/LibriSpeech は ASR 用コーパスであり、音声区間の正確なタイムスタンプを持たない

**評価アプローチ:**
本ベンチマークでは、VAD の「精度」を**下流タスク（ASR）の WER/CER への影響**で間接的に評価する:

```
VAD が良い → セグメント境界が適切 → ASR 精度が高い
VAD が悪い → 音声の切り落とし/ノイズ混入 → ASR 精度が低下
```

**将来の拡張（必要に応じて）:**
- 基準 VAD（例: Silero）との交差比較（IoU, Segment Agreement Rate）
- 手動アノテーションによるサブセット評価

### 6.4 Raw vs Normalized メトリクス（Phase B 実装予定）

**背景:**
正規化前後の両方のメトリクスを報告することで、トレースアビリティを向上させる。

| メトリクス | 説明 | 用途 |
|-----------|------|------|
| **WER/CER (Normalized)** | 正規化後（句読点除去、小文字化）| エンジン比較の主指標 |
| **WER/CER (Raw)** | 正規化前（元のテキスト）| 句読点予測精度の把握 |

**実装例:**
```python
@dataclass
class BenchmarkResult:
    # Primary metrics (normalized)
    wer: float | None = None
    cer: float | None = None

    # Raw metrics (traceability)
    wer_raw: float | None = None
    cer_raw: float | None = None
```

**出力例:**
```
Engine        CER    CER(raw)   RTF   VRAM
-----------  -----  ---------  ----  ------
reazonspeech  3.2%     8.1%    0.15  2048MB
whispers2t    5.1%    12.3%    0.08  3072MB
```

**メリット:**
- 正規化の影響を定量的に把握
- 句読点予測精度の比較が可能
- デバッグ・分析時に有用

### 6.5 統計的厳密性

**ユースケース:** VAD × ASR の組み合わせ比較、結果は公開（論文ほどの厳密性は不要）

#### 決定事項: 2モード制

| モード | 実行回数 | 用途 | CLI |
|--------|---------|------|-----|
| **単一実行** | 1回 | 開発・CI・スクリプトテスト | `--runs 1`（デフォルト） |
| **複数実行** | 3回 | 公開用ベンチマーク | `--runs 3` |

#### メトリクス別の測定方法

| メトリクス | 測定方法 | 理由 |
|-----------|---------|------|
| **RTF** | ウォームアップ1回 + N回測定 | GPU状態による変動を排除 |
| **WER/CER** | ファイルごとに1回 | 決定的（同じ入力=同じ出力） |
| **VRAM** | 1回のみ | ほぼ固定値 |

#### RTF の変動要因

| 要因 | 影響度 | 説明 |
|------|--------|------|
| GPU ウォームアップ | 高 | 初回推論はCUDAカーネルコンパイルで遅い |
| GPU サーマルスロットリング | 中 | 温度上昇でクロック低下 |
| メモリ状態 | 低 | VRAM断片化、キャッシュ状態 |

**ウォームアップ後の期待変動幅:** ±5-10%（専用マシンでは ±2-5%）

#### Warm-up タイミング

**決定事項:** エンジンごとに1回（最初のファイル処理前）

| 方式 | 採用 | 理由 |
|------|------|------|
| エンジンごと | ✅ | 高速、GPU状態の安定化に十分 |
| ファイルごと | ❌ | 時間がかかりすぎる（100ファイル = 100回の warm-up） |

#### 実装例

```python
def benchmark_engine(engine, dataset, runs=1):
    """エンジンのベンチマーク実行"""
    # エンジンごとに1回の warm-up（最初のファイルで実行）
    first_file = dataset.files[0]
    engine.transcribe(first_file.audio, first_file.sample_rate)

    results = []
    for audio_file in dataset:
        # RTF測定（runs回実行して平均を取る）
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            transcript, _ = engine.transcribe(audio_file.audio, audio_file.sample_rate)
            times.append(time.perf_counter() - start)

        mean_time = statistics.mean(times)
        rtf = mean_time / audio_file.duration

        results.append({
            "file_id": audio_file.stem,
            "transcript": transcript,
            "rtf": rtf,
        })

    return results
```

#### 出力例（複数実行モード）

```
=== ASR Benchmark Results (3 runs) ===

--- Japanese ---
Engine        CER    RTF (mean±std)    VRAM
-----------  -----  ----------------  ------
reazonspeech  3.2%   0.15 ± 0.01     2048MB
parakeet_ja   4.5%   0.12 ± 0.02     3584MB
whispers2t    5.1%   0.08 ± 0.01     1536MB
```

#### CLI 使用例

```bash
# 開発時（高速）
python -m benchmarks.asr --mode quick

# 公開用（信頼性重視）
python -m benchmarks.asr --mode standard --runs 3
```

#### Phase B 実装決定事項

- `--runs` オプションを Phase B で実装
- デフォルト: `--runs 1`（単一実行）
- 複数実行時は RTF の `mean` を記録（`std/min/max` は必要に応じて後から追加）

### 6.6 エンジン×言語の互換性チェック

**決定事項:** 非対応言語のデータは自動スキップ

エンジンが対応していない言語のデータセットは、エラーではなく自動的にスキップする。

```python
# benchmarks/common/datasets.py
def get_files_for_engine(self, engine_id: str) -> Iterator[AudioFile]:
    from engines.metadata import EngineMetadata
    info = EngineMetadata.get(engine_id)
    supported = info.supported_languages if info else None

    for f in self.files:
        if supported is None or f.language in supported:
            yield f
        else:
            logger.debug(f"Skipping {f.path}: {engine_id} doesn't support {f.language}")
```

**メリット:**
- ユーザーフレンドリー（明示的な除外指定が不要）
- `--engine reazonspeech --language ja en` 指定時、`reazonspeech` は `en` を自動スキップ

### 6.7 エラーハンドリング戦略

**決定事項:** デフォルトはスキップ＋警告

| ケース | 挙動 | 理由 |
|--------|------|------|
| エンジンロード失敗 | スキップ＋警告 | 他エンジンの結果は取得したい |
| 音声ファイル読み込み失敗 | スキップ＋警告 | 他ファイルの結果は取得したい |
| 推論中エラー | スキップ＋警告 | 同上 |

**実装例:**

```python
try:
    engine = engine_manager.get_engine(engine_id, device, language)
except Exception as e:
    logger.warning(f"Failed to load {engine_id}: {e}")
    continue  # 次のエンジンへ
```

**将来の拡張:** `--strict` オプションで例外を発生させるモードを追加可能

### 6.8 進捗表示

**決定事項:** `tqdm` を使用した進捗バー表示

Full モード（60分+）を多用することを考慮し、`tqdm` による進捗表示を実装。

```python
from tqdm import tqdm

for engine_id in tqdm(engines, desc="Engines"):
    for audio_file in tqdm(dataset, desc=f"  {engine_id}", leave=False):
        result = benchmark_single(engine_id, audio_file)
```

**依存関係:** `tqdm` を `benchmark` extra に追加

### 6.9 既知の制限事項

#### 日本語ひらがな/漢字表記揺れ

**問題:**
ASR出力とリファレンスで文字種（ひらがな/漢字）が異なる場合、CERに影響する。

```
リファレンス: "食べる"（漢字）
ASR出力:     "たべる"（ひらがな）
→ CER = 33%（意味的には同一）
```

**評価:**
- **影響度: 低** - 現代の日本語ASRは漢字変換後のテキストを出力
- **対応: ドキュメント化のみ** - ASRの言語モデルが異なる表記を選択した場合は、実質的な差異として扱う

**理由:**
1. ASRモデルは内部で言語モデルによる漢字変換を行う
2. リファレンスコーパス（JSUT等）も漢字混じりの自然な日本語
3. 表記の違いは字幕品質に影響するため、エラーとしてカウントするのは妥当

**注記:**
一部の表記揺れ（「子供」vs「子ども」等）は真のエラーではないが、全体のCERへの影響は限定的。

---

## 7. データセット

### 7.1 ディレクトリ構造

**決定事項:** 言語別フォルダ構造に統一（`audio/` と `prepared/` で一貫性を持たせる）

```
tests/assets/
├── audio/                    # git追跡（quickモード用、数ファイル）
│   ├── ja/
│   │   ├── jsut_basic5000_0001.wav
│   │   └── jsut_basic5000_0001.txt
│   └── en/
│       ├── librispeech_1089-134686-0001.wav
│       └── librispeech_1089-134686-0001.txt
│
├── prepared/                 # git無視（スクリプトで生成）
│   ├── ja/                   # 変換済み日本語データ
│   │   ├── jsut_basic5000_0002.wav
│   │   └── ...
│   └── en/                   # 変換済み英語データ
│       └── ...
│
├── source/                   # git無視（ソースコーパス）
│   ├── jsut/jsut_ver1.1/     # 3.4GB - JSUT v1.1
│   └── librispeech/test-clean/  # 358MB - LibriSpeech
│
└── README.md
```

### 7.2 構造変更の影響

現在の `tests/assets/audio/` はフラット構造のため、言語別フォルダへの移行が必要。

**現在の構造:**
```
tests/assets/audio/
├── jsut_basic5000_0001_ja.wav
├── jsut_basic5000_0001_ja.txt
├── librispeech_test-clean_1089-134686-0001_en.wav
└── librispeech_test-clean_1089-134686-0001_en.txt
```

**変更後:**
```
tests/assets/audio/
├── ja/
│   ├── jsut_basic5000_0001.wav
│   └── jsut_basic5000_0001.txt
└── en/
    ├── librispeech_1089-134686-0001.wav
    └── librispeech_1089-134686-0001.txt
```

**影響を受けるファイル:**

| ファイル | 変更内容 |
|---------|---------|
| `tests/audio_sources/test_file_source.py` | パス更新 |
| `tests/core/test_text_normalization.py` | パス更新 |
| `tests/integration/engines/test_smoke_engines.py` | パス更新 |
| `tests/integration/realtime/test_mock_realtime_flow.py` | パス更新 |
| `tests/integration/realtime/test_e2e_realtime_flow.py` | パス更新 |
| `examples/realtime/basic_file_transcription.py` | パス更新 |
| `examples/realtime/custom_vad_config.py` | パス更新 |
| `examples/realtime/callback_api.py` | パス更新 |
| `examples/README.md` | パス更新 |
| `tests/assets/README.md` | ドキュメント更新 |
| `tests/utils/text_normalization.py` | `get_language_from_filename()` 更新 |

**実装タスク:** Phase A-2 に含める

### 7.3 ソースデータセット

| データセット | 言語 | 形式 | トランスクリプト | ライセンス |
|-------------|------|------|-----------------|-----------|
| JSUT v1.1 | ja | WAV | `ID:テキスト` | 非商用 |
| LibriSpeech test-clean | en | FLAC | `ID TEXT` | CC BY 4.0 |

### 7.4 統一フォーマット仕様

変換スクリプトで以下のフォーマットに統一：

| 項目 | 仕様 |
|------|------|
| 音声形式 | WAV, 16kHz, mono, 16bit |
| 正規化 | ピーク -1dBFS |
| ファイル名 | `{corpus}_{subset}_{id}.wav` |
| トランスクリプト | 同名 `.txt`、UTF-8、1行、末尾改行 |
| フォルダ | `{lang}/` (ja, en) |

### 7.5 実行モードとデータセット

| モード | データソース | ファイル数 | 用途 |
|--------|-------------|-----------|------|
| `quick` | `audio/` | ja:2, en:2 | CI smoke test |
| `standard` | `prepared/` | ja:100, en:100 (調整可) | ローカル開発 |
| `full` | `prepared/` | 全ファイル | 本格ベンチマーク |

### 7.6 変換スクリプト

```bash
# standard モード（各言語100ファイル）
python scripts/prepare_benchmark_data.py --mode standard

# full モード（全ファイル）
python scripts/prepare_benchmark_data.py --mode full

# カスタム
python scripts/prepare_benchmark_data.py --ja-limit 500 --en-limit 200
```

スクリプトの処理内容：

1. **JSUT**: `transcript_utf8.txt` 読み込み → WAV 正規化 → `prepared/ja/` へ出力
2. **LibriSpeech**: `*.trans.txt` 読み込み → FLAC→WAV 変換 + 正規化 → `prepared/en/` へ出力

### 7.7 .gitignore 設定

```gitignore
# ソースコーパス（大規模）
tests/assets/source/

# 生成データ（ライセンス問題）
tests/assets/prepared/
```

---

## 8. 実装ステップ

### 概要: Phase A/B/C アプローチ

| Phase | 内容 | ステータス |
|-------|------|----------|
| **A** | 基盤構築 | ✅ 完了 |
| **B** | ASR ベンチマーク | ✅ 完了 |
| **C** | VAD ベンチマーク | 🔜 次 |

**実装理由:**
1. **動作確認が先**: 壊れたエンジンでベンチマークしても無意味
2. **データセットが先**: ベンチマークの価値はデータ品質で決まる
3. **ASR が先**: VAD は ASR の上に構築（依存関係）

---

### Phase A: 基盤構築 (✅ 完了)

| タスク | PR | 内容 |
|--------|-----|------|
| A-1 | #91-95 | 全10エンジンの smoke test 完了 |
| A-2a | #97 | `tests/assets/audio/` を言語別フォルダに再構成 |
| A-2b | #98 | `scripts/prepare_benchmark_data.py` 作成 |
| A-3 | #100 | `benchmarks/common/` 実装 |
| A-4 | #100 | `pyproject.toml` に benchmark extra 追加 |

---

### Phase B: ASR ベンチマーク (✅ 完了)

**対象:** `benchmarks/asr/`

| タスク | PR | 内容 |
|--------|-----|------|
| B-1 | #103 | `ASRBenchmarkRunner` 実装 |
| B-2 | #103 | CLI 実装 (`python -m benchmarks.asr`) |
| B-3 | #103 | 動作確認とベンチマーク実行 |
| B-4 | #104 | CI ワークフロー設定 (`.github/workflows/benchmark.yml`) |
| B-5 | #105 | CI: CUDA PyTorch インストール修正 |
| B-6 | #108 | VRAM 累積測定バグ修正 (Issue #107) |

**検証結果:** Quick mode 実行成功、VRAM 測定正常 (~2450MB/エンジン)

**CLI 使用例:**
```bash
python -m benchmarks.asr --mode quick
python -m benchmarks.asr --engine parakeet_ja whispers2t_large_v3 --language ja
python -m benchmarks.asr --mode standard --runs 3 --output results.json
```

---

### Phase C: VAD ベンチマーク (🔜 進行中)

**対象:** `benchmarks/vad/`

| タスク | PR | 内容 | ステータス |
|--------|-----|------|----------|
| C-1 | #110, #114 | VAD バックエンド実装（Section 5.2 参照） | ✅ 完了 |
| C-2 | - | `VADBenchmarkRunner` 実装（C-2～C-7 セクション参照） | 🔜 次 |
| C-3 | - | CI ワークフロー作成 (`vad-benchmark.yml`) | 📋 待機 |

**CLI 使用例:**
```bash
python -m benchmarks.vad --vad silero_v6 javad_precise --asr parakeet_ja
python -m benchmarks.vad --mode standard --format markdown
```

#### C-1: VAD バックエンド実装方針

##### 設計決定サマリー

| 項目 | 決定 | 理由 |
|------|------|------|
| フレームサイズ | 動的（Protocol に `frame_size` 追加） | 各バックエンドが最適サイズで動作 |
| JaVAD | ベンチマーク専用 | 640ms+ レイテンシ、Protocol 不適合 |
| 入力フォーマット | バックエンド内部で変換 | 標準アダプターパターン |
| 選択 API | コンストラクタ（現行維持） | Pythonic、型安全 |
| デフォルト VAD | Silero（暫定） | ベンチマーク後に再検討 (#96) |

##### ディレクトリ構造

```
本番用（livecap_core/vad/backends/）:
├── __init__.py     # VADBackend Protocol（拡張版）
├── silero.py       # SileroVADBackend (既存)
├── webrtc.py       # WebRTCVADBackend (新規)
└── tenvad.py       # TenVADBackend (新規、使用時警告)

ベンチマーク専用（benchmarks/vad/backends/）:
└── javad.py        # JaVADPipeline (バッチ処理専用)
```

##### VADBackend Protocol（拡張版）

```python
class VADBackend(Protocol):
    """VAD バックエンドのプロトコル。"""

    def process(self, audio: np.ndarray) -> float:
        """音声フレームを処理して発話確率を返す。

        Args:
            audio: float32 音声データ（長さは frame_size samples）

        Returns:
            発話確率 (0.0-1.0)
        """
        ...

    def reset(self) -> None:
        """内部状態をリセット。"""
        ...

    @property
    def frame_size(self) -> int:
        """16kHz での推奨フレームサイズ（samples）"""
        ...

    @property
    def name(self) -> str:
        """バックエンド識別子"""
        ...
```

##### VADProcessor の変更

```python
class VADProcessor:
    def __init__(self, backend: VADBackend = None):
        self._backend = backend or self._create_default_backend()
        self._frame_size = self._backend.frame_size  # 動的に決定
```

##### フレームサイズ処理

| Backend | ネイティブサイズ | VADProcessor 対応 |
|---------|----------------|------------------|
| Silero | 512 samples (32ms) | `frame_size=512` |
| WebRTC | 160/320/480 samples | `frame_size=320` (20ms 推奨) |
| TenVAD | 160/256 samples | `frame_size=256` (16ms) |
| JaVAD | 10,240+ samples | **Protocol 非準拠**（下記参照） |

##### JaVAD: ベンチマーク専用

**理由:**
1. **アーキテクチャの根本的違い**: JaVAD は per-frame probability ではなく segments を返す
2. **レイテンシ**: 最小 640ms はリアルタイム用途に不適
3. **実装コスト**: Protocol 準拠には複雑な変換ロジックが必要、精度低下リスクあり
4. **ベンチマーク目的**: 真の性能を測定するにはネイティブ API が最適

**将来の検討:**
ベンチマーク結果で JaVAD の価値が明確になった場合、以下を検討:
- ファイル処理専用オプションとして提供
- ストリーミング対応の実装（需要があれば）

```python
# benchmarks/vad/backends/javad.py
class JaVADPipeline:
    """JaVAD バッチ処理（ベンチマーク専用）

    ⚠️ ストリーミング非対応
    リアルタイム用途には silero, webrtc, tenvad を使用してください。
    """

    def __init__(self, model: str = "balanced"):
        """
        Args:
            model: "tiny" (640ms), "balanced" (1920ms), "precise" (3840ms)
        """
        ...

    def process_audio(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
        """音声全体を処理してセグメントを返す。"""
        from javad import Processor
        processor = Processor(model=self.model)
        return processor.intervals(audio)
```

##### 入力フォーマット変換

各バックエンドが内部で変換を処理:

```python
class WebRTCVADBackend:
    def process(self, audio: np.ndarray) -> float:
        # float32 [-1, 1] → int16 bytes（WebRTC 要求フォーマット）
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return float(self._vad.is_speech(audio_bytes, 16000))

class TenVADBackend:
    def process(self, audio: np.ndarray) -> float:
        # float32 → int16（TenVAD 要求フォーマット）
        audio_int16 = (audio * 32767).astype(np.int16)
        prob, flag = self._vad.process(audio_int16)
        return prob
```

##### TenVAD 警告対応

```python
# livecap_core/vad/backends/tenvad.py
class TenVADBackend:
    """TenVAD バックエンド。"""

    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        logger.warning(
            "TenVAD has limited license terms. "
            "Please review the license before use: "
            "https://github.com/TEN-framework/ten-vad"
        )
        ...
```

##### デフォルト VAD と自動選択

**現時点:** Silero を暫定デフォルトとして維持（既存互換性）

**将来（Phase C 後）:** 言語に応じた自動選択を実装 → #96 で検討

```python
# 将来の実装案
class VADConfig:
    backend: str = "auto"  # "auto", "silero", "webrtc", "tenvad"

def get_optimal_vad(language: str) -> str:
    """ベンチマーク結果に基づく最適 VAD を返す"""
    # Phase C 完了後に実装
    OPTIMAL_VAD = {
        "ja": "???",  # ベンチマーク結果で決定
        "en": "???",
        "default": "silero",
    }
    return OPTIMAL_VAD.get(language, OPTIMAL_VAD["default"])
```

#### C-2: VADBenchmarkRunner 実装設計

##### ベンチマーク専用統一インターフェース

Protocol準拠 VAD (Silero, WebRTC, TenVAD) と JaVAD (Protocol非準拠) の両方を
同一インターフェースで扱うため、ベンチマーク専用の Protocol を定義。

```python
# benchmarks/vad/backends/base.py
class VADBenchmarkBackend(Protocol):
    """ベンチマーク用 VAD バックエンドの統一インターフェース。

    Protocol準拠 VAD と JaVAD の両方を同じ方法で扱える。
    """

    def process_audio(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[tuple[float, float]]:
        """音声全体を処理してセグメントを返す。

        Args:
            audio: float32形式の音声データ
            sample_rate: サンプルレート

        Returns:
            セグメントのリスト [(start_time, end_time), ...]
            時間は秒単位
        """
        ...

    @property
    def name(self) -> str:
        """バックエンド識別子"""
        ...

    @property
    def config(self) -> dict:
        """レポート用の設定パラメータを返す。

        Returns:
            VAD 固有のパラメータ辞書
            例: {"mode": 3, "frame_duration_ms": 20}
        """
        ...
```

##### 各バックエンドの config 実装例

```python
# WebRTC VAD
class WebRTCVAD:
    @property
    def config(self) -> dict:
        return {
            "mode": self._mode,
            "frame_duration_ms": self._frame_duration_ms,
        }

# TenVAD
class TenVAD:
    @property
    def config(self) -> dict:
        return {
            "hop_size": self._hop_size,
            "threshold": self._threshold,
        }

# Silero VAD
class SileroVAD:
    @property
    def config(self) -> dict:
        return {
            "threshold": self._threshold,
        }

# JaVAD Pipeline
class JaVADPipeline:
    WINDOW_SIZES = {"tiny": 640, "balanced": 1920, "precise": 3840}

    @property
    def config(self) -> dict:
        return {
            "model": self._model,
            "window_ms": self.WINDOW_SIZES[self._model],
        }
```

##### VADProcessorWrapper（Protocol準拠 VAD 用）

本番用 VADProcessor を使用して、ベンチマーク用インターフェースを提供。

```python
# benchmarks/vad/backends/processor_wrapper.py
class VADProcessorWrapper:
    """VADProcessor をベンチマーク用インターフェースでラップ。

    Protocol準拠の VADBackend (Silero, WebRTC, TenVAD) を
    VADBenchmarkBackend インターフェースで使用可能にする。
    """

    SAMPLE_RATE = 16000  # 全バックエンド共通

    def __init__(self, backend: VADBackend, config: VADConfig | None = None):
        self._processor = VADProcessor(config=config, backend=backend)
        self._backend = backend

    def process_audio(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[tuple[float, float]]:
        """音声全体を処理してセグメントを返す。

        ⚠️ 時間精度のため、先に16kHzにリサンプルしてから処理する。
        """
        # 1. 先に全体を16kHzにリサンプル（時間基準を統一）
        if sample_rate != self.SAMPLE_RATE:
            audio_16k = self._resample(audio, sample_rate, self.SAMPLE_RATE)
        else:
            audio_16k = audio

        # 2. frame_size の整数倍でチャンク処理
        frame_size = self._backend.frame_size
        chunk_size = frame_size * 100  # 例: 512 * 100 = 51200 samples ≈ 3.2秒

        segments = []
        self._processor.reset()

        for i in range(0, len(audio_16k), chunk_size):
            chunk = audio_16k[i:i+chunk_size]
            vad_segments = self._processor.process_chunk(chunk, self.SAMPLE_RATE)
            for seg in vad_segments:
                if seg.is_final:
                    segments.append((seg.start_time, seg.end_time))

        # 残りのセグメントを取得
        final = self._processor.finalize()
        if final:
            segments.append((final.start_time, final.end_time))

        return segments

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """リサンプリング（実装は librosa または scipy を使用）"""
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    @property
    def name(self) -> str:
        return self._backend.name
```

**設計理由:**
1. **ベンチマーク目的に適合**: ファイル単位のバッチ処理が前提
2. **JaVAD との自然な統合**: ネイティブインターフェースがそのまま使える
3. **Runner の簡潔さ**: 単一の処理フローで全 VAD を扱える
4. **本番コードへの影響なし**: `livecap_core/vad/` は変更不要

##### 非16kHz入力時のリサンプリング戦略

**問題:**
チャンク境界ごとにリサンプリングすると、サンプル数の丸め誤差が蓄積し、
セグメントの start/end 時刻が元音声の実時間と微妙にずれる可能性がある。

**解決策:**
1. **全体を先に16kHzにリサンプル**: チャンク処理前に統一
2. **frame_size の整数倍でチャンク化**: フレーム境界のずれを防止
3. **時間計算は16kHz基準**: `start_time = sample_index / 16000`

**実装上の注意:**
```python
# ❌ 悪い例: チャンクごとにリサンプル（時間ずれの原因）
for chunk in chunks:
    chunk_16k = resample(chunk, orig_sr, 16000)  # 毎回丸め誤差
    process(chunk_16k)

# ✅ 良い例: 先に全体をリサンプル
audio_16k = resample(full_audio, orig_sr, 16000)  # 一度だけ
for chunk in split_into_chunks(audio_16k, frame_size * N):
    process(chunk)
```

##### セグメント結合戦略

VAD で検出したセグメントを ASR に渡した後、各セグメントの文字起こし結果を結合する際の戦略:

**言語ベース結合:**
```python
def combine_segments(transcripts: list[str], language: str) -> str:
    """セグメント結果を言語に応じて結合。"""
    if language == "ja":
        # 日本語: スペースなし
        return "".join(transcripts)
    else:
        # 英語等: スペース区切り
        return " ".join(transcripts)
```

**理由:**
- 日本語は単語間にスペースがないため、連結のみ
- 英語等は単語境界にスペースが必要

#### C-3: Quick Mode 構成

**決定事項:** 全 VAD 構成 + 最小データで高速に全体をカバー

| 項目 | 構成 |
|------|------|
| VAD | 全 9 構成 (Silero, WebRTC×4, JaVAD×3, TenVAD) |
| ASR | 言語別 1-2 エンジン (ja: parakeet_ja, en: parakeet) |
| データ | ja: 1-2 ファイル, en: 1-2 ファイル |
| 推定時間 | ~3-5 分 |

**理由:**
- VAD の網羅性を優先（バックエンド実装の検証が主目的）
- ASR は動作確認済みのため最小限でOK
- 全 VAD × 少数データで問題を早期検出

#### C-4: 実装詳細の確定事項

##### VADProcessor フレームサイズ対応

**決定:** インスタンス変数として初期化時に設定（選択肢 A）

```python
class VADProcessor:
    SAMPLE_RATE: int = 16000  # 固定（全バックエンド共通）

    def __init__(self, backend: VADBackend = None):
        self._backend = backend or self._create_default_backend()
        self._frame_size = self._backend.frame_size  # インスタンス変数

    @property
    def frame_size(self) -> int:
        return self._frame_size
```

**理由:** フレームサイズは実行中に変わらないため、初期化時の一度の取得で十分。

##### VADBackend の cleanup() メソッド

**決定:** 不要（`reset()` で十分）

**理由:**
- Silero: `reset_states()` で状態リセット、モデルは torch が管理
- WebRTC: 純粋な C 拡張、明示的解放不要
- TenVAD: 軽量、明示的解放不要
- 必要になれば後から追加可能（Protocol は後方互換で拡張可能）

##### サンプルレート統一

**決定:** 16kHz 統一

| バックエンド | ネイティブ対応 | 実装方針 |
|-------------|--------------|---------|
| Silero | 8kHz, 16kHz | 16kHz 使用 |
| WebRTC | 8-48kHz | 16kHz 使用 |
| TenVAD | 16kHz only | 16kHz 使用 |

**理由:**
- `VADProcessor` の既存リサンプリングロジックを活用
- 各バックエンドは 16kHz 入力を前提として実装
- WebRTC の 8kHz 最適化は将来の検討事項

##### VAD ベンチマーク結果構造

**決定:** 既存 `BenchmarkResult` を拡張

```python
@dataclass
class BenchmarkResult:
    # 既存フィールド（ASR）
    engine: str
    language: str
    audio_file: str
    transcript: str
    reference: str
    wer: float | None = None
    cer: float | None = None
    rtf: float | None = None
    audio_duration_s: float | None = None
    processing_time_s: float | None = None
    # ... memory fields ...

    # VAD 拡張フィールド（オプショナル）
    vad: str | None = None                     # VAD バックエンド名
    vad_config: dict | None = None             # VAD 設定パラメータ（再現性確保用）
    vad_rtf: float | None = None               # VAD 処理の RTF
    segments_count: int | None = None          # 検出セグメント数
    avg_segment_duration_s: float | None = None  # 平均セグメント長（秒）
    speech_ratio: float | None = None          # 音声区間の割合（診断用）
```

**C-2 で追加するフィールド:**
- `vad_rtf`: VAD 処理速度の評価に必要
- `segments_count`: セグメント分割数の把握に必要
- `avg_segment_duration_s`: セグメント粒度の評価に必要
- `speech_ratio`: VAD の傾向把握に必要（積極的/保守的）
- `vad_config`: VAD パラメータの記録（再現性・診断能力の確保）

**speech_ratio の意義:**
```
speech_ratio = sum(segment_durations) / audio_duration

speech_ratio が高い → 積極的に音声検出（False Positive 傾向）
speech_ratio が低い → 保守的に検出（False Negative 傾向）
```

**vad_config の意義:**
- 再現性の確保: パラメータが記録されていれば同じ条件で再実行可能
- 診断能力の向上: WER が悪い原因を VAD パラメータから推測可能
- 将来の比較: パラメータ調整実験の際に基準となる

**用途:**
- `vad=None`: ASR 単体ベンチマーク
- `vad="silero"`: VAD+ASR 統合ベンチマーク

##### CI ワークフロー構成

**決定:** 分離方式（`asr-benchmark.yml` + `vad-benchmark.yml`）

```
.github/workflows/
├── asr-benchmark.yml     # ASR 単体ベンチマーク（既存）
├── vad-benchmark.yml     # VAD ベンチマーク（C-3 で新規作成）
└── core-tests.yml        # 既存
```

**ASR Benchmark (`asr-benchmark.yml`):**
- VAD は Silero 固定（デフォルト）
- ASR エンジンの比較が目的

**VAD Benchmark (`vad-benchmark.yml`):**
- ASR は言語別に固定（ja: parakeet_ja, en: parakeet）
- VAD バックエンドの比較が目的

```yaml
# .github/workflows/vad-benchmark.yml
name: VAD Benchmark

on:
  workflow_dispatch:
    inputs:
      mode:
        type: choice
        options: [quick, standard, full]
        default: quick
      language:
        type: choice
        options: [ja, en, both]
        default: both
      vad:
        description: 'Specific VAD (comma-separated, empty=all)'
        required: false
```

**分離の理由:**
1. **概念的分離**: ASR/VAD は異なる比較目的
2. **パラメータ簡潔化**: 各ワークフローに必要なオプションのみ
3. **独立実行**: 用途に応じて個別にトリガー可能
4. **障害分離**: 一方の失敗が他方に影響しない
5. **GitHub Actions UI**: 目的が明確で見つけやすい

**コード重複への対処:**
- 現時点: 共通セットアップ（~30行）の重複は許容範囲
- 将来: 問題になれば Composite Action に抽出

##### ProgressReporter の拡張

**決定:** 既存 ProgressReporter を拡張（新クラス作成ではない）

```python
class ProgressReporter:
    def engine_started(
        self,
        engine_id: str,
        language: str,
        files_count: int,
        vad_name: str | None = None,  # 追加: VAD名（VADベンチマーク時のみ）
    ) -> None:
        ...
```

**Step Summary の変更（VAD ベンチマーク時）:**
```
| # | VAD | Engine | Lang | Files | WER | CER | RTF | Time | Status |
|---|-----|--------|------|-------|-----|-----|-----|------|--------|
| 1 | silero | parakeet_ja | ja | 100/100 | 4.2% | 2.1% | 0.15 | 45s | ✅ |
| 2 | webrtc_mode3 | parakeet_ja | ja | 100/100 | 4.5% | 2.3% | 0.14 | 43s | ✅ |
```

**拡張の理由:**
1. **コード再利用**: 進捗表示、Step Summary、ETA計算のロジックを継承
2. **一貫性**: ASR/VAD 両方で同じ見た目のレポート
3. **シンプル**: 新クラス作成より変更量が少ない

**進捗表示フォーマット（VAD ベンチマーク時）:**
```
[1/9] silero + parakeet_ja (ja): Processing 100 files...
      ████████████████████░░░░ 80/100 files (2m 15s remaining)
```

VAD名 + ASR名 の組み合わせで表示し、何を評価中か明確にする。

#### C-5: VAD Factory 設計

**決定:** ハイブリッド方式（Registry + Factory、キャッシュなし）

```python
# benchmarks/vad/factory.py

# Registry: VAD 構成の定義
VAD_REGISTRY: dict[str, dict] = {
    # Protocol準拠 VAD (VADProcessorWrapper で使用)
    "silero": {"type": "protocol", "backend_class": "SileroVAD", "params": {}},
    "webrtc_mode0": {"type": "protocol", "backend_class": "WebRTCVAD", "params": {"mode": 0}},
    "webrtc_mode1": {"type": "protocol", "backend_class": "WebRTCVAD", "params": {"mode": 1}},
    "webrtc_mode2": {"type": "protocol", "backend_class": "WebRTCVAD", "params": {"mode": 2}},
    "webrtc_mode3": {"type": "protocol", "backend_class": "WebRTCVAD", "params": {"mode": 3}},
    "tenvad": {"type": "protocol", "backend_class": "TenVAD", "params": {}},
    # JaVAD (直接 process_audio を持つ)
    "javad_tiny": {"type": "javad", "model": "tiny"},
    "javad_balanced": {"type": "javad", "model": "balanced"},
    "javad_precise": {"type": "javad", "model": "precise"},
}

def create_vad(vad_id: str) -> VADBenchmarkBackend:
    """VAD バックエンドを生成する。

    毎回新しいインスタンスを生成（キャッシュなし）。

    Args:
        vad_id: VAD 識別子（VAD_REGISTRY のキー）

    Returns:
        VADBenchmarkBackend を実装するインスタンス

    Raises:
        ValueError: 不明な vad_id
    """
    if vad_id not in VAD_REGISTRY:
        raise ValueError(f"Unknown VAD: {vad_id}. Available: {list(VAD_REGISTRY.keys())}")

    config = VAD_REGISTRY[vad_id]

    if config["type"] == "javad":
        from benchmarks.vad.backends.javad import JaVADPipeline
        return JaVADPipeline(model=config["model"])
    else:
        # Protocol準拠 VAD
        backend = _create_protocol_backend(config)
        return VADProcessorWrapper(backend)

def get_all_vad_ids() -> list[str]:
    """利用可能な全 VAD ID を返す。"""
    return list(VAD_REGISTRY.keys())
```

**キャッシュなしの理由:**
1. **状態汚染の回避**: `reset()` があるが、完全なクリーンスレートが望ましい
2. **メモリ管理の簡素化**: VAD は軽量、毎回生成しても問題なし
3. **テスト容易性**: 各テストで独立したインスタンス
4. **ASR エンジンとの違い**: ASR は重量級（数GB）、VAD は軽量（数MB）

#### C-6: 空セグメント・短セグメントの処理

**決定:**
- **0 セグメント → 空文字列の transcript**: エラーではなく正常なケース
- **短いセグメントはフィルタなし**: VAD の判断をそのまま採用

```python
def benchmark_file(
    self,
    vad: VADBenchmarkBackend,
    engine: TranscriptionEngine,
    audio_file: AudioFile,
) -> BenchmarkResult:
    """1ファイルの VAD + ASR ベンチマーク。"""

    # VAD 処理
    vad_start = time.perf_counter()
    segments = vad.process_audio(audio_file.audio, audio_file.sample_rate)
    vad_time = time.perf_counter() - vad_start

    # 空セグメントの場合
    if not segments:
        return BenchmarkResult(
            engine=engine_id,
            vad=vad.name,
            vad_config=vad.config,  # VAD パラメータを記録
            language=audio_file.language,
            audio_file=audio_file.stem,
            transcript="",  # 空文字列
            reference=audio_file.transcript,
            wer=calculate_wer(audio_file.transcript, ""),  # 参照との比較
            cer=calculate_cer(audio_file.transcript, ""),
            vad_rtf=vad_time / audio_file.duration,
            rtf=0.0,  # ASR 未実行
            segments_count=0,
            avg_segment_duration_s=0.0,
            speech_ratio=0.0,  # 音声なし
        )

    # 各セグメントを ASR で処理（短いセグメントもそのまま）
    transcripts = []
    asr_total_time = 0.0

    for start, end in segments:
        segment_audio = extract_segment(audio_file.audio, start, end, audio_file.sample_rate)
        asr_start = time.perf_counter()
        transcript, _ = engine.transcribe(segment_audio, audio_file.sample_rate)
        asr_total_time += time.perf_counter() - asr_start
        transcripts.append(transcript)

    # 結果の結合
    full_transcript = combine_segments(transcripts, audio_file.language)

    # セグメント統計を計算
    total_speech_duration = sum(e - s for s, e in segments)
    speech_ratio = total_speech_duration / audio_file.duration

    return BenchmarkResult(
        engine=engine_id,
        vad=vad.name,
        vad_config=vad.config,  # VAD パラメータを記録
        language=audio_file.language,
        audio_file=audio_file.stem,
        transcript=full_transcript,
        reference=audio_file.transcript,
        wer=calculate_wer(audio_file.transcript, full_transcript),
        cer=calculate_cer(audio_file.transcript, full_transcript),
        vad_rtf=vad_time / audio_file.duration,  # VAD のみ
        rtf=asr_total_time / audio_file.duration,  # ASR のみ
        segments_count=len(segments),
        avg_segment_duration_s=total_speech_duration / len(segments),
        speech_ratio=speech_ratio,  # 音声区間の割合
    )
```

**設計理由:**
- **空セグメント**: 無音ファイルは存在し得る。WER/CER で「全削除」として評価
- **短セグメント**: フィルタすると VAD の問題を隠蔽。VAD 設定（`min_speech_ms` 等）で調整すべき
- **RTF 分離**: VAD と ASR のボトルネック特定が容易
- **speech_ratio**: VAD の傾向（積極的/保守的）を定量化
- **vad_config**: 再現性と診断能力の確保

#### C-7: Quick Mode データソース

**決定:** 既存の DatasetManager をそのまま使用

```python
# benchmarks/vad/runner.py
class VADBenchmarkRunner:
    def __init__(self, config: VADBenchmarkConfig):
        self.dataset_manager = DatasetManager()  # 既存を再利用

    def _benchmark_language(self, language: str) -> None:
        # mode に応じて適切なデータセットを取得
        dataset = self.dataset_manager.get_dataset(language, mode=self.config.mode)
        # quick → tests/assets/audio/{lang}/
        # standard/full → tests/assets/prepared/{lang}/
```

**理由:**
- ASR ベンチマークと同じデータソースを使用することで一貫性を確保
- DatasetManager は mode に応じてパスを切り替える機能を既に持っている
- 新規実装不要

#### 依存関係追加

`pyproject.toml` への追加:

```toml
[project.optional-dependencies]
# 既存（変更なし）
vad = ["silero-vad>=5.1"]

# 新規: 個別バックエンド
vad-webrtc = ["webrtcvad>=2.0.10"]  # 軽量、torch 不要
vad-tenvad = ["ten-vad"]            # 軽量
vad-javad = ["javad"]               # torch 共有（ベンチマーク用）

# 全バックエンド
vad-all = [
    "livecap-core[vad]",
    "livecap-core[vad-webrtc]",
    "livecap-core[vad-tenvad]",
    "livecap-core[vad-javad]",
]

# ベンチマーク
benchmark = [
    "livecap-core[vad-all]",
    "jiwer>=3.0",
    # ... 既存 ...
]
```

**依存関係の分離理由:**
- `webrtcvad` は非常に軽量（C 拡張のみ、ML フレームワーク不要）
- ユーザーは必要なバックエンドのみインストール可能
- `[vad]` は Silero のまま（既存互換性維持）

---

## 9. CLI インターフェース

### 9.1 ASR ベンチマーク

```bash
# 全エンジン比較
python -m benchmarks.asr --all

# 言語別比較
python -m benchmarks.asr --language ja

# 特定エンジン
python -m benchmarks.asr --engine reazonspeech parakeet_ja whispers2t_base

# 実行モード
python -m benchmarks.asr --mode quick    # デフォルト2エンジン
python -m benchmarks.asr --mode standard # 言語別全エンジン
python -m benchmarks.asr --mode full     # 全エンジン×全言語

# 出力形式
python -m benchmarks.asr --output results.json --format json
python -m benchmarks.asr --output report.md --format markdown
```

### 9.2 VAD ベンチマーク

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

### 9.3 統合実行

```bash
# 両方のベンチマークを実行
python -m benchmarks --type both --mode standard

# ASRのみ
python -m benchmarks --type asr --mode quick

# VADのみ
python -m benchmarks --type vad --mode full
```

---

## 10. 出力フォーマット

### 10.1 出力ディレクトリ構造

**決定事項:** タイムスタンプ付きディレクトリで複数回の実行結果を保持

```
benchmark_results/
  {YYYYMMDD_HHMMSS}_{mode}/     # 例: 20250126_143052_quick/
    summary.md                   # 全体サマリー（Markdown）
    raw/
      {engine}_{lang}.csv        # 例: parakeet_ja_ja.csv, whispers2t_large_v3_en.csv
```

**役割分担:**
- `summary.md`: 人間が読むレポート（集約結果、Best/Fastest等）
- `raw/*.csv`: 分析用データ（各ファイルの詳細結果）

### 10.2 生データ形式（CSV）

**決定事項:** CSV形式（JSONはデータが膨れがちなため）

**ファイル単位:** エンジン×言語ごとに1ファイル

**CSV構造:**
```csv
file_id,reference,transcript,cer,wer,rtf,duration_sec
JSUT_basic5000_0001,水をマレーシアから買わなければならないのです,水をマレーシアから買わなければならないのです,0.0000,0.0000,0.12,3.45
JSUT_basic5000_0002,よくよく調べればつまらない話だと思う,よくよく調べれ詰らない話だと思う,0.0526,0.1429,0.15,2.89
```

**カラム定義:**

| カラム | 説明 |
|--------|------|
| `file_id` | ファイル識別子 |
| `reference` | 教師文字列（正解） |
| `transcript` | 文字起こし結果 |
| `cer` | Character Error Rate（正規化後） |
| `wer` | Word Error Rate（正規化後） |
| `rtf` | Real-Time Factor |
| `duration_sec` | 音声の長さ（秒） |

**メリット:**
- スプレッドシート（Excel等）での分析が容易
- 外部ツール（diff等）での詳細比較が可能
- `reference` と `transcript` を並列表示することで比較が可能

### 10.3 サマリーレポート（Markdown）

**内容:**
- ベンチマーク実行情報（日時、モード、実行回数）
- エンジン×言語ごとの集約結果テーブル
- Best by language（最高精度）
- Fastest（最高速）
- Lowest VRAM（最小メモリ）

**出力例:**
```markdown
# ASR Benchmark Report

**Date:** 2025-01-26 14:30:52
**Mode:** standard
**Runs:** 3

## Results by Language

### Japanese (ja)

| Engine | CER | WER | RTF (mean±std) | VRAM |
|--------|-----|-----|----------------|------|
| parakeet_ja | 3.2% | 8.1% | 0.12 ± 0.01 | 3584MB |
| whispers2t_large_v3 | 4.1% | 9.5% | 0.15 ± 0.02 | 1536MB |

**Best CER:** parakeet_ja (3.2%)
**Fastest:** parakeet_ja (RTF 0.12)

### English (en)
...

## Summary

- **Total files:** 200
- **Total duration:** 1234.5 sec
- **Errors/Skipped:** 2
```

### 10.4 VAD ベンチマーク出力（Phase C）

#### CSV 構造

VAD ベンチマークでは追加カラムを含む:

```csv
file_id,vad,asr,reference,transcript,cer,wer,rtf,vad_rtf,segments_count,speech_ratio,duration_sec
```

| カラム | 説明 |
|--------|------|
| `vad` | VAD バックエンド名 |
| `asr` | ASR エンジン名 |
| `vad_rtf` | VAD 処理の Real-Time Factor |
| `segments_count` | 検出セグメント数 |
| `speech_ratio` | 音声区間の割合（0.0-1.0） |

#### サマリーレポートの VAD 設定テーブル

**再現性と診断能力の確保のため、VAD の設定パラメータをレポートに含める。**

```markdown
# VAD Benchmark Report

**Date:** 2025-01-28 14:30:52
**Mode:** standard

## VAD Configurations

| ID | Backend | Parameters |
|----|---------|------------|
| silero | SileroVAD | threshold=0.5 |
| webrtc_mode0 | WebRTCVAD | mode=0, frame_duration_ms=20 |
| webrtc_mode1 | WebRTCVAD | mode=1, frame_duration_ms=20 |
| webrtc_mode2 | WebRTCVAD | mode=2, frame_duration_ms=20 |
| webrtc_mode3 | WebRTCVAD | mode=3, frame_duration_ms=20 |
| tenvad | TenVAD | hop_size=256, threshold=0.5 |
| javad_tiny | JaVADPipeline | model=tiny, window_ms=640 |
| javad_balanced | JaVADPipeline | model=balanced, window_ms=1920 |
| javad_precise | JaVADPipeline | model=precise, window_ms=3840 |

## Results by Language

### Japanese (ja)

| VAD | ASR | CER | WER | RTF | VAD RTF | Segments | Speech Ratio |
|-----|-----|-----|-----|-----|---------|----------|--------------|
| silero | parakeet_ja | 3.2% | 8.1% | 0.12 | 0.02 | 45 | 0.72 |
| webrtc_mode3 | parakeet_ja | 4.5% | 9.8% | 0.12 | 0.01 | 52 | 0.68 |
| javad_precise | parakeet_ja | 3.0% | 7.9% | 0.12 | 0.05 | 38 | 0.75 |

**Best CER:** javad_precise + parakeet_ja (3.0%)
**Fastest VAD:** webrtc_mode3 (VAD RTF 0.01)
**Highest Speech Ratio:** javad_precise (0.75)
```

**パラメータテーブルの意義:**
1. **再現性**: 同じパラメータで再実行可能
2. **診断**: WER 差異の原因を推測可能（例: mode 0 vs mode 3）
3. **比較**: パラメータ変更時の影響を追跡可能

---

## 11. CI ワークフロー

### 11.1 ワークフロー設計

**構成:** 分離方式（ASR と VAD を別ワークフローに分離）

```
.github/workflows/
├── asr-benchmark.yml     # ASR 単体ベンチマーク（既存）
├── vad-benchmark.yml     # VAD ベンチマーク（C-3 で新規作成）
└── core-tests.yml        # 既存
```

**ASR Benchmark (`asr-benchmark.yml`):**
- 目的: ASR エンジンの比較
- VAD: Silero 固定（デフォルト）
- パラメータ: `mode`, `language`, `engine`

**VAD Benchmark (`vad-benchmark.yml`):**
- 目的: VAD バックエンドの比較
- ASR: 言語別に固定（ja: parakeet_ja, en: parakeet）
- パラメータ: `mode`, `language`, `vad`

**共通:**
- トリガー: `workflow_dispatch` (手動実行)
- 実行環境: `[self-hosted, windows]` (RTX 4090)
- 処理フロー:
  1. Checkout → FFmpeg setup → Python environment (`uv sync`)
  2. Benchmark 実行 → `results.json` 出力
  3. Report 生成 → `report.md` 出力
  4. Artifact upload + GitHub Step Summary

### 11.2 実行モード詳細

| モード | ASR Benchmark | VAD Benchmark | 推定時間 |
|--------|--------------|---------------|---------|
| `quick` | 4 (言語別2) | 18 (9 VAD × 1 ASR × 2 lang) | ~3-5分 |
| `standard` | 10 (言語別全) | 36-54 (9 VAD × 2-3 ASR × 2 lang) | ~20分 |
| `full` | 20 (全組み合わせ) | 72+ (9 VAD × 全ASR × 2 lang) | ~60分 |

---

## 12. 依存関係

### 12.1 pyproject.toml 追加

> **Note**: 依存関係の詳細な構成は [セクション 8: 依存関係追加](#依存関係追加) を参照してください。
> `pyproject.toml` の更新は C-1 実装時に行います。

**採用する構成（モジュラー方式）:**

```toml
[project.optional-dependencies]
# 個別 VAD バックエンド（ユーザーは必要なもののみ選択可能）
vad = ["silero-vad>=5.1"]           # 既存（暫定デフォルト）
vad-webrtc = ["webrtcvad>=2.0.10"]  # 軽量、torch 不要
vad-tenvad = ["ten-vad"]            # 軽量（使用時警告表示）
vad-javad = ["javad"]               # ベンチマーク用

# 全 VAD バックエンド
vad-all = [
    "livecap-core[vad]",
    "livecap-core[vad-webrtc]",
    "livecap-core[vad-tenvad]",
    "livecap-core[vad-javad]",
]

# ベンチマーク
benchmark = [
    "livecap-core[vad-all]",
    # Metrics
    "jiwer>=3.0",
    # Reporting
    "matplotlib",
    "pandas",
    "tabulate",
    # Progress display
    "tqdm>=4.0",
    # Profiling
    "memory_profiler",
]
```

**モジュラー方式の利点:**
- ユーザーは必要な VAD のみインストール可能
- `webrtcvad` は非常に軽量（C 拡張のみ、ML フレームワーク不要）
- `[vad]` は Silero のまま（既存互換性維持）
- ベンチマーク実行時は `[benchmark]` で全 VAD が自動インストール

### 12.2 既存依存の活用

- `engines/` - ASR エンジン実装
- `engines/metadata.py` - エンジンメタデータ
- `engines/engine_factory.py` - エンジン生成
- `tests/utils/text_normalization.py` - テキスト正規化

---

## 13. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| TenVAD ライセンス問題 | ライセンス条件限定的 | 使用時に警告メッセージを表示 |
| 大規模モデルのメモリ不足 | テスト失敗 | エンジンごとにメモリ解放、順次実行 |
| 全組み合わせの実行時間 | CI タイムアウト | モード分離（quick/standard/full） |
| エンジン依存関係の競合 | インストール失敗 | `engines-nemo` と `engines-torch` を分離 |

---

## 14. 将来の拡張

### 14.1 ノイズ耐性評価

- DEMAND ノイズデータセットとの混合
- SNR 別の精度評価

### 14.2 リアルタイム性能評価

- ストリーミング処理のレイテンシ測定
- メモリ使用量の時系列分析

### 14.3 多言語拡張

- de, fr, es などの追加言語
- 言語検出精度の評価

---

## 15. 参考資料

- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [Silero VAD Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)
- [JaVAD GitHub](https://github.com/skrbnv/javad)
- [TenVAD GitHub](https://github.com/TEN-framework/ten-vad)
- [jiwer (WER calculation)](https://github.com/jitsi/jiwer)
- `engines/metadata.py` - エンジンメタデータ定義
- `docs/reference/vad-comparison.md` - VAD 比較調査
- `tests/integration/engines/test_smoke_engines.py` - 既存エンジンテスト
