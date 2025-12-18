# Issue #74 実装計画: 依存関係整理・CLI・パッケージ名変更

## 概要

Issue #74 は livecap-core リファクタリングの Phase 6 として、CLI コマンドの実装とパッケージ名変更を行う。

**親 Issue:** #64 [Epic] livecap-cli リファクタリング

## 現状分析

### 1. pyproject.toml (既存構造)

```toml
# 現在の optional-dependencies
"engines-nemo"       # NeMo系エンジン (canary, parakeet, voxtral)
"engines-torch"      # PyTorch系エンジン (reazonspeech)
"translation"        # Google翻訳 (deep-translator)
"translation-local"  # OPUS-MT ローカル翻訳
"translation-riva"   # Riva-4B 翻訳
"dev"                # 開発ツール (pytest)
"benchmark"          # ベンチマーク (jiwer, tabulate)
"optimization"       # 最適化 (optuna, plotly)
```

**Issue #74 の目標構造との差分:**

| 目標 | 現状 | 対応 |
|------|------|------|
| `engine-whisper` | コア依存に含む | ✅ 維持 |
| `engine-sherpa` | コア依存に含む | ✅ 維持 |
| `engine-torch` | `engines-torch` | ✅ 既存 |
| `engine-nemo` | `engines-nemo` | ✅ 既存 |
| `translation-google` | `translation` | 🔄 リネーム検討 |
| `translation-riva` | `translation-riva` | ✅ 既存 |
| `recommended` | なし | ➕ 追加 |
| `all` | なし | ➕ 追加 |

### 2. CLI (既存機能)

```bash
# 現在のコマンド
livecap-core --info           # インストール診断
livecap-core --ensure-ffmpeg  # FFmpeg確保
livecap-core --as-json        # JSON出力
```

**Issue #74 の目標コマンド:**

| コマンド | 現状 | 実装難易度 |
|----------|------|-----------|
| `livecap-cli transcribe --realtime --mic <id>` | なし | 中 (ロジックは examples/ に存在) |
| `livecap-cli transcribe --realtime --system` | なし | 高 (システム音声キャプチャ未実装) |
| `livecap-cli transcribe <file> -o <output>` | なし | 低 (FileTranscriptionPipeline 利用) |
| `livecap-cli devices` | なし | 低 (MicrophoneSource.list_devices()) |
| `livecap-cli engines` | なし | 低 (EngineMetadata.get_all()) |

### 3. 利用可能なエンジン・翻訳器

**ASR エンジン (6種):**
- `whispers2t` - WhisperS2T (CTranslate2)
- `reazonspeech` - ReazonSpeech K2 (CPU専用)
- `canary` - NVIDIA Canary 1B Flash
- `parakeet` / `parakeet_ja` - NVIDIA Parakeet TDT
- `voxtral` - Mistral Voxtral Mini 3B

**翻訳器 (3種):**
- `google` - Google Translate (deep-translator)
- `opus_mt` - OPUS-MT (CTranslate2)
- `riva_instruct` - Riva-Translate-4B-Instruct

### 4. システム音声キャプチャの課題

`--system` オプションにはシステム音声のキャプチャが必要。これはプラットフォーム依存:

| プラットフォーム | 方法 | 難易度 |
|-----------------|------|--------|
| Windows | WASAPI loopback | 中 |
| macOS | BlackHole/Soundflower 経由 | 外部依存 |
| Linux | PulseAudio monitor | 中 |

**推奨:** Phase 6 では `--system` をスコープ外とし、将来課題とする。

---

## 実装計画

### Phase 6A: pyproject.toml 整理 (低リスク)

**目的:** extras の整理と `recommended` / `all` メタエクストラの追加

**変更内容:**

```toml
[project.optional-dependencies]
# 既存（変更なし）
"engines-nemo" = [...]
"engines-torch" = [...]
"translation" = [...]
"translation-local" = [...]
"translation-riva" = [...]
"dev" = [...]
"benchmark" = [...]
"optimization" = [...]

# 新規追加
"recommended" = [
  "livecap-core[translation]",  # Google翻訳
]
"all" = [
  "livecap-core[engines-nemo,engines-torch,translation,translation-local,translation-riva,benchmark,optimization]",
]
```

**完了条件:**
- [ ] `pip install livecap-core[recommended]` が動作
- [ ] `pip install livecap-core[all]` が動作
- [ ] 既存の extras が引き続き動作

### Phase 6B: CLI コマンド実装 (中リスク)

**目的:** `transcribe`, `devices`, `engines` コマンドの実装

#### 6B-1: サブコマンド構造の導入

現在の argparse を拡張し、サブコマンド構造を導入:

```bash
livecap-core info              # 現在の --info 相当
livecap-core devices           # オーディオデバイス一覧
livecap-core engines           # ASRエンジン一覧
livecap-core translators       # 翻訳器一覧
livecap-core transcribe [args] # 文字起こし
```

#### 6B-2: devices コマンド

```python
def cmd_devices(args):
    from livecap_core import MicrophoneSource
    devices = MicrophoneSource.list_devices()
    for dev in devices:
        default = " (default)" if dev.is_default else ""
        print(f"[{dev.index}] {dev.name}{default}")
```

#### 6B-3: engines コマンド

```python
def cmd_engines(args):
    from livecap_core.engines.metadata import EngineMetadata
    for engine_id, meta in EngineMetadata.get_all().items():
        print(f"{engine_id}: {meta.display_name}")
```

#### 6B-4: translators コマンド (追加提案)

```python
def cmd_translators(args):
    from livecap_core.translation.metadata import TranslatorMetadata
    for tid, info in TranslatorMetadata.get_all().items():
        gpu = " (GPU)" if info.requires_gpu else ""
        print(f"{tid}: {info.display_name}{gpu}")
```

#### 6B-5: transcribe コマンド

**マイク入力 (リアルタイム):**

```bash
livecap-core transcribe --realtime --mic 0 \
  --engine whispers2t --device cuda --language ja
```

実装: `examples/realtime/async_microphone.py` のロジックを CLI に統合

**ファイル入力:**

```bash
livecap-core transcribe input.mp4 -o output.srt \
  --engine whispers2t --device cuda --language ja
```

実装: `FileTranscriptionPipeline` を利用

**オプション:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--engine` | ASRエンジンID | `whispers2t` |
| `--device` | デバイス (auto/cuda/cpu) | `auto` |
| `--language` | 入力言語 | `ja` |
| `--model-size` | WhisperS2Tモデルサイズ | `base` |
| `--translate` | 翻訳器ID | なし |
| `--target-lang` | 翻訳先言語 | `en` |

**完了条件:**
- [ ] `livecap-core devices` が動作
- [ ] `livecap-core engines` が動作
- [ ] `livecap-core translators` が動作
- [ ] `livecap-core transcribe --realtime --mic 0` が動作
- [ ] `livecap-core transcribe input.mp4 -o output.srt` が動作

### Phase 6C: パッケージ名変更 (高リスク)

**目的:** `livecap-core` → `livecap-cli` へのリネーム

**影響範囲:**
1. `pyproject.toml` の `name` フィールド
2. `project.scripts` のエントリポイント名
3. PyPI への新規パッケージ公開
4. ドキュメント・README の更新
5. CI/CD の更新

**リスク:**
- 既存ユーザーへの影響（pip install 名が変更）
- PyPI での新規パッケージ登録が必要
- インポートパス `livecap_core` は**変更しない**（互換性維持）

**推奨:** Phase 6C は 6A/6B 完了後に慎重に実施

**変更内容:**

```toml
[project]
name = "livecap-cli"  # 変更

[project.scripts]
livecap-cli = "livecap_core.cli:main"  # 変更
# livecap-core も互換性のため残す
livecap-core = "livecap_core.cli:main"
```

**完了条件:**
- [ ] `pip install livecap-cli` が動作
- [ ] `livecap-cli transcribe ...` が動作
- [ ] 旧名 `livecap-core` も引き続き動作（互換性）

---

## スコープ外 (将来課題)

### システム音声キャプチャ (`--system`)

プラットフォーム依存性が高く、Phase 6 では実装しない。

**将来的な実装方針:**
1. WASAPI loopback (Windows)
2. PulseAudio monitor (Linux)
3. 外部ツール連携 (macOS: BlackHole)

### VAD オプション

現在は Silero VAD がデフォルト。CLI での VAD バックエンド選択は将来課題。

---

## 実装順序とリスク評価

```
Phase 6A (pyproject.toml整理)  [低リスク, 0.5日]
    ↓
Phase 6B (CLIコマンド実装)     [中リスク, 1-2日]
    ↓
Phase 6C (パッケージ名変更)    [高リスク, 0.5日]
```

**推奨:** 6A → 6B → 6C の順で、各 Phase を別 PR として作成

---

## 完了条件チェックリスト

### Phase 6A
- [ ] `recommended` extras 追加
- [ ] `all` extras 追加
- [ ] 既存テスト通過

### Phase 6B
- [ ] サブコマンド構造導入
- [ ] `devices` コマンド実装
- [ ] `engines` コマンド実装
- [ ] `translators` コマンド実装
- [ ] `transcribe --realtime --mic` 実装
- [ ] `transcribe <file> -o <output>` 実装
- [ ] ユニットテスト追加
- [ ] 既存テスト通過

### Phase 6C
- [ ] pyproject.toml 更新
- [ ] エントリポイント更新
- [ ] ドキュメント更新
- [ ] 既存テスト通過

---

## 関連ファイル

| ファイル | 役割 |
|---------|------|
| `pyproject.toml` | パッケージ設定 |
| `livecap_core/cli.py` | CLI エントリポイント |
| `livecap_core/audio_sources/microphone.py` | マイク入力 |
| `livecap_core/transcription/file_pipeline.py` | ファイル文字起こし |
| `livecap_core/transcription/stream.py` | リアルタイム文字起こし |
| `livecap_core/engines/metadata.py` | エンジンメタデータ |
| `livecap_core/translation/metadata.py` | 翻訳器メタデータ |
| `examples/realtime/async_microphone.py` | マイク入力サンプル |

---

## 参考: CLI 使用例 (目標)

```bash
# インストール
pip install livecap-cli[recommended]

# デバイス一覧
livecap-cli devices

# エンジン一覧
livecap-cli engines

# 翻訳器一覧
livecap-cli translators

# リアルタイム文字起こし (マイク)
livecap-cli transcribe --realtime --mic 0 --engine whispers2t --language ja

# ファイル文字起こし
livecap-cli transcribe input.mp4 -o output.srt --engine whispers2t

# 翻訳付き文字起こし
livecap-cli transcribe input.mp4 -o output.srt --translate google --target-lang en
```
