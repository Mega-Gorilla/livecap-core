#!/usr/bin/env python3
"""API 概要デモ

livecap-cli の主要 API を確認するサンプルです。

Usage:
    python examples/library/api_overview.py
"""

from livecap_cli import (
    EngineFactory,
    MicrophoneSource,
    VADConfig,
)


def list_audio_devices():
    """利用可能なオーディオデバイスを一覧表示"""
    print("=" * 60)
    print("Audio Devices")
    print("=" * 60)

    try:
        devices = MicrophoneSource.list_devices()
        if not devices:
            print("No audio input devices found.")
            return

        for dev in devices:
            default_mark = " (default)" if dev.is_default else ""
            print(f"[{dev.index}] {dev.name}")
            print(f"    Channels: {dev.channels}")
            print(f"    Sample Rate: {dev.sample_rate}{default_mark}")
    except Exception as e:
        print(f"Error listing devices: {e}")
        print("Note: PortAudio may not be installed.")
    print()


def list_engines():
    """利用可能なエンジンを一覧表示"""
    print("=" * 60)
    print("Available Engines")
    print("=" * 60)

    engines = EngineFactory.get_available_engines()
    for engine_id, info in engines.items():
        print(f"{engine_id}:")
        print(f"    Name: {info['name']}")
        print(f"    Description: {info['description'][:60]}...")
    print()


def show_engine_info(engine_id: str = "whispers2t"):
    """特定エンジンの詳細情報を表示"""
    print("=" * 60)
    print(f"Engine Info: {engine_id}")
    print("=" * 60)

    info = EngineFactory.get_engine_info(engine_id)
    if info is None:
        print(f"Engine '{engine_id}' not found.")
        return

    print(f"Name: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Supported Languages: {', '.join(info['supported_languages'][:10])}...")
    print(f"Default Params: {info['default_params']}")

    if info.get("available_model_sizes"):
        print(f"Available Model Sizes: {', '.join(info['available_model_sizes'])}")
    print()


def list_engines_for_language(lang_code: str = "ja"):
    """特定言語に対応したエンジンを一覧表示"""
    print("=" * 60)
    print(f"Engines for Language: {lang_code}")
    print("=" * 60)

    engines = EngineFactory.get_engines_for_language(lang_code)
    if not engines:
        print(f"No engines found for language '{lang_code}'.")
        return

    for engine_id, info in engines.items():
        print(f"- {engine_id}: {info['name']}")
    print()


def show_vad_config():
    """VADConfig のデフォルト値を表示"""
    print("=" * 60)
    print("VADConfig Defaults")
    print("=" * 60)

    config = VADConfig()
    print(f"threshold: {config.threshold}")
    print(f"neg_threshold: {config.neg_threshold} (effective: {config.get_neg_threshold():.2f})")
    print(f"min_speech_ms: {config.min_speech_ms}")
    print(f"min_silence_ms: {config.min_silence_ms}")
    print(f"speech_pad_ms: {config.speech_pad_ms}")
    print(f"max_speech_ms: {config.max_speech_ms}")
    print(f"interim_min_duration_ms: {config.interim_min_duration_ms}")
    print(f"interim_interval_ms: {config.interim_interval_ms}")
    print()


def main():
    """メイン関数"""
    print("\nlivecap-cli API Overview\n")

    list_audio_devices()
    list_engines()
    show_engine_info("whispers2t")
    list_engines_for_language("ja")
    list_engines_for_language("en")
    show_vad_config()


if __name__ == "__main__":
    main()
