"""マイク音声ソース

システムマイクからのリアルタイム音声キャプチャ。
sounddevice ライブラリを使用。

WASAPI 対応:
    Windows では WASAPI デバイスはネイティブサンプルレート（多くは 48000Hz）
    のみサポートするため、デバイスのネイティブサンプルレートで開き、
    内部でリサンプリングを行う。
"""

from __future__ import annotations

import logging
import queue
import sys
from math import gcd
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy import signal

from .base import AudioSource, DeviceInfo

logger = logging.getLogger(__name__)


class MicrophoneSource(AudioSource):
    """
    マイクからの音声ストリーム

    システムのデフォルトマイクまたは指定デバイスから
    音声をキャプチャする。

    Args:
        device: デバイスインデックスまたは名前。None でデフォルト。
        sample_rate: サンプリングレート (Hz)
        chunk_ms: チャンクサイズ（ミリ秒）

    Usage:
        # デフォルトマイクから
        with MicrophoneSource() as mic:
            for chunk in mic:
                process(chunk)

        # 特定デバイスから
        devices = MicrophoneSource.list_devices()
        with MicrophoneSource(device=devices[0].index) as mic:
            for chunk in mic:
                process(chunk)
    """

    def __init__(
        self,
        device: Optional[int | str] = None,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
    ):
        super().__init__(sample_rate=sample_rate, chunk_ms=chunk_ms)
        self.device = device
        self._stream: Optional[sd.InputStream] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        # デバイスのネイティブサンプルレート（start() で設定）
        self._device_sample_rate: Optional[int] = None
        # リサンプリング用のキャッシュ（up/down 比率）
        self._resample_up: int = 1
        self._resample_down: int = 1

    def start(self) -> None:
        """マイクキャプチャを開始"""
        if self._is_active:
            return

        # デバイスのネイティブサンプルレートを取得
        self._device_sample_rate = self._get_device_sample_rate(self.device)

        # リサンプリング比率を計算（効率的な整数比）
        if self._device_sample_rate != self.sample_rate:
            g = gcd(self._device_sample_rate, self.sample_rate)
            self._resample_up = self.sample_rate // g
            self._resample_down = self._device_sample_rate // g
        else:
            self._resample_up = 1
            self._resample_down = 1

        # デバイスサンプルレートでのチャンクサイズを計算
        device_chunk_size = int(self._device_sample_rate * self.chunk_ms / 1000)

        def callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: sd.CallbackFlags,
        ) -> None:
            """sounddevice コールバック"""
            if status:
                logger.warning(f"Audio callback status: {status}")
            # float32 モノラルに変換してキューに追加
            audio = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            self._queue.put(audio.flatten().astype(np.float32))

        try:
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self._device_sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=device_chunk_size,
                callback=callback,
            )
            self._stream.start()
            self._is_active = True

            device_name = self._get_device_name()
            if self._device_sample_rate != self.sample_rate:
                logger.info(
                    f"MicrophoneSource started: {device_name} "
                    f"(device: {self._device_sample_rate}Hz -> output: {self.sample_rate}Hz, "
                    f"{self.chunk_ms}ms chunks)"
                )
            else:
                logger.info(
                    f"MicrophoneSource started: {device_name} "
                    f"({self.sample_rate}Hz, {self.chunk_ms}ms chunks)"
                )
        except sd.PortAudioError as e:
            logger.error(f"Failed to start microphone: {e}")
            raise RuntimeError(f"Failed to start microphone: {e}") from e

    def stop(self) -> None:
        """マイクキャプチャを停止"""
        if not self._is_active:
            return

        self._is_active = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # キューをクリア
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        logger.info("MicrophoneSource stopped")

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        次のチャンクを読み取り

        Args:
            timeout: タイムアウト（秒）。None で無限待機。

        Returns:
            音声チャンク（float32, 指定サンプルレート）。タイムアウト時は None。
        """
        if not self._is_active:
            return None

        try:
            chunk = self._queue.get(timeout=timeout)
            # リサンプリング（必要な場合）
            if self._resample_up != 1 or self._resample_down != 1:
                chunk = self._resample(chunk)
            return chunk
        except queue.Empty:
            return None

    @classmethod
    def list_devices(cls, prefer_wasapi: bool = False) -> list[DeviceInfo]:
        """
        利用可能な入力デバイス一覧を取得

        Args:
            prefer_wasapi: True の場合、Windows で WASAPI デバイスのみ返す。
                          Linux/macOS では無視される。

        Returns:
            入力チャンネルを持つデバイスのリスト
        """
        devices: list[DeviceInfo] = []
        try:
            default_device = sd.default.device[0]  # 入力デバイスのデフォルト
        except Exception:
            default_device = None

        # ホスト API 情報を取得（デバイスインデックス → API 名のマッピング）
        device_to_api: dict[int, str] = {}
        wasapi_devices: Optional[set[int]] = None
        wasapi_default_input: Optional[int] = None
        try:
            for api in sd.query_hostapis():
                api_name = api["name"]
                for dev_idx in api["devices"]:
                    device_to_api[dev_idx] = api_name
                # Windows WASAPI のデバイスを記録
                if prefer_wasapi and sys.platform == "win32" and "WASAPI" in api_name:
                    wasapi_devices = set(api["devices"])
                    wasapi_default_input = api.get("default_input_device")
        except Exception as e:
            logger.debug(f"Failed to query host APIs: {e}")

        # prefer_wasapi モードでは WASAPI のデフォルトを使用
        if wasapi_default_input is not None:
            default_device = wasapi_default_input

        try:
            for i, dev in enumerate(sd.query_devices()):
                # 入力チャンネルがあるデバイスのみ
                if dev["max_input_channels"] <= 0:
                    continue

                # WASAPI フィルタ（Windows で prefer_wasapi=True の場合）
                if wasapi_devices is not None and i not in wasapi_devices:
                    continue

                devices.append(
                    DeviceInfo(
                        index=i,
                        name=dev["name"],
                        channels=dev["max_input_channels"],
                        sample_rate=int(dev["default_samplerate"]),
                        is_default=(i == default_device),
                        host_api=device_to_api.get(i),
                    )
                )
        except sd.PortAudioError as e:
            logger.warning(f"Failed to query devices: {e}")

        return devices

    def _get_device_name(self) -> str:
        """現在のデバイス名を取得"""
        try:
            if self.device is None:
                default_idx = sd.default.device[0]
                if default_idx is not None:
                    return sd.query_devices(default_idx)["name"]
                return "default"
            elif isinstance(self.device, int):
                return sd.query_devices(self.device)["name"]
            else:
                return str(self.device)
        except Exception:
            return "unknown"

    def _get_device_sample_rate(self, device: Optional[int | str]) -> int:
        """
        デバイスのネイティブサンプルレートを取得

        Args:
            device: デバイスインデックスまたは名前。None でデフォルト。

        Returns:
            デバイスのデフォルトサンプルレート
        """
        try:
            if device is None:
                dev_info = sd.query_devices(kind="input")
            else:
                dev_info = sd.query_devices(device)
            return int(dev_info["default_samplerate"])
        except Exception as e:
            logger.warning(f"Failed to get device sample rate: {e}, using 16000Hz")
            return 16000

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """
        リサンプリング（デバイスサンプルレート → 出力サンプルレート）

        効率的なポリフェーズリサンプリングを使用。

        Args:
            audio: 入力音声データ（デバイスサンプルレート）

        Returns:
            リサンプリング後の音声データ（出力サンプルレート）
        """
        resampled = signal.resample_poly(audio, self._resample_up, self._resample_down)
        return resampled.astype(np.float32)
