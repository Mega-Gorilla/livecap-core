"""Unit tests for MicrophoneSource.

These tests use mocks since actual hardware is not available in CI.
Skipped via conftest.py if sounddevice/PortAudio is not available.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from livecap_cli.audio_sources import MicrophoneSource


class TestMicrophoneSourceBasics:
    """MicrophoneSource 基本機能テスト"""

    def test_create_microphone_source(self):
        """MicrophoneSource 作成"""
        source = MicrophoneSource()
        assert source.device is None
        assert source.sample_rate == 16000
        assert source.chunk_ms == 100
        assert source.is_active is False

    def test_create_with_device_index(self):
        """デバイスインデックス指定で作成"""
        source = MicrophoneSource(device=0)
        assert source.device == 0

    def test_create_with_device_name(self):
        """デバイス名指定で作成"""
        source = MicrophoneSource(device="Test Device")
        assert source.device == "Test Device"

    def test_create_with_custom_params(self):
        """カスタムパラメータで作成"""
        source = MicrophoneSource(
            device=1,
            sample_rate=22050,
            chunk_ms=50,
        )
        assert source.device == 1
        assert source.sample_rate == 22050
        assert source.chunk_ms == 50


class TestMicrophoneSourceListDevices:
    """MicrophoneSource list_devices テスト"""

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_list_devices(self, mock_sd):
        """list_devices() がデバイス一覧を返す"""
        mock_sd.query_devices.return_value = [
            {
                "name": "Input Device 1",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000,
            },
            {
                "name": "Output Device",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
            {
                "name": "Input Device 2",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000,
            },
        ]
        mock_sd.query_hostapis.return_value = [
            {"name": "MME", "devices": [0, 1, 2]},
        ]
        mock_sd.default.device = (0, 1)  # (input, output)

        devices = MicrophoneSource.list_devices()

        # 入力デバイスのみ返す
        assert len(devices) == 2
        assert devices[0].name == "Input Device 1"
        assert devices[0].channels == 2
        assert devices[0].is_default is True
        assert devices[0].host_api == "MME"
        assert devices[1].name == "Input Device 2"
        assert devices[1].channels == 1
        assert devices[1].is_default is False
        assert devices[1].host_api == "MME"

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_list_devices_empty(self, mock_sd):
        """入力デバイスがない場合"""
        mock_sd.query_devices.return_value = [
            {
                "name": "Output Only",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
        ]
        mock_sd.query_hostapis.return_value = [
            {"name": "MME", "devices": [0]},
        ]
        mock_sd.default.device = (None, 0)

        devices = MicrophoneSource.list_devices()
        assert devices == []

    @patch("livecap_cli.audio_sources.microphone.sys")
    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_list_devices_prefer_wasapi_on_windows(self, mock_sd, mock_sys):
        """Windows で prefer_wasapi=True の場合、WASAPI デバイスのみ返す"""
        mock_sys.platform = "win32"
        mock_sd.query_devices.return_value = [
            # MME devices
            {
                "name": "Device A",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000,
            },
            # WASAPI devices
            {
                "name": "Device A",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000,
            },
            {
                "name": "Device B",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000,
            },
        ]
        mock_sd.query_hostapis.return_value = [
            {"name": "MME", "devices": [0], "default_input_device": 0},
            {"name": "Windows WASAPI", "devices": [1, 2], "default_input_device": 2},
        ]
        mock_sd.default.device = (0, None)  # グローバルデフォルトは MME

        # prefer_wasapi=True の場合
        devices = MicrophoneSource.list_devices(prefer_wasapi=True)
        assert len(devices) == 2
        assert all(d.host_api == "Windows WASAPI" for d in devices)
        assert devices[0].index == 1
        assert devices[1].index == 2
        # WASAPI のデフォルト（index=2）が is_default=True
        assert devices[0].is_default is False
        assert devices[1].is_default is True

    @patch("livecap_cli.audio_sources.microphone.sys")
    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_list_devices_prefer_wasapi_ignored_on_linux(self, mock_sd, mock_sys):
        """Linux では prefer_wasapi は無視される"""
        mock_sys.platform = "linux"
        mock_sd.query_devices.return_value = [
            {
                "name": "ALSA Device",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000,
            },
        ]
        mock_sd.query_hostapis.return_value = [
            {"name": "ALSA", "devices": [0]},
        ]
        mock_sd.default.device = (0, None)

        # Linux では prefer_wasapi=True でも全デバイスを返す
        devices = MicrophoneSource.list_devices(prefer_wasapi=True)
        assert len(devices) == 1
        assert devices[0].host_api == "ALSA"


class TestMicrophoneSourceStartStop:
    """MicrophoneSource start/stop テスト（モック使用）"""

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_start_creates_stream(self, mock_sd):
        """start() がストリームを作成"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 48000,
        }

        source = MicrophoneSource()
        source.start()

        mock_sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()
        assert source.is_active is True

        source.stop()

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_stop_closes_stream(self, mock_sd):
        """stop() がストリームを閉じる"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 48000,
        }

        source = MicrophoneSource()
        source.start()
        source.stop()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert source.is_active is False

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_double_start_is_safe(self, mock_sd):
        """二重 start() は安全"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 48000,
        }

        source = MicrophoneSource()
        source.start()
        source.start()  # Should not create second stream

        # InputStream は 1 回だけ呼ばれる
        assert mock_sd.InputStream.call_count == 1

        source.stop()

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_double_stop_is_safe(self, mock_sd):
        """二重 stop() は安全"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 48000,
        }

        source = MicrophoneSource()
        source.start()
        source.stop()
        source.stop()  # Should not raise

        assert source.is_active is False


class TestMicrophoneSourceRead:
    """MicrophoneSource read テスト（モック使用）"""

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_read_returns_from_queue_no_resample(self, mock_sd):
        """read() がキューからデータを返す（リサンプリングなし）"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        # デバイスサンプルレートと出力サンプルレートが同じ
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 16000,
        }

        source = MicrophoneSource(sample_rate=16000)
        source.start()

        # キューに直接データを追加
        test_audio = np.zeros(1600, dtype=np.float32)
        source._queue.put(test_audio)

        chunk = source.read(timeout=1.0)
        assert chunk is not None
        assert len(chunk) == 1600

        source.stop()

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_read_returns_none_on_timeout(self, mock_sd):
        """タイムアウトで None"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 48000,
        }

        source = MicrophoneSource()
        source.start()

        # キューは空
        chunk = source.read(timeout=0.01)
        assert chunk is None

        source.stop()

    def test_read_returns_none_when_not_active(self):
        """非アクティブ時は None"""
        source = MicrophoneSource()
        chunk = source.read()
        assert chunk is None


class TestMicrophoneSourceContextManager:
    """MicrophoneSource コンテキストマネージャ テスト"""

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_context_manager(self, mock_sd):
        """コンテキストマネージャとして使用"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 48000,
        }

        with MicrophoneSource() as source:
            assert source.is_active is True

        assert source.is_active is False
        mock_stream.stop.assert_called()
        mock_stream.close.assert_called()


class TestMicrophoneSourceResampling:
    """MicrophoneSource リサンプリング テスト"""

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_start_uses_device_native_samplerate(self, mock_sd):
        """start() がデバイスのネイティブサンプルレートで InputStream を開く"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 48000,
        }

        source = MicrophoneSource(sample_rate=16000)
        source.start()

        # InputStream は 48000Hz で開かれる
        call_kwargs = mock_sd.InputStream.call_args[1]
        assert call_kwargs["samplerate"] == 48000
        # blocksize は 48000Hz の 100ms = 4800 samples
        assert call_kwargs["blocksize"] == 4800

        source.stop()

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_resample_ratio_calculated_correctly(self, mock_sd):
        """リサンプリング比率が正しく計算される"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 48000,
        }

        source = MicrophoneSource(sample_rate=16000)
        source.start()

        # 48000 -> 16000 の比率は 1:3 (up=1, down=3)
        assert source._resample_up == 1
        assert source._resample_down == 3

        source.stop()

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_read_resamples_audio(self, mock_sd):
        """read() がリサンプリングを行う"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 48000,
        }

        source = MicrophoneSource(sample_rate=16000, chunk_ms=100)
        source.start()

        # 48000Hz の 100ms = 4800 samples
        test_audio = np.sin(
            2 * np.pi * 440 * np.arange(4800) / 48000
        ).astype(np.float32)
        source._queue.put(test_audio)

        chunk = source.read(timeout=1.0)
        assert chunk is not None
        # リサンプリング後は 16000Hz の 100ms = 1600 samples
        assert len(chunk) == 1600
        assert chunk.dtype == np.float32

        source.stop()

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_no_resample_when_samplerates_match(self, mock_sd):
        """サンプルレートが同じ場合はリサンプリングしない"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Test Device",
            "default_samplerate": 16000,
        }

        source = MicrophoneSource(sample_rate=16000)
        source.start()

        # リサンプリング比率は 1:1
        assert source._resample_up == 1
        assert source._resample_down == 1

        # キューに追加したデータがそのまま返される
        test_audio = np.zeros(1600, dtype=np.float32)
        source._queue.put(test_audio)

        chunk = source.read(timeout=1.0)
        assert chunk is not None
        assert len(chunk) == 1600

        source.stop()

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_resample_various_ratios(self, mock_sd):
        """様々なサンプルレート比率でリサンプリングが正しく動作する"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)

        test_cases = [
            (44100, 16000),  # CD品質 -> ASR
            (48000, 16000),  # 標準 -> ASR
            (96000, 16000),  # ハイレゾ -> ASR
        ]

        for device_rate, output_rate in test_cases:
            mock_sd.query_devices.return_value = {
                "name": "Test Device",
                "default_samplerate": device_rate,
            }

            source = MicrophoneSource(sample_rate=output_rate, chunk_ms=100)
            source.start()

            # デバイスサンプルレートの 100ms
            device_chunk_size = int(device_rate * 100 / 1000)
            test_audio = np.zeros(device_chunk_size, dtype=np.float32)
            source._queue.put(test_audio)

            chunk = source.read(timeout=1.0)
            assert chunk is not None
            # 出力サンプルレートの 100ms
            expected_size = int(output_rate * 100 / 1000)
            assert len(chunk) == expected_size, (
                f"Failed for {device_rate}->{output_rate}: "
                f"expected {expected_size}, got {len(chunk)}"
            )

            source.stop()
