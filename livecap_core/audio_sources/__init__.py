"""Audio source module for livecap_core.

Provides abstract base class and concrete implementations for audio input.
"""

from .base import AudioSource, DeviceInfo
from .file import FileSource
from .microphone import MicrophoneSource

__all__ = [
    "AudioSource",
    "DeviceInfo",
    "FileSource",
    "MicrophoneSource",
]
