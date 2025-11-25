"""Pytest configuration for audio_sources tests.

Conditionally skips MicrophoneSource tests if PortAudio is not available.
"""


def pytest_ignore_collect(collection_path, config):
    """Skip test_microphone_source.py if sounddevice can't be imported."""
    if collection_path.name == "test_microphone_source.py":
        try:
            import sounddevice  # noqa: F401

            return False  # Don't ignore, run the tests
        except OSError:
            # PortAudio not found - skip this file
            return True
    return False
