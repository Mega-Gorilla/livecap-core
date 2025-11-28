#!/usr/bin/env python
"""TenVAD Debug Script

Detailed investigation of TenVAD behavior, especially:
- Instance creation
- reset() behavior (recreate instance)
- process() after reset

Run: python scripts/debug_tenvad.py
"""

from __future__ import annotations

import sys
import traceback
import warnings

import numpy as np


def create_test_audio(duration_s: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    """Create test audio (sine wave)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    return audio.astype(np.float32)


def test_tenvad_basic():
    """Test basic TenVAD operations."""
    print("=" * 60)
    print("TEST 1: Basic TenVAD creation and process")
    print("=" * 60)

    try:
        from livecap_core.vad.backends.tenvad import TenVAD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            vad = TenVAD(hop_size=256, threshold=0.5)

        print(f"  Created TenVAD: name={vad.name}, frame_size={vad.frame_size}")
        print(f"  Internal _vad: {vad._vad}")
        print(f"  _vad type: {type(vad._vad)}")

        # Test process with valid audio
        audio = create_test_audio(0.1)  # 100ms = 1600 samples
        print(f"\n  Test audio: shape={audio.shape}, dtype={audio.dtype}")
        print(f"  Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

        # Process frame by frame
        frame_size = vad.frame_size
        n_frames = len(audio) // frame_size
        print(f"\n  Processing {n_frames} frames (frame_size={frame_size})")

        for i in range(n_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            print(f"    Frame {i}: shape={frame.shape}, ", end="")

            # Convert to int16 like TenVAD.process() does
            frame_int16 = (frame * 32767).astype(np.int16)
            print(f"int16 range=[{frame_int16.min()}, {frame_int16.max()}], ", end="")

            try:
                prob = vad.process(frame)
                print(f"probability={prob:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                return False

        print("\n  [PASS] Basic TenVAD test passed")
        return True

    except ImportError as e:
        print(f"  [SKIP] TenVAD not installed: {e}")
        return None
    except OSError as e:
        print(f"  [SKIP] TenVAD native library error: {e}")
        return None


def test_tenvad_reset():
    """Test TenVAD reset behavior."""
    print("\n" + "=" * 60)
    print("TEST 2: TenVAD reset() behavior")
    print("=" * 60)

    try:
        from livecap_core.vad.backends.tenvad import TenVAD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            vad = TenVAD(hop_size=256, threshold=0.5)

        print(f"  Initial _vad id: {id(vad._vad)}")
        original_id = id(vad._vad)

        # Process some frames first
        audio = create_test_audio(0.1)
        frame = audio[: vad.frame_size]
        prob = vad.process(frame)
        print(f"  Process before reset: prob={prob:.4f}")

        # Reset
        print("\n  Calling reset()...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            vad.reset()

        print(f"  After reset _vad id: {id(vad._vad)}")
        new_id = id(vad._vad)

        if original_id != new_id:
            print("  [INFO] _vad instance was recreated (expected)")
        else:
            print("  [WARN] _vad instance was NOT recreated")

        print(f"  _vad type: {type(vad._vad)}")

        print("\n  [PASS] Reset test passed")
        return True

    except ImportError as e:
        print(f"  [SKIP] TenVAD not installed: {e}")
        return None
    except OSError as e:
        print(f"  [SKIP] TenVAD native library error: {e}")
        return None


def test_tenvad_process_after_reset():
    """Test TenVAD process after reset - this is where the error occurs."""
    print("\n" + "=" * 60)
    print("TEST 3: TenVAD process() AFTER reset() (critical test)")
    print("=" * 60)

    try:
        from livecap_core.vad.backends.tenvad import TenVAD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            vad = TenVAD(hop_size=256, threshold=0.5)

        print(f"  Created TenVAD, _vad id: {id(vad._vad)}")

        # Process before reset
        audio = create_test_audio(0.5)  # 500ms
        frame = audio[: vad.frame_size]
        print(f"\n  Frame shape: {frame.shape}, dtype: {frame.dtype}")

        prob1 = vad.process(frame)
        print(f"  Process BEFORE reset: prob={prob1:.4f}")

        # Reset (this recreates the instance)
        print("\n  Calling reset()...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            vad.reset()
        print(f"  After reset, _vad id: {id(vad._vad)}")

        # Process after reset - THIS IS WHERE THE ERROR SHOULD OCCUR
        print("\n  Processing AFTER reset (this should trigger the error)...")
        frame2 = audio[vad.frame_size : 2 * vad.frame_size]
        print(f"  Frame2 shape: {frame2.shape}, dtype: {frame2.dtype}")
        print(f"  Frame2 range: [{frame2.min():.4f}, {frame2.max():.4f}]")

        # Convert to int16 for debugging
        frame2_int16 = (frame2 * 32767).astype(np.int16)
        print(f"  Frame2 int16 range: [{frame2_int16.min()}, {frame2_int16.max()}]")

        try:
            prob2 = vad.process(frame2)
            print(f"  Process AFTER reset: prob={prob2:.4f}")
            print("\n  [PASS] Process after reset succeeded!")
            return True
        except Exception as e:
            print(f"\n  [FAIL] Process after reset FAILED: {e}")
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"  [SKIP] TenVAD not installed: {e}")
        return None
    except OSError as e:
        print(f"  [SKIP] TenVAD native library error: {e}")
        return None


def test_ten_vad_raw():
    """Test raw ten_vad.TenVad directly."""
    print("\n" + "=" * 60)
    print("TEST 4: Raw ten_vad.TenVad test (bypassing our wrapper)")
    print("=" * 60)

    try:
        from ten_vad import TenVad

        print("  Creating first TenVad instance...")
        vad1 = TenVad(hop_size=256, threshold=0.5)
        print(f"  vad1 id: {id(vad1)}")
        print(f"  vad1 attributes: {dir(vad1)}")

        # Create test audio
        audio = create_test_audio(0.1)
        frame_int16 = (audio[:256] * 32767).astype(np.int16)
        print(f"\n  Test frame: shape={frame_int16.shape}, dtype={frame_int16.dtype}")

        # Process with vad1
        print("  Processing with vad1...")
        result1 = vad1.process(frame_int16)
        print(f"  vad1.process() result: {result1}")

        # Create second instance (simulating what reset() does)
        print("\n  Creating second TenVad instance (simulating reset)...")
        vad2 = TenVad(hop_size=256, threshold=0.5)
        print(f"  vad2 id: {id(vad2)}")

        # Process with vad2
        print("  Processing with vad2...")
        try:
            result2 = vad2.process(frame_int16)
            print(f"  vad2.process() result: {result2}")
            print("\n  [PASS] Raw ten_vad test passed")
            return True
        except Exception as e:
            print(f"\n  [FAIL] vad2.process() failed: {e}")
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"  [SKIP] ten_vad not installed: {e}")
        return None
    except OSError as e:
        print(f"  [SKIP] ten_vad native library error: {e}")
        return None


def test_multiple_instances():
    """Test creating multiple TenVAD instances."""
    print("\n" + "=" * 60)
    print("TEST 5: Multiple TenVAD instances")
    print("=" * 60)

    try:
        from livecap_core.vad.backends.tenvad import TenVAD

        audio = create_test_audio(0.1)
        frame = audio[:256]

        instances = []
        for i in range(3):
            print(f"\n  Creating instance {i+1}...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                vad = TenVAD(hop_size=256, threshold=0.5)
            print(f"    _vad id: {id(vad._vad)}")
            instances.append(vad)

            # Process with each instance
            try:
                prob = vad.process(frame)
                print(f"    Process result: prob={prob:.4f}")
            except Exception as e:
                print(f"    Process FAILED: {e}")
                return False

        # Process again with all instances
        print("\n  Processing again with all instances:")
        for i, vad in enumerate(instances):
            try:
                prob = vad.process(frame)
                print(f"    Instance {i+1}: prob={prob:.4f}")
            except Exception as e:
                print(f"    Instance {i+1} FAILED: {e}")
                return False

        print("\n  [PASS] Multiple instances test passed")
        return True

    except ImportError as e:
        print(f"  [SKIP] TenVAD not installed: {e}")
        return None
    except OSError as e:
        print(f"  [SKIP] TenVAD native library error: {e}")
        return None


def main():
    """Run all TenVAD debug tests."""
    print("TenVAD Debug Script")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")

    try:
        import ten_vad

        print(f"ten_vad: {ten_vad.__version__ if hasattr(ten_vad, '__version__') else 'unknown'}")
    except ImportError:
        print("ten_vad: NOT INSTALLED")

    results = {}
    results["basic"] = test_tenvad_basic()
    results["reset"] = test_tenvad_reset()
    results["process_after_reset"] = test_tenvad_process_after_reset()
    results["raw_ten_vad"] = test_ten_vad_raw()
    results["multiple_instances"] = test_multiple_instances()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name}: {status}")

    # Determine exit code
    failures = [r for r in results.values() if r is False]
    if failures:
        print(f"\n{len(failures)} test(s) FAILED")
        sys.exit(1)
    else:
        print("\nAll tests passed or skipped")
        sys.exit(0)


if __name__ == "__main__":
    main()
