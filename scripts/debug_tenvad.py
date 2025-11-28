#!/usr/bin/env python
"""TenVAD Debug Script

Direct testing of ten_vad.TenVad (raw library) to investigate:
- Instance creation
- Multiple instance behavior (simulating reset)
- process() after creating new instance

Run: python scripts/debug_tenvad.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np


def create_test_audio_int16(duration_s: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    """Create test audio (sine wave) as int16."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16


def test_ten_vad_basic():
    """Test basic ten_vad.TenVad operations."""
    print("=" * 60)
    print("TEST 1: Basic ten_vad.TenVad creation and process")
    print("=" * 60)

    try:
        from ten_vad import TenVad

        print("  Creating TenVad instance...")
        vad = TenVad(hop_size=256, threshold=0.5)
        print(f"  Created: {vad}")
        print(f"  Type: {type(vad)}")

        # Check attributes
        print("\n  Instance attributes:")
        for attr in dir(vad):
            if not attr.startswith("__"):
                try:
                    val = getattr(vad, attr)
                    if not callable(val):
                        print(f"    {attr} = {val}")
                except Exception:
                    pass

        # Test process with valid audio
        audio = create_test_audio_int16(0.1)  # 100ms
        print(f"\n  Test audio: shape={audio.shape}, dtype={audio.dtype}")
        print(f"  Audio range: [{audio.min()}, {audio.max()}]")

        # Process single frame
        frame = audio[:256]
        print(f"\n  Processing frame: shape={frame.shape}")

        result = vad.process(frame)
        print(f"  Result: {result}")
        print(f"  Result type: {type(result)}")

        print("\n  [PASS] Basic test passed")
        return True

    except ImportError as e:
        print(f"  [SKIP] ten_vad not installed: {e}")
        return None
    except OSError as e:
        print(f"  [SKIP] ten_vad native library error: {e}")
        return None
    except Exception as e:
        print(f"  [FAIL] Unexpected error: {e}")
        traceback.print_exc()
        return False


def test_ten_vad_multiple_instances():
    """Test creating multiple TenVad instances (simulates reset behavior)."""
    print("\n" + "=" * 60)
    print("TEST 2: Multiple TenVad instances (simulates reset)")
    print("=" * 60)

    try:
        from ten_vad import TenVad

        audio = create_test_audio_int16(0.1)
        frame = audio[:256]

        # Create first instance
        print("  Creating first TenVad instance...")
        vad1 = TenVad(hop_size=256, threshold=0.5)
        print(f"  vad1 id: {id(vad1)}")

        # Process with first instance
        result1 = vad1.process(frame)
        print(f"  vad1.process() result: {result1}")

        # Create second instance (simulating what reset() does)
        print("\n  Creating second TenVad instance (like reset)...")
        vad2 = TenVad(hop_size=256, threshold=0.5)
        print(f"  vad2 id: {id(vad2)}")

        # Process with second instance
        print("  Processing with vad2...")
        try:
            result2 = vad2.process(frame)
            print(f"  vad2.process() result: {result2}")
        except Exception as e:
            print(f"  [ERROR] vad2.process() failed: {e}")
            traceback.print_exc()
            return False

        # Create third instance
        print("\n  Creating third TenVad instance...")
        vad3 = TenVad(hop_size=256, threshold=0.5)
        print(f"  vad3 id: {id(vad3)}")

        # Process with third instance
        print("  Processing with vad3...")
        try:
            result3 = vad3.process(frame)
            print(f"  vad3.process() result: {result3}")
        except Exception as e:
            print(f"  [ERROR] vad3.process() failed: {e}")
            traceback.print_exc()
            return False

        print("\n  [PASS] Multiple instances test passed")
        return True

    except ImportError as e:
        print(f"  [SKIP] ten_vad not installed: {e}")
        return None
    except OSError as e:
        print(f"  [SKIP] ten_vad native library error: {e}")
        return None


def test_ten_vad_sequential_process():
    """Test sequential processing with single instance."""
    print("\n" + "=" * 60)
    print("TEST 3: Sequential process() calls on single instance")
    print("=" * 60)

    try:
        from ten_vad import TenVad

        audio = create_test_audio_int16(0.5)  # 500ms = 8000 samples at 16kHz

        print("  Creating TenVad instance...")
        vad = TenVad(hop_size=256, threshold=0.5)

        # Process multiple frames
        n_frames = len(audio) // 256
        print(f"\n  Processing {n_frames} frames sequentially:")

        for i in range(min(n_frames, 10)):  # Process up to 10 frames
            frame = audio[i * 256 : (i + 1) * 256]
            try:
                result = vad.process(frame)
                print(f"    Frame {i}: result={result}")
            except Exception as e:
                print(f"    Frame {i}: ERROR - {e}")
                traceback.print_exc()
                return False

        print("\n  [PASS] Sequential process test passed")
        return True

    except ImportError as e:
        print(f"  [SKIP] ten_vad not installed: {e}")
        return None
    except OSError as e:
        print(f"  [SKIP] ten_vad native library error: {e}")
        return None


def test_ten_vad_instance_replacement():
    """Test replacing instance while first is still in scope."""
    print("\n" + "=" * 60)
    print("TEST 4: Instance replacement (first still in scope)")
    print("=" * 60)

    try:
        from ten_vad import TenVad

        audio = create_test_audio_int16(0.1)
        frame = audio[:256]

        instances = []

        # Create multiple instances, keeping all in scope
        for i in range(3):
            print(f"\n  Creating instance {i+1}...")
            vad = TenVad(hop_size=256, threshold=0.5)
            instances.append(vad)
            print(f"    id: {id(vad)}")

            # Process immediately after creation
            try:
                result = vad.process(frame)
                print(f"    process() result: {result}")
            except Exception as e:
                print(f"    process() ERROR: {e}")
                traceback.print_exc()
                return False

        # Process again with all instances
        print("\n  Processing again with all instances:")
        for i, vad in enumerate(instances):
            try:
                result = vad.process(frame)
                print(f"    Instance {i+1}: result={result}")
            except Exception as e:
                print(f"    Instance {i+1}: ERROR - {e}")
                traceback.print_exc()
                return False

        print("\n  [PASS] Instance replacement test passed")
        return True

    except ImportError as e:
        print(f"  [SKIP] ten_vad not installed: {e}")
        return None
    except OSError as e:
        print(f"  [SKIP] ten_vad native library error: {e}")
        return None


def test_ten_vad_del_and_recreate():
    """Test deleting instance and creating new one."""
    print("\n" + "=" * 60)
    print("TEST 5: Delete and recreate (closest to reset behavior)")
    print("=" * 60)

    try:
        from ten_vad import TenVad
        import gc

        audio = create_test_audio_int16(0.1)
        frame = audio[:256]

        # Create and use first instance
        print("  Creating first instance...")
        vad = TenVad(hop_size=256, threshold=0.5)
        print(f"  vad id: {id(vad)}")

        result1 = vad.process(frame)
        print(f"  process() result: {result1}")

        # Delete first instance
        print("\n  Deleting instance (del vad)...")
        del vad
        gc.collect()
        print("  Garbage collection done")

        # Create second instance
        print("\n  Creating second instance...")
        vad = TenVad(hop_size=256, threshold=0.5)
        print(f"  vad id: {id(vad)}")

        # Process with second instance
        print("  Processing with new instance...")
        try:
            result2 = vad.process(frame)
            print(f"  process() result: {result2}")
        except Exception as e:
            print(f"  process() ERROR: {e}")
            traceback.print_exc()
            return False

        print("\n  [PASS] Delete and recreate test passed")
        return True

    except ImportError as e:
        print(f"  [SKIP] ten_vad not installed: {e}")
        return None
    except OSError as e:
        print(f"  [SKIP] ten_vad native library error: {e}")
        return None


def main():
    """Run all TenVAD debug tests."""
    print("TenVAD Debug Script (Direct ten_vad testing)")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")

    try:
        import ten_vad

        print(f"ten_vad: {getattr(ten_vad, '__version__', 'unknown version')}")
        print(f"ten_vad location: {ten_vad.__file__}")
    except ImportError:
        print("ten_vad: NOT INSTALLED")

    results = {}
    results["basic"] = test_ten_vad_basic()
    results["multiple_instances"] = test_ten_vad_multiple_instances()
    results["sequential_process"] = test_ten_vad_sequential_process()
    results["instance_replacement"] = test_ten_vad_instance_replacement()
    results["del_and_recreate"] = test_ten_vad_del_and_recreate()

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
