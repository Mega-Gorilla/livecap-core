# Test assets

This directory holds lightweight audio fixtures plus optional source corpora used to generate them.

## Layout
- Tracked smoke clips live in `tests/assets/audio/` (short WAV, 16 kHz mono).
- Source corpora are **not** tracked and should be placed under `tests/assets/source/`:
  - `tests/assets/source/jsut/jsut_ver1.1/` — JSUT v1.1 (https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
  - `tests/assets/source/librispeech/test-clean/` — LibriSpeech test-clean (https://www.openslr.org/12)

## Naming (tracked clips)
- `<corpus>_<subset>_<utterance>_<lang>.wav` (and same stem for `.txt` transcripts).
- Examples: `librispeech_test-clean_1089-134686-0001_en.wav`, `jsut_basic5000_0001_ja.wav`.

## Transcript format
- UTF-8 plain text, single line, trailing newline, stem matches the WAV (`<corpus>_<subset>_<utterance>_<lang>.txt`).
- Content mirrors the source corpus text as-is (LibriSpeech: uppercase, original punctuation; JSUT: original Japanese punctuation).
- No extra quoting or metadata in the `.txt`. If metadata is needed later, add a parallel `<stem>.meta.json` instead of changing the `.txt` contents.

## Generation & normalization
- Target: 3–8 seconds, 16 kHz mono, peak around -1 dBFS, keep ~0.2–0.3 s of leading/trailing silence.
- LibriSpeech arrives as FLAC; convert and trim in one step, e.g.\
  `ffmpeg -i tests/assets/source/librispeech/test-clean/LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac -ss 0 -t 4 -ac 1 -ar 16000 -af "volume=-1dB" tests/assets/audio/librispeech_test-clean_1089-134686-0001_en.wav`
- JSUT is WAV; trim/normalize directly, e.g.\
  `ffmpeg -i tests/assets/source/jsut/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav -ss 0 -t 6 -ac 1 -ar 16000 -af "volume=-1dB" tests/assets/audio/jsut_basic5000_0001_ja.wav`

## Optional environment variables for benchmarks
- `LIVECAP_LIBRISPEECH_DIR` and `LIVECAP_JSUT_DIR` can point to larger local corpora for benchmark-style tests; tests should skip gracefully when these are absent.

## Licensing and attribution
- LibriSpeech test-clean
  - License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
  - Source: https://www.openslr.org/12
  - Commercial use: permitted
- JSUT v1.1
  - License: research / non-commercial only
  - Source: https://sites.google.com/site/shinnosuketakamichi/publication/jsut
  - Commercial use: requires separate permission
- Derived short clips in `tests/assets/audio/` inherit the licenses of their source corpora. Use only for development/CI; provide alternative fixtures or obtain permission for commercial deployments.
