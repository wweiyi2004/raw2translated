# Project Plan

## Status Snapshot (2026-06-08)

Done:

- CLI shell: `probe` / `process` / `translate` / `export-subtitle` / `mux`.
- Transcript data model with `language`, `metadata`, and `schema_version`, backward compatible.
- `faster-whisper` ASR and `pyannote.audio` diarization behind optional-dependency providers.
- Local-first translation loop: `memory` and `glossary` providers, `transcript.translated.json`.
- Subtitle export with `original` / `translated` / `bilingual` text modes.
- ruff config and a light GitHub Actions CI (`dev` extra only).
- Tkinter desktop GUI (`raw2translated gui`) with Process / Editor / Export tabs — a first
  cut of the Phase 7 review UI.

Next:

- Forced alignment (WhisperX) for tighter timing — see Phase 3.
- Character voiceprints — see Phase 5.
- A real machine-translation provider behind the existing `TranslationProvider` interface.
- Grow the GUI editor: audio/video playback per line, low-confidence filtering, mux to MKV.
- Batch processing of multiple episodes.

## Product Boundary

First build a local-first subtitle production tool:

- Japanese ASR
- speaker-separated transcript
- character confirmation and local voiceprint library
- Chinese translation with project glossary and style cards
- `ASS` / `SRT` export
- original audio preserved

Do not include these in the first version:

- public model training from user anime files
- voice actor cloning
- automatic Chinese dubbing
- cloud ingestion of full episodes

## Phase 1: CLI MVP

Command:

```powershell
raw2translated process .\episode.mkv --out .\output
```

Acceptance criteria:

- probe media metadata
- extract 16 kHz mono analysis audio
- write a stable transcript JSON file
- export placeholder `ASS` and `SRT`
- keep all model-backed steps behind provider interfaces

Status:

- implemented as project shell
- `faster-whisper` can now populate `transcript.raw.json`
- optional `Demucs` preprocessing can create `media/dialogue.wav` for ASR
- `pyannote.audio` can now populate speaker labels in `transcript.speaker.json`

## Phase 2: Japanese ASR

Planned providers:

- `faster-whisper`
- `kotoba-whisper`
- `ReazonSpeech`

Outputs:

- `transcript.raw.json`
- ASR confidence per segment where available
- low-confidence segment list

Command:

```powershell
pip install -e ".[asr]"
raw2translated process .\episode.mkv --out .\output --asr faster-whisper --lang ja
```

CPU smoke test:

```powershell
raw2translated process .\episode.mkv --out .\output --asr faster-whisper --asr-model small --asr-device cpu --asr-compute-type int8
```

Optional dialogue enhancement:

```powershell
pip install -e ".[preprocess]"
raw2translated process .\episode.mkv --out .\output --preprocess demucs --asr faster-whisper --lang ja
```

Design rule:

- ASR may use `media/dialogue.wav`.
- Speaker diarization should use `media/analysis.wav` unless a future quality test proves otherwise.

## Phase 3: Forced Alignment

Planned provider:

- `WhisperX`

Goal:

- improve subtitle start/end timing
- optionally produce word-level timing for future editing UI

## Phase 4: Speaker Diarization

Planned provider:

- `pyannote.audio`

Goal:

- produce `SPEAKER_00`, `SPEAKER_01`, etc.
- merge speaker turns into ASR segments by time overlap
- mark overlapping speech and low-confidence speaker attribution

Command:

```powershell
pip install -e ".[diarization]"
$env:HF_TOKEN="hf_..."
raw2translated process .\episode.mkv --out .\output --asr faster-whisper --diarization pyannote --lang ja
```

Current default model:

```text
pyannote/speaker-diarization-community-1
```

For legacy pyannote.audio 3.x environments:

```powershell
raw2translated process .\episode.mkv --out .\output --asr faster-whisper --diarization pyannote --diarization-model pyannote/speaker-diarization-3.1
```

## Phase 5: Character Voiceprints

Workflow:

```text
speaker cluster
 -> representative lines
 -> user binds speaker to character
 -> save voice embedding locally
 -> match future episodes against project voiceprints
```

Project artifacts:

- `characters.yaml`
- local embeddings
- confidence thresholds
- correction history

## Phase 6: Translation Style

Translation input should include:

- current Japanese segment
- previous and next context
- speaker / character
- character style card
- glossary
- translation memory
- target subtitle style

Primary goal:

- consistent names, terms, catchphrases, and relationship words
- readable subtitle length
- anime-aware but not noisy Chinese

## Phase 7: Review UI

Expected UI:

- drag in video
- timeline subtitle list
- click a line to play the matching audio/video range
- edit speaker, character, Japanese text, Chinese text
- filter low-confidence lines
- export `ASS`, `SRT`, and muxed `MKV`
