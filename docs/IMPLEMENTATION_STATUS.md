# Implementation Status

This document tracks what is implemented, what is still missing for the MVP goal,
the plan for the current round of work, and the things that are explicitly out of scope.

Last updated: 2026-06-08.

## Goal of this round

Complete the local-first loop:

```text
raw video/audio -> transcript -> translated transcript -> ASS/SRT subtitles
```

without a large rewrite, building on the existing `cli.py`, `pipeline.py`,
`models.py`, `translation.py`, and `subtitles.py`.

## Implemented before this round

- `probe` / `process` / `export-subtitle` / `mux` CLI commands.
- ffmpeg/ffprobe wrappers for media inspection and analysis-audio extraction.
- `faster-whisper` ASR provider (optional dependency, behind an interface).
- `pyannote.audio` diarization provider (optional dependency, behind an interface).
- Demucs vocal preprocessing (optional dependency, behind an interface).
- `EpisodeTranscript` / `TranscriptSegment` data model with JSON read/write.
- `transcript.raw.json` (ASR only) and `transcript.speaker.json` (speaker-labelled).
- ASS / SRT export, including a `--bilingual` flag.
- Unit tests for models, subtitles, pipeline, asr, diarization, preprocess (mock/fake based).

## Implemented in this round

- **Data model**: `language`, `schema_version`, and `metadata` per segment; helpers for
  source/translated text; round-trip and backward-compatibility tests.
- **Translation providers**: `TranslationProvider` protocol with `NullTranslationProvider`,
  `TranslationMemoryProvider` (local JSON memory), and `GlossaryTranslationProvider`
  (local glossary substitution), plus a `build_translation_provider` factory. Untranslated
  lines keep their source text and are marked, never dropped.
- **CLI**: a new `translate` subcommand and optional `process` translation flags
  (`--translate`, `--translation-memory`, `--glossary`, `--target-lang`). The manifest records
  the translation provider, target language, and output path.
- **Subtitles**: `--text-mode original|translated|bilingual` for both ASS and SRT, with
  `--bilingual` kept as a compatibility alias. Untranslated lines never render `None`.
- **Engineering**: ruff configuration, a GitHub Actions workflow that installs only the light
  `dev` extra, and refreshed README / PROJECT_PLAN / DATA_MODEL docs.
- **Desktop GUI** (`gui.py`): a Tkinter app with Process / Editor / Export tabs, launched via
  `raw2translated gui`. All logic lives in a Tk-free `GuiController` that is unit-tested
  headless; Tkinter is imported lazily so importing the module never needs a display.
- **Editor enhancements**: filter the table to only flagged lines (untranslated / has notes /
  low confidence), per-line `ffplay` preview, and mux a subtitle into a video — all backed by
  headless-testable `GuiController` methods.
- **Batch processing**: `raw2translated batch <dir>` (and `pipeline.process_batch`) runs the
  pipeline over every media file in a directory into per-file output subdirectories, recording
  per-file failures instead of aborting the whole batch.

- **Real translation backend**: `OpenAITranslationProvider`, an OpenAI-compatible
  chat-completions provider built on stdlib `urllib` (no new dependency). It is never the
  default, reads the API key from an env var (never CLI args/manifest), accepts any
  `--api-base` for local LLM gateways, and injects the glossary into the system prompt. The
  HTTP transport is injectable so it is unit-tested with no network.

- **Character naming**: `characters.py` binds diarization clusters (`SPEAKER_00`) to character
  names via a local JSON map, populating the `character` field. Exposed as the
  `assign-characters` command and a `process` / `batch` `--characters` option.

## Still missing / future work

- Forced alignment (WhisperX) for tighter subtitle timing.
- Character *voiceprints* (auto-binding clusters across episodes); the manual character map
  is implemented, automatic voiceprint matching is not.
- `openai` translation wired into `process`/`batch` (currently the local providers only;
  use the standalone `translate` command for the LLM backend).
- A richer review/editing UI (waveform, word-level timing).

## Out of scope (explicitly not done)

- Bundling real anime clips, copyrighted media, or model weights.
- Uploading user media to any cloud service by default.
- Training public models from user footage.
- Automatic Chinese dubbing / voice cloning.

## Acceptance criteria for this round

- `python -m unittest discover -s tests` passes.
- `python -m ruff check src tests` passes.
- Translation loop exists: transcript JSON -> translated JSON.
- Subtitle loop exists: translated JSON -> ASS/SRT.
- README and docs updated.
- Work committed in stages and pushed to `improve-translation-mvp`.

## Verification notes (real models)

Unit tests use mock/fake providers and never touch the network, GPU, ffmpeg, or
Hugging Face. Real verification of ASR/diarization requires installing the optional
extras (`.[asr]`, `.[diarization]`, `.[preprocess]`), an `HF_TOKEN` for gated pyannote
models, ffmpeg on PATH, and ideally a CUDA GPU. Those steps are environment-dependent
and are not exercised by CI.
