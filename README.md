# raw2translated

`raw2translated` is a local-first pipeline for turning unsubtitled Japanese anime episodes into structured transcripts and subtitle files.

The first milestone is intentionally narrow:

- extract and inspect media with `ffmpeg` / `ffprobe`
- produce a stable transcript JSON format
- export `ASS` / `SRT` subtitles
- leave ASR, diarization, character voiceprints, and translation behind replaceable interfaces

It does not train a public model on user media. Project-level learning should stay local: character voiceprints, glossary entries, translation memory, and user corrections belong to the user's project directory.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
raw2translated --help
```

Probe a video:

```powershell
raw2translated probe .\input.mkv
```

Install only the ASR dependency:

```powershell
pip install -e ".[asr]"
```

Create a project output folder and extract analysis audio:

```powershell
raw2translated process .\input.mkv --out .\output
```

Process a video with no subtitles by transcribing the Japanese audio:

```powershell
raw2translated process .\input.mkv --out .\output --asr faster-whisper --lang ja
```

Optionally enhance dialogue for ASR with Demucs:

```powershell
pip install -e ".[preprocess]"
raw2translated process .\input.mkv --out .\output --preprocess demucs --asr faster-whisper --lang ja
```

This writes `media/dialogue.wav` for ASR. Speaker diarization still uses `media/analysis.wav` by default so voice identity is less distorted.

Install diarization support:

```powershell
pip install -e ".[diarization]"
```

Run ASR and assign lines to speaker clusters:

```powershell
$env:HF_TOKEN="hf_..."
raw2translated process .\input.mkv --out .\output --asr faster-whisper --diarization pyannote --lang ja
```

If you use a pyannote.audio 3.x environment, switch the model:

```powershell
raw2translated process .\input.mkv --out .\output --asr faster-whisper --diarization pyannote --diarization-model pyannote/speaker-diarization-3.1
```

For a faster CPU smoke test, use a smaller model:

```powershell
raw2translated process .\input.mkv --out .\output --asr faster-whisper --asr-model small --asr-device cpu --asr-compute-type int8
```

Export subtitles from a transcript JSON:

```powershell
raw2translated export-subtitle .\output\transcript.speaker.json --format ass --out .\output\episode.ass
```

## MVP Pipeline

```text
video input
 -> ffprobe metadata
 -> ffmpeg analysis audio extraction
 -> ASR provider
 -> forced alignment provider
 -> speaker diarization provider
 -> character voiceprint matching
 -> translation provider
 -> ASS/SRT export
 -> MKV muxing
```

The current code implements the stable shell around that pipeline. `faster-whisper` ASR and `pyannote.audio` diarization are wired in as optional model-backed providers. Character matching, alignment, and translation are still pending.

## Output Layout

```text
output/
  manifest.json
  media/
    analysis.wav
    separation_input.wav
    dialogue.wav
  transcript.raw.json
  transcript.speaker.json
  subtitles/
    episode.ass
    episode.srt
```

## Development

Run the standard-library test suite:

```powershell
python -m unittest discover -s tests
```

Use `docs/PROJECT_PLAN.md` for the staged roadmap and `docs/DATA_MODEL.md` for transcript schema notes.
