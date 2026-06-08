# Data Model

The transcript JSON is the central interchange format.

```json
{
  "media_path": "episode.mkv",
  "language": "ja",
  "segments": [
    {
      "start": 72.31,
      "end": 74.08,
      "speaker": "SPEAKER_00",
      "character": "女主A",
      "text_ja": "何してるの？",
      "text_zh": "你在干什么？",
      "speaker_confidence": 0.82,
      "asr_confidence": 0.91,
      "notes": []
    }
  ],
  "metadata": {}
}
```

## Field Notes

- `speaker` is the diarization cluster name for the current episode.
- `character` is the project-level character identity, set manually or by voiceprint matching.
- `text_ja` should be the Japanese transcript after ASR and alignment.
- `text_zh` should be the final subtitle translation after glossary and style passes.
- `speaker_confidence` is confidence in speaker or character attribution.
- `asr_confidence` is confidence in the Japanese text.
- `notes` can track warnings such as overlapping speech, uncertain translation, or noisy audio.

## Local Learning

Project-level learning should produce separate artifacts:

```text
characters.yaml
glossary.yaml
translation_memory.sqlite
voiceprints/
```

These artifacts should not be merged into a public model unless the user has explicitly licensed the data for that purpose.

## Diarization Output

When speaker diarization is enabled, `diarization.json` stores raw speaker turns:

```json
{
  "turns": [
    {
      "start": 12.31,
      "end": 14.08,
      "speaker": "SPEAKER_00",
      "confidence": null
    }
  ]
}
```

`transcript.raw.json` preserves the ASR-only transcript. `transcript.speaker.json` applies the dominant speaker turn to each ASR segment by time overlap.

## Audio Artifacts

The pipeline may produce multiple audio files with different purposes:

```text
media/analysis.wav          # 16k mono, default ASR and diarization input
media/separation_input.wav  # 44.1k stereo, temporary input for Demucs
media/dialogue.wav          # 16k mono, Demucs vocals stem converted for ASR
```

When preprocessing is enabled, ASR uses `dialogue.wav`; diarization still uses `analysis.wav`.
