# Data Model

The transcript JSON is the central interchange format.

```json
{
  "schema_version": 1,
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
      "language": "ja",
      "speaker_confidence": 0.82,
      "asr_confidence": 0.91,
      "notes": [],
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

## Field Notes

- `schema_version` is the transcript format version. It defaults to `1` and is filled in
  automatically when an older transcript without the field is loaded.
- `speaker` is the diarization cluster name for the current episode.
- `character` is the project-level character identity, set manually or by voiceprint matching.
- `text_ja` is the source (Japanese) transcript after ASR and alignment. It is exposed in code
  as `TranscriptSegment.source_text`.
- `text_zh` is the target-language translation after glossary and style passes. It is exposed in
  code as `TranscriptSegment.translated_text` and is `null` until translation runs.
- `language` is the source language of the segment text (`ja` by default).
- `speaker_confidence` is confidence in speaker or character attribution.
- `asr_confidence` is confidence in the Japanese text.
- `notes` can track warnings such as overlapping speech, uncertain translation, or noisy audio.
  For example a line that was not matched by any translation rule is tagged `untranslated`.
- `metadata` is a free-form per-segment map; the translation step records details such as the
  provider name and target language here.

## Backward compatibility

Old `transcript.raw.json` and `transcript.speaker.json` files written before
`schema_version`, per-segment `language`, and per-segment `metadata` existed still load
correctly: missing fields fall back to their defaults (`schema_version = 1`,
`language = "ja"`, `metadata = {}`, `text_zh = null`). A round-trip
(object -> JSON -> object) preserves every field.

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
