import json
import tempfile
import unittest
from pathlib import Path

from raw2translated.models import EpisodeTranscript, TranscriptSegment


class ModelTests(unittest.TestCase):
    def test_segment_rejects_negative_duration(self) -> None:
        with self.assertRaises(ValueError):
            TranscriptSegment(start=2.0, end=1.0)

    def test_episode_json_roundtrip(self) -> None:
        transcript = EpisodeTranscript(
            media_path="episode.mkv",
            segments=[
                TranscriptSegment(
                    start=1.2,
                    end=3.4,
                    speaker="SPEAKER_00",
                    character="女主A",
                    text_ja="何してるの？",
                    text_zh="你在干什么？",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcript.json"
            transcript.write_json(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            loaded = EpisodeTranscript.from_dict(data)

        self.assertEqual(loaded.segments[0].display_speaker, "女主A")
        self.assertEqual(loaded.segments[0].text_zh, "你在干什么？")

    def test_full_segment_roundtrip_preserves_fields(self) -> None:
        segment = TranscriptSegment(
            start=1.2,
            end=3.4,
            speaker="SPEAKER_00",
            character="女主A",
            text_ja="何してるの？",
            text_zh="你在干什么？",
            language="ja",
            speaker_confidence=0.82,
            asr_confidence=0.91,
            notes=["overlap"],
            metadata={"source": "memory"},
        )
        transcript = EpisodeTranscript(media_path="episode.mkv", segments=[segment])

        restored = EpisodeTranscript.from_dict(transcript.to_dict())
        restored_segment = restored.segments[0]

        self.assertEqual(restored.schema_version, transcript.schema_version)
        self.assertEqual(restored_segment.language, "ja")
        self.assertEqual(restored_segment.metadata, {"source": "memory"})
        self.assertEqual(restored_segment.notes, ["overlap"])
        self.assertEqual(restored_segment.asr_confidence, 0.91)
        self.assertTrue(restored_segment.is_translated)
        self.assertEqual(restored_segment.source_text, "何してるの？")
        self.assertEqual(restored_segment.translated_text, "你在干什么？")

    def test_legacy_json_without_new_fields_still_loads(self) -> None:
        # Mimics an old transcript.raw.json written before language/metadata/schema_version
        # and before any translation existed.
        legacy = {
            "media_path": "episode.mkv",
            "language": "ja",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "speaker": "SPEAKER_00",
                    "text_ja": "おはよう",
                }
            ],
            "metadata": {},
        }

        transcript = EpisodeTranscript.from_dict(legacy)
        segment = transcript.segments[0]

        self.assertEqual(transcript.schema_version, 1)
        self.assertEqual(segment.language, "ja")
        self.assertEqual(segment.metadata, {})
        self.assertIsNone(segment.text_zh)
        self.assertFalse(segment.is_translated)
        self.assertEqual(segment.source_text, "おはよう")


if __name__ == "__main__":
    unittest.main()

