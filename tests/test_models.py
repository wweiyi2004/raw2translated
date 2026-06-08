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


if __name__ == "__main__":
    unittest.main()

