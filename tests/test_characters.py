import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from raw2translated.characters import (
    CharacterMapError,
    apply_character_map,
    load_character_map,
)
from raw2translated.cli import main
from raw2translated.diarization import SpeakerTurn
from raw2translated.models import EpisodeTranscript, TranscriptSegment
from raw2translated.pipeline import ProcessOptions, process_episode

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=0.0, end=1.0, speaker="SPEAKER_00", text_ja="a"),
        TranscriptSegment(start=1.0, end=2.0, speaker="SPEAKER_01", text_ja="b"),
        TranscriptSegment(start=2.0, end=3.0, speaker="SPEAKER_99", text_ja="c"),
    ]


class ApplyMapTests(unittest.TestCase):
    def test_apply_sets_character_and_counts(self) -> None:
        segments = _segments()
        assigned = apply_character_map(segments, {"SPEAKER_00": "女主A", "SPEAKER_01": "男主B"})
        self.assertEqual(assigned, 2)
        self.assertEqual(segments[0].character, "女主A")
        self.assertEqual(segments[0].display_speaker, "女主A")
        # Unmapped speaker is left untouched, not blanked.
        self.assertIsNone(segments[2].character)
        self.assertEqual(segments[2].display_speaker, "SPEAKER_99")


class LoadMapTests(unittest.TestCase):
    def test_load_flat_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            path.write_text(json.dumps({"SPEAKER_00": "女主A"}), encoding="utf-8")
            mapping = load_character_map(path)
        self.assertEqual(mapping["SPEAKER_00"], "女主A")

    def test_load_example_config_list_shape(self) -> None:
        mapping = load_character_map(CONFIGS / "characters.example.json")
        self.assertEqual(mapping["SPEAKER_00"], "女主A")

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(CharacterMapError):
            load_character_map(Path("nope.json"))

    def test_invalid_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{nope", encoding="utf-8")
            with self.assertRaises(CharacterMapError):
                load_character_map(path)


class AssignCharactersCliTests(unittest.TestCase):
    def test_cli_assign_characters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "transcript.json"
            out = root / "out.json"
            EpisodeTranscript(segments=_segments()).write_json(src)
            code = main(
                [
                    "assign-characters",
                    str(src),
                    "--out",
                    str(out),
                    "--map",
                    str(CONFIGS / "characters.example.json"),
                ]
            )
            self.assertEqual(code, 0)
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["segments"][0]["character"], "女主A")
        self.assertEqual(data["segments"][1]["character"], "男主B")

    def test_cli_assign_characters_bad_map_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "transcript.json"
            EpisodeTranscript(segments=_segments()).write_json(src)
            code = main(
                ["assign-characters", str(src), "--out", str(root / "o.json"), "--map", str(root / "nope.json")]
            )
        self.assertEqual(code, 2)


class ProcessCharacterMapTests(unittest.TestCase):
    def test_process_applies_character_map(self) -> None:
        class FakeAsr:
            def transcribe(self, audio_path, *, language="ja"):
                return [TranscriptSegment(start=0.0, end=2.0, speaker="UNKNOWN", text_ja="やあ")]

        class FakeDiar:
            def diarize(self, audio_path):
                return [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "episode.mkv"
            input_path.write_bytes(b"x")
            char_map = root / "map.json"
            char_map.write_text(json.dumps({"SPEAKER_00": "女主A"}), encoding="utf-8")
            audio = root / "output" / "media" / "analysis.wav"

            with patch("raw2translated.pipeline.probe_media", return_value={}), patch(
                "raw2translated.pipeline.extract_analysis_audio", return_value=audio
            ), patch("raw2translated.pipeline.build_asr_provider", return_value=FakeAsr()), patch(
                "raw2translated.pipeline.build_diarization_provider", return_value=FakeDiar()
            ):
                result = process_episode(
                    input_path,
                    ProcessOptions(
                        output_dir=root / "output",
                        asr_provider="faster-whisper",
                        diarization_provider="pyannote",
                        character_map=char_map,
                    ),
                )

            speaker = json.loads(result.speaker_transcript_path.read_text(encoding="utf-8"))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(speaker["segments"][0]["character"], "女主A")
        self.assertEqual(manifest["pipeline"]["characters"], "assigned:1")


if __name__ == "__main__":
    unittest.main()
