import json
import tempfile
import unittest
from pathlib import Path

from raw2translated.cli import build_parser, main
from raw2translated.models import EpisodeTranscript, TranscriptSegment

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write_transcript(path: Path) -> None:
    transcript = EpisodeTranscript(
        media_path="episode.mkv",
        segments=[
            TranscriptSegment(start=0.0, end=1.0, speaker="SPEAKER_00", text_ja="おはよう"),
            TranscriptSegment(start=1.0, end=2.0, speaker="SPEAKER_01", text_ja="未収録のセリフ"),
        ],
    )
    transcript.write_json(path)


class ParserTests(unittest.TestCase):
    def test_translate_subcommand_parses(self) -> None:
        args = build_parser().parse_args(
            ["translate", "in.json", "--out", "out.json", "--provider", "memory", "--memory", "m.json"]
        )
        self.assertEqual(args.command, "translate")
        self.assertEqual(args.provider, "memory")
        self.assertEqual(args.memory, Path("m.json"))

    def test_process_translate_defaults_to_none(self) -> None:
        args = build_parser().parse_args(["process", "in.mkv"])
        self.assertEqual(args.translate_provider, "none")
        self.assertEqual(args.target_lang, "zh-CN")


class TranslateCommandTests(unittest.TestCase):
    def test_translate_with_memory_writes_translated_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "transcript.speaker.json"
            out = root / "transcript.translated.json"
            _write_transcript(src)

            code = main(
                [
                    "translate",
                    str(src),
                    "--out",
                    str(out),
                    "--provider",
                    "memory",
                    "--memory",
                    str(CONFIGS / "translation_memory.example.json"),
                ]
            )
            self.assertEqual(code, 0)
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(data["segments"][0]["text_zh"], "早上好")
        # Unmatched line keeps its source and is flagged, never dropped.
        self.assertIsNone(data["segments"][1]["text_zh"])
        self.assertIn("untranslated", data["segments"][1]["notes"])

    def test_translate_missing_memory_path_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "transcript.json"
            _write_transcript(src)
            code = main(["translate", str(src), "--out", str(root / "o.json"), "--provider", "memory"])
        self.assertEqual(code, 2)

    def test_translate_missing_input_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "o.json"
            code = main(["translate", str(Path(tmp) / "nope.json"), "--out", str(out), "--provider", "none"])
        self.assertEqual(code, 2)


class ProcessTranslateTests(unittest.TestCase):
    def test_process_dry_run_without_translation_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "episode.mkv"
            input_path.write_bytes(b"not a real media file")
            code = main(["process", str(input_path), "--out", str(root / "output"), "--dry-run"])
            self.assertEqual(code, 0)
            manifest = json.loads((root / "output" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline"]["translation"], "skipped")
            self.assertFalse((root / "output" / "transcript.translated.json").exists())

    def test_process_dry_run_with_memory_records_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "episode.mkv"
            input_path.write_bytes(b"not a real media file")
            code = main(
                [
                    "process",
                    str(input_path),
                    "--out",
                    str(root / "output"),
                    "--dry-run",
                    "--translate",
                    "memory",
                    "--translation-memory",
                    str(CONFIGS / "translation_memory.example.json"),
                ]
            )
            self.assertEqual(code, 0)
            manifest = json.loads((root / "output" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline"]["translation_provider"], "memory")
            self.assertEqual(manifest["pipeline"]["translation_target_lang"], "zh-CN")
            self.assertTrue((root / "output" / "transcript.translated.json").exists())


if __name__ == "__main__":
    unittest.main()
