import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from raw2translated.alignment import (
    AlignmentProviderError,
    WhisperXAlignmentProvider,
    _apply_aligned_timing,
    build_alignment_provider,
)
from raw2translated.models import TranscriptSegment
from raw2translated.pipeline import ProcessOptions, process_episode


class BuildProviderTests(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(build_alignment_provider("none"))

    def test_whisperx_returns_provider(self) -> None:
        self.assertIsInstance(build_alignment_provider("whisperx"), WhisperXAlignmentProvider)

    def test_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_alignment_provider("aeneas")


class WhisperXMissingDepTests(unittest.TestCase):
    def test_align_without_whisperx_raises_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "a.wav"
            audio.write_bytes(b"x")
            provider = WhisperXAlignmentProvider()
            segments = [TranscriptSegment(start=0.0, end=1.0, text_ja="はい")]
            with self.assertRaises(AlignmentProviderError):
                provider.align(segments, audio)

    def test_align_empty_segments_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "a.wav"
            audio.write_bytes(b"x")
            # No segments -> returns immediately without importing whisperx.
            self.assertEqual(WhisperXAlignmentProvider().align([], audio), [])


class ApplyAlignedTimingTests(unittest.TestCase):
    def test_refines_timing_and_records_words(self) -> None:
        segments = [TranscriptSegment(start=0.0, end=2.0, text_ja="はい")]
        aligned = [
            {
                "start": 0.4,
                "end": 1.6,
                "words": [{"word": "はい", "start": 0.4, "end": 1.6}],
            }
        ]
        _apply_aligned_timing(segments, aligned)
        self.assertEqual(segments[0].start, 0.4)
        self.assertEqual(segments[0].end, 1.6)
        self.assertEqual(segments[0].metadata["words"][0]["word"], "はい")

    def test_missing_times_keep_original(self) -> None:
        segments = [TranscriptSegment(start=0.0, end=2.0, text_ja="はい")]
        _apply_aligned_timing(segments, [{"start": None, "end": None}])
        self.assertEqual(segments[0].start, 0.0)
        self.assertEqual(segments[0].end, 2.0)


class PipelineAlignmentTests(unittest.TestCase):
    def test_process_applies_alignment_and_updates_manifest(self) -> None:
        class FakeAsr:
            def transcribe(self, audio_path, *, language="ja"):
                return [TranscriptSegment(start=0.0, end=2.0, text_ja="はい")]

        class FakeAligner:
            def align(self, segments, audio_path, *, language="ja"):
                for s in segments:
                    s.start = 0.5
                    s.end = 1.5
                return segments

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "ep.mkv"
            input_path.write_bytes(b"x")
            audio = root / "output" / "media" / "analysis.wav"

            with patch("raw2translated.pipeline.probe_media", return_value={}), patch(
                "raw2translated.pipeline.extract_analysis_audio", return_value=audio
            ), patch("raw2translated.pipeline.build_asr_provider", return_value=FakeAsr()), patch(
                "raw2translated.pipeline.build_alignment_provider", return_value=FakeAligner()
            ):
                result = process_episode(
                    input_path,
                    ProcessOptions(
                        output_dir=root / "output",
                        asr_provider="faster-whisper",
                        alignment_provider="whisperx",
                    ),
                )

            raw = json.loads(result.raw_transcript_path.read_text(encoding="utf-8"))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(raw["segments"][0]["start"], 0.5)
        self.assertEqual(raw["segments"][0]["end"], 1.5)
        self.assertEqual(manifest["pipeline"]["alignment"], "completed")
        self.assertEqual(manifest["pipeline"]["alignment_provider"], "whisperx")

    def test_dry_run_marks_alignment_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "ep.mkv"
            input_path.write_bytes(b"x")
            result = process_episode(
                input_path,
                ProcessOptions(
                    output_dir=root / "output",
                    dry_run=True,
                    asr_provider="faster-whisper",
                    alignment_provider="whisperx",
                ),
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["pipeline"]["alignment"], "dry_run")


if __name__ == "__main__":
    unittest.main()
