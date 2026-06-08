import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from raw2translated.diarization import SpeakerTurn
from raw2translated.models import TranscriptSegment
from raw2translated.pipeline import ProcessOptions, process_episode


class PipelineTests(unittest.TestCase):
    def test_dry_run_writes_placeholder_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "episode.mkv"
            input_path.write_bytes(b"not a real media file")

            result = process_episode(
                input_path,
                ProcessOptions(
                    output_dir=root / "output",
                    dry_run=True,
                    asr_provider="faster-whisper",
                    diarization_provider="pyannote",
                ),
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            transcript = json.loads(result.raw_transcript_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["pipeline"]["asr"], "dry_run")
        self.assertEqual(manifest["pipeline"]["diarization"], "dry_run")
        self.assertEqual(transcript["metadata"]["status"], "placeholder")
        self.assertTrue(result.subtitle_ass_path.name.endswith(".ass"))

    def test_process_keeps_raw_and_speaker_transcripts_separate(self) -> None:
        class FakeAsrProvider:
            def transcribe(self, audio_path: Path, *, language: str = "ja") -> list[TranscriptSegment]:
                return [
                    TranscriptSegment(
                        start=0.0,
                        end=2.0,
                        speaker="UNKNOWN",
                        text_ja="何してるの？",
                    )
                ]

        class FakeDiarizationProvider:
            def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
                return [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "episode.mkv"
            input_path.write_bytes(b"not a real media file")
            audio_path = root / "output" / "media" / "analysis.wav"

            with patch("raw2translated.pipeline.probe_media", return_value={}), patch(
                "raw2translated.pipeline.extract_analysis_audio",
                return_value=audio_path,
            ), patch("raw2translated.pipeline.build_asr_provider", return_value=FakeAsrProvider()), patch(
                "raw2translated.pipeline.build_diarization_provider",
                return_value=FakeDiarizationProvider(),
            ):
                result = process_episode(
                    input_path,
                    ProcessOptions(
                        output_dir=root / "output",
                        asr_provider="faster-whisper",
                        diarization_provider="pyannote",
                    ),
                )

            raw = json.loads(result.raw_transcript_path.read_text(encoding="utf-8"))
            speaker = json.loads(result.speaker_transcript_path.read_text(encoding="utf-8"))

        self.assertEqual(raw["segments"][0]["speaker"], "UNKNOWN")
        self.assertEqual(speaker["segments"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(speaker["segments"][0]["speaker_confidence"], 1.0)

    def test_preprocess_audio_is_used_for_asr_not_diarization(self) -> None:
        seen: dict[str, Path] = {}

        class FakePreprocessor:
            def preprocess(
                self,
                input_audio: Path,
                output_audio: Path,
                *,
                work_dir: Path,
                overwrite: bool = False,
            ) -> Path:
                seen["preprocess_input"] = input_audio
                seen["preprocess_output"] = output_audio
                return output_audio

        class FakeAsrProvider:
            def transcribe(self, audio_path: Path, *, language: str = "ja") -> list[TranscriptSegment]:
                seen["asr_audio"] = audio_path
                return [TranscriptSegment(start=0.0, end=1.0, text_ja="はい")]

        class FakeDiarizationProvider:
            def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
                seen["diarization_audio"] = audio_path
                return [SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00")]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "episode.mkv"
            input_path.write_bytes(b"not a real media file")

            with patch("raw2translated.pipeline.probe_media", return_value={}), patch(
                "raw2translated.pipeline.extract_analysis_audio",
                side_effect=lambda _input, output, **_kwargs: output,
            ), patch("raw2translated.pipeline.build_audio_preprocessor", return_value=FakePreprocessor()), patch(
                "raw2translated.pipeline.build_asr_provider",
                return_value=FakeAsrProvider(),
            ), patch(
                "raw2translated.pipeline.build_diarization_provider",
                return_value=FakeDiarizationProvider(),
            ):
                result = process_episode(
                    input_path,
                    ProcessOptions(
                        output_dir=root / "output",
                        preprocess_provider="demucs",
                        asr_provider="faster-whisper",
                        diarization_provider="pyannote",
                    ),
                )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(seen["preprocess_input"].name, "separation_input.wav")
        self.assertEqual(seen["asr_audio"].name, "dialogue.wav")
        self.assertEqual(seen["diarization_audio"].name, "analysis.wav")
        self.assertEqual(manifest["pipeline"]["preprocess"], "completed")
        self.assertEqual(manifest["media"]["asr_audio"], str(root / "output" / "media" / "dialogue.wav"))


if __name__ == "__main__":
    unittest.main()
