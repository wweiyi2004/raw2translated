import unittest

from raw2translated.diarization import (
    PyannoteDiarizationProvider,
    SpeakerTurn,
    build_diarization_provider,
    dominant_speaker_for_segment,
    speaker_overlap_ratio,
)


class DiarizationTests(unittest.TestCase):
    def test_build_none_provider(self) -> None:
        self.assertIsNone(build_diarization_provider("none"))

    def test_build_pyannote_provider(self) -> None:
        provider = build_diarization_provider("pyannote", model="pyannote/speaker-diarization-3.1")
        self.assertIsInstance(provider, PyannoteDiarizationProvider)

    def test_reject_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            build_diarization_provider("unknown")

    def test_dominant_speaker_for_segment(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=1.0, end=4.0, speaker="SPEAKER_01"),
        ]
        best = dominant_speaker_for_segment(0.5, 3.0, turns)

        self.assertIsNotNone(best)
        self.assertEqual(best.speaker, "SPEAKER_01")
        self.assertAlmostEqual(speaker_overlap_ratio(0.5, 3.0, best), 0.8)


if __name__ == "__main__":
    unittest.main()

