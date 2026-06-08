import unittest

from raw2translated.asr import FasterWhisperAsrProvider, build_asr_provider


class AsrTests(unittest.TestCase):
    def test_build_none_provider(self) -> None:
        self.assertIsNone(build_asr_provider("none"))

    def test_build_faster_whisper_provider(self) -> None:
        provider = build_asr_provider("faster-whisper", model="tiny", device="cpu")
        self.assertIsInstance(provider, FasterWhisperAsrProvider)

    def test_reject_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            build_asr_provider("unknown")


if __name__ == "__main__":
    unittest.main()

