import tempfile
import unittest
from pathlib import Path

from raw2translated.preprocess import DemucsAudioPreprocessor, _find_demucs_stem, build_audio_preprocessor


class PreprocessTests(unittest.TestCase):
    def test_build_none_provider(self) -> None:
        self.assertIsNone(build_audio_preprocessor("none"))

    def test_build_demucs_provider(self) -> None:
        provider = build_audio_preprocessor("demucs", model="htdemucs", device="cpu")
        self.assertIsInstance(provider, DemucsAudioPreprocessor)

    def test_reject_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            build_audio_preprocessor("unknown")

    def test_find_demucs_stem_expected_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "htdemucs" / "episode" / "vocals.wav"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"wav")

            found = _find_demucs_stem(root, "htdemucs", "episode", "vocals")

        self.assertEqual(found, expected)


if __name__ == "__main__":
    unittest.main()

