import tempfile
import unittest
from pathlib import Path

from raw2translated.gui import GuiController
from raw2translated.models import EpisodeTranscript, TranscriptSegment
from raw2translated.pipeline import ProcessOptions

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _transcript() -> EpisodeTranscript:
    return EpisodeTranscript(
        media_path="episode.mkv",
        segments=[
            TranscriptSegment(start=0.0, end=1.0, speaker="SPEAKER_00", text_ja="おはよう"),
            TranscriptSegment(
                start=1.0, end=2.0, speaker="SPEAKER_01", text_ja="未収録のセリフ", notes=["untranslated"]
            ),
        ],
    )


class ControllerEditingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = GuiController()
        self.controller.set_transcript(_transcript())

    def test_rows_expose_display_fields(self) -> None:
        rows = self.controller.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].source, "おはよう")
        self.assertEqual(rows[0].translation, "")
        self.assertEqual(rows[1].notes, "untranslated")

    def test_update_translation_sets_text_and_clears_flag(self) -> None:
        self.controller.update_translation(1, "未收录的台词")
        segment = self.controller.segment(1)
        self.assertEqual(segment.text_zh, "未收录的台词")
        self.assertNotIn("untranslated", segment.notes)

    def test_update_translation_empty_clears_text(self) -> None:
        self.controller.update_translation(0, "   ")
        self.assertIsNone(self.controller.segment(0).text_zh)

    def test_save_and_reload_roundtrip(self) -> None:
        self.controller.update_translation(0, "早上好")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edited.json"
            self.controller.save_transcript(path)
            reloaded = GuiController()
            reloaded.load_transcript(path)
        self.assertEqual(reloaded.segment(0).text_zh, "早上好")


class ControllerTranslateTests(unittest.TestCase):
    def test_run_translate_with_memory(self) -> None:
        controller = GuiController()
        controller.set_transcript(_transcript())
        translated, total = controller.run_translate(
            "memory",
            memory_path=CONFIGS / "translation_memory.example.json",
        )
        self.assertEqual(total, 2)
        self.assertEqual(translated, 1)
        self.assertEqual(controller.segment(0).text_zh, "早上好")
        # Unmatched line is preserved, not dropped.
        self.assertIsNone(controller.segment(1).text_zh)


class ControllerExportTests(unittest.TestCase):
    def test_export_bilingual_ass_and_srt(self) -> None:
        controller = GuiController()
        transcript = _transcript()
        transcript.segments[0].text_zh = "早上好"
        controller.set_transcript(transcript)
        with tempfile.TemporaryDirectory() as tmp:
            ass = controller.export_subtitle(Path(tmp) / "out.ass", fmt="ass", text_mode="bilingual")
            srt = controller.export_subtitle(Path(tmp) / "out.srt", fmt="srt", text_mode="translated")
            ass_text = ass.read_text(encoding="utf-8")
            srt_text = srt.read_text(encoding="utf-8")
        self.assertIn("早上好", ass_text)
        self.assertIn("おはよう", ass_text)
        # Untranslated line falls back to source; never "None".
        self.assertIn("未収録のセリフ", srt_text)
        self.assertNotIn("None", srt_text)


class ControllerProcessTests(unittest.TestCase):
    def test_run_process_dry_run_loads_transcript(self) -> None:
        controller = GuiController()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "episode.mkv"
            input_path.write_bytes(b"not a real media file")
            result = controller.run_process(
                input_path,
                ProcessOptions(output_dir=root / "output", dry_run=True),
            )
            self.assertTrue(result.manifest_path.exists())
        # The speaker transcript was loaded into the controller.
        self.assertIsNotNone(controller.transcript)


class ModuleImportTests(unittest.TestCase):
    def test_importing_gui_does_not_require_tkinter(self) -> None:
        # Importing the module must not import tkinter (it is lazy inside launch()),
        # so the test simply confirms the controller is usable without a display.
        import raw2translated.gui as gui

        self.assertTrue(hasattr(gui, "GuiController"))
        self.assertTrue(hasattr(gui, "launch"))


if __name__ == "__main__":
    unittest.main()
