import unittest

from raw2translated.models import TranscriptSegment
from raw2translated.subtitles import format_ass_time, format_srt_time, segments_to_ass, segments_to_srt


class SubtitleTests(unittest.TestCase):
    def test_time_formatting(self) -> None:
        self.assertEqual(format_ass_time(3661.235), "1:01:01.24")
        self.assertEqual(format_srt_time(3661.235), "01:01:01,235")

    def test_ass_export_contains_dialogue(self) -> None:
        ass = segments_to_ass(
            [
                TranscriptSegment(
                    start=1,
                    end=2.5,
                    speaker="SPEAKER_00",
                    text_ja="べ、別に",
                    text_zh="才、才不是",
                )
            ],
            bilingual=True,
        )
        self.assertIn("[Events]", ass)
        self.assertIn("Dialogue: 0,0:00:01.00,0:00:02.50", ass)
        self.assertIn("才、才不是", ass)

    def test_srt_export(self) -> None:
        srt = segments_to_srt(
            [
                TranscriptSegment(
                    start=1,
                    end=2.5,
                    speaker="SPEAKER_00",
                    text_ja="何してるの？",
                )
            ]
        )
        self.assertIn("00:00:01,000 --> 00:00:02,500", srt)
        self.assertIn("何してるの？", srt)

    def _bilingual_segment(self) -> TranscriptSegment:
        return TranscriptSegment(
            start=1,
            end=2.5,
            speaker="SPEAKER_00",
            text_ja="べ、別に",
            text_zh="才、才不是",
        )

    def test_ass_translated_mode_excludes_source(self) -> None:
        ass = segments_to_ass([self._bilingual_segment()], text_mode="translated")
        self.assertIn("才、才不是", ass)
        self.assertNotIn("べ、別に", ass)

    def test_ass_bilingual_mode_includes_both(self) -> None:
        ass = segments_to_ass([self._bilingual_segment()], text_mode="bilingual")
        self.assertIn("才、才不是\\Nべ、別に", ass)

    def test_srt_translated_mode(self) -> None:
        srt = segments_to_srt([self._bilingual_segment()], text_mode="translated")
        self.assertIn("才、才不是", srt)
        self.assertNotIn("べ、別に", srt)

    def test_srt_bilingual_mode_two_lines(self) -> None:
        srt = segments_to_srt([self._bilingual_segment()], text_mode="bilingual")
        self.assertIn("才、才不是\nべ、別に", srt)

    def test_untranslated_line_falls_back_to_source_no_none(self) -> None:
        segment = TranscriptSegment(start=0, end=1, text_ja="おはよう", text_zh=None)
        for mode in ("translated", "bilingual"):
            ass = segments_to_ass([segment], text_mode=mode)
            srt = segments_to_srt([segment], text_mode=mode)
            self.assertIn("おはよう", ass)
            self.assertIn("おはよう", srt)
            self.assertNotIn("None", ass)
            self.assertNotIn("None", srt)

    def test_invalid_text_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            segments_to_ass([self._bilingual_segment()], text_mode="quadlingual")


if __name__ == "__main__":
    unittest.main()

