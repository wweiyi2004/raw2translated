import json
import tempfile
import unittest
from pathlib import Path

from raw2translated.models import TranscriptSegment
from raw2translated.translation import (
    GlossaryTranslationProvider,
    NullTranslationProvider,
    OpenAITranslationProvider,
    TranslationError,
    TranslationMemoryProvider,
    TranslationTransportError,
    build_translation_provider,
)


class _FakeTransport:
    """Records calls and returns a canned OpenAI-style response."""

    def __init__(self, reply: str = "译文", *, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[dict] = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload})
        if self.fail:
            raise TranslationTransportError("boom")
        return {"choices": [{"message": {"content": self.reply}}]}


def _segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=0.0, end=1.0, text_ja="おはよう"),
        TranscriptSegment(start=1.0, end=2.0, text_ja="未収録のセリフ"),
    ]


class NullProviderTests(unittest.TestCase):
    def test_null_provider_keeps_source_and_marks_untranslated(self) -> None:
        segments = NullTranslationProvider().translate(_segments())
        self.assertIsNone(segments[0].text_zh)
        self.assertIn("untranslated", segments[0].notes)
        self.assertFalse(segments[0].metadata["translation"]["translated"])


class MemoryProviderTests(unittest.TestCase):
    def test_memory_translates_known_lines(self) -> None:
        provider = TranslationMemoryProvider({"おはよう": "早上好"})
        segments = provider.translate(_segments())

        self.assertEqual(segments[0].text_zh, "早上好")
        self.assertTrue(segments[0].metadata["translation"]["translated"])
        self.assertEqual(segments[0].metadata["translation"]["target_lang"], "zh-CN")

    def test_memory_keeps_unmatched_source_and_marks_it(self) -> None:
        provider = TranslationMemoryProvider({"おはよう": "早上好"})
        segments = provider.translate(_segments())

        # Unmatched line is never dropped and keeps its source text.
        self.assertEqual(segments[1].source_text, "未収録のセリフ")
        self.assertIsNone(segments[1].text_zh)
        self.assertIn("untranslated", segments[1].notes)

    def test_memory_from_file_supports_flat_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            path.write_text(json.dumps({"おはよう": "早上好"}), encoding="utf-8")
            provider = TranslationMemoryProvider.from_file(path)
        self.assertEqual(provider.memory["おはよう"], "早上好")


class GlossaryProviderTests(unittest.TestCase):
    def test_glossary_substitutes_terms(self) -> None:
        provider = GlossaryTranslationProvider({"先輩": "前辈"})
        segments = [TranscriptSegment(start=0.0, end=1.0, text_ja="先輩、おはよう")]
        provider.translate(segments)
        self.assertIn("前辈", segments[0].text_zh)

    def test_glossary_prefers_longer_terms(self) -> None:
        provider = GlossaryTranslationProvider({"先輩": "前辈", "先輩方": "各位前辈"})
        segments = [TranscriptSegment(start=0.0, end=1.0, text_ja="先輩方")]
        provider.translate(segments)
        self.assertEqual(segments[0].text_zh, "各位前辈")

    def test_glossary_marks_unmatched(self) -> None:
        provider = GlossaryTranslationProvider({"先輩": "前辈"})
        segments = [TranscriptSegment(start=0.0, end=1.0, text_ja="こんにちは")]
        provider.translate(segments)
        self.assertIsNone(segments[0].text_zh)
        self.assertIn("untranslated", segments[0].notes)


class OpenAIProviderTests(unittest.TestCase):
    def test_translates_untranslated_lines(self) -> None:
        transport = _FakeTransport(reply="早上好")
        provider = OpenAITranslationProvider(api_key="sk-test", transport=transport)
        segments = [TranscriptSegment(start=0.0, end=1.0, text_ja="おはよう")]
        provider.translate(segments)
        self.assertEqual(segments[0].text_zh, "早上好")
        self.assertTrue(segments[0].metadata["translation"]["translated"])
        self.assertEqual(len(transport.calls), 1)
        # The user message carries the source line.
        messages = transport.calls[0]["payload"]["messages"]
        self.assertEqual(messages[-1]["content"], "おはよう")

    def test_glossary_is_in_system_prompt(self) -> None:
        provider = OpenAITranslationProvider(
            api_key="sk-test",
            glossary={"先輩": "前辈"},
            transport=_FakeTransport(),
        )
        prompt = provider.system_prompt("zh-CN")
        self.assertIn("先輩 => 前辈", prompt)
        self.assertIn("zh-CN", prompt)

    def test_transport_error_keeps_source_and_continues(self) -> None:
        transport = _FakeTransport(fail=True)
        provider = OpenAITranslationProvider(api_key="sk-test", transport=transport)
        segments = [TranscriptSegment(start=0.0, end=1.0, text_ja="おはよう")]
        provider.translate(segments)
        self.assertIsNone(segments[0].text_zh)
        self.assertIn("untranslated", segments[0].notes)
        self.assertIn("error", segments[0].metadata["translation"])

    def test_already_translated_line_is_skipped(self) -> None:
        transport = _FakeTransport()
        provider = OpenAITranslationProvider(api_key="sk-test", transport=transport)
        segments = [TranscriptSegment(start=0.0, end=1.0, text_ja="おはよう", text_zh="早上好")]
        provider.translate(segments)
        self.assertEqual(len(transport.calls), 0)

    def test_unexpected_response_raises(self) -> None:
        def bad_transport(url, headers, payload, timeout):
            return {"unexpected": True}

        provider = OpenAITranslationProvider(api_key="sk-test", transport=bad_transport)
        with self.assertRaises(TranslationError):
            provider.translate([TranscriptSegment(start=0.0, end=1.0, text_ja="おはよう")])


class FactoryTests(unittest.TestCase):
    def test_build_none_returns_null_provider(self) -> None:
        self.assertIsInstance(build_translation_provider("none"), NullTranslationProvider)

    def test_build_memory_requires_path(self) -> None:
        with self.assertRaises(TranslationError):
            build_translation_provider("memory")

    def test_build_unknown_provider_raises(self) -> None:
        with self.assertRaises(TranslationError):
            build_translation_provider("gpt-9000")

    def test_build_openai_requires_api_key(self) -> None:
        with self.assertRaises(TranslationError):
            build_translation_provider("openai")

    def test_build_openai_with_key_and_transport_translates(self) -> None:
        provider = build_translation_provider(
            "openai",
            api_key="sk-test",
            transport=_FakeTransport(reply="早上好"),
        )
        segments = [TranscriptSegment(start=0.0, end=1.0, text_ja="おはよう")]
        provider.translate(segments)
        self.assertEqual(segments[0].text_zh, "早上好")

    def test_build_memory_from_example_config(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs" / "translation_memory.example.json"
        provider = build_translation_provider("memory", memory_path=config)
        segments = [TranscriptSegment(start=0.0, end=1.0, text_ja="おはよう")]
        provider.translate(segments)
        self.assertEqual(segments[0].text_zh, "早上好")

    def test_build_glossary_from_example_config(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs" / "glossary.example.json"
        provider = build_translation_provider("glossary", glossary_path=config)
        segments = [TranscriptSegment(start=0.0, end=1.0, text_ja="先輩、ありがとう")]
        provider.translate(segments)
        self.assertEqual(segments[0].text_zh, "前辈、谢谢")

    def test_invalid_json_raises_translation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(TranslationError):
                build_translation_provider("memory", memory_path=path)

    def test_missing_file_raises_translation_error(self) -> None:
        with self.assertRaises(TranslationError):
            build_translation_provider("memory", memory_path=Path("does-not-exist.json"))


if __name__ == "__main__":
    unittest.main()
