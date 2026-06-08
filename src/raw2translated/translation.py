from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .models import TranscriptSegment

UNTRANSLATED_NOTE = "untranslated"


class TranslationError(ValueError):
    """Raised when a translation provider cannot be built or run."""


class TranslationProvider(Protocol):
    name: str

    def translate(
        self,
        segments: list[TranscriptSegment],
        *,
        target_lang: str = "zh-CN",
    ) -> list[TranscriptSegment]:
        """Return segments with ``text_zh`` populated where possible.

        Implementations must never drop a segment. A line that cannot be
        translated keeps its source text and is tagged ``untranslated``.
        """


class NotConfiguredTranslationProvider:
    """Legacy provider kept for backward compatibility.

    Prefer :class:`NullTranslationProvider`, which is a no-op instead of raising.
    """

    name = "not-configured"

    def translate(
        self,
        segments: list[TranscriptSegment],
        *,
        target_lang: str = "zh-CN",
    ) -> list[TranscriptSegment]:
        raise RuntimeError("No translation provider configured.")


def _mark_untranslated(segment: TranscriptSegment, provider: str) -> None:
    if UNTRANSLATED_NOTE not in segment.notes:
        segment.notes.append(UNTRANSLATED_NOTE)
    segment.metadata["translation"] = {"provider": provider, "translated": False}


def _mark_translated(segment: TranscriptSegment, provider: str, target_lang: str) -> None:
    if UNTRANSLATED_NOTE in segment.notes:
        segment.notes.remove(UNTRANSLATED_NOTE)
    segment.metadata["translation"] = {
        "provider": provider,
        "translated": True,
        "target_lang": target_lang,
    }


class NullTranslationProvider:
    """Keeps the source text and produces no translation.

    Lines remain untranslated but are never dropped. This is the default so the
    pipeline works with no API key, model, or network access.
    """

    name = "null"

    def translate(
        self,
        segments: list[TranscriptSegment],
        *,
        target_lang: str = "zh-CN",
    ) -> list[TranscriptSegment]:
        for segment in segments:
            if not segment.is_translated:
                _mark_untranslated(segment, self.name)
        return segments


class TranslationMemoryProvider:
    """Translate by looking up exact source text in a local translation memory.

    The memory maps a source string to its target translation. Anything not in
    the memory keeps its source text and is tagged ``untranslated``.
    """

    name = "memory"

    def __init__(self, memory: dict[str, str]) -> None:
        self.memory = dict(memory)

    @classmethod
    def from_file(cls, path: Path) -> "TranslationMemoryProvider":
        return cls(_load_memory(path))

    def translate(
        self,
        segments: list[TranscriptSegment],
        *,
        target_lang: str = "zh-CN",
    ) -> list[TranscriptSegment]:
        for segment in segments:
            source = (segment.source_text or "").strip()
            target = self.memory.get(source)
            if target is None:
                # Allow the raw (unstripped) key to match as well.
                target = self.memory.get(segment.source_text or "")
            if target:
                segment.text_zh = target
                _mark_translated(segment, self.name, target_lang)
            elif not segment.is_translated:
                _mark_untranslated(segment, self.name)
        return segments


class GlossaryTranslationProvider:
    """Substring substitution from a local glossary.

    This is a deterministic, offline placeholder rather than a real translator:
    every glossary term found in the source is replaced by its target. A line
    where at least one term matched is treated as (partially) translated; a line
    with no matches keeps its source text and is tagged ``untranslated``.
    """

    name = "glossary"

    def __init__(self, terms: dict[str, str]) -> None:
        # Replace longer source terms first so they win over shorter substrings.
        self.terms = dict(sorted(terms.items(), key=lambda item: len(item[0]), reverse=True))

    @classmethod
    def from_file(cls, path: Path) -> "GlossaryTranslationProvider":
        return cls(_load_glossary(path))

    def translate(
        self,
        segments: list[TranscriptSegment],
        *,
        target_lang: str = "zh-CN",
    ) -> list[TranscriptSegment]:
        for segment in segments:
            source = segment.source_text or ""
            rendered = source
            matched = False
            for src_term, tgt_term in self.terms.items():
                if src_term and src_term in rendered:
                    rendered = rendered.replace(src_term, tgt_term)
                    matched = True
            if matched:
                segment.text_zh = rendered
                _mark_translated(segment, self.name, target_lang)
            elif not segment.is_translated:
                _mark_untranslated(segment, self.name)
        return segments


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise TranslationError(f"translation config not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranslationError(f"invalid JSON in {path}: {exc}") from exc


def _coerce_mapping(data: Any, path: Path) -> dict[str, str]:
    """Accept either a flat ``{source: target}`` dict or a list of entries.

    List entries use ``source``/``target`` keys, matching the glossary YAML shape.
    """
    if isinstance(data, dict):
        if "terms" in data and isinstance(data["terms"], list):
            data = data["terms"]
        elif "entries" in data and isinstance(data["entries"], list):
            data = data["entries"]
        else:
            return {str(key): str(value) for key, value in data.items()}
    if isinstance(data, list):
        mapping: dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                raise TranslationError(f"expected object entries in {path}, got {item!r}")
            source = item.get("source")
            target = item.get("target")
            if source is None or target is None:
                raise TranslationError(f"entry in {path} needs 'source' and 'target': {item!r}")
            mapping[str(source)] = str(target)
        return mapping
    raise TranslationError(f"unsupported translation config shape in {path}: {type(data).__name__}")


def _load_memory(path: Path) -> dict[str, str]:
    return _coerce_mapping(_load_json(path), path)


def _load_glossary(path: Path) -> dict[str, str]:
    return _coerce_mapping(_load_json(path), path)


def build_translation_provider(
    name: str,
    *,
    memory_path: Path | None = None,
    glossary_path: Path | None = None,
) -> TranslationProvider:
    """Factory for the local-first translation providers.

    ``name`` is one of ``none``/``null``, ``memory``, ``glossary``.
    """
    if name in ("none", "null"):
        return NullTranslationProvider()
    if name == "memory":
        if memory_path is None:
            raise TranslationError("provider 'memory' requires --translation-memory PATH")
        return TranslationMemoryProvider.from_file(memory_path)
    if name == "glossary":
        if glossary_path is None:
            raise TranslationError("provider 'glossary' requires --glossary PATH")
        return GlossaryTranslationProvider.from_file(glossary_path)
    raise TranslationError(f"unsupported translation provider: {name}")
