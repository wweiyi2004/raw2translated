from __future__ import annotations

from typing import Protocol

from .models import TranscriptSegment


class TranslationProvider(Protocol):
    def translate(
        self,
        segments: list[TranscriptSegment],
        *,
        style: str = "anime-subgroup",
    ) -> list[TranscriptSegment]:
        """Return segments with text_zh populated."""


class NotConfiguredTranslationProvider:
    def translate(
        self,
        segments: list[TranscriptSegment],
        *,
        style: str = "anime-subgroup",
    ) -> list[TranscriptSegment]:
        raise RuntimeError("No translation provider configured.")

