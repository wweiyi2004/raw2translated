"""Bind diarization speaker clusters to project character names.

Diarization produces anonymous clusters such as ``SPEAKER_00``. A character map
turns those into stable character identities (``女主A``) that the model already
carries on :attr:`TranscriptSegment.character`. The map is a local project
artifact — nothing here uploads or learns from media.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import TranscriptSegment


class CharacterMapError(ValueError):
    """Raised when a character map cannot be loaded."""


def load_character_map(path: Path) -> dict[str, str]:
    """Load a ``{speaker_label: character_name}`` map from JSON.

    Accepts either a flat object, or a list of ``{"speaker": ..., "character": ...}``
    entries (the same shape style as the glossary configs).
    """
    path = Path(path)
    if not path.exists():
        raise CharacterMapError(f"character map not found: {path}")
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CharacterMapError(f"invalid JSON in {path}: {exc}") from exc
    return _coerce_character_map(data, path)


def _coerce_character_map(data: Any, path: Path) -> dict[str, str]:
    if isinstance(data, dict):
        if isinstance(data.get("characters"), list):
            data = data["characters"]
        else:
            return {str(speaker): str(name) for speaker, name in data.items()}
    if isinstance(data, list):
        mapping: dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                raise CharacterMapError(f"expected object entries in {path}, got {item!r}")
            speaker = item.get("speaker")
            character = item.get("character")
            if speaker is None or character is None:
                raise CharacterMapError(
                    f"entry in {path} needs 'speaker' and 'character': {item!r}"
                )
            mapping[str(speaker)] = str(character)
        return mapping
    raise CharacterMapError(f"unsupported character map shape in {path}: {type(data).__name__}")


def apply_character_map(segments: list[TranscriptSegment], mapping: dict[str, str]) -> int:
    """Set ``character`` on every segment whose speaker is in the map.

    Returns the number of segments that were assigned a character.
    """
    assigned = 0
    for segment in segments:
        name = mapping.get(segment.speaker)
        if name:
            segment.character = name
            assigned += 1
    return assigned
