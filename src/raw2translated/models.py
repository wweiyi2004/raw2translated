from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str = "UNKNOWN"
    character: str | None = None
    text_ja: str = ""
    text_zh: str | None = None
    speaker_confidence: float | None = None
    asr_confidence: float | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.start = float(self.start)
        self.end = float(self.end)
        if self.end < self.start:
            raise ValueError(f"segment end must be >= start: {self.start} > {self.end}")

    @property
    def display_speaker(self) -> str:
        return self.character or self.speaker

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            start=data["start"],
            end=data["end"],
            speaker=data.get("speaker", "UNKNOWN"),
            character=data.get("character"),
            text_ja=data.get("text_ja", ""),
            text_zh=data.get("text_zh"),
            speaker_confidence=data.get("speaker_confidence"),
            asr_confidence=data.get("asr_confidence"),
            notes=list(data.get("notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeTranscript:
    media_path: str | None = None
    language: str = "ja"
    segments: list[TranscriptSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, *, media_path: Path | None = None, language: str = "ja") -> "EpisodeTranscript":
        return cls(media_path=str(media_path) if media_path else None, language=language, segments=[])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeTranscript":
        return cls(
            media_path=data.get("media_path"),
            language=data.get("language", "ja"),
            segments=[TranscriptSegment.from_dict(item) for item in data.get("segments", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "EpisodeTranscript":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_path": self.media_path,
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": self.metadata,
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

