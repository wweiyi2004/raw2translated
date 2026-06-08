from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import TranscriptSegment


class AsrProviderError(RuntimeError):
    pass


class AsrProvider(Protocol):
    def transcribe(self, audio_path: Path, *, language: str = "ja") -> list[TranscriptSegment]:
        """Return time-coded Japanese transcript segments."""


@dataclass(frozen=True)
class FasterWhisperConfig:
    model: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "default"
    vad_filter: bool = True
    beam_size: int = 5
    initial_prompt: str | None = None


class NotConfiguredAsrProvider:
    def transcribe(self, audio_path: Path, *, language: str = "ja") -> list[TranscriptSegment]:
        raise AsrProviderError(
            "No ASR provider configured. Planned providers: faster-whisper, kotoba-whisper, ReazonSpeech."
        )


class FasterWhisperAsrProvider:
    def __init__(self, config: FasterWhisperConfig | None = None) -> None:
        self.config = config or FasterWhisperConfig()

    def transcribe(self, audio_path: Path, *, language: str = "ja") -> list[TranscriptSegment]:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AsrProviderError(
                "faster-whisper is not installed. Install model dependencies with: "
                'pip install -e ".[asr]"'
            ) from exc

        model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        kwargs = {
            "language": language,
            "vad_filter": self.config.vad_filter,
            "beam_size": self.config.beam_size,
        }
        if self.config.initial_prompt:
            kwargs["initial_prompt"] = self.config.initial_prompt

        raw_segments, info = model.transcribe(str(audio_path), **kwargs)
        result: list[TranscriptSegment] = []
        for raw_segment in raw_segments:
            text = raw_segment.text.strip()
            if not text:
                continue
            notes = []
            no_speech_prob = getattr(raw_segment, "no_speech_prob", None)
            if no_speech_prob is not None and no_speech_prob >= 0.5:
                notes.append("high_no_speech_probability")
            result.append(
                TranscriptSegment(
                    start=raw_segment.start,
                    end=raw_segment.end,
                    speaker="UNKNOWN",
                    text_ja=text,
                    asr_confidence=_confidence_from_avg_logprob(
                        getattr(raw_segment, "avg_logprob", None)
                    ),
                    notes=notes,
                )
            )

        detected_language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)
        if result and detected_language and detected_language != language:
            result[0].notes.extend(
                [
                    f"detected_language={detected_language}",
                    f"language_probability={language_probability}",
                ]
            )
        return result


def build_asr_provider(
    provider: str,
    *,
    model: str = "large-v3-turbo",
    device: str = "auto",
    compute_type: str = "default",
    vad_filter: bool = True,
    beam_size: int = 5,
    initial_prompt: str | None = None,
) -> AsrProvider | None:
    if provider == "none":
        return None
    if provider == "faster-whisper":
        return FasterWhisperAsrProvider(
            FasterWhisperConfig(
                model=model,
                device=device,
                compute_type=compute_type,
                vad_filter=vad_filter,
                beam_size=beam_size,
                initial_prompt=initial_prompt,
            )
        )
    raise ValueError(f"unknown ASR provider: {provider}")


def _confidence_from_avg_logprob(avg_logprob: float | None) -> float | None:
    if avg_logprob is None:
        return None
    return round(max(0.0, min(1.0, math.exp(avg_logprob))), 4)
