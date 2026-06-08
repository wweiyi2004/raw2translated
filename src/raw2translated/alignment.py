"""Forced-alignment providers that refine ASR segment timing.

Alignment takes the ASR transcript plus the audio and produces tighter
start/end times (and, with a real backend, word-level timing). Like the ASR and
diarization stages it lives behind a provider interface: the default is ``none``
and the real WhisperX backend is an optional dependency, imported lazily so the
package works — and tests run — without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import TranscriptSegment


class AlignmentProviderError(RuntimeError):
    pass


class AlignmentProvider(Protocol):
    def align(
        self,
        segments: list[TranscriptSegment],
        audio_path: Path,
        *,
        language: str = "ja",
    ) -> list[TranscriptSegment]:
        """Return segments with refined timing."""


@dataclass(frozen=True)
class WhisperXAlignmentConfig:
    model: str = "WAV2VEC2_ASR_LARGE_LV60K_960H"
    device: str = "auto"


class NotConfiguredAlignmentProvider:
    def align(
        self,
        segments: list[TranscriptSegment],
        audio_path: Path,
        *,
        language: str = "ja",
    ) -> list[TranscriptSegment]:
        raise AlignmentProviderError(
            "No alignment provider configured. Planned provider: WhisperX."
        )


class WhisperXAlignmentProvider:
    def __init__(self, config: WhisperXAlignmentConfig | None = None) -> None:
        self.config = config or WhisperXAlignmentConfig()

    def align(
        self,
        segments: list[TranscriptSegment],
        audio_path: Path,
        *,
        language: str = "ja",
    ) -> list[TranscriptSegment]:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if not segments:
            return segments

        try:
            import whisperx
        except ImportError as exc:
            raise AlignmentProviderError(
                "whisperx is not installed. Install alignment dependencies with: "
                'pip install -e ".[align]"'
            ) from exc

        device = _resolve_device(self.config.device)
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_name=self.config.model or None,
        )
        transcription = [
            {"start": segment.start, "end": segment.end, "text": segment.text_ja}
            for segment in segments
        ]
        aligned = whisperx.align(
            transcription,
            model_a,
            metadata,
            str(audio_path),
            device,
            return_char_alignments=False,
        )
        return _apply_aligned_timing(segments, aligned.get("segments", []))


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _apply_aligned_timing(
    segments: list[TranscriptSegment],
    aligned_segments: list[dict],
) -> list[TranscriptSegment]:
    """Copy refined start/end (and word timing) onto the original segments.

    Pairing is positional: WhisperX returns one aligned entry per input segment.
    Segments without a usable aligned counterpart keep their original timing.
    """
    for segment, aligned in zip(segments, aligned_segments, strict=False):
        start = aligned.get("start")
        end = aligned.get("end")
        if start is None or end is None:
            continue
        start = float(start)
        end = float(end)
        if end < start:
            continue
        segment.start = start
        segment.end = end
        words = aligned.get("words")
        if words:
            segment.metadata["words"] = [
                {"word": w.get("word"), "start": w.get("start"), "end": w.get("end")}
                for w in words
            ]
        if "alignment_skipped" in segment.notes:
            segment.notes.remove("alignment_skipped")
    return segments


def build_alignment_provider(
    provider: str,
    *,
    model: str = "WAV2VEC2_ASR_LARGE_LV60K_960H",
    device: str = "auto",
) -> AlignmentProvider | None:
    if provider == "none":
        return None
    if provider == "whisperx":
        return WhisperXAlignmentProvider(
            WhisperXAlignmentConfig(model=model, device=device)
        )
    raise ValueError(f"unknown alignment provider: {provider}")
