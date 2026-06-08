from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class DiarizationProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str
    confidence: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PyannoteDiarizationConfig:
    model: str = "pyannote/speaker-diarization-community-1"
    token: str | None = None
    device: str = "auto"
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class DiarizationProvider(Protocol):
    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        """Return speaker turns for an episode audio file."""


class NotConfiguredDiarizationProvider:
    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        raise DiarizationProviderError(
            "No diarization provider configured. Planned provider: pyannote.audio."
        )


class PyannoteDiarizationProvider:
    def __init__(self, config: PyannoteDiarizationConfig | None = None) -> None:
        self.config = config or PyannoteDiarizationConfig()

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationProviderError(
                "pyannote.audio is not installed. Install diarization dependencies with: "
                'pip install -e ".[diarization]"'
            ) from exc

        token = self.config.token or _env_token()
        pipeline = _load_pyannote_pipeline(Pipeline, self.config.model, token)
        _move_pipeline_to_device(pipeline, self.config.device)

        kwargs = {}
        if self.config.num_speakers is not None:
            kwargs["num_speakers"] = self.config.num_speakers
        if self.config.min_speakers is not None:
            kwargs["min_speakers"] = self.config.min_speakers
        if self.config.max_speakers is not None:
            kwargs["max_speakers"] = self.config.max_speakers

        try:
            diarization = pipeline(str(audio_path), **kwargs)
        except Exception as exc:
            raise DiarizationProviderError(f"pyannote diarization failed: {exc}") from exc

        turns: list[SpeakerTurn] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=str(speaker),
                    confidence=None,
                )
            )
        return turns


def dominant_speaker_for_segment(start: float, end: float, turns: list[SpeakerTurn]) -> SpeakerTurn | None:
    best: SpeakerTurn | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best = turn
            best_overlap = overlap
    return best


def speaker_overlap_ratio(start: float, end: float, turn: SpeakerTurn | None) -> float:
    duration = max(0.0, end - start)
    if duration == 0.0 or turn is None:
        return 0.0
    overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
    return min(1.0, overlap / duration)


def build_diarization_provider(
    provider: str,
    *,
    model: str = "pyannote/speaker-diarization-community-1",
    token: str | None = None,
    device: str = "auto",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> DiarizationProvider | None:
    if provider == "none":
        return None
    if provider == "pyannote":
        return PyannoteDiarizationProvider(
            PyannoteDiarizationConfig(
                model=model,
                token=token,
                device=device,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        )
    raise ValueError(f"unknown diarization provider: {provider}")


def _env_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("PYANNOTE_AUTH_TOKEN")
    )


def _load_pyannote_pipeline(pipeline_cls, model: str, token: str | None):
    try:
        return pipeline_cls.from_pretrained(model, token=token)
    except TypeError:
        try:
            return pipeline_cls.from_pretrained(model, use_auth_token=token)
        except Exception as exc:
            raise DiarizationProviderError(
                "Could not load pyannote pipeline. Check the model name, access token, and that you accepted "
                "the model conditions on Hugging Face."
            ) from exc
    except Exception as exc:
        raise DiarizationProviderError(
            "Could not load pyannote pipeline. Check the model name, access token, and that you accepted "
            "the model conditions on Hugging Face."
        ) from exc


def _move_pipeline_to_device(pipeline, device: str) -> None:
    if device == "cpu":
        return

    try:
        import torch
    except ImportError as exc:
        if device == "auto":
            return
        raise DiarizationProviderError("Torch is required to select a diarization device.") from exc

    if device == "auto":
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        return

    try:
        pipeline.to(torch.device(device))
    except Exception as exc:
        raise DiarizationProviderError(
            f"Could not move pyannote pipeline to device {device!r}: {exc}"
        ) from exc
