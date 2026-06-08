from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .ffmpeg import convert_audio


class AudioPreprocessError(RuntimeError):
    pass


class AudioPreprocessor(Protocol):
    def preprocess(
        self,
        input_audio: Path,
        output_audio: Path,
        *,
        work_dir: Path,
        overwrite: bool = False,
    ) -> Path:
        """Return an ASR-ready audio file."""


@dataclass(frozen=True)
class DemucsPreprocessConfig:
    model: str = "htdemucs"
    stem: str = "vocals"
    device: str = "auto"
    segment: int | None = None
    shifts: int = 0
    jobs: int | None = None


class DemucsAudioPreprocessor:
    def __init__(self, config: DemucsPreprocessConfig | None = None) -> None:
        self.config = config or DemucsPreprocessConfig()

    def preprocess(
        self,
        input_audio: Path,
        output_audio: Path,
        *,
        work_dir: Path,
        overwrite: bool = False,
    ) -> Path:
        if not input_audio.exists():
            raise FileNotFoundError(input_audio)
        if output_audio.exists() and not overwrite:
            return output_audio

        work_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "--two-stems",
            self.config.stem,
            "-n",
            self.config.model,
            "-o",
            str(work_dir),
        ]
        if self.config.device != "auto":
            command.extend(["-d", self.config.device])
        if self.config.segment is not None:
            command.extend(["--segment", str(self.config.segment)])
        if self.config.shifts > 0:
            command.extend(["--shifts", str(self.config.shifts)])
        if self.config.jobs is not None:
            command.extend(["-j", str(self.config.jobs)])
        command.append(str(input_audio))

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise AudioPreprocessError(
                stderr
                or 'Demucs failed. Install preprocessing dependencies with: pip install -e ".[preprocess]"'
            )

        stem_audio = _find_demucs_stem(work_dir, self.config.model, input_audio.stem, self.config.stem)
        return convert_audio(stem_audio, output_audio, channels=1, sample_rate=16000, overwrite=True)


def build_audio_preprocessor(
    provider: str,
    *,
    model: str = "htdemucs",
    device: str = "auto",
    segment: int | None = None,
    shifts: int = 0,
    jobs: int | None = None,
) -> AudioPreprocessor | None:
    if provider == "none":
        return None
    if provider == "demucs":
        return DemucsAudioPreprocessor(
            DemucsPreprocessConfig(
                model=model,
                device=device,
                segment=segment,
                shifts=shifts,
                jobs=jobs,
            )
        )
    raise ValueError(f"unknown audio preprocessing provider: {provider}")


def _find_demucs_stem(work_dir: Path, model: str, track_name: str, stem: str) -> Path:
    expected = work_dir / model / track_name / f"{stem}.wav"
    if expected.exists():
        return expected

    candidates = sorted(work_dir.rglob(f"{stem}.wav"))
    if candidates:
        return candidates[0]

    raise AudioPreprocessError(f"Demucs finished but did not produce {stem}.wav under {work_dir}")
