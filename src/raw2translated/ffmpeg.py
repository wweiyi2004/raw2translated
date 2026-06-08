from __future__ import annotations

import json
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def probe_media(input_path: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(input_path),
    ]
    completed = _run(command)
    return json.loads(completed.stdout)


def extract_analysis_audio(
    input_path: Path,
    output_path: Path,
    *,
    audio_stream: int | None = None,
    channels: int = 1,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg"]
    if overwrite:
        command.append("-y")
    else:
        command.append("-n")
    command.extend(["-i", str(input_path)])
    if audio_stream is not None:
        command.extend(["-map", f"0:a:{audio_stream}"])
    command.extend(["-vn", "-ac", str(channels), "-ar", str(sample_rate), "-f", "wav", str(output_path)])
    _run(command)
    return output_path


def convert_audio(
    input_path: Path,
    output_path: Path,
    *,
    channels: int = 1,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg"]
    if overwrite:
        command.append("-y")
    else:
        command.append("-n")
    command.extend(
        [
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(output_path),
        ]
    )
    _run(command)
    return output_path


def mux_subtitle(
    input_path: Path, subtitle_path: Path, output_path: Path, *, overwrite: bool = False
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not subtitle_path.exists():
        raise FileNotFoundError(subtitle_path)
    if output_path.exists() and not overwrite:
        raise FFmpegError(f"output exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg"]
    if overwrite:
        command.append("-y")
    else:
        command.append("-n")
    command.extend(
        [
            "-i",
            str(input_path),
            "-i",
            str(subtitle_path),
            "-map",
            "0",
            "-map",
            "1",
            "-c",
            "copy",
            str(output_path),
        ]
    )
    _run(command)
    return output_path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise FFmpegError(f"missing executable: {command[0]}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise FFmpegError(stderr or f"{command[0]} exited with code {completed.returncode}")
    return completed
