from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asr import build_asr_provider
from .diarization import (
    SpeakerTurn,
    build_diarization_provider,
    dominant_speaker_for_segment,
    speaker_overlap_ratio,
)
from .ffmpeg import extract_analysis_audio, probe_media
from .models import EpisodeTranscript
from .preprocess import build_audio_preprocessor
from .subtitles import segments_to_ass, segments_to_srt
from .translation import build_translation_provider


@dataclass(frozen=True)
class ProcessOptions:
    output_dir: Path
    language: str = "ja"
    dry_run: bool = False
    overwrite: bool = False
    asr_provider: str = "none"
    asr_model: str = "large-v3-turbo"
    asr_device: str = "auto"
    asr_compute_type: str = "default"
    vad_filter: bool = True
    beam_size: int = 5
    initial_prompt: str | None = None
    preprocess_provider: str = "none"
    preprocess_model: str = "htdemucs"
    preprocess_device: str = "auto"
    preprocess_segment: int | None = None
    preprocess_shifts: int = 0
    preprocess_jobs: int | None = None
    diarization_provider: str = "none"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_token: str | None = None
    diarization_device: str = "auto"
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    speaker_min_overlap: float = 0.2
    translate_provider: str = "none"
    translation_memory: Path | None = None
    glossary: Path | None = None
    target_lang: str = "zh-CN"


@dataclass(frozen=True)
class ProcessResult:
    manifest_path: Path
    raw_transcript_path: Path
    speaker_transcript_path: Path
    diarization_path: Path | None
    subtitle_ass_path: Path
    subtitle_srt_path: Path
    audio_path: Path | None
    asr_audio_path: Path | None
    translated_transcript_path: Path | None = None


def process_episode(input_path: Path, options: ProcessOptions) -> ProcessResult:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir = options.output_dir
    media_dir = output_dir / "media"
    subtitle_dir = output_dir / "subtitles"
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any]
    try:
        metadata = probe_media(input_path)
    except Exception as exc:
        if not options.dry_run:
            raise
        metadata = {"probe_error": str(exc)}

    audio_path = None
    separation_input_path = None
    if not options.dry_run:
        audio_path = extract_analysis_audio(
            input_path,
            media_dir / "analysis.wav",
            overwrite=options.overwrite,
        )

    asr_audio_path = audio_path
    preprocess_status = "skipped"
    if audio_path is not None and options.preprocess_provider != "none":
        separation_input_path = extract_analysis_audio(
            input_path,
            media_dir / "separation_input.wav",
            channels=2,
            sample_rate=44100,
            overwrite=options.overwrite,
        )
        preprocessor = build_audio_preprocessor(
            options.preprocess_provider,
            model=options.preprocess_model,
            device=options.preprocess_device,
            segment=options.preprocess_segment,
            shifts=options.preprocess_shifts,
            jobs=options.preprocess_jobs,
        )
        if preprocessor is not None:
            asr_audio_path = preprocessor.preprocess(
                separation_input_path,
                media_dir / "dialogue.wav",
                work_dir=output_dir / "preprocess",
                overwrite=options.overwrite,
            )
            preprocess_status = "completed"
    elif options.dry_run and options.preprocess_provider != "none":
        preprocess_status = "dry_run"

    transcript = EpisodeTranscript.empty(media_path=input_path, language=options.language)
    asr_status = "skipped"
    if asr_audio_path is not None and options.asr_provider != "none":
        provider = build_asr_provider(
            options.asr_provider,
            model=options.asr_model,
            device=options.asr_device,
            compute_type=options.asr_compute_type,
            vad_filter=options.vad_filter,
            beam_size=options.beam_size,
            initial_prompt=options.initial_prompt,
        )
        if provider is not None:
            transcript.segments = provider.transcribe(asr_audio_path, language=options.language)
            transcript.metadata["status"] = "asr_completed"
            transcript.metadata["asr"] = {
                "provider": options.asr_provider,
                "model": options.asr_model,
                "audio": str(asr_audio_path),
                "device": options.asr_device,
                "compute_type": options.asr_compute_type,
                "vad_filter": options.vad_filter,
                "beam_size": options.beam_size,
            }
            asr_status = "completed"
    else:
        transcript.metadata["status"] = "placeholder"
        transcript.metadata["next_steps"] = [
            "Run ASR provider",
            "Run speaker diarization provider",
            "Merge speaker turns with ASR segments",
            "Translate with project glossary and style cards",
        ]
        if options.dry_run:
            asr_status = "dry_run"

    raw_transcript = copy.deepcopy(transcript)

    diarization_status = "skipped"
    diarization_path = None
    speaker_turns: list[SpeakerTurn] = []
    if audio_path is not None and options.diarization_provider != "none":
        provider = build_diarization_provider(
            options.diarization_provider,
            model=options.diarization_model,
            token=options.diarization_token,
            device=options.diarization_device,
            num_speakers=options.num_speakers,
            min_speakers=options.min_speakers,
            max_speakers=options.max_speakers,
        )
        if provider is not None:
            speaker_turns = provider.diarize(audio_path)
            _apply_speaker_turns(
                transcript,
                speaker_turns,
                min_overlap=options.speaker_min_overlap,
            )
            diarization_status = "completed"
            transcript.metadata["diarization"] = {
                "provider": options.diarization_provider,
                "model": options.diarization_model,
                "device": options.diarization_device,
                "num_speakers": options.num_speakers,
                "min_speakers": options.min_speakers,
                "max_speakers": options.max_speakers,
                "speaker_min_overlap": options.speaker_min_overlap,
            }
            diarization_path = output_dir / "diarization.json"
            _write_speaker_turns(diarization_path, speaker_turns)
    elif options.dry_run and options.diarization_provider != "none":
        diarization_status = "dry_run"

    translation_status = "skipped"
    translated_transcript = None
    translated_transcript_path = None
    if options.translate_provider != "none":
        provider = build_translation_provider(
            options.translate_provider,
            memory_path=options.translation_memory,
            glossary_path=options.glossary,
        )
        translated_transcript = copy.deepcopy(transcript)
        translated_transcript.segments = provider.translate(
            translated_transcript.segments,
            target_lang=options.target_lang,
        )
        translated_transcript.metadata["translation"] = {
            "provider": options.translate_provider,
            "target_lang": options.target_lang,
        }
        translation_status = "completed" if transcript.segments else "no_segments"

    manifest = {
        "input": str(input_path),
        "language": options.language,
        "dry_run": options.dry_run,
        "media": {
            "analysis_audio": str(audio_path) if audio_path else None,
            "separation_input_audio": str(separation_input_path) if separation_input_path else None,
            "asr_audio": str(asr_audio_path) if asr_audio_path else None,
        },
        "ffprobe": metadata,
        "pipeline": {
            "preprocess": preprocess_status,
            "preprocess_provider": options.preprocess_provider,
            "asr": asr_status,
            "asr_provider": options.asr_provider,
            "alignment": "pending",
            "diarization": diarization_status,
            "diarization_provider": options.diarization_provider,
            "translation": translation_status,
            "translation_provider": options.translate_provider,
            "translation_target_lang": (
                options.target_lang if options.translate_provider != "none" else None
            ),
            "translation_output": (
                str(output_dir / "transcript.translated.json")
                if translated_transcript is not None
                else None
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_transcript_path = raw_transcript.write_json(output_dir / "transcript.raw.json")
    speaker_transcript_path = transcript.write_json(output_dir / "transcript.speaker.json")

    subtitle_source = translated_transcript if translated_transcript is not None else transcript
    if translated_transcript is not None:
        translated_transcript_path = translated_transcript.write_json(
            output_dir / "transcript.translated.json"
        )

    subtitle_ass_path = subtitle_dir / "episode.ass"
    subtitle_srt_path = subtitle_dir / "episode.srt"
    subtitle_ass_path.write_text(segments_to_ass(subtitle_source.segments), encoding="utf-8")
    subtitle_srt_path.write_text(segments_to_srt(subtitle_source.segments), encoding="utf-8")

    return ProcessResult(
        manifest_path=manifest_path,
        raw_transcript_path=raw_transcript_path,
        speaker_transcript_path=speaker_transcript_path,
        diarization_path=diarization_path,
        subtitle_ass_path=subtitle_ass_path,
        subtitle_srt_path=subtitle_srt_path,
        audio_path=audio_path,
        asr_audio_path=asr_audio_path,
        translated_transcript_path=translated_transcript_path,
    )


def _apply_speaker_turns(
    transcript: EpisodeTranscript,
    turns: list[SpeakerTurn],
    *,
    min_overlap: float,
) -> None:
    if not transcript.segments:
        transcript.metadata.setdefault("warnings", []).append("diarization_has_no_asr_segments_to_label")
        return

    for segment in transcript.segments:
        turn = dominant_speaker_for_segment(segment.start, segment.end, turns)
        ratio = speaker_overlap_ratio(segment.start, segment.end, turn)
        if turn is None or ratio < min_overlap:
            segment.notes.append("speaker_unassigned")
            segment.speaker_confidence = round(ratio, 4)
            continue
        segment.speaker = turn.speaker
        segment.speaker_confidence = round(ratio, 4)
        if ratio < 0.5:
            segment.notes.append("low_speaker_overlap")


def _write_speaker_turns(path: Path, turns: list[SpeakerTurn]) -> Path:
    payload = {"turns": [turn.to_dict() for turn in turns]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
