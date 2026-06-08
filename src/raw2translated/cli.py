from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .asr import AsrProviderError
from .diarization import DiarizationProviderError
from .ffmpeg import FFmpegError, mux_subtitle, probe_media
from .models import EpisodeTranscript
from .pipeline import ProcessOptions, process_episode
from .preprocess import AudioPreprocessError
from .subtitles import segments_to_ass, segments_to_srt
from .translation import TranslationError, build_translation_provider


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "probe":
            return _probe(args)
        if args.command == "process":
            return _process(args)
        if args.command == "gui":
            return _gui(args)
        if args.command == "translate":
            return _translate(args)
        if args.command == "export-subtitle":
            return _export_subtitle(args)
        if args.command == "mux":
            return _mux(args)
    except FFmpegError as exc:
        print(f"ffmpeg error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"file not found: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    except AsrProviderError as exc:
        print(f"asr error: {exc}", file=sys.stderr)
        return 2
    except DiarizationProviderError as exc:
        print(f"diarization error: {exc}", file=sys.stderr)
        return 2
    except AudioPreprocessError as exc:
        print(f"audio preprocess error: {exc}", file=sys.stderr)
        return 2
    except TranslationError as exc:
        print(f"translation error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raw2translated",
        description="Local anime transcription, speaker attribution, and subtitle pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    probe = subparsers.add_parser("probe", help="Inspect media with ffprobe.")
    probe.add_argument("input", type=Path)
    probe.add_argument("--json", action="store_true", help="Print raw ffprobe JSON.")

    process = subparsers.add_parser("process", help="Create pipeline output files for an episode.")
    process.add_argument("input", type=Path)
    process.add_argument("--out", type=Path, default=Path("output"))
    process.add_argument("--lang", default="ja")
    process.add_argument(
        "--asr",
        choices=["none", "faster-whisper"],
        default="none",
        help="ASR provider to use for videos without subtitles.",
    )
    process.add_argument("--asr-model", default="large-v3-turbo", help="Model name for the ASR provider.")
    process.add_argument("--asr-device", default="auto", help="ASR device, for example auto, cpu, or cuda.")
    process.add_argument(
        "--asr-compute-type",
        default="default",
        help="ASR compute type, for example default, int8, float16, or int8_float16.",
    )
    process.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable faster-whisper VAD filtering.",
    )
    process.add_argument("--beam-size", type=int, default=5)
    process.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional Japanese prompt to bias ASR toward names, terms, or anime dialogue.",
    )
    process.add_argument(
        "--preprocess",
        choices=["none", "demucs"],
        default="none",
        help="Optional audio preprocessing before ASR. Diarization still uses analysis.wav.",
    )
    process.add_argument("--preprocess-model", default="htdemucs", help="Demucs model name.")
    process.add_argument(
        "--preprocess-device",
        default="auto",
        help="Preprocessing device, for example auto, cpu, or cuda.",
    )
    process.add_argument("--preprocess-segment", type=int, default=None)
    process.add_argument("--preprocess-shifts", type=int, default=0)
    process.add_argument("--preprocess-jobs", type=int, default=None)
    process.add_argument(
        "--diarization",
        choices=["none", "pyannote"],
        default="none",
        help="Speaker diarization provider for assigning ASR lines to SPEAKER_XX.",
    )
    process.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
        help="Diarization model name. Use pyannote/speaker-diarization-3.1 for legacy pyannote.audio 3.x.",
    )
    process.add_argument(
        "--diarization-token",
        default=None,
        help="Hugging Face token for gated pyannote models. If omitted, HF_TOKEN/HUGGINGFACE_TOKEN is used.",
    )
    process.add_argument(
        "--diarization-device",
        default="auto",
        help="Diarization device, for example auto, cpu, or cuda.",
    )
    process.add_argument("--num-speakers", type=int, default=None)
    process.add_argument("--min-speakers", type=int, default=None)
    process.add_argument("--max-speakers", type=int, default=None)
    process.add_argument(
        "--speaker-min-overlap",
        type=float,
        default=0.2,
        help="Minimum ASR segment overlap ratio required to assign a speaker.",
    )
    process.add_argument(
        "--translate",
        dest="translate_provider",
        choices=["none", "memory", "glossary"],
        default="none",
        help="Local translation provider. Default none keeps the original behavior.",
    )
    process.add_argument(
        "--translation-memory",
        type=Path,
        default=None,
        help="Path to a translation memory JSON file (for --translate memory).",
    )
    process.add_argument(
        "--glossary",
        type=Path,
        default=None,
        help="Path to a glossary JSON file (for --translate glossary).",
    )
    process.add_argument(
        "--target-lang",
        default="zh-CN",
        help="Target translation language tag recorded in the transcript and manifest.",
    )
    process.add_argument(
        "--dry-run",
        action="store_true",
        help="Create metadata and placeholder transcript files without extracting audio.",
    )
    process.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated files when they already exist.",
    )

    translate = subparsers.add_parser(
        "translate",
        help="Translate a transcript JSON into a translated transcript JSON.",
    )
    translate.add_argument("transcript", type=Path)
    translate.add_argument("--out", type=Path, required=True)
    translate.add_argument(
        "--provider",
        choices=["none", "memory", "glossary"],
        default="memory",
        help="Local translation provider.",
    )
    translate.add_argument(
        "--memory",
        type=Path,
        default=None,
        help="Path to a translation memory JSON file (for --provider memory).",
    )
    translate.add_argument(
        "--glossary",
        type=Path,
        default=None,
        help="Path to a glossary JSON file (for --provider glossary).",
    )
    translate.add_argument("--target-lang", default="zh-CN")

    export = subparsers.add_parser("export-subtitle", help="Export ASS or SRT from transcript JSON.")
    export.add_argument("transcript", type=Path)
    export.add_argument("--format", choices=["ass", "srt"], default="ass")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--title", default="raw2translated")
    export.add_argument(
        "--text-mode",
        choices=["original", "translated", "bilingual"],
        default=None,
        help="Subtitle text: original (Japanese), translated (Chinese), or bilingual.",
    )
    export.add_argument(
        "--bilingual",
        action="store_true",
        help="Compatibility alias for --text-mode bilingual.",
    )

    subparsers.add_parser("gui", help="Launch the desktop GUI (Tkinter).")

    mux = subparsers.add_parser("mux", help="Mux a subtitle file into a video container.")
    mux.add_argument("input", type=Path)
    mux.add_argument("subtitle", type=Path)
    mux.add_argument("--out", type=Path, required=True)
    mux.add_argument("--overwrite", action="store_true")

    return parser


def _probe(args: argparse.Namespace) -> int:
    metadata = probe_media(args.input)
    if args.json:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    fmt = metadata.get("format", {})
    streams = metadata.get("streams", [])
    print(f"file: {args.input}")
    print(f"duration: {fmt.get('duration', 'unknown')} seconds")
    print(f"format: {fmt.get('format_long_name', fmt.get('format_name', 'unknown'))}")
    for stream in streams:
        index = stream.get("index")
        codec_type = stream.get("codec_type")
        codec_name = stream.get("codec_name", "unknown")
        language = stream.get("tags", {}).get("language", "und")
        print(f"stream #{index}: {codec_type} {codec_name} lang={language}")
    return 0


def _process(args: argparse.Namespace) -> int:
    result = process_episode(
        args.input,
        ProcessOptions(
            output_dir=args.out,
            language=args.lang,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            asr_provider=args.asr,
            asr_model=args.asr_model,
            asr_device=args.asr_device,
            asr_compute_type=args.asr_compute_type,
            vad_filter=args.vad_filter,
            beam_size=args.beam_size,
            initial_prompt=args.initial_prompt,
            preprocess_provider=args.preprocess,
            preprocess_model=args.preprocess_model,
            preprocess_device=args.preprocess_device,
            preprocess_segment=args.preprocess_segment,
            preprocess_shifts=args.preprocess_shifts,
            preprocess_jobs=args.preprocess_jobs,
            diarization_provider=args.diarization,
            diarization_model=args.diarization_model,
            diarization_token=args.diarization_token,
            diarization_device=args.diarization_device,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            speaker_min_overlap=args.speaker_min_overlap,
            translate_provider=args.translate_provider,
            translation_memory=args.translation_memory,
            glossary=args.glossary,
            target_lang=args.target_lang,
        ),
    )
    print(f"manifest: {result.manifest_path}")
    print(f"raw transcript: {result.raw_transcript_path}")
    print(f"speaker transcript: {result.speaker_transcript_path}")
    if result.diarization_path is not None:
        print(f"diarization: {result.diarization_path}")
    if result.translated_transcript_path is not None:
        print(f"translated transcript: {result.translated_transcript_path}")
    print(f"subtitles: {result.subtitle_ass_path}, {result.subtitle_srt_path}")
    if result.audio_path is not None:
        print(f"analysis audio: {result.audio_path}")
    if result.asr_audio_path is not None and result.asr_audio_path != result.audio_path:
        print(f"asr audio: {result.asr_audio_path}")
    return 0


def _gui(args: argparse.Namespace) -> int:
    try:
        from .gui import launch
    except ImportError as exc:  # pragma: no cover - tkinter missing on this build
        print(f"GUI unavailable (tkinter not installed): {exc}", file=sys.stderr)
        return 2
    return launch()


def _translate(args: argparse.Namespace) -> int:
    transcript = EpisodeTranscript.from_json_file(args.transcript)
    provider = build_translation_provider(
        args.provider,
        memory_path=args.memory,
        glossary_path=args.glossary,
    )
    transcript.segments = provider.translate(transcript.segments, target_lang=args.target_lang)
    transcript.metadata["translation"] = {
        "provider": args.provider,
        "target_lang": args.target_lang,
    }
    transcript.write_json(args.out)
    translated = sum(1 for segment in transcript.segments if segment.is_translated)
    print(f"{args.out} ({translated}/{len(transcript.segments)} segments translated)")
    return 0


def _export_subtitle(args: argparse.Namespace) -> int:
    transcript = EpisodeTranscript.from_json_file(args.transcript)
    text_mode = args.text_mode
    if text_mode is None and args.bilingual:
        text_mode = "bilingual"
    if text_mode is None:
        text_mode = "translated"
    if args.format == "ass":
        text = segments_to_ass(transcript.segments, title=args.title, text_mode=text_mode)
    else:
        text = segments_to_srt(transcript.segments, text_mode=text_mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(args.out)
    return 0


def _mux(args: argparse.Namespace) -> int:
    mux_subtitle(args.input, args.subtitle, args.out, overwrite=args.overwrite)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
