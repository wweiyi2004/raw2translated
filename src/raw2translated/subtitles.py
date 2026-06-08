from __future__ import annotations

from .models import TranscriptSegment

TextMode = str  # one of: "original", "translated", "bilingual"
_VALID_TEXT_MODES = ("original", "translated", "bilingual")


def _resolve_text_mode(text_mode: str | None, bilingual: bool) -> str:
    if text_mode is None:
        return "bilingual" if bilingual else "translated"
    if text_mode not in _VALID_TEXT_MODES:
        raise ValueError(f"unsupported text mode: {text_mode}; expected one of {_VALID_TEXT_MODES}")
    return text_mode


def segments_to_ass(
    segments: list[TranscriptSegment],
    *,
    title: str = "raw2translated",
    bilingual: bool = False,
    text_mode: str | None = None,
) -> str:
    lines = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Microsoft YaHei,48,&H00FFFFFF,&H000000FF,&H001E1E1E,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,0,2,60,60,42,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    mode = _resolve_text_mode(text_mode, bilingual)
    for segment in segments:
        text = _render_text(segment, mode=mode)
        lines.append(
            "Dialogue: 0,{start},{end},Default,{name},0,0,0,,{text}".format(
                start=format_ass_time(segment.start),
                end=format_ass_time(segment.end),
                name=_escape_ass_field(segment.display_speaker),
                text=escape_ass_text(text),
            )
        )
    return "\n".join(lines) + "\n"


def segments_to_srt(
    segments: list[TranscriptSegment],
    *,
    bilingual: bool = False,
    text_mode: str | None = None,
) -> str:
    blocks: list[str] = []
    mode = _resolve_text_mode(text_mode, bilingual)
    for index, segment in enumerate(segments, start=1):
        text = _render_text(segment, mode=mode).replace("\\N", "\n")
        blocks.append(
            f"{index}\n"
            f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, csecs = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{csecs:02d}"


def format_srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def escape_ass_text(text: str) -> str:
    return text.replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _render_text(segment: TranscriptSegment, *, mode: str) -> str:
    source = segment.text_ja or ""
    translation = segment.text_zh or ""

    if mode == "original":
        return source
    if mode == "translated":
        # Fall back to the source so an untranslated line is never rendered as None/empty.
        return translation or source
    # bilingual: Chinese on top, Japanese below. Drop a side when it is missing so we
    # never emit a stray blank line or "None".
    if translation and source:
        return f"{translation}\\N{source}"
    return translation or source


def _escape_ass_field(text: str) -> str:
    return text.replace(",", " ")

