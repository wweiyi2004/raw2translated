"""Local-first desktop GUI for raw2translated.

The GUI is split into two layers:

* :class:`GuiController` holds all logic and is completely free of Tkinter, so it
  can be unit-tested headless (no display, no `_tkinter`).
* :func:`launch` builds the Tkinter window and wires its widgets to a controller.
  Tkinter is imported lazily inside ``launch`` so importing this module never
  requires a display or the ``_tkinter`` extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import EpisodeTranscript, TranscriptSegment
from .pipeline import ProcessOptions, ProcessResult, process_episode
from .subtitles import segments_to_ass, segments_to_srt
from .translation import UNTRANSLATED_NOTE, build_translation_provider

# Dark palette used by the custom fallback theme (when sv-ttk is not installed)
# and for the plain ``tk`` widgets (e.g. the log Text) that ttk styling can't reach.
DARK_PALETTE = {
    "bg": "#1e1e2e",
    "surface": "#262637",
    "surface_alt": "#2e2e42",
    "border": "#3a3a52",
    "text": "#cdd6f4",
    "subtext": "#a6adc8",
    "accent": "#7c5cff",
    "accent_active": "#9277ff",
    "accent_text": "#ffffff",
    "field": "#2a2a3c",
    "selection": "#3a3358",
}

# Microsoft YaHei UI renders both Chinese and Latin crisply on Windows.
UI_FONT = ("Microsoft YaHei UI", 10)
UI_FONT_BOLD = ("Microsoft YaHei UI", 10, "bold")
UI_FONT_TITLE = ("Microsoft YaHei UI", 15, "bold")
UI_FONT_SECTION = ("Microsoft YaHei UI", 11, "bold")
UI_FONT_MONO = ("Consolas", 10)


@dataclass
class SegmentRow:
    """A flat, display-friendly view of one segment for the editor table."""

    index: int
    start: float
    end: float
    speaker: str
    source: str
    translation: str
    notes: str


class GuiController:
    """Tk-free logic backing the desktop GUI.

    Every method here is safe to call from unit tests; nothing touches Tkinter.
    """

    def __init__(self) -> None:
        self.transcript: EpisodeTranscript | None = None
        self.transcript_path: Path | None = None

    # -- transcript loading / saving ------------------------------------

    def load_transcript(self, path: Path) -> EpisodeTranscript:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        self.transcript = EpisodeTranscript.from_json_file(path)
        self.transcript_path = path
        return self.transcript

    def set_transcript(self, transcript: EpisodeTranscript, path: Path | None = None) -> None:
        self.transcript = transcript
        self.transcript_path = Path(path) if path is not None else None

    def save_transcript(self, path: Path | None = None) -> Path:
        if self.transcript is None:
            raise ValueError("no transcript loaded")
        target = Path(path) if path is not None else self.transcript_path
        if target is None:
            raise ValueError("no path to save to")
        self.transcript.write_json(target)
        self.transcript_path = target
        return target

    # -- editor ----------------------------------------------------------

    def is_flagged(self, segment: TranscriptSegment, *, confidence_threshold: float | None = None) -> bool:
        """A segment worth reviewing: untranslated, carrying notes, or low confidence."""
        if not segment.is_translated:
            return True
        if segment.notes:
            return True
        if confidence_threshold is not None:
            for value in (segment.speaker_confidence, segment.asr_confidence):
                if value is not None and value < confidence_threshold:
                    return True
        return False

    def rows(
        self,
        *,
        only_flagged: bool = False,
        confidence_threshold: float | None = None,
    ) -> list[SegmentRow]:
        if self.transcript is None:
            return []
        rows: list[SegmentRow] = []
        for i, segment in enumerate(self.transcript.segments):
            if only_flagged and not self.is_flagged(segment, confidence_threshold=confidence_threshold):
                continue
            rows.append(
                SegmentRow(
                    index=i,
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.display_speaker,
                    source=segment.text_ja,
                    translation=segment.text_zh or "",
                    notes=", ".join(segment.notes),
                )
            )
        return rows

    def segment(self, index: int) -> TranscriptSegment:
        if self.transcript is None:
            raise ValueError("no transcript loaded")
        return self.transcript.segments[index]

    def update_translation(self, index: int, text: str) -> None:
        """Set a segment's Chinese text from the editor.

        An empty edit clears the translation; a non-empty edit drops the
        ``untranslated`` flag so the line stops being reported as missing.
        """
        segment = self.segment(index)
        cleaned = text.strip()
        if cleaned:
            segment.text_zh = cleaned
            if UNTRANSLATED_NOTE in segment.notes:
                segment.notes.remove(UNTRANSLATED_NOTE)
        else:
            segment.text_zh = None

    # -- translation -----------------------------------------------------

    def run_translate(
        self,
        provider: str,
        *,
        memory_path: Path | None = None,
        glossary_path: Path | None = None,
        target_lang: str = "zh-CN",
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> tuple[int, int]:
        """Translate the loaded transcript in place. Returns (translated, total)."""
        if self.transcript is None:
            raise ValueError("no transcript loaded")
        translator = build_translation_provider(
            provider,
            memory_path=memory_path,
            glossary_path=glossary_path,
            model=model,
            api_base=api_base,
            api_key=api_key,
        )
        self.transcript.segments = translator.translate(
            self.transcript.segments,
            target_lang=target_lang,
        )
        self.transcript.metadata["translation"] = {
            "provider": provider,
            "target_lang": target_lang,
        }
        total = len(self.transcript.segments)
        translated = sum(1 for s in self.transcript.segments if s.is_translated)
        return translated, total

    # -- character naming ------------------------------------------------

    def apply_characters(self, map_path: Path) -> int:
        """Apply a speaker -> character map to the loaded transcript.

        Returns the number of segments that were assigned a character.
        """
        from .characters import apply_character_map, load_character_map

        if self.transcript is None:
            raise ValueError("no transcript loaded")
        mapping = load_character_map(Path(map_path))
        return apply_character_map(self.transcript.segments, mapping)

    # -- subtitle export -------------------------------------------------

    def export_subtitle(
        self,
        path: Path,
        *,
        fmt: str = "ass",
        text_mode: str = "bilingual",
        title: str = "raw2translated",
    ) -> Path:
        if self.transcript is None:
            raise ValueError("no transcript loaded")
        path = Path(path)
        if fmt == "ass":
            text = segments_to_ass(self.transcript.segments, title=title, text_mode=text_mode)
        elif fmt == "srt":
            text = segments_to_srt(self.transcript.segments, text_mode=text_mode)
        else:
            raise ValueError(f"unsupported subtitle format: {fmt}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    # -- per-line playback ----------------------------------------------

    def build_play_command(
        self,
        index: int,
        *,
        player: str = "ffplay",
        media_path: Path | str | None = None,
        pad: float = 0.0,
    ) -> list[str]:
        """Build an ``ffplay`` command that plays one segment's time range.

        The GUI runs this with :mod:`subprocess`; the command is built here so it
        can be unit-tested without launching a player.
        """
        segment = self.segment(index)
        media = media_path or (self.transcript.media_path if self.transcript else None)
        if not media:
            raise ValueError("no media path is associated with this transcript")
        start = max(0.0, segment.start - pad)
        duration = max(0.0, (segment.end - segment.start) + 2 * pad)
        return [
            player,
            "-autoexit",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            str(media),
        ]

    # -- mux -------------------------------------------------------------

    def mux(
        self,
        subtitle_path: Path,
        output_path: Path,
        *,
        input_path: Path | str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Mux a subtitle file into the (loaded or given) media container."""
        from .ffmpeg import mux_subtitle

        media = input_path or (self.transcript.media_path if self.transcript else None)
        if not media:
            raise ValueError("no input media to mux into")
        return mux_subtitle(
            Path(media),
            Path(subtitle_path),
            Path(output_path),
            overwrite=overwrite,
        )

    # -- process ---------------------------------------------------------

    def run_process(self, input_path: Path, options: ProcessOptions) -> ProcessResult:
        """Run the full pipeline and load its best transcript into the editor."""
        result = process_episode(Path(input_path), options)
        best = result.translated_transcript_path or result.speaker_transcript_path
        if best is not None and Path(best).exists():
            self.load_transcript(Path(best))
        return result


def _enable_dpi_awareness() -> None:  # pragma: no cover - Windows display only
    """Tell Windows we handle DPI ourselves so text is rendered crisp, not stretched.

    Without this, Windows bitmap-stretches the window on high-DPI screens, which is
    what makes default Tkinter text look blurry. Must run before ``tk.Tk()``.
    """
    import sys

    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _install_theme(root) -> str:  # pragma: no cover - requires a display
    """Apply a modern theme. Prefer sv-ttk (Windows 11 look), else a dark fallback.

    Returns the name of the theme that was applied.
    """
    try:
        import sv_ttk

        sv_ttk.set_theme("dark")
        return "sv-ttk"
    except Exception:
        _apply_dark_theme(root)
        return "custom-dark"


def _apply_dark_theme(root) -> None:  # pragma: no cover - requires a display
    from tkinter import ttk

    p = DARK_PALETTE
    root.configure(bg=p["bg"])
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=p["bg"], foreground=p["text"], font=UI_FONT)
    style.configure("TFrame", background=p["bg"])
    style.configure("TLabel", background=p["bg"], foreground=p["text"])
    style.configure("Subtle.TLabel", background=p["bg"], foreground=p["subtext"])
    style.configure("Title.TLabel", background=p["bg"], foreground=p["text"], font=UI_FONT_TITLE)

    style.configure(
        "TButton",
        background=p["surface_alt"],
        foreground=p["text"],
        bordercolor=p["border"],
        focuscolor=p["accent"],
        relief="flat",
        padding=(12, 7),
    )
    style.map(
        "TButton",
        background=[("active", p["border"]), ("pressed", p["border"])],
        foreground=[("disabled", p["subtext"])],
    )
    style.configure(
        "Accent.TButton",
        background=p["accent"],
        foreground=p["accent_text"],
        bordercolor=p["accent"],
        relief="flat",
        padding=(14, 8),
        font=UI_FONT_BOLD,
    )
    style.map(
        "Accent.TButton",
        background=[("active", p["accent_active"]), ("pressed", p["accent_active"])],
        foreground=[("disabled", p["subtext"])],
    )

    style.configure("TNotebook", background=p["bg"], bordercolor=p["bg"], tabmargins=(6, 6, 6, 0))
    style.configure(
        "TNotebook.Tab",
        background=p["bg"],
        foreground=p["subtext"],
        padding=(16, 8),
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", p["surface"])],
        foreground=[("selected", p["text"])],
    )

    for widget in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(
            widget,
            fieldbackground=p["field"],
            background=p["field"],
            foreground=p["text"],
            bordercolor=p["border"],
            insertcolor=p["text"],
            arrowcolor=p["subtext"],
            padding=6,
        )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", p["field"])],
        foreground=[("readonly", p["text"])],
    )

    style.configure(
        "TCheckbutton",
        background=p["bg"],
        foreground=p["text"],
        focuscolor=p["accent"],
    )
    style.map("TCheckbutton", background=[("active", p["bg"])])

    style.configure("TSeparator", background=p["border"])

    style.configure(
        "Treeview",
        background=p["surface"],
        fieldbackground=p["surface"],
        foreground=p["text"],
        bordercolor=p["border"],
        borderwidth=0,
        rowheight=26,
    )
    style.map("Treeview", background=[("selected", p["selection"])], foreground=[("selected", p["text"])])
    style.configure(
        "Treeview.Heading",
        background=p["surface_alt"],
        foreground=p["subtext"],
        relief="flat",
        font=UI_FONT_BOLD,
        padding=6,
    )
    style.map("Treeview.Heading", background=[("active", p["border"])])


def launch(argv: list[str] | None = None) -> int:  # pragma: no cover - requires a display
    """Build and run the Tkinter window. Returns a process exit code."""
    import queue
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    _enable_dpi_awareness()

    controller = GuiController()

    root = tk.Tk()
    root.title("raw2translated")
    root.geometry("1040x720")
    root.minsize(860, 580)

    # Match Tk's point-to-pixel scaling to the real screen DPI so fonts are crisp
    # and correctly sized on high-DPI displays.
    try:
        root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72.0)
    except Exception:
        pass

    theme = _install_theme(root)

    # Custom style names used below, configured for whichever theme is active.
    style = ttk.Style(root)
    style.configure("Title.TLabel", font=UI_FONT_TITLE)
    style.configure("Subtle.TLabel", font=UI_FONT, foreground=DARK_PALETTE["subtext"])
    style.configure("Section.TLabel", font=UI_FONT_SECTION, foreground=DARK_PALETTE["accent"])
    # Use the CJK-capable font and a comfortable row height for the editor table,
    # overriding whatever the active theme set.
    style.configure("Treeview", font=UI_FONT, rowheight=30)
    style.configure("Treeview.Heading", font=UI_FONT_BOLD)

    def _section(parent, title: str, *, expand: bool = False):
        """A titled group: a bold heading followed by an indented content frame."""
        ttk.Label(parent, text=title, style="Section.TLabel").pack(anchor="w", pady=(10, 2))
        body = ttk.Frame(parent)
        body.pack(fill="both" if expand else "x", expand=expand, padx=(2, 0))
        return body

    header = ttk.Frame(root, padding=(20, 16, 20, 6))
    header.pack(fill="x")
    ttk.Label(header, text="raw2translated", style="Title.TLabel").pack(side="left")
    ttk.Label(
        header,
        text="本地优先 · 日语动画转录 · 翻译 · 字幕",
        style="Subtle.TLabel",
    ).pack(side="left", padx=12)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=16, pady=(4, 16))

    log_queue: queue.Queue[str] = queue.Queue()
    log_queue.put(f"主题：{theme}（装 sv-ttk 可获得 Windows 11 风格）")

    # ---- Process tab ----
    process_tab = ttk.Frame(notebook, padding=20)
    notebook.add(process_tab, text="  处理  ")

    state = {
        "input": tk.StringVar(),
        "output": tk.StringVar(value="output"),
        "asr": tk.StringVar(value="none"),
        "alignment": tk.StringVar(value="none"),
        "translate": tk.StringVar(value="none"),
        "memory": tk.StringVar(),
        "glossary": tk.StringVar(),
        "characters": tk.StringVar(),
        "target_lang": tk.StringVar(value="zh-CN"),
        "dry_run": tk.BooleanVar(value=True),
    }

    def _pick_file(var: tk.StringVar) -> None:
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _pick_dir(var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _row(parent: ttk.Frame, label: str, var: tk.StringVar, picker=None) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=3)
        ttk.Label(frame, text=label, width=12).pack(side="left")
        ttk.Entry(frame, textvariable=var).pack(side="left", fill="x", expand=True)
        if picker is not None:
            ttk.Button(frame, text="浏览…", width=6, command=picker).pack(side="left", padx=(6, 0))

    def _combo_row(parent: ttk.Frame, label: str, var: tk.StringVar, values: list[str]) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=3)
        ttk.Label(frame, text=label, width=12).pack(side="left")
        ttk.Combobox(frame, textvariable=var, values=values, state="readonly", width=22).pack(side="left")

    io_box = _section(process_tab, "输入 / 输出")
    _row(io_box, "输入视频", state["input"], lambda: _pick_file(state["input"]))
    _row(io_box, "输出目录", state["output"], lambda: _pick_dir(state["output"]))

    pipe_box = _section(process_tab, "流水线")
    _combo_row(pipe_box, "语音识别", state["asr"], ["none", "faster-whisper"])
    _combo_row(pipe_box, "强制对齐", state["alignment"], ["none", "whisperx"])

    tr_box = _section(process_tab, "翻译与角色")
    _combo_row(tr_box, "翻译方式", state["translate"], ["none", "memory", "glossary"])
    _row(tr_box, "翻译记忆", state["memory"], lambda: _pick_file(state["memory"]))
    _row(tr_box, "术语表", state["glossary"], lambda: _pick_file(state["glossary"]))
    _row(tr_box, "角色映射", state["characters"], lambda: _pick_file(state["characters"]))
    _row(tr_box, "目标语言", state["target_lang"])

    ttk.Checkbutton(
        process_tab,
        text="空跑（dry run，不解码音频，仅生成占位文件）",
        variable=state["dry_run"],
    ).pack(anchor="w", pady=(12, 4))

    log_box = _section(process_tab, "运行日志", expand=True)
    log_text = tk.Text(
        log_box,
        height=9,
        wrap="word",
        bg=DARK_PALETTE["field"],
        fg=DARK_PALETTE["text"],
        insertbackground=DARK_PALETTE["text"],
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=DARK_PALETTE["border"],
        padx=10,
        pady=8,
        font=UI_FONT_MONO,
    )
    log_text.pack(fill="both", expand=True)

    def _log(message: str) -> None:
        log_text.insert("end", message + "\n")
        log_text.see("end")

    def _poll_log() -> None:
        try:
            while True:
                _log(log_queue.get_nowait())
        except queue.Empty:
            pass
        root.after(100, _poll_log)

    run_button = ttk.Button(process_tab, text="▶  运行流水线", style="Accent.TButton")

    def _run_pipeline() -> None:
        if not state["input"].get():
            messagebox.showerror("raw2translated", "请先选择一个输入视频文件。")
            return
        options = ProcessOptions(
            output_dir=Path(state["output"].get() or "output"),
            dry_run=state["dry_run"].get(),
            asr_provider=state["asr"].get(),
            alignment_provider=state["alignment"].get(),
            translate_provider=state["translate"].get(),
            translation_memory=Path(state["memory"].get()) if state["memory"].get() else None,
            glossary=Path(state["glossary"].get()) if state["glossary"].get() else None,
            character_map=Path(state["characters"].get()) if state["characters"].get() else None,
            target_lang=state["target_lang"].get() or "zh-CN",
        )
        run_button.config(state="disabled")
        log_queue.put(f"开始处理：{state['input'].get()} …")

        def _worker() -> None:
            try:
                result = controller.run_process(Path(state["input"].get()), options)
                log_queue.put(f"清单：{result.manifest_path}")
                if result.translated_transcript_path:
                    log_queue.put(f"译文转录：{result.translated_transcript_path}")
                log_queue.put("完成。已把转录载入「字幕编辑」标签页。")
                root.after(0, _refresh_editor)
            except Exception as exc:  # noqa: BLE001 - surface any failure to the log
                log_queue.put(f"错误：{exc}")
            finally:
                root.after(0, lambda: run_button.config(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    run_button.config(command=_run_pipeline)
    run_button.pack(anchor="e", pady=(12, 0))

    # ---- Editor tab ----
    editor_tab = ttk.Frame(notebook, padding=20)
    notebook.add(editor_tab, text="  字幕编辑  ")

    only_flagged = tk.BooleanVar(value=False)
    filter_frame = ttk.Frame(editor_tab)
    filter_frame.pack(fill="x", pady=(0, 6))
    ttk.Checkbutton(
        filter_frame,
        text="仅显示待处理（未翻译 / 有备注 / 低置信度）",
        variable=only_flagged,
        command=lambda: _refresh_editor(),
    ).pack(side="left")

    columns = ("index", "start", "end", "speaker", "source", "translation", "notes")
    headers = {
        "index": "#",
        "start": "开始",
        "end": "结束",
        "speaker": "说话人",
        "source": "原文",
        "translation": "译文",
        "notes": "备注",
    }
    tree = ttk.Treeview(editor_tab, columns=columns, show="headings", height=14)
    widths = {
        "index": 44,
        "start": 72,
        "end": 72,
        "speaker": 110,
        "source": 280,
        "translation": 280,
        "notes": 120,
    }
    for col in columns:
        tree.heading(col, text=headers[col])
        tree.column(col, width=widths[col], anchor="w")
    tree.pack(fill="both", expand=True, pady=4)

    def _refresh_editor() -> None:
        tree.delete(*tree.get_children())
        for r in controller.rows(only_flagged=only_flagged.get(), confidence_threshold=0.5):
            tree.insert(
                "",
                "end",
                iid=str(r.index),
                values=(
                    r.index,
                    f"{r.start:.2f}",
                    f"{r.end:.2f}",
                    r.speaker,
                    r.source,
                    r.translation,
                    r.notes,
                ),
            )

    edit_frame = ttk.Frame(editor_tab)
    edit_frame.pack(fill="x", pady=6)
    ttk.Label(edit_frame, text="译文：").pack(side="left")
    edit_var = tk.StringVar()
    edit_entry = ttk.Entry(edit_frame, textvariable=edit_var)
    edit_entry.pack(side="left", fill="x", expand=True, padx=4)

    def _on_select(_event=None) -> None:
        selection = tree.selection()
        if not selection:
            return
        index = int(selection[0])
        edit_var.set(controller.segment(index).text_zh or "")

    def _apply_edit() -> None:
        selection = tree.selection()
        if not selection:
            return
        index = int(selection[0])
        controller.update_translation(index, edit_var.get())
        _refresh_editor()
        tree.selection_set(str(index))

    tree.bind("<<TreeviewSelect>>", _on_select)
    ttk.Button(edit_frame, text="应用", command=_apply_edit, style="Accent.TButton").pack(side="left")

    button_frame = ttk.Frame(editor_tab)
    button_frame.pack(fill="x", pady=4)

    def _open_transcript() -> None:
        path = filedialog.askopenfilename(filetypes=[("转录 JSON", "*.json"), ("所有文件", "*.*")])
        if path:
            controller.load_transcript(Path(path))
            _refresh_editor()

    def _save_transcript() -> None:
        if controller.transcript is None:
            messagebox.showinfo("raw2translated", "请先载入一个转录文件。")
            return
        path = controller.transcript_path
        if path is None:
            chosen = filedialog.asksaveasfilename(defaultextension=".json")
            if not chosen:
                return
            path = Path(chosen)
        controller.save_transcript(path)
        messagebox.showinfo("raw2translated", f"已保存 {path}")

    def _play_selected() -> None:
        import subprocess

        selection = tree.selection()
        if not selection:
            return
        index = int(selection[0])
        try:
            command = controller.build_play_command(index, pad=0.25)
        except ValueError as exc:
            messagebox.showinfo("raw2translated", str(exc))
            return
        try:
            subprocess.Popen(command)  # noqa: S603 - command built from local data
        except FileNotFoundError:
            messagebox.showerror(
                "raw2translated",
                "未找到 ffplay。请安装 ffmpeg（自带 ffplay）以试听音频。",
            )

    def _load_characters() -> None:
        if controller.transcript is None:
            messagebox.showinfo("raw2translated", "请先载入一个转录文件。")
            return
        path = filedialog.askopenfilename(filetypes=[("角色映射 JSON", "*.json"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            assigned = controller.apply_characters(Path(path))
        except Exception as exc:  # noqa: BLE001 - surface to the user
            messagebox.showerror("raw2translated", str(exc))
            return
        _refresh_editor()
        messagebox.showinfo("raw2translated", f"已为 {assigned} 条字幕分配角色。")

    ttk.Button(button_frame, text="打开转录…", command=_open_transcript).pack(side="left")
    ttk.Button(button_frame, text="保存转录", command=_save_transcript).pack(side="left", padx=6)
    ttk.Button(button_frame, text="载入角色映射", command=_load_characters).pack(side="left")
    ttk.Button(button_frame, text="试听选中", command=_play_selected).pack(side="left", padx=6)

    # ---- Export tab ----
    export_tab = ttk.Frame(notebook, padding=20)
    notebook.add(export_tab, text="  导出  ")

    export_state = {
        "format": tk.StringVar(value="ass"),
        "text_mode": tk.StringVar(value="bilingual"),
        "out": tk.StringVar(value="output/subtitles/episode.ass"),
    }

    sub_box = _section(export_tab, "导出字幕")

    def _combo_row_ex(parent, label: str, var: tk.StringVar, values: list[str]) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=3)
        ttk.Label(frame, text=label, width=12).pack(side="left")
        ttk.Combobox(frame, textvariable=var, values=values, state="readonly", width=22).pack(side="left")

    _combo_row_ex(sub_box, "格式", export_state["format"], ["ass", "srt"])
    _combo_row_ex(sub_box, "文本模式", export_state["text_mode"], ["original", "translated", "bilingual"])

    def _pick_export_out() -> None:
        chosen = filedialog.asksaveasfilename(
            defaultextension=f".{export_state['format'].get()}"
        )
        if chosen:
            export_state["out"].set(chosen)

    _row(sub_box, "输出文件", export_state["out"], _pick_export_out)

    def _do_export() -> None:
        if controller.transcript is None:
            messagebox.showinfo("raw2translated", "请先载入或生成一个转录文件。")
            return
        try:
            out = controller.export_subtitle(
                Path(export_state["out"].get()),
                fmt=export_state["format"].get(),
                text_mode=export_state["text_mode"].get(),
            )
            messagebox.showinfo("raw2translated", f"已导出 {out}")
        except Exception as exc:  # noqa: BLE001 - surface to the user
            messagebox.showerror("raw2translated", str(exc))

    ttk.Button(sub_box, text="导出字幕", command=_do_export, style="Accent.TButton").pack(
        anchor="e", pady=(8, 0)
    )

    mux_box = _section(export_tab, "封装进视频（需 ffmpeg）")
    ttk.Label(
        mux_box,
        text="把字幕轨道封装进视频容器，原视频不重新编码。",
        style="Subtle.TLabel",
    ).pack(anchor="w", pady=(0, 4))

    mux_state = {
        "input": tk.StringVar(),
        "subtitle": tk.StringVar(),
        "out": tk.StringVar(value="output/episode.muxed.mkv"),
    }
    _row(mux_box, "输入视频", mux_state["input"], lambda: _pick_file(mux_state["input"]))
    _row(mux_box, "字幕文件", mux_state["subtitle"], lambda: _pick_file(mux_state["subtitle"]))
    _row(mux_box, "输出文件", mux_state["out"])

    def _do_mux() -> None:
        if not mux_state["subtitle"].get():
            messagebox.showinfo("raw2translated", "请选择要封装的字幕文件。")
            return
        try:
            out = controller.mux(
                Path(mux_state["subtitle"].get()),
                Path(mux_state["out"].get()),
                input_path=Path(mux_state["input"].get()) if mux_state["input"].get() else None,
                overwrite=True,
            )
            messagebox.showinfo("raw2translated", f"已封装 {out}")
        except FileNotFoundError:
            messagebox.showerror(
                "raw2translated",
                "未找到 ffmpeg 或输入文件。请安装 ffmpeg 并检查路径。",
            )
        except Exception as exc:  # noqa: BLE001 - surface to the user
            messagebox.showerror("raw2translated", str(exc))

    ttk.Button(mux_box, text="封装进视频", command=_do_mux).pack(anchor="e", pady=(8, 0))

    _poll_log()
    root.mainloop()
    return 0
