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

    def rows(self) -> list[SegmentRow]:
        if self.transcript is None:
            return []
        return [
            SegmentRow(
                index=i,
                start=segment.start,
                end=segment.end,
                speaker=segment.display_speaker,
                source=segment.text_ja,
                translation=segment.text_zh or "",
                notes=", ".join(segment.notes),
            )
            for i, segment in enumerate(self.transcript.segments)
        ]

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
    ) -> tuple[int, int]:
        """Translate the loaded transcript in place. Returns (translated, total)."""
        if self.transcript is None:
            raise ValueError("no transcript loaded")
        translator = build_translation_provider(
            provider,
            memory_path=memory_path,
            glossary_path=glossary_path,
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

    # -- process ---------------------------------------------------------

    def run_process(self, input_path: Path, options: ProcessOptions) -> ProcessResult:
        """Run the full pipeline and load its best transcript into the editor."""
        result = process_episode(Path(input_path), options)
        best = result.translated_transcript_path or result.speaker_transcript_path
        if best is not None and Path(best).exists():
            self.load_transcript(Path(best))
        return result


def launch(argv: list[str] | None = None) -> int:  # pragma: no cover - requires a display
    """Build and run the Tkinter window. Returns a process exit code."""
    import queue
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    controller = GuiController()

    root = tk.Tk()
    root.title("raw2translated")
    root.geometry("960x640")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    log_queue: queue.Queue[str] = queue.Queue()

    # ---- Process tab ----
    process_tab = ttk.Frame(notebook)
    notebook.add(process_tab, text="Process")

    state = {
        "input": tk.StringVar(),
        "output": tk.StringVar(value="output"),
        "asr": tk.StringVar(value="none"),
        "translate": tk.StringVar(value="none"),
        "memory": tk.StringVar(),
        "glossary": tk.StringVar(),
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
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label, width=16).pack(side="left")
        ttk.Entry(frame, textvariable=var).pack(side="left", fill="x", expand=True)
        if picker is not None:
            ttk.Button(frame, text="...", width=3, command=picker).pack(side="left", padx=2)

    _row(process_tab, "Input media", state["input"], lambda: _pick_file(state["input"]))
    _row(process_tab, "Output dir", state["output"], lambda: _pick_dir(state["output"]))

    asr_frame = ttk.Frame(process_tab)
    asr_frame.pack(fill="x", pady=2)
    ttk.Label(asr_frame, text="ASR", width=16).pack(side="left")
    ttk.Combobox(
        asr_frame,
        textvariable=state["asr"],
        values=["none", "faster-whisper"],
        state="readonly",
    ).pack(side="left")

    tr_frame = ttk.Frame(process_tab)
    tr_frame.pack(fill="x", pady=2)
    ttk.Label(tr_frame, text="Translate", width=16).pack(side="left")
    ttk.Combobox(
        tr_frame,
        textvariable=state["translate"],
        values=["none", "memory", "glossary"],
        state="readonly",
    ).pack(side="left")

    _row(process_tab, "Translation mem", state["memory"], lambda: _pick_file(state["memory"]))
    _row(process_tab, "Glossary", state["glossary"], lambda: _pick_file(state["glossary"]))
    _row(process_tab, "Target lang", state["target_lang"])

    ttk.Checkbutton(process_tab, text="Dry run", variable=state["dry_run"]).pack(anchor="w", pady=2)

    log_text = tk.Text(process_tab, height=10, wrap="word")
    log_text.pack(fill="both", expand=True, pady=4)

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

    run_button = ttk.Button(process_tab, text="Run pipeline")

    def _run_pipeline() -> None:
        if not state["input"].get():
            messagebox.showerror("raw2translated", "Please choose an input media file.")
            return
        options = ProcessOptions(
            output_dir=Path(state["output"].get() or "output"),
            dry_run=state["dry_run"].get(),
            asr_provider=state["asr"].get(),
            translate_provider=state["translate"].get(),
            translation_memory=Path(state["memory"].get()) if state["memory"].get() else None,
            glossary=Path(state["glossary"].get()) if state["glossary"].get() else None,
            target_lang=state["target_lang"].get() or "zh-CN",
        )
        run_button.config(state="disabled")
        log_queue.put(f"Running pipeline on {state['input'].get()} ...")

        def _worker() -> None:
            try:
                result = controller.run_process(Path(state["input"].get()), options)
                log_queue.put(f"manifest: {result.manifest_path}")
                if result.translated_transcript_path:
                    log_queue.put(f"translated: {result.translated_transcript_path}")
                log_queue.put("Done. Loading transcript into the Editor tab.")
                root.after(0, _refresh_editor)
            except Exception as exc:  # noqa: BLE001 - surface any failure to the log
                log_queue.put(f"ERROR: {exc}")
            finally:
                root.after(0, lambda: run_button.config(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    run_button.config(command=_run_pipeline)
    run_button.pack(anchor="e")

    # ---- Editor tab ----
    editor_tab = ttk.Frame(notebook)
    notebook.add(editor_tab, text="Editor")

    columns = ("index", "start", "end", "speaker", "source", "translation", "notes")
    tree = ttk.Treeview(editor_tab, columns=columns, show="headings", height=14)
    widths = {
        "index": 40,
        "start": 70,
        "end": 70,
        "speaker": 110,
        "source": 260,
        "translation": 260,
        "notes": 110,
    }
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=widths[col], anchor="w")
    tree.pack(fill="both", expand=True, pady=4)

    def _refresh_editor() -> None:
        tree.delete(*tree.get_children())
        for r in controller.rows():
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
    edit_frame.pack(fill="x", pady=4)
    ttk.Label(edit_frame, text="Translation:").pack(side="left")
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
    ttk.Button(edit_frame, text="Apply", command=_apply_edit).pack(side="left")

    button_frame = ttk.Frame(editor_tab)
    button_frame.pack(fill="x", pady=4)

    def _open_transcript() -> None:
        path = filedialog.askopenfilename(filetypes=[("Transcript JSON", "*.json"), ("All", "*.*")])
        if path:
            controller.load_transcript(Path(path))
            _refresh_editor()

    def _save_transcript() -> None:
        if controller.transcript is None:
            messagebox.showinfo("raw2translated", "Load a transcript first.")
            return
        path = controller.transcript_path
        if path is None:
            chosen = filedialog.asksaveasfilename(defaultextension=".json")
            if not chosen:
                return
            path = Path(chosen)
        controller.save_transcript(path)
        messagebox.showinfo("raw2translated", f"Saved {path}")

    ttk.Button(button_frame, text="Open transcript...", command=_open_transcript).pack(side="left")
    ttk.Button(button_frame, text="Save transcript", command=_save_transcript).pack(side="left", padx=4)

    # ---- Export tab ----
    export_tab = ttk.Frame(notebook)
    notebook.add(export_tab, text="Export")

    export_state = {
        "format": tk.StringVar(value="ass"),
        "text_mode": tk.StringVar(value="bilingual"),
        "out": tk.StringVar(value="output/subtitles/episode.ass"),
    }

    fmt_frame = ttk.Frame(export_tab)
    fmt_frame.pack(fill="x", pady=2)
    ttk.Label(fmt_frame, text="Format", width=16).pack(side="left")
    ttk.Combobox(
        fmt_frame,
        textvariable=export_state["format"],
        values=["ass", "srt"],
        state="readonly",
    ).pack(side="left")

    mode_frame = ttk.Frame(export_tab)
    mode_frame.pack(fill="x", pady=2)
    ttk.Label(mode_frame, text="Text mode", width=16).pack(side="left")
    ttk.Combobox(
        mode_frame,
        textvariable=export_state["text_mode"],
        values=["original", "translated", "bilingual"],
        state="readonly",
    ).pack(side="left")

    def _pick_export_out() -> None:
        chosen = filedialog.asksaveasfilename(
            defaultextension=f".{export_state['format'].get()}"
        )
        if chosen:
            export_state["out"].set(chosen)

    _row(export_tab, "Output file", export_state["out"], _pick_export_out)

    def _do_export() -> None:
        if controller.transcript is None:
            messagebox.showinfo("raw2translated", "Load or generate a transcript first.")
            return
        try:
            out = controller.export_subtitle(
                Path(export_state["out"].get()),
                fmt=export_state["format"].get(),
                text_mode=export_state["text_mode"].get(),
            )
            messagebox.showinfo("raw2translated", f"Exported {out}")
        except Exception as exc:  # noqa: BLE001 - surface to the user
            messagebox.showerror("raw2translated", str(exc))

    ttk.Button(export_tab, text="Export subtitle", command=_do_export).pack(anchor="e", pady=4)

    _poll_log()
    root.mainloop()
    return 0
