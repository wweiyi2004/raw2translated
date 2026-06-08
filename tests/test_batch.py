import tempfile
import unittest
from pathlib import Path

from raw2translated.cli import main
from raw2translated.pipeline import ProcessOptions, discover_media, process_batch


def _make_media_dir(root: Path) -> Path:
    media_dir = root / "episodes"
    media_dir.mkdir()
    (media_dir / "ep01.mkv").write_bytes(b"x")
    (media_dir / "ep02.mp4").write_bytes(b"x")
    (media_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    (media_dir / "cover.png").write_bytes(b"x")
    return media_dir


class DiscoverMediaTests(unittest.TestCase):
    def test_discover_only_media_extensions_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media_dir = _make_media_dir(Path(tmp))
            found = discover_media(media_dir)
        names = [p.name for p in found]
        self.assertEqual(names, ["ep01.mkv", "ep02.mp4"])

    def test_discover_missing_dir_raises(self) -> None:
        with self.assertRaises(NotADirectoryError):
            discover_media(Path("does-not-exist-dir"))


class ProcessBatchTests(unittest.TestCase):
    def test_batch_dry_run_creates_per_file_output_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_dir = _make_media_dir(root)
            out_root = root / "out"
            items = process_batch(
                media_dir,
                out_root,
                ProcessOptions(output_dir=out_root, dry_run=True),
            )
            self.assertEqual(len(items), 2)
            for item in items:
                self.assertIsNone(item.error)
                self.assertTrue((item.output_dir / "manifest.json").exists())
            self.assertTrue((out_root / "ep01").is_dir())
            self.assertTrue((out_root / "ep02").is_dir())


class BatchCliTests(unittest.TestCase):
    def test_cli_batch_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_dir = _make_media_dir(root)
            code = main(["batch", str(media_dir), "--out", str(root / "out"), "--dry-run"])
        self.assertEqual(code, 0)

    def test_cli_batch_empty_dir_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            code = main(["batch", str(empty), "--out", str(Path(tmp) / "out"), "--dry-run"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
