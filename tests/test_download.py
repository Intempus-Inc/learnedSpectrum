"""Offline archive extraction tests; fixtures contain no dataset records."""

import importlib.util
import io
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download.py"
spec = importlib.util.spec_from_file_location("archive_downloader_under_test", SCRIPT)
download = importlib.util.module_from_spec(spec)
# The extraction tests do not use either network/progress dependency.
requests_stub = types.ModuleType("requests")
requests_stub.get = mock.Mock(side_effect=AssertionError("Network use is forbidden"))
tqdm_stub = types.ModuleType("tqdm")
tqdm_stub.tqdm = mock.Mock(side_effect=AssertionError("Downloads are not tested"))
original_path = list(sys.path)
try:
    with mock.patch.dict(sys.modules, {"requests": requests_stub, "tqdm": tqdm_stub}):
        spec.loader.exec_module(download)
finally:
    sys.path[:] = original_path
download.logger.disabled = True


def make_archive(path, entries):
    """Entries are (name, kind, content-or-link-target)."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "w") as archive:
            for name, kind, content in entries:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                modes = {"file": stat.S_IFREG, "dir": stat.S_IFDIR,
                         "symlink": stat.S_IFLNK, "fifo": stat.S_IFIFO,
                         "char": stat.S_IFCHR, "block": stat.S_IFBLK}
                info.external_attr = (modes[kind] | 0o644) << 16
                archive.writestr(info, content)
    else:
        with tarfile.open(path, "w:gz") as archive:
            for name, kind, content in entries:
                info = tarfile.TarInfo(name)
                kinds = {"file": tarfile.REGTYPE, "dir": tarfile.DIRTYPE,
                         "symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE,
                         "fifo": tarfile.FIFOTYPE, "char": tarfile.CHRTYPE,
                         "block": tarfile.BLKTYPE}
                info.type = kinds[kind]
                if kind == "file":
                    payload = content.encode("utf-8")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                else:
                    if kind in ("symlink", "hardlink"):
                        info.linkname = content
                    archive.addfile(info)


class ArchiveExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def paths(self, label, suffix):
        case = self.base / label
        case.mkdir()
        return case / ("archive" + suffix), case / "output"

    def test_extracts_regular_files_and_directories_without_extractall(self):
        for suffix in (".zip", ".tgz", ".tar.gz"):
            with self.subTest(suffix=suffix):
                archive, target = self.paths(suffix.replace(".", "_"), suffix)
                make_archive(archive, [("./", "dir", ""), ("nested/", "dir", ""),
                                       ("nested/data.txt", "file", "plain data"),
                                       ("empty.txt", "file", "")])
                with mock.patch.object(tarfile.TarFile, "extractall", side_effect=AssertionError("Unsafe extraction API")), \
                     mock.patch.object(zipfile.ZipFile, "extractall", side_effect=AssertionError("Unsafe extraction API")):
                    download.extract_archive(archive, target)
                self.assertEqual((target / "nested" / "data.txt").read_text(), "plain data")
                self.assertEqual((target / "empty.txt").read_bytes(), b"")

    def test_rejects_traversal_absolute_and_windows_paths_before_writing_files(self):
        names = ["../output-other/escaped.txt", "nested/../../escaped.txt",
                 "/absolute.txt", "C:/escaped.txt", "C:escaped.txt",
                 "nested\\escaped.txt", "nested/file:stream", "nested/../data.txt",
                 "nested/.. /escaped.txt", "nested/.../escaped.txt", "nested/NUL.txt",
                 "."]
        for suffix in (".zip", ".tgz"):
            for index, name in enumerate(names):
                with self.subTest(suffix=suffix, name=name):
                    archive, target = self.paths("invalid{}{}".format(suffix, index), suffix)
                    make_archive(archive, [("valid.txt", "file", "must not extract"), (name, "file", "bad")])
                    with self.assertRaises(ValueError):
                        download.extract_archive(archive, target)
                    self.assertFalse((target / "valid.txt").exists())
                    self.assertFalse((target.parent / "output-other" / "escaped.txt").exists())
                    self.assertFalse((target.parent / "escaped.txt").exists())

    def test_rejects_archive_links_and_special_files(self):
        for suffix in (".zip", ".tgz"):
            kinds = ["symlink", "fifo", "char", "block"]
            if suffix == ".tgz":
                kinds.append("hardlink")
            for kind in kinds:
                with self.subTest(suffix=suffix, kind=kind):
                    archive, target = self.paths(kind + suffix, suffix)
                    make_archive(archive, [("valid.txt", "file", "must not extract"),
                                           ("link", kind, "../outside"),
                                           ("link/escaped.txt", "file", "bad")])
                    with self.assertRaises(ValueError):
                        download.extract_archive(archive, target)
                    self.assertFalse((target / "valid.txt").exists())
                    self.assertFalse((target.parent / "outside" / "escaped.txt").exists())

    def test_rejects_even_links_to_another_archive_member(self):
        archive, target = self.paths("internal-link", ".tgz")
        make_archive(archive, [("original.txt", "file", "data"),
                               ("alias.txt", "hardlink", "original.txt")])
        with self.assertRaises(ValueError):
            download.extract_archive(archive, target)
        self.assertFalse((target / "original.txt").exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_rejects_existing_destination_symlink(self):
        for suffix in (".zip", ".tgz"):
            with self.subTest(suffix=suffix):
                archive, target = self.paths("root-link" + suffix, suffix)
                outside = target.parent / "outside"
                outside.mkdir()
                target.symlink_to(outside, target_is_directory=True)
                make_archive(archive, [("escaped.txt", "file", "bad")])
                with self.assertRaises(ValueError):
                    download.extract_archive(archive, target)
                self.assertFalse((outside / "escaped.txt").exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_rejects_existing_symlink_parents_and_files(self):
        for suffix in (".zip", ".tgz"):
            for parent_link in (True, False):
                with self.subTest(suffix=suffix, parent_link=parent_link):
                    archive, target = self.paths("existing-link{}{}".format(suffix, parent_link), suffix)
                    target.mkdir()
                    outside = target.parent / "outside"
                    outside.mkdir()
                    protected = outside / "data.txt"
                    protected.write_text("preserved")
                    if parent_link:
                        (target / "nested").symlink_to(outside, target_is_directory=True)
                    else:
                        (target / "nested").mkdir()
                        (target / "nested" / "data.txt").symlink_to(protected)
                    make_archive(archive, [("nested/data.txt", "file", "replacement")])
                    with self.assertRaises(ValueError):
                        download.extract_archive(archive, target)
                    self.assertEqual(protected.read_text(), "preserved")

    def test_overwrites_plain_existing_files(self):
        for suffix in (".zip", ".tgz"):
            with self.subTest(suffix=suffix):
                archive, target = self.paths("plain-existing" + suffix, suffix)
                target.mkdir()
                (target / "data.txt").write_text("old")
                make_archive(archive, [("data.txt", "file", "new")])
                download.extract_archive(archive, target)
                self.assertEqual((target / "data.txt").read_text(), "new")

    @unittest.skipUnless(hasattr(os, "link"), "hard links are unavailable")
    def test_existing_hard_link_does_not_overwrite_its_external_inode(self):
        for suffix in (".zip", ".tgz"):
            with self.subTest(suffix=suffix):
                archive, target = self.paths("existing-hardlink" + suffix, suffix)
                target.mkdir()
                outside = target.parent / "outside.txt"
                outside.write_text("preserved")
                os.link(str(outside), str(target / "data.txt"))
                make_archive(archive, [("data.txt", "file", "replacement")])
                download.extract_archive(archive, target)
                self.assertEqual(outside.read_text(), "preserved")
                self.assertEqual((target / "data.txt").read_text(), "replacement")

    def test_failed_file_copy_preserves_existing_file_and_cleans_temporary_file(self):
        for suffix in (".zip", ".tgz"):
            with self.subTest(suffix=suffix):
                archive, target = self.paths("failed-copy" + suffix, suffix)
                target.mkdir()
                (target / "data.txt").write_text("preserved")
                make_archive(archive, [("data.txt", "file", "replacement")])
                with mock.patch.object(download.shutil, "copyfileobj", side_effect=OSError("Synthetic copy failure")):
                    with self.assertRaises(OSError):
                        download.extract_archive(archive, target)
                self.assertEqual((target / "data.txt").read_text(), "preserved")
                self.assertEqual([item.name for item in target.iterdir()], ["data.txt"])


if __name__ == "__main__":
    unittest.main()
