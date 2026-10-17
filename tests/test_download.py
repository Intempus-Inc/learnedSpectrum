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

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_resolved_containment_catches_aliases_not_reported_as_symlinks(self):
        archive, target = self.paths("resolved-alias", ".zip")
        target.mkdir()
        outside = target.parent / "outside"
        outside.mkdir()
        (target / "alias").symlink_to(outside, target_is_directory=True)
        make_archive(archive, [("alias/escaped.txt", "file", "bad")])
        # Models a directory alias such as a junction without claiming Windows execution.
        with mock.patch.object(Path, "is_symlink", return_value=False):
            with self.assertRaises(ValueError):
                download.extract_archive(archive, target)
        self.assertFalse((outside / "escaped.txt").exists())

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


class DatasetReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.raw = Path(self.temporary.name) / "raw"
        self.raw.mkdir()
        self.target = self.raw / "synthetic"
        self.target.mkdir()
        (self.target / "original.txt").write_text("original dataset")
        self.shared = self.raw / "temp"
        self.shared.mkdir()
        (self.shared / "unrelated.txt").write_text("keep this")
        self.download_paths = []
        datasets = {"synthetic": {"url": "https://example.invalid/synthetic.zip", "description": "Synthetic fixture"}}
        for patch in (mock.patch.object(download, "setup_data_directories", return_value=self.raw),
                      mock.patch.object(download, "DATASETS", datasets)):
            patch.start()
            self.addCleanup(patch.stop)

    def fake_download(self, url, destination):
        self.download_paths.append(destination)
        make_archive(destination, [("replacement.txt", "file", "replacement dataset")])

    def assert_original(self):
        self.assertEqual((self.target / "original.txt").read_text(), "original dataset")
        self.assertEqual([path.name for path in self.target.iterdir()], ["original.txt"])
        self.assertEqual((self.shared / "unrelated.txt").read_text(), "keep this")

    def assert_no_staging(self):
        self.assertEqual(sorted(path.name for path in self.raw.iterdir()), ["synthetic", "temp"])

    def test_nonforced_existing_dataset_is_skipped(self):
        with mock.patch.object(download, "download_file") as fetch:
            download.download_datasets()
        fetch.assert_not_called()
        self.assert_original()
        self.assert_no_staging()

    def test_failed_forced_download_preserves_existing_dataset_and_shared_temp(self):
        def fail_download(url, destination):
            destination.write_bytes(b"partial archive")
            raise OSError("Synthetic download failure")
        with mock.patch.object(download, "download_file", side_effect=fail_download):
            with self.assertRaises(SystemExit) as raised:
                download.download_datasets(force_download=True)
        self.assertEqual(raised.exception.code, 1)
        self.assert_original()
        self.assert_no_staging()

    def test_failed_forced_extraction_preserves_existing_dataset_and_shared_temp(self):
        def fail_extract(archive, destination):
            self.assertNotEqual(destination, self.target)
            destination.mkdir()
            (destination / "partial.txt").write_text("partial extraction")
            raise ValueError("Synthetic extraction failure")
        with mock.patch.object(download, "download_file", side_effect=self.fake_download), \
             mock.patch.object(download, "extract_archive", side_effect=fail_extract):
            with self.assertRaises(SystemExit):
                download.download_datasets(force_download=True)
        self.assert_original()
        self.assert_no_staging()

    def test_successful_forced_refresh_replaces_only_after_complete_extraction(self):
        original_extract = download.extract_archive
        def inspect_extract(archive, destination):
            self.assert_original()
            self.assertNotEqual(destination, self.target)
            original_extract(archive, destination)
            self.assert_original()
        with mock.patch.object(download, "download_file", side_effect=self.fake_download), \
             mock.patch.object(download, "extract_archive", side_effect=inspect_extract):
            download.download_datasets(force_download=True)
        self.assertEqual((self.target / "replacement.txt").read_text(), "replacement dataset")
        self.assertFalse((self.target / "original.txt").exists())
        self.assertEqual((self.shared / "unrelated.txt").read_text(), "keep this")
        self.assert_no_staging()

    def test_failed_install_restores_previous_dataset(self):
        real_replace = Path.replace
        def fail_install(source, destination):
            if source.name == "extracted":
                raise OSError("Synthetic installation failure")
            return real_replace(source, destination)
        with mock.patch.object(download, "download_file", side_effect=self.fake_download), \
             mock.patch.object(Path, "replace", fail_install):
            with self.assertRaises(SystemExit):
                download.download_datasets(force_download=True)
        self.assert_original()
        self.assert_no_staging()

    def test_failed_rollback_preserves_backup_for_manual_recovery(self):
        real_replace = Path.replace
        def fail_install_and_rollback(source, destination):
            if source.name in ("extracted", "previous"):
                raise OSError("Synthetic rename failure")
            return real_replace(source, destination)
        with mock.patch.object(download, "download_file", side_effect=self.fake_download), \
             mock.patch.object(Path, "replace", fail_install_and_rollback):
            with self.assertRaises(SystemExit):
                download.download_datasets(force_download=True)
        backups = list(self.raw.glob(".synthetic-*/previous/original.txt"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), "original dataset")
        self.assertEqual((self.shared / "unrelated.txt").read_text(), "keep this")

    def test_fresh_datasets_use_distinct_staging_directories(self):
        datasets = {name: {"url": "https://example.invalid/fixture.zip", "description": "Synthetic fixture"}
                    for name in ("new-one", "new-two")}
        with mock.patch.object(download, "DATASETS", datasets), \
             mock.patch.object(download, "download_file", side_effect=self.fake_download):
            download.download_datasets()
        self.assertEqual(len({path.parent for path in self.download_paths}), 2)
        for path in self.download_paths:
            self.assertEqual(path.parent.parent, self.raw)
            self.assertNotEqual(path.parent, self.shared)
            self.assertFalse(path.parent.exists())
        for name in datasets:
            self.assertEqual((self.raw / name / "replacement.txt").read_text(), "replacement dataset")
        self.assert_original()


if __name__ == "__main__":
    unittest.main()
