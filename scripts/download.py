"""
Dataset downloader for LearnedSpectrum
Downloads and extracts required OpenFMRI datasets
"""

import os
import sys
import logging
from pathlib import Path, PurePosixPath, PureWindowsPath
import requests
from tqdm import tqdm
import zipfile
import tarfile
import shutil
import stat
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dataset information
DATASETS = {
    "ds000002": {
        "url": "https://s3.amazonaws.com/openneuro/ds000002/ds000002_R2.0.5/compressed/ds000002_R2.0.5_raw.zip",
        "description": "Classification Learning",
        "tr": 2.0,
        "stage_map": lambda f: 0.25 if "run-2" in str(f) else 0.0,  # prob learning
    },
    "ds000011": {
        "url": "https://s3.amazonaws.com/openneuro/ds000011/ds000011_R2.0.1/compressed/ds000011_R2.0.1_raw.zip",
        "description": "Mixed-gambles Task",
        "tr": 1.5,
        "stage_map": lambda f: 0.5 if "run-2" in str(f) else 0.25,  # det learning
    },
    "ds000017": {
        "url": "https://s3.amazonaws.com/openneuro/ds000017/ds000017_R2.0.1/compressed/ds000017_R2.0.1.zip",
        "description": "Classification Learning and Reversal",
        "tr": 2.5,
        "stage_map": lambda f: 0.75 if "run-2" in str(f) else 0.5,  # reversal
    },
    "ds000052": {
        "url": "https://s3.amazonaws.com/openneuro/ds000052/ds000052_R2.0.0/compressed/ds052_R2.0.0_01-14.tgz",
        "description": "Classification Learning and Stop-signal",
        "tr": 2.0,
        "stage_map": lambda f: (
            1.0
            if ("reversal" in str(f) and "run-2" in str(f))
            else 0.67
            if ("reversal" in str(f))
            else 0.33
            if "run-2" in str(f)
            else 0.0
        ),  # full spectrum
    },
}


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)


def _member_path(root: Path, name: str) -> Path:
    """Reject paths that can escape or follow existing links under root."""
    path = PurePosixPath(name)
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or ".." in path.parts
        or any(
            part.endswith((".", " ")) or PureWindowsPath(part).is_reserved()
            for part in path.parts
        )
    ):
        raise ValueError(f"Unsafe archive path: {name!r}")
    target = root
    for part in path.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"Archive path follows an existing symlink: {name!r}")
    try:
        target.resolve().relative_to(root)
    except ValueError:
        raise ValueError(f"Archive path resolves outside the extraction directory: {name!r}") from None
    return target


def _copy_member(source, target: Path) -> None:
    """Replace a file without writing through an existing hard link."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(target.parent), delete=False) as output:
            temporary = Path(output.name)
            shutil.copyfileobj(source, output)
        os.replace(str(temporary), str(target))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def extract_archive(archive_path: Path, extract_path: Path) -> None:
    """Extract regular files and directories; reject links and special files.

    Archive ownership and permission metadata are not applied. The destination
    must not be modified by another process during extraction.
    """
    logger.info(f"Extracting {archive_path} to {extract_path}")
    if extract_path.is_symlink():
        raise ValueError("Extraction directory must not be a symlink")
    extract_path.mkdir(parents=True, exist_ok=True)
    root = extract_path.resolve()

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            members = []
            for member in zip_ref.infolist():
                kind = stat.S_IFMT(member.external_attr >> 16)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError(f"Unsupported ZIP member type: {member.filename!r}")
                target = _member_path(root, member.filename)
                is_directory = member.is_dir() or kind == stat.S_IFDIR
                if target == root and not is_directory:
                    raise ValueError("Archive file must not replace the extraction directory")
                members.append((member, target, is_directory))
            for member, target, is_directory in members:
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    with zip_ref.open(member) as source:
                        _copy_member(source, target)
    elif archive_path.suffix in [".gz", ".tgz"] or archive_path.name.endswith(
        ".tar.gz"
    ):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            members = []
            for member in tar_ref.getmembers():
                if not member.isdir() and member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    raise ValueError(f"Unsupported TAR member type: {member.name!r}")
                target = _member_path(root, member.name)
                if target == root and not member.isdir():
                    raise ValueError("Archive file must not replace the extraction directory")
                members.append((member, target))
            for member, target in members:
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    with tar_ref.extractfile(member) as source:
                        _copy_member(source, target)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


def setup_data_directories() -> Path:
    """
    Create necessary data directories
    """
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    cleaned_dir = data_dir / "cleaned"

    for directory in [raw_dir, processed_dir, cleaned_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return raw_dir


def get_file_extension(url: str) -> str:
    """
    Get the file extension from the URL
    """
    if url.endswith(".zip"):
        return ".zip"
    elif url.endswith(".tgz"):
        return ".tgz"
    elif url.endswith(".tar.gz"):
        return ".tar.gz"
    else:
        raise ValueError(f"Unsupported file format in URL: {url}")


def download_datasets(force_download: bool = False) -> None:
    """
    Download into unique staging directories and replace completed datasets.

    Only one process may refresh a given dataset at a time.
    """
    raw_dir = setup_data_directories()
    success = True

    for dataset_id, info in DATASETS.items():
        dataset_dir = raw_dir / dataset_id

        if (dataset_dir.exists() or dataset_dir.is_symlink()) and not force_download:
            logger.info(f"Dataset {dataset_id} already exists. Skipping...")
            continue

        logger.info(f"Processing {dataset_id} - {info['description']}")

        temp_dir = Path(tempfile.mkdtemp(prefix=f".{dataset_id}-", dir=str(raw_dir)))
        backup = temp_dir / "previous"
        installed = False

        try:
            # Get correct file extension from URL
            file_ext = get_file_extension(info["url"])
            # Download archive with correct extension
            archive_path = temp_dir / f"{dataset_id}{file_ext}"
            download_file(info["url"], archive_path)

            staged = temp_dir / "extracted"
            extract_archive(archive_path, staged)

            try:
                if dataset_dir.exists() or dataset_dir.is_symlink():
                    dataset_dir.replace(backup)
                staged.replace(dataset_dir)
            except BaseException:
                if backup.exists() or backup.is_symlink():
                    backup.replace(dataset_dir)
                raise
            installed = True

            logger.info(f"Successfully downloaded and extracted {dataset_id}")

        except Exception as e:
            success = False
            logger.error(f"Error processing {dataset_id}: {str(e)}")

        finally:
            if not installed and (backup.exists() or backup.is_symlink()):
                logger.error(f"Previous dataset preserved at {backup}; restore it before retrying.")
            else:
                shutil.rmtree(temp_dir)

    if success:
        logger.info("All datasets downloaded successfully")
    else:
        logger.error("Some datasets failed to download. Please check the errors above.")
        sys.exit(1)


def main():
    """
    Main execution function
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Download OpenFMRI datasets for LearnedSpectrum"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force download even if datasets exist"
    )
    args = parser.parse_args()

    try:
        download_datasets(force_download=args.force)
        logger.info("Dataset download completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
