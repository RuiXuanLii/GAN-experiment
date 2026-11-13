"""Prepare official CIFAR-10 Python batches without third-party dependencies."""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import sys
import tarfile
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ROOT = r"D:\ESTR3114\project\CIFAR-10\dataset"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
BASE_FOLDER = "cifar-10-batches-py"
# Official torchvision 0.20 metadata; see docs/cifar10-windows-zh.md.
FILE_MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
}


def resolve_root(root=None):
    if root is None:
        if os.name != "nt":
            raise ValueError(
                "The default destination is a Windows D: drive. Run this script "
                "on Windows, or supply --root with a local path explicitly."
            )
        root = DEFAULT_ROOT
    if os.name != "nt" and PureWindowsPath(str(root)).drive:
        raise ValueError("A Windows drive path cannot be used here; supply --root with a local path.")
    return Path(root).expanduser()


def _md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(root):
    """Match all official batch bytes; their known image counts need no unpickling."""
    folder = Path(root) / BASE_FOLDER
    if folder.is_symlink():
        raise ValueError(f"Dataset folder must not be a symbolic link: {folder}")
    for filename, checksum in FILE_MD5.items():
        path = folder / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular dataset file: {path}")
        if _md5(path) != checksum:
            raise ValueError(f"MD5 mismatch for dataset file: {path}")
    return {"train_images": 50000, "test_images": 10000, "verified_files": 7}


def download_archive(root, *, url=None, expected_md5=None):
    """Keep interrupted bytes in .part; publish an archive only after MD5 verification."""
    root = Path(root)
    url = URL if url is None else url
    expected_md5 = ARCHIVE_MD5 if expected_md5 is None else expected_md5
    archive = root / ARCHIVE_NAME
    partial = root / (ARCHIVE_NAME + ".part")
    if archive.is_file() and _md5(archive) == expected_md5:
        return archive
    root.mkdir(parents=True, exist_ok=True)
    if partial.is_file() and _md5(partial) == expected_md5:
        os.replace(partial, archive)
        return archive
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "ESTR3114-CIFAR10-preparation", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = urlopen(Request(url, headers=headers), timeout=60)
    except HTTPError as error:
        if error.code != 416 or not offset:
            raise
        error.close()
        # A stale/oversized partial cannot be resumed. Retry once from zero.
        partial.unlink()
        offset = 0
        headers.pop("Range")
        response = urlopen(Request(url, headers=headers), timeout=60)
    with response:
        status = response.getcode()
        expected_total = None
        if status == 206:
            content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if (not content_range or int(content_range[1]) != offset
                    or not offset <= int(content_range[2]) < int(content_range[3])):
                raise ValueError("Invalid Content-Range; partial download was not modified.")
            expected_total = int(content_range[3])
        elif status == 200:
            offset = 0  # A server ignoring Range returned the whole archive.
        else:
            raise ValueError(f"Unexpected download HTTP status: {status}")
        content_length = response.headers.get("Content-Length")
        expected_length = int(content_length) if content_length is not None else None
        downloaded = offset
        print(f"Downloading {url}\nDestination: {archive}", flush=True)
        with partial.open("ab" if offset else "wb") as target:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                previous = downloaded
                downloaded += len(chunk)
                if downloaded // (16 * 1024 * 1024) > previous // (16 * 1024 * 1024):
                    print(f"Downloaded {downloaded / (1024 * 1024):.0f} MiB", flush=True)
        if ((expected_length is not None and downloaded - offset < expected_length)
                or (expected_total is not None and downloaded < expected_total)):
            raise OSError("Incomplete HTTP response; .part kept. Run the same command again to resume.")
    if _md5(partial) != expected_md5:
        partial.unlink()
        raise ValueError("Archive MD5 mismatch; invalid .part removed. Run the command again to retry.")
    os.replace(partial, archive)
    return archive


def extract_archive(archive_path, root, *, expected_md5=None):
    """Stage and verify required regular files before replacing dataset files."""
    expected_md5 = ARCHIVE_MD5 if expected_md5 is None else expected_md5
    if _md5(archive_path) != expected_md5:
        raise ValueError(f"Archive MD5 mismatch: {archive_path}")
    root = Path(root)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (path.is_absolute() or not path.parts or path.parts[0] != BASE_FOLDER
                    or ".." in path.parts or "\\" in member.name or ":" in member.name
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if len(path.parts) == 2 and path.name in FILE_MD5:
                if not member.isfile() or path.name in members:
                    raise ValueError(f"Unsafe duplicate/non-file member: {member.name}")
                members[path.name] = member
        missing = set(FILE_MD5) - members.keys()
        if missing:
            raise ValueError(f"Missing archive files: {', '.join(sorted(missing))}")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cifar10-", dir=root) as staging:
            staged_folder = Path(staging) / BASE_FOLDER
            staged_folder.mkdir()
            for filename, member in members.items():
                with archive.extractfile(member) as source, (staged_folder / filename).open("wb") as target:
                    shutil.copyfileobj(source, target)
            verify_dataset(staging)
            folder = root / BASE_FOLDER
            if folder.is_symlink():
                raise ValueError(f"Dataset folder must not be a symbolic link: {folder}")
            folder.mkdir(exist_ok=True)
            for filename in FILE_MD5:
                os.replace(staged_folder / filename, folder / filename)
    return verify_dataset(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help=f"Dataset parent folder (Windows default: {DEFAULT_ROOT})")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing extracted files without downloading or writing")
    args = parser.parse_args(argv)
    try:
        root = resolve_root(args.root)
        if args.verify_only:
            summary = verify_dataset(root)
        else:
            try:
                summary = verify_dataset(root)
            except ValueError:
                archive = download_archive(root)
                print("Archive MD5 verified; extracting required files...", flush=True)
                summary = extract_archive(archive, root)
        print(f"Verified CIFAR-10: {summary['train_images']:,} training and {summary['test_images']:,} test images "
              f"({summary['verified_files']} official files).\nDataset root: {root}")
    except (ValueError, OSError, URLError, tarfile.TarError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Download interrupted. Run the same command again to resume the .part file.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
