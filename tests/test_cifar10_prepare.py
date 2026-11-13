"""CIFAR preparation tests use temporary fixtures, never the real dataset."""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading

import pytest

from scripts import prepare_cifar10 as prepare


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_cifar10.py"


@pytest.mark.skipif(os.name == "nt", reason="POSIX default-root guard")
def test_default_windows_root_fails_without_creating_a_fake_drive(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "Windows" in result.stderr and "--root" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX Windows-path guard")
def test_explicit_windows_root_is_also_rejected_on_posix():
    with pytest.raises(ValueError, match="Windows"):
        prepare.resolve_root(r"D:\ESTR3114\project\CIFAR-10\dataset")


def test_explicit_local_root_is_permitted_on_posix(tmp_path):
    assert prepare.resolve_root(tmp_path) == tmp_path


@pytest.fixture
def tiny_batches(monkeypatch):
    names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    content = {name: f"tiny checked fixture for {name}".encode() for name in names}
    monkeypatch.setattr(prepare, "FILE_MD5", {
        name: hashlib.md5(payload).hexdigest() for name, payload in content.items()
    }, raising=False)
    return content


def write_batches(root, content):
    folder = root / "cifar-10-batches-py"
    folder.mkdir(parents=True)
    for name, payload in content.items():
        (folder / name).write_bytes(payload)
    return folder


def test_verify_batches_requires_every_official_batch_and_metadata(tmp_path, tiny_batches):
    folder = write_batches(tmp_path, tiny_batches)
    assert prepare.verify_dataset(tmp_path) == {
        "train_images": 50000, "test_images": 10000, "verified_files": 7,
    }
    (folder / "test_batch").unlink()
    with pytest.raises(ValueError, match="test_batch"):
        prepare.verify_dataset(tmp_path)


def test_verify_batches_detects_corruption(tmp_path, tiny_batches):
    folder = write_batches(tmp_path, tiny_batches)
    (folder / "data_batch_3").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="data_batch_3"):
        prepare.verify_dataset(tmp_path)


def test_verify_only_missing_root_has_no_network_or_filesystem_side_effect(tmp_path, capsys):
    missing = tmp_path / "missing"
    assert prepare.main(["--root", str(missing), "--verify-only"]) == 1
    assert "Missing" in capsys.readouterr().err
    assert not missing.exists()


def make_archive(path, content, extra=None):
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in content.items():
            member = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        if extra is not None:
            archive.addfile(extra, io.BytesIO(b"x") if extra.size else None)
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_extract_checks_archive_hash_before_creating_output(tmp_path, tiny_batches):
    archive = tmp_path / "fixture.tar.gz"
    make_archive(archive, tiny_batches)
    target = tmp_path / "dataset"
    with pytest.raises(ValueError, match="MD5"):
        prepare.extract_archive(archive, target, expected_md5="0" * 32)
    assert not target.exists()


def test_extract_publishes_only_complete_verified_batches(tmp_path, tiny_batches):
    archive = tmp_path / "fixture.tar.gz"
    digest = make_archive(archive, tiny_batches)
    target = tmp_path / "dataset"
    prepare.extract_archive(archive, target, expected_md5=digest)
    assert (target / "cifar-10-batches-py" / "data_batch_1").read_bytes() == tiny_batches["data_batch_1"]
    assert prepare.verify_dataset(target)["verified_files"] == 7
    assert not list(target.glob(".cifar10-*"))


def test_extract_does_not_replace_existing_files_with_corrupt_batches(tmp_path, tiny_batches):
    target = tmp_path / "dataset"
    folder = write_batches(target, tiny_batches)
    corrupt = dict(tiny_batches, data_batch_1=b"corrupt")
    archive = tmp_path / "fixture.tar.gz"
    digest = make_archive(archive, corrupt)
    with pytest.raises(ValueError, match="data_batch_1"):
        prepare.extract_archive(archive, target, expected_md5=digest)
    assert (folder / "data_batch_1").read_bytes() == tiny_batches["data_batch_1"]


@pytest.mark.parametrize("member_name,symlink", [
    ("../escape", False),
    ("cifar-10-batches-py/../../escape", False),
    ("cifar-10-batches-py/..\\escape", False),
    ("cifar-10-batches-py/link", True),
])
def test_extract_rejects_unsafe_members(tmp_path, tiny_batches, member_name, symlink):
    archive = tmp_path / "fixture.tar.gz"
    member = tarfile.TarInfo(member_name)
    if symlink:
        member.type = tarfile.SYMTYPE
        member.linkname = "../../escape"
    else:
        member.size = 1
    digest = make_archive(archive, tiny_batches, extra=member)
    target = tmp_path / "dataset"
    with pytest.raises(ValueError, match="Unsafe"):
        prepare.extract_archive(archive, target, expected_md5=digest)
    assert not (tmp_path / "escape").exists()
    assert not (target / "cifar-10-batches-py").exists()


@contextmanager
def download_server(payload, *, ignore_range=False, content_range_offset=0, truncate_at=None, segment_size=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            if offset >= len(payload):
                self.send_response(416)
                self.end_headers()
                return
            if not ignore_range and offset:
                end = min(offset + segment_size, len(payload)) if segment_size else len(payload)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {offset + content_range_offset}-{end-1}/{len(payload)}")
            else:
                offset = 0
                end = len(payload)
                self.send_response(200)
            self.send_header("Content-Length", str(end - offset))
            self.end_headers()
            self.wfile.write(payload[offset:truncate_at if truncate_at is not None else end])

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/cifar-10-python.tar.gz"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("ignore_range", [False, True])
def test_download_resumes_or_restarts_when_server_ignores_range(tmp_path, ignore_range):
    payload = b"small HTTP fixture for resumable download"
    partial = tmp_path / "cifar-10-python.tar.gz.part"
    partial.write_bytes(payload[:9])
    with download_server(payload, ignore_range=ignore_range) as url:
        archive = prepare.download_archive(tmp_path, url=url, expected_md5=hashlib.md5(payload).hexdigest())
    assert archive.read_bytes() == payload
    assert not partial.exists()


def test_download_rejects_a_wrong_resumed_range_without_appending(tmp_path):
    payload = b"small HTTP fixture for malformed range"
    partial = tmp_path / "cifar-10-python.tar.gz.part"
    partial.write_bytes(payload[:9])
    with download_server(payload, content_range_offset=1) as url:
        with pytest.raises(ValueError, match="Content-Range"):
            prepare.download_archive(tmp_path, url=url, expected_md5=hashlib.md5(payload).hexdigest())
    assert partial.read_bytes() == payload[:9]
    assert not (tmp_path / "cifar-10-python.tar.gz").exists()


def test_download_rejects_wrong_checksum_and_discards_bad_partial(tmp_path):
    with download_server(b"not CIFAR-10") as url:
        with pytest.raises(ValueError, match="MD5"):
            prepare.download_archive(tmp_path, url=url, expected_md5="0" * 32)
    assert not (tmp_path / "cifar-10-python.tar.gz").exists()
    assert not (tmp_path / "cifar-10-python.tar.gz.part").exists()


def test_interrupted_download_preserves_partial_for_the_next_run(tmp_path):
    payload = b"temporary interruption with a resumable payload"
    digest = hashlib.md5(payload).hexdigest()
    with download_server(payload, truncate_at=12) as url:
        with pytest.raises(OSError, match="Incomplete"):
            prepare.download_archive(tmp_path, url=url, expected_md5=digest)
    assert (tmp_path / "cifar-10-python.tar.gz.part").read_bytes() == payload[:12]
    with download_server(payload) as url:
        archive = prepare.download_archive(tmp_path, url=url, expected_md5=digest)
    assert archive.read_bytes() == payload


def test_unresumable_oversized_partial_restarts_from_zero(tmp_path):
    payload = b"small complete file"
    (tmp_path / "cifar-10-python.tar.gz.part").write_bytes(b"stale partial bigger than the server's file")
    with download_server(payload) as url:
        archive = prepare.download_archive(tmp_path, url=url, expected_md5=hashlib.md5(payload).hexdigest())
    assert archive.read_bytes() == payload


def test_short_valid_range_response_keeps_received_bytes_for_another_resume(tmp_path):
    payload = b"server supplies only one segment of the remaining file"
    digest = hashlib.md5(payload).hexdigest()
    partial = tmp_path / "cifar-10-python.tar.gz.part"
    partial.write_bytes(payload[:9])
    with download_server(payload, segment_size=6) as url:
        with pytest.raises(OSError, match="Incomplete"):
            prepare.download_archive(tmp_path, url=url, expected_md5=digest)
    assert partial.read_bytes() == payload[:15]
    with download_server(payload) as url:
        assert prepare.download_archive(tmp_path, url=url, expected_md5=digest).read_bytes() == payload


def test_main_downloads_verifies_and_extracts_tiny_archive(tmp_path, tiny_batches, monkeypatch, capsys):
    archive = tmp_path / "fixture.tar.gz"
    digest = make_archive(archive, tiny_batches)
    monkeypatch.setattr(prepare, "ARCHIVE_MD5", digest)
    with download_server(archive.read_bytes()) as url:
        monkeypatch.setattr(prepare, "URL", url)
        assert prepare.main(["--root", str(tmp_path / "dataset")]) == 0
    assert (tmp_path / "dataset" / "cifar-10-batches-py" / "test_batch").read_bytes() == tiny_batches["test_batch"]
    assert "50,000" in capsys.readouterr().out


def test_already_verified_archive_is_reused_without_network(tmp_path):
    archive = tmp_path / "cifar-10-python.tar.gz"
    archive.write_bytes(b"checked archive")
    result = prepare.download_archive(tmp_path, url="http://invalid.invalid", expected_md5="ae76268fc877fc277f3ec4c0a7dd0681")
    assert result == archive


def test_completed_partial_is_promoted_without_network(tmp_path):
    partial = tmp_path / "cifar-10-python.tar.gz.part"
    partial.write_bytes(b"checked archive")
    result = prepare.download_archive(tmp_path, url="http://invalid.invalid", expected_md5="ae76268fc877fc277f3ec4c0a7dd0681")
    assert result.read_bytes() == b"checked archive"
    assert not partial.exists()


def test_verified_dataset_skips_download_on_rerun(tmp_path, tiny_batches, capsys):
    write_batches(tmp_path, tiny_batches)
    assert prepare.main(["--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "50,000" in output and "10,000" in output
    assert not (tmp_path / "cifar-10-python.tar.gz").exists()


@pytest.mark.parametrize("stage", ["pilot", "tuning", "main"])
def test_windows_configs_preserve_protocol_defaults_except_dataset_location(stage):
    config_root = SCRIPT.parents[1] / "configs"
    windows_path = config_root / "windows" / f"cifar10_{stage}.json"
    assert windows_path.is_file(), f"Windows {stage} config is missing"
    windows = json.loads(windows_path.read_text())
    generic = json.loads((config_root / f"cifar10_{stage}.json").read_text())
    assert windows["data_root"] == r"D:\ESTR3114\project\CIFAR-10\dataset"
    assert windows["download"] is False
    windows["data_root"] = generic["data_root"]
    assert windows == generic
