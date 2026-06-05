import sys
from pathlib import Path
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))


def test_checksum_matches_correct_file(tmp_path):
    from download_models_numcc import checksum_matches
    f = tmp_path / "file.pth"
    f.write_bytes(b"fake model weights")
    expected = hashlib.sha256(b"fake model weights").hexdigest()
    assert checksum_matches(f, expected) is True


def test_checksum_matches_wrong_content(tmp_path):
    from download_models_numcc import checksum_matches
    f = tmp_path / "file.pth"
    f.write_bytes(b"different content")
    wrong_hash = hashlib.sha256(b"original content").hexdigest()
    assert checksum_matches(f, wrong_hash) is False


def test_checksum_matches_missing_file(tmp_path):
    from download_models_numcc import checksum_matches
    assert checksum_matches(tmp_path / "nonexistent.pth", "abc123") is False
