"""
Downloads P2C and NU-MCC model weights to ~/models/numcc/ on the host.
Run once before the first docker run.

Usage:
    python tools/download_models_numcc.py
"""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODELS_ROOT = Path.home() / "models" / "numcc"

# Update URLs and checksums after locating official release assets from the repos.
# For HuggingFace checkpoints replace urllib calls with huggingface_hub.hf_hub_download.
CHECKPOINTS = {
    "p2c/p2c_checkpoint.pth": {
        "url": "https://github.com/CuiRuikai/Partial2Complete/releases/download/v1.0/p2c_checkpoint.pth",
        "sha256": None,  # fill in: sha256sum ~/models/numcc/p2c/p2c_checkpoint.pth
    },
    "numcc/numcc_checkpoint.pth": {
        "url": "https://github.com/sail-sg/numcc/releases/download/v1.0/numcc_checkpoint.pth",
        "sha256": None,  # fill in: sha256sum ~/models/numcc/numcc/numcc_checkpoint.pth
    },
}


def checksum_matches(path: Path, expected_sha256: str) -> bool:
    """Return True if file exists and sha256 matches expected_sha256."""
    if not path.exists():
        return False
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} from {url}")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print()


def main():
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {MODELS_ROOT}\n")

    for rel_path, info in CHECKPOINTS.items():
        dest = MODELS_ROOT / rel_path
        sha256 = info["sha256"]

        if sha256 and checksum_matches(dest, sha256):
            print(f"[skip] {rel_path} — already downloaded and verified")
            continue

        if dest.exists() and sha256 is None:
            print(f"[skip] {rel_path} — exists (no checksum configured)")
            continue

        print(f"[download] {rel_path}")
        try:
            download_file(info["url"], dest)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            print(f"  Download manually and place at: {dest}", file=sys.stderr)
            sys.exit(1)

        if sha256 and not checksum_matches(dest, sha256):
            print(f"  ERROR: checksum mismatch for {dest}", file=sys.stderr)
            sys.exit(1)

    print("\nAll models ready.")
    print(f"Mount with: -v {MODELS_ROOT}:/opt/models:ro")


if __name__ == "__main__":
    main()
