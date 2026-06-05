"""
Downloads P2C and NU-MCC model weights to ~/models/numcc/ on the host.
Run once before the first docker run.

Usage:
    python tools/download_models_numcc.py

Sources:
    P2C:    Google Drive  (github.com/CuiRuikai/Partial2Complete)
    NU-MCC: AWS S3        (github.com/sail-sg/numcc) — CO3D-V2 checkpoint
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_ROOT = Path.home() / "models" / "numcc"

CHECKPOINTS = {
    # P2C — Google Drive requires gdown; install with: pip install gdown
    "p2c/p2c_checkpoint.pth": {
        "gdrive_id": "1Cj2E2bhx7WsKxg1FMBysJajt4xIL8PD4",
        "url": None,
        "sha256": "184064288a5f41735345f7d88b1e86185739ca22a591d522337c70702f2047d5",
    },
    # NU-MCC — CO3D-V2 pretrained (best for single-object reconstruction)
    "numcc/numcc_checkpoint.pth": {
        "gdrive_id": None,
        "url": "https://numcc.s3.us-west-1.amazonaws.com/udf-ep99.pth",
        "sha256": "8e92765aefe6d81b084c5e0f495d7a11dcf9b13aeede5815986a6ddd3ab21b49",
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


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_direct(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading from {url}")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print()


def download_gdrive(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        print("  ERROR: gdown is required for Google Drive downloads.", file=sys.stderr)
        print("  Install with: pip install gdown", file=sys.stderr)
        sys.exit(1)
    print(f"  Downloading from Google Drive (id={file_id})")
    gdown.download(id=file_id, output=str(dest), quiet=False)


def main():
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {MODELS_ROOT}\n")

    for rel_path, info in CHECKPOINTS.items():
        dest = MODELS_ROOT / rel_path
        sha256 = info.get("sha256")

        if sha256 and checksum_matches(dest, sha256):
            print(f"[skip] {rel_path} — already downloaded and verified")
            continue

        if dest.exists() and sha256 is None:
            print(f"[skip] {rel_path} — exists (sha256 not yet configured)")
            continue

        print(f"[download] {rel_path}")
        try:
            if info.get("gdrive_id"):
                download_gdrive(info["gdrive_id"], dest)
            else:
                download_direct(info["url"], dest)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            print(f"  Download manually and place at: {dest}", file=sys.stderr)
            sys.exit(1)

        actual_sha256 = compute_sha256(dest)
        print(f"  SHA256: {actual_sha256}")
        if sha256 and actual_sha256 != sha256:
            print(f"  ERROR: checksum mismatch for {dest}", file=sys.stderr)
            sys.exit(1)

    print("\nAll models ready.")
    print(f"Mount with: -v {MODELS_ROOT}:/opt/models:ro")


if __name__ == "__main__":
    main()
