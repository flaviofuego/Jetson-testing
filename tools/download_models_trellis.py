"""
Downloads TRELLIS models to ~/models/huggingface on the host.
Run once on the Jetson before the first docker run.

Usage:
    python3 tools/download_models_trellis.py
"""
import os
os.environ["HF_HOME"] = os.path.expanduser("~/models/huggingface")

from huggingface_hub import snapshot_download
import rembg

print("Downloading TRELLIS-image-large weights (~3 GB)...")
snapshot_download("JeffreyXiang/TRELLIS-image-large")

print("Downloading rembg U2-Net model (~176 MB)...")
rembg.new_session()

print("\nAll TRELLIS models downloaded to ~/models/huggingface")
print("Mount with: -v ~/models/huggingface:/root/.cache/huggingface")
