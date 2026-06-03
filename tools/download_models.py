"""
Downloads TripoSR models to ~/models/huggingface on the host.
Run once on the Jetson before the first docker run.

Usage:
    python3 tools/download_models.py
"""
import os
os.environ["HF_HOME"] = os.path.expanduser("~/models/huggingface")

from huggingface_hub import hf_hub_download
import rembg

print("Downloading TripoSR weights (~1.7 GB)...")
hf_hub_download("stabilityai/TripoSR", "config.yaml")
hf_hub_download("stabilityai/TripoSR", "model.ckpt")

print("Downloading DINO-ViT config...")
hf_hub_download("facebook/dino-vitb16", "config.json")

print("Downloading rembg U2-Net model (~176 MB)...")
rembg.new_session()

print("\nAll models downloaded to ~/models/huggingface")
print("Mount with: -v ~/models/huggingface:/root/.cache/huggingface")
