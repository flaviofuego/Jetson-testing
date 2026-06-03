from pathlib import Path

import numpy as np
import rembg
from PIL import Image


def preprocess_image(path: Path | str, foreground_ratio: float = 0.85) -> Image.Image:
    from tsr.utils import remove_background, resize_foreground

    img = Image.open(path)
    session = rembg.new_session()
    img = remove_background(img, session)
    img = resize_foreground(img, foreground_ratio)
    # Composite onto gray (0.5) — the background TripoSR was trained on
    arr = np.array(img, dtype=np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + 0.5 * (1.0 - arr[:, :, 3:4])
    return Image.fromarray((rgb * 255.0).astype(np.uint8))
