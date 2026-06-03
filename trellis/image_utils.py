from pathlib import Path
import numpy as np
import rembg
from PIL import Image

_SIZE = 518  # TRELLIS DINO encoder input size


def preprocess_image(path: Path | str, foreground_ratio: float = 0.85) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    session = rembg.new_session()
    img = rembg.remove(img, session=session)
    img = _resize_foreground(img, foreground_ratio)
    img = img.resize((_SIZE, _SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + 1.0 * (1.0 - arr[:, :, 3:4])
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def _resize_foreground(img: Image.Image, ratio: float) -> Image.Image:
    arr = np.array(img)
    mask = arr[:, :, 3] > 25
    if not mask.any():
        return img
    rows, cols = np.where(mask)
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    fg = img.crop((cmin, rmin, cmax + 1, rmax + 1))
    size = int(max(fg.size) / ratio)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = ((size - fg.width) // 2, (size - fg.height) // 2)
    canvas.paste(fg, offset)
    return canvas
