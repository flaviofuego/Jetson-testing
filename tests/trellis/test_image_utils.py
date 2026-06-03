import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "trellis"))


def test_preprocess_returns_rgb_518x518(tmp_path):
    from image_utils import preprocess_image
    img = Image.new("RGBA", (256, 256), (200, 100, 50, 255))
    p = tmp_path / "test.png"
    img.save(str(p))
    result = preprocess_image(p)
    assert result.size == (518, 518)
    assert result.mode == "RGB"


def test_preprocess_background_is_white(tmp_path):
    from image_utils import preprocess_image
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    p = tmp_path / "transparent.png"
    img.save(str(p))
    result = preprocess_image(p)
    arr = np.array(result)
    assert arr.mean() > 200
