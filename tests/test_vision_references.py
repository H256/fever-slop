import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from feverslop.prompting.vision_references import prepare_vision_image


class VisionReferenceImageTests(unittest.TestCase):
    def test_prepare_vision_image_bounds_longest_side_without_upscaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            large = Path(tmp) / "large.png"
            small = Path(tmp) / "small.png"
            Image.new("RGB", (2048, 1024), "red").save(large)
            Image.new("RGB", (320, 240), "blue").save(small)

            large_mime, large_bytes = prepare_vision_image(large)
            small_mime, small_bytes = prepare_vision_image(small)

            self.assertEqual("image/jpeg", large_mime)
            self.assertEqual("image/jpeg", small_mime)
            with Image.open(BytesIO(large_bytes)) as prepared:
                self.assertEqual((1024, 512), prepared.size)
            with Image.open(BytesIO(small_bytes)) as prepared:
                self.assertEqual((320, 240), prepared.size)
            with Image.open(large) as source:
                self.assertEqual((2048, 1024), source.size)


if __name__ == "__main__":
    unittest.main()
