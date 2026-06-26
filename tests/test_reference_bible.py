import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from feverslop.application.reference_bible import ReferenceBibleGenerator, ReferenceLocation, ReferenceSubject


class FakeImageBackend:
    def render_image(self, request):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        output = request.output_dir / f"scene_{request.scene_number:04}.png"
        Image.new("RGB", (32, 24), color=(request.scene_number * 20, 0, 0)).save(output)
        return output


class ReferenceBibleTests(unittest.TestCase):
    def test_generator_writes_manifest_views_and_sheet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            generator = ReferenceBibleGenerator(backend=FakeImageBackend(), output_dir=output_dir)

            manifest_path = generator.generate_subject_bible(
                ReferenceSubject(
                    id="singer",
                    name="Mara",
                    role="lead",
                    visual_description="silver hair",
                    image_prompt="portrait of Mara",
                )
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("singer", manifest["id"])
            self.assertTrue((manifest_path.parent / "sheet.png").exists())
            self.assertEqual(["hero", "front", "left", "right", "closeup"], [view["name"] for view in manifest["views"]])

    def test_generator_writes_location_manifest_views_and_sheet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            generator = ReferenceBibleGenerator(backend=FakeImageBackend(), output_dir=output_dir)

            manifest_path = generator.generate_location_bible(
                ReferenceLocation(
                    id="stage",
                    name="Mirror Stage",
                    visual_description="black stage",
                    image_prompt="wide mirror stage",
                )
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("location", manifest["kind"])
            self.assertEqual("stage", manifest["id"])
            self.assertTrue((manifest_path.parent / "sheet.png").exists())
