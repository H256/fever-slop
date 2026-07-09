import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from feverslop.application.movie_references import SceneReferenceSheetBuilder
from feverslop.application.reference_bible import (
    _fit_contain_image,
    _panel_position_label,
    _type_label,
    compose_scene_reference_sheet,
    generate_scene_sheet_description,
)


class FitContainImageTests(unittest.TestCase):

    def test_fit_contain_wide_into_tall_letterboxes(self):
        src = Image.new("RGB", (200, 100), color=(255, 0, 0))
        result = _fit_contain_image(src, (100, 200))
        self.assertEqual((100, 200), result.size)
        self.assertEqual((255, 255, 255), result.getpixel((0, 0)))
        self.assertEqual((255, 0, 0), result.getpixel((50, 90)))

    def test_fit_contain_tall_into_wide_letterboxes(self):
        src = Image.new("RGB", (100, 200), color=(0, 0, 255))
        result = _fit_contain_image(src, (200, 100))
        self.assertEqual((200, 100), result.size)
        self.assertEqual((255, 255, 255), result.getpixel((0, 0)))
        self.assertEqual((0, 0, 255), result.getpixel((90, 50)))

    def test_fit_contain_same_size_returns_copy(self):
        src = Image.new("RGB", (100, 100), color=(0, 255, 0))
        result = _fit_contain_image(src, (100, 100))
        self.assertEqual((100, 100), result.size)
        self.assertEqual((0, 255, 0), result.getpixel((0, 0)))
        self.assertEqual((0, 255, 0), result.getpixel((99, 99)))
        self.assertIsNot(src, result)

    def test_fit_contain_smaller_fits_without_scaling_up(self):
        src = Image.new("RGB", (50, 50), color=(128, 128, 128))
        result = _fit_contain_image(src, (100, 100))
        self.assertEqual((100, 100), result.size)
        self.assertEqual((255, 255, 255), result.getpixel((0, 0)))
        self.assertEqual((128, 128, 128), result.getpixel((25, 25)))

    def test_fit_contain_output_dimensions_exact(self):
        for target in ((300, 200), (200, 300), (100, 100), (1, 1)):
            src = Image.new("RGB", (50, 80), color=(100, 50, 50))
            result = _fit_contain_image(src, target)
            self.assertEqual(target, result.size, f"Failed for target {target}")

    def test_fit_contain_centered_placement(self):
        src = Image.new("RGB", (100, 50), color=(200, 100, 50))
        result = _fit_contain_image(src, (200, 200))
        self.assertEqual((200, 200), result.size)
        scaled_w, scaled_h = 100, 50
        offset_x = (200 - scaled_w) // 2
        offset_y = (200 - scaled_h) // 2
        self.assertEqual((255, 255, 255), result.getpixel((0, 0)))
        self.assertEqual((255, 255, 255), result.getpixel((0, 199)))
        self.assertEqual((200, 100, 50), result.getpixel((offset_x, offset_y)))
        self.assertEqual((200, 100, 50), result.getpixel((offset_x + scaled_w - 1, offset_y + scaled_h - 1)))


class ComposeSceneReferenceSheetTests(unittest.TestCase):

    @staticmethod
    def _make_image(w: int, h: int, color: tuple[int, int, int], tmp: Path) -> Path:
        p = tmp / f"img_{w}x{h}_{color[0]}_{color[1]}_{color[2]}.png"
        Image.new("RGB", (w, h), color=color).save(p)
        return p

    def test_compose_scene_sheet_basic_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imgs = [
                self._make_image(200, 200, (255, 0, 0), tmp),
                self._make_image(200, 200, (0, 255, 0), tmp),
                self._make_image(200, 200, (0, 0, 255), tmp),
            ]
            out = tmp / "output" / "scene_sheet.png"
            result = compose_scene_reference_sheet(imgs, out, size=(600, 400))
            self.assertEqual(out, result)
            self.assertTrue(out.exists())
            with Image.open(out) as sheet:
                self.assertEqual((600, 400), sheet.size)

    def test_compose_scene_sheet_heterogeneous_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imgs = [
                self._make_image(300, 100, (255, 0, 0), tmp),
                self._make_image(100, 300, (0, 255, 0), tmp),
                self._make_image(200, 200, (0, 0, 255), tmp),
            ]
            out = tmp / "hetero.png"
            compose_scene_reference_sheet(imgs, out, size=(600, 400))
            with Image.open(out) as sheet:
                self.assertEqual((600, 400), sheet.size)

    def test_compose_scene_sheet_single_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img = self._make_image(100, 100, (128, 0, 128), tmp)
            out = tmp / "single.png"
            compose_scene_reference_sheet([img], out, size=(400, 300))
            with Image.open(out) as sheet:
                self.assertEqual((400, 300), sheet.size)
                self.assertEqual((255, 255, 255), sheet.getpixel((0, 0)))
                self.assertEqual((255, 255, 255), sheet.getpixel((399, 0)))

    def test_compose_scene_sheet_explicit_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imgs = [
                self._make_image(100, 100, (255, 0, 0), tmp),
                self._make_image(100, 100, (0, 255, 0), tmp),
                self._make_image(100, 100, (0, 0, 255), tmp),
                self._make_image(100, 100, (255, 255, 0), tmp),
                self._make_image(100, 100, (255, 0, 255), tmp),
            ]
            out = tmp / "explicit_cols.png"
            compose_scene_reference_sheet(imgs, out, size=(300, 400), columns=2)
            with Image.open(out) as sheet:
                self.assertEqual((300, 400), sheet.size)

    def test_compose_scene_sheet_empty_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "empty.png"
            with self.assertRaises(ValueError):
                compose_scene_reference_sheet([], out)

    def test_compose_scene_sheet_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img = self._make_image(100, 100, (100, 100, 100), tmp)
            out = tmp / "deep" / "nested" / "dir" / "sheet.png"
            self.assertFalse(out.parent.exists())
            compose_scene_reference_sheet([img], out)
            self.assertTrue(out.exists())


class SceneReferenceSheetBuilderTests(unittest.TestCase):

    @staticmethod
    def _make_image(w: int, h: int, color: tuple[int, int, int], parent: Path) -> Path:
        p = parent / f"img_{w}x{h}_{color[0]}_{color[1]}_{color[2]}.png"
        Image.new("RGB", (w, h), color=color).save(p)
        return p

    def _make_project(self, tmp: Path) -> Path:
        movie = tmp / "movie"
        refs = movie / "references"
        actors_dir = refs / "actors" / "actor_1"
        actors_dir.mkdir(parents=True, exist_ok=True)
        locs_dir = refs / "locations" / "loc_1"
        locs_dir.mkdir(parents=True, exist_ok=True)
        return tmp

    def test_builder_basic_shot_with_actor_and_location(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            actor_sheet = self._make_image(200, 200, (255, 0, 0), tmp / "movie" / "references" / "actors" / "actor_1")
            loc_sheet = self._make_image(200, 200, (0, 0, 255), tmp / "movie" / "references" / "locations" / "loc_1")

            manifest = {
                "actors": [{"id": "actor_1", "name": "Alice", "msr_sheet_path": actor_sheet.relative_to(tmp).as_posix(), "sheet_path": actor_sheet.relative_to(tmp).as_posix()}],
                "locations": [{"id": "loc_1", "name": "Room", "msr_sheet_path": loc_sheet.relative_to(tmp).as_posix(), "sheet_path": loc_sheet.relative_to(tmp).as_posix()}],
            }
            shot = {"shot_id": "shot_001", "scene": 1, "reference_ids": {"actors": ["actor_1"], "location": "loc_1"}}

            builder = SceneReferenceSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertEqual(2, result["image_count"])
            self.assertEqual("movie/scene_sheets/shot_001_scene.png", result["sheet_path"])
            self.assertTrue((tmp / "movie" / "scene_sheets" / "shot_001_scene.png").exists())
            self.assertEqual(2, len(result["images"]))

    def test_builder_no_references_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            manifest = {"actors": [], "locations": []}
            shot = {"shot_id": "shot_002", "scene": 1}

            builder = SceneReferenceSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertEqual("", result["sheet_path"])
            self.assertEqual(0, result["image_count"])
            self.assertEqual([], result["images"])

    def test_builder_missing_files_filtered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            actor_sheet = self._make_image(200, 200, (255, 0, 0), tmp / "movie" / "references" / "actors" / "actor_1")

            manifest = {
                "actors": [{"id": "actor_1", "name": "Alice", "msr_sheet_path": "movie/references/actors/actor_1/nonexistent.png", "sheet_path": actor_sheet.relative_to(tmp).as_posix()}],
                "locations": [{"id": "loc_1", "name": "Room", "msr_sheet_path": "movie/references/locations/loc_1/nonexistent.png", "sheet_path": "movie/references/locations/loc_1/nonexistent.png"}],
            }
            shot = {"shot_id": "shot_003", "scene": 1, "reference_ids": {"actors": ["actor_1"], "location": "loc_1"}}

            builder = SceneReferenceSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertEqual(1, result["image_count"])
            self.assertTrue(result["sheet_path"])

    def test_builder_relative_path_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            short_name = "short_filename.png"
            prefixed = tmp / "movie" / "references" / short_name
            Image.new("RGB", (200, 200), color=(0, 255, 0)).save(prefixed)

            manifest = {
                "actors": [{"id": "actor_1", "name": "Alice", "msr_sheet_path": short_name, "sheet_path": short_name}],
                "locations": [],
            }
            shot = {"shot_id": "shot_004", "scene": 1, "reference_ids": {"actors": ["actor_1"], "location": ""}}

            builder = SceneReferenceSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertEqual(1, result["image_count"])
            self.assertTrue(result["images"][0]["path"].startswith("movie/references/"))

    def test_builder_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            actor_sheet = self._make_image(200, 200, (0, 255, 0), tmp / "movie" / "references" / "actors" / "actor_1")

            manifest = {
                "actors": [{"id": "actor_1", "name": "Alice", "msr_sheet_path": actor_sheet.relative_to(tmp).as_posix(), "sheet_path": actor_sheet.relative_to(tmp).as_posix()}],
                "locations": [],
            }
            scene_sheets = tmp / "movie" / "scene_sheets"
            self.assertFalse(scene_sheets.exists())
            shot = {"shot_id": "shot_005", "scene": 1, "reference_ids": {"actors": ["actor_1"]}}

            builder = SceneReferenceSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertTrue(scene_sheets.exists())
            self.assertEqual(1, result["image_count"])

    def test_builder_returns_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            actor_sheet = self._make_image(200, 200, (255, 255, 0), tmp / "movie" / "references" / "actors" / "actor_1")

            manifest = {
                "actors": [{"id": "actor_1", "name": "Alice", "msr_sheet_path": actor_sheet.relative_to(tmp).as_posix(), "sheet_path": actor_sheet.relative_to(tmp).as_posix()}],
                "locations": [],
            }
            shot = {"shot_id": "shot_006", "scene": 2, "reference_ids": {"actors": ["actor_1"]}}

            builder = SceneReferenceSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertFalse(result["sheet_path"].startswith("/"))
            self.assertNotIn("\\", result["sheet_path"])
            self.assertTrue(result["sheet_path"].startswith("movie/scene_sheets/"))


class MockSceneBuilder:
    def __init__(self, sheet_path="movie/scene_sheets/test_scene.png"):
        self.sheet_path = sheet_path
        self.calls = []

    def build(self, shot):
        self.calls.append(shot)
        return {
            "sheet_path": self.sheet_path,
            "image_count": 1,
            "images": [],
        }


class MSREnrichmentSceneSheetWiringTests(unittest.TestCase):

    def _make_project(self, tmp):
        movie = tmp / "movie"
        refs = movie / "references"
        refs.mkdir(parents=True, exist_ok=True)
        (movie / "bible.json").write_text(
            '{"runtime_constraints": {"fps": 24}, "actors": [], "locations": []}',
            encoding="utf-8",
        )
        (movie / "render_plan.json").write_text(
            json.dumps({
                "resolution": {"width": 1280, "height": 720},
                "shots": [
                    {
                        "shot_id": "shot_001",
                        "scene": 1,
                        "duration_seconds": 2,
                        "description": "A test scene",
                        "reference_ids": {"actors": ["actor_1"], "location": "loc_1"},
                    }
                ],
            }),
            encoding="utf-8",
        )
        (refs / "manifest.json").write_text(
            json.dumps({
                "actors": [{"id": "actor_1", "name": "Alice"}],
                "locations": [{"id": "loc_1", "name": "Room"}],
            }),
            encoding="utf-8",
        )
        return tmp

    def test_enrichment_wires_scene_sheet(self):
        from feverslop.application import movie_msr_enrichment

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)

            original_import = movie_msr_enrichment.SceneReferenceSheetBuilder
            mock_builder = MockSceneBuilder("movie/scene_sheets/shot_001_scene.png")
            movie_msr_enrichment.SceneReferenceSheetBuilder = lambda *a, **kw: mock_builder
            try:
                movie_msr_enrichment.enrich_movie_render_plan_with_msr_prompts(
                    project_dir=tmp,
                )
                output = json.loads((tmp / "movie" / "render_plan_msr.json").read_text(encoding="utf-8"))
            finally:
                movie_msr_enrichment.SceneReferenceSheetBuilder = original_import

            self.assertEqual(1, len(output["shots"]))
            self.assertIn("scene_reference_sheet", output["shots"][0])
            self.assertEqual("movie/scene_sheets/shot_001_scene.png", output["shots"][0]["scene_reference_sheet"])
            self.assertEqual(1, len(mock_builder.calls))

    def test_enrichment_scene_sheet_none_when_no_refs(self):
        from feverslop.application import movie_msr_enrichment

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            movie = tmp / "movie"
            refs = movie / "references"
            refs.mkdir(parents=True, exist_ok=True)
            (movie / "bible.json").write_text(
                '{"runtime_constraints": {"fps": 24}, "actors": [], "locations": []}',
                encoding="utf-8",
            )
            (movie / "render_plan.json").write_text(
                json.dumps({
                    "resolution": {"width": 1280, "height": 720},
                    "shots": [
                        {
                            "shot_id": "shot_001",
                            "scene": 1,
                            "duration_seconds": 2,
                            "description": "A test scene",
                            "reference_ids": {"actors": [], "location": ""},
                        }
                    ],
                }),
                encoding="utf-8",
            )
            (refs / "manifest.json").write_text(
                json.dumps({"actors": [], "locations": []}),
                encoding="utf-8",
            )

            movie_msr_enrichment.enrich_movie_render_plan_with_msr_prompts(project_dir=tmp)
            output = json.loads((tmp / "movie" / "render_plan_msr.json").read_text(encoding="utf-8"))

            self.assertEqual(1, len(output["shots"]))
            self.assertIn("scene_reference_sheet", output["shots"][0])
            self.assertEqual("", output["shots"][0]["scene_reference_sheet"])

    def test_enrichment_scene_sheet_in_ltx_config(self):
        from feverslop.application import movie_msr_enrichment

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)

            original_import = movie_msr_enrichment.SceneReferenceSheetBuilder
            movie_msr_enrichment.SceneReferenceSheetBuilder = lambda *a, **kw: MockSceneBuilder("movie/scene_sheets/shot_001_scene.png")
            try:
                movie_msr_enrichment.enrich_movie_render_plan_with_msr_prompts(
                    project_dir=tmp,
                )
                output = json.loads((tmp / "movie" / "render_plan_msr.json").read_text(encoding="utf-8"))
            finally:
                movie_msr_enrichment.SceneReferenceSheetBuilder = original_import

            shot = output["shots"][0]
            self.assertIn("scene_reference_sheet", shot)
            ltx = shot.get("ltx") or {}
            self.assertNotIn("scene_reference_sheet", ltx)


class PanelPositionLabelTests(unittest.TestCase):

    def test_single_image_full(self):
        self.assertEqual("Full", _panel_position_label(0, 0, 1, 1, 0, 1))

    def test_single_row_two_cols(self):
        self.assertEqual("Left", _panel_position_label(0, 0, 1, 2, 0, 2))
        self.assertEqual("Right", _panel_position_label(0, 1, 1, 2, 1, 2))

    def test_two_rows_two_cols(self):
        self.assertEqual("Top Row Left", _panel_position_label(0, 0, 2, 2, 0, 4))
        self.assertEqual("Top Row Right", _panel_position_label(0, 1, 2, 2, 1, 4))
        self.assertEqual("Bottom Row Left", _panel_position_label(1, 0, 2, 2, 2, 4))
        self.assertEqual("Bottom Row Right", _panel_position_label(1, 1, 2, 2, 3, 4))

    def test_three_images_two_cols_last_row_single(self):
        self.assertEqual("Top Row Left", _panel_position_label(0, 0, 2, 2, 0, 3))
        self.assertEqual("Top Row Right", _panel_position_label(0, 1, 2, 2, 1, 3))
        self.assertEqual("Bottom Row", _panel_position_label(1, 0, 2, 2, 2, 3))

    def test_single_row_three_cols(self):
        self.assertEqual("Left", _panel_position_label(0, 0, 1, 3, 0, 3))
        self.assertEqual("Center", _panel_position_label(0, 1, 1, 3, 1, 3))
        self.assertEqual("Right", _panel_position_label(0, 2, 1, 3, 2, 3))

    def test_single_row_four_cols(self):
        self.assertEqual("Left", _panel_position_label(0, 0, 1, 4, 0, 4))
        self.assertEqual("Center-Left", _panel_position_label(0, 1, 1, 4, 1, 4))
        self.assertEqual("Center-Right", _panel_position_label(0, 2, 1, 4, 2, 4))
        self.assertEqual("Right", _panel_position_label(0, 3, 1, 4, 3, 4))

    def test_three_rows_middle(self):
        self.assertEqual("Top Row Left", _panel_position_label(0, 0, 3, 2, 0, 6))
        self.assertEqual("Middle Row Right", _panel_position_label(1, 1, 3, 2, 3, 6))
        self.assertEqual("Bottom Row Right", _panel_position_label(2, 1, 3, 2, 5, 6))

    def test_many_rows_uses_numbered(self):
        self.assertEqual("Row 3 Left", _panel_position_label(2, 0, 5, 2, 4, 10))
        self.assertEqual("Bottom Row Right", _panel_position_label(4, 1, 5, 2, 9, 10))


class TypeLabelTests(unittest.TestCase):

    def test_actor_becomes_character(self):
        self.assertEqual("Character", _type_label("actor"))

    def test_location_becomes_setting(self):
        self.assertEqual("Setting", _type_label("location"))

    def test_prop_stays_prop(self):
        self.assertEqual("Prop", _type_label("prop"))

    def test_unknown_titlecased(self):
        self.assertEqual("Other", _type_label("other"))


class SceneSheetDescriptionTests(unittest.TestCase):

    def test_empty_images_returns_empty(self):
        self.assertEqual("", generate_scene_sheet_description([], 2, (1280, 704)))

    def test_single_actor(self):
        images = [{"type": "actor", "visual_description": "a young woman with long brown hair"}]
        desc = generate_scene_sheet_description(images, 1, (1280, 704))
        self.assertIn("### Reference Sheet Description", desc)
        self.assertIn("**Full (Character):** a young woman with long brown hair", desc)

    def test_two_images_left_right(self):
        images = [
            {"type": "actor", "visual_description": "a woman with brown hair"},
            {"type": "location", "visual_description": "a cobblestone alley"},
        ]
        desc = generate_scene_sheet_description(images, 2, (1280, 704))
        lines = desc.split("\n")
        self.assertIn("**Left (Character):** a woman with brown hair", lines)
        self.assertIn("**Right (Setting):** a cobblestone alley", lines)

    def test_three_images_last_row_single(self):
        images = [
            {"type": "actor", "visual_description": "actor one"},
            {"type": "actor", "visual_description": "actor two"},
            {"type": "location", "visual_description": "a garden"},
        ]
        desc = generate_scene_sheet_description(images, 2, (1280, 704))
        lines = desc.split("\n")
        self.assertIn("**Top Row Left (Character):** actor one", lines)
        self.assertIn("**Top Row Right (Character):** actor two", lines)
        self.assertIn("**Bottom Row (Setting):** a garden", lines)

    def test_fallback_to_name_when_no_visual_description(self):
        images = [{"type": "actor", "name": "Alice"}]
        desc = generate_scene_sheet_description(images, 1, (1280, 704))
        self.assertIn("Alice", desc)

    def test_description_starts_with_header(self):
        images = [{"type": "actor", "visual_description": "test"}]
        desc = generate_scene_sheet_description(images, 1, (1280, 704))
        self.assertTrue(desc.startswith("### Reference Sheet Description"))


class SceneReferenceSheetBuilderDescriptionTests(unittest.TestCase):

    @staticmethod
    def _make_image(w: int, h: int, color: tuple[int, int, int], parent: Path) -> Path:
        p = parent / f"img_{w}x{h}_{color[0]}_{color[1]}_{color[2]}.png"
        Image.new("RGB", (w, h), color=color).save(p)
        return p

    def _make_project(self, tmp: Path) -> Path:
        movie = tmp / "movie"
        refs = movie / "references"
        actors_dir = refs / "actors" / "actor_1"
        actors_dir.mkdir(parents=True, exist_ok=True)
        locs_dir = refs / "locations" / "loc_1"
        locs_dir.mkdir(parents=True, exist_ok=True)
        return tmp

    def test_builder_returns_description_with_visual_descriptions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            actor_sheet = self._make_image(200, 200, (255, 0, 0), tmp / "movie" / "references" / "actors" / "actor_1")
            loc_sheet = self._make_image(200, 200, (0, 0, 255), tmp / "movie" / "references" / "locations" / "loc_1")

            manifest = {
                "actors": [{
                    "id": "actor_1",
                    "name": "Alice",
                    "visual_description": "a young woman with long brown hair",
                    "msr_sheet_path": actor_sheet.relative_to(tmp).as_posix(),
                    "sheet_path": actor_sheet.relative_to(tmp).as_posix(),
                }],
                "locations": [{
                    "id": "loc_1",
                    "name": "Garden",
                    "visual_description": "a bright sunlit garden with tall trees",
                    "msr_sheet_path": loc_sheet.relative_to(tmp).as_posix(),
                    "sheet_path": loc_sheet.relative_to(tmp).as_posix(),
                }],
            }
            shot = {"shot_id": "shot_001", "scene": 1, "reference_ids": {"actors": ["actor_1"], "location": "loc_1"}}

            builder = SceneReferenceSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertIn("scene_reference_sheet_description", result)
            desc = result["scene_reference_sheet_description"]
            self.assertIn("### Reference Sheet Description", desc)
            self.assertIn("Character", desc)
            self.assertIn("Setting", desc)
            self.assertIn("young woman with long brown hair", desc)
            self.assertIn("bright sunlit garden", desc)

    def test_builder_empty_no_description(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            manifest = {"actors": [], "locations": []}
            shot = {"shot_id": "shot_002", "scene": 1}

            builder = SceneReferenceSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertIn("scene_reference_sheet_description", result)
            self.assertEqual("", result["scene_reference_sheet_description"])


class MSREnrichmentSceneSheetDescriptionTests(unittest.TestCase):

    def test_enrichment_passes_description_to_shot(self):
        from feverslop.application import movie_msr_enrichment

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            movie = tmp / "movie"
            refs = movie / "references"
            refs.mkdir(parents=True, exist_ok=True)
            (movie / "bible.json").write_text(
                '{"runtime_constraints": {"fps": 24}, "actors": [], "locations": []}',
                encoding="utf-8",
            )
            (movie / "render_plan.json").write_text(
                json.dumps({
                    "resolution": {"width": 1280, "height": 720},
                    "shots": [
                        {
                            "shot_id": "shot_001",
                            "scene": 1,
                            "duration_seconds": 2,
                            "description": "A test scene",
                            "reference_ids": {"actors": [], "location": ""},
                        }
                    ],
                }),
                encoding="utf-8",
            )
            (refs / "manifest.json").write_text(
                json.dumps({"actors": [], "locations": []}),
                encoding="utf-8",
            )

            original_builder = movie_msr_enrichment.SceneReferenceSheetBuilder

            class DescBuilder:
                def build(self, shot):
                    return {
                        "sheet_path": "movie/scene_sheets/shot_001_scene.png",
                        "image_count": 1,
                        "images": [],
                        "scene_reference_sheet_description": "### Reference Sheet Description\n**Full (Character):** test description",
                    }

            movie_msr_enrichment.SceneReferenceSheetBuilder = lambda *a, **kw: DescBuilder()
            try:
                movie_msr_enrichment.enrich_movie_render_plan_with_msr_prompts(project_dir=tmp)
                output = json.loads((tmp / "movie" / "render_plan_msr.json").read_text(encoding="utf-8"))
            finally:
                movie_msr_enrichment.SceneReferenceSheetBuilder = original_builder

            shot = output["shots"][0]
            self.assertIn("scene_reference_sheet_description", shot)
            self.assertIn("### Reference Sheet Description", shot["scene_reference_sheet_description"])
            self.assertIn("scene_reference_sheet_description", shot.get("ltx", {}))
