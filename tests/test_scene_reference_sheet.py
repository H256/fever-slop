import json
import math
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from feverslop.application.movie_ingredients_sheets import (
    IngredientsSceneSheetBuilder,
    enrich_movie_render_plan_with_ingredients_sheets,
)
from feverslop.application.reference_bible import (
    _fit_contain_image,
    _panel_position_label,
    _type_label,
    compose_scene_reference_sheet,
    generate_scene_sheet_description,
    ingredients_sheet_size,
)


class FitContainImageTests(unittest.TestCase):

    def test_fit_contain_wide_into_tall_letterboxes(self):
        src = Image.new("RGB", (200, 100), color=(255, 0, 0))
        result = _fit_contain_image(src, (100, 200))
        self.assertEqual((100, 200), result.size)
        self.assertEqual((0, 0, 0), result.getpixel((0, 0)))
        self.assertEqual((255, 0, 0), result.getpixel((50, 90)))

    def test_fit_contain_tall_into_wide_letterboxes(self):
        src = Image.new("RGB", (100, 200), color=(0, 0, 255))
        result = _fit_contain_image(src, (200, 100))
        self.assertEqual((200, 100), result.size)
        self.assertEqual((0, 0, 0), result.getpixel((0, 0)))
        self.assertEqual((0, 0, 255), result.getpixel((90, 50)))


class IngredientsSheetSizeTests(unittest.TestCase):

    def test_uses_two_times_project_size_and_expands_to_twelve_by_seven(self):
        self.assertEqual((3840, 2240), ingredients_sheet_size(1920, 1088))

    def test_never_shrinks_below_scaled_project_dimensions(self):
        width, height = ingredients_sheet_size(1280, 720, 2.0)
        self.assertGreaterEqual(width, 2560)
        self.assertGreaterEqual(height, 1440)
        self.assertEqual(width * 7, height * 12)

    def test_fit_contain_same_size_returns_copy(self):
        src = Image.new("RGB", (100, 100), color=(0, 255, 0))
        result = _fit_contain_image(src, (100, 100))
        self.assertEqual((100, 100), result.size)
        self.assertEqual((0, 255, 0), result.getpixel((0, 0)))
        self.assertEqual((0, 255, 0), result.getpixel((99, 99)))
        self.assertIsNot(src, result)

    def test_fit_contain_smaller_scales_up_to_fill(self):
        src = Image.new("RGB", (50, 50), color=(128, 128, 128))
        result = _fit_contain_image(src, (100, 100))
        self.assertEqual((100, 100), result.size)
        self.assertEqual((128, 128, 128), result.getpixel((0, 0)))
        self.assertEqual((128, 128, 128), result.getpixel((99, 99)))

    def test_fit_contain_output_dimensions_exact(self):
        for target in ((300, 200), (200, 300), (100, 100), (1, 1)):
            src = Image.new("RGB", (50, 80), color=(100, 50, 50))
            result = _fit_contain_image(src, target)
            self.assertEqual(target, result.size, f"Failed for target {target}")

    def test_fit_contain_centered_placement(self):
        src = Image.new("RGB", (100, 50), color=(200, 100, 50))
        result = _fit_contain_image(src, (200, 200))
        self.assertEqual((200, 200), result.size)
        self.assertEqual((0, 0, 0), result.getpixel((0, 0)))
        self.assertEqual((0, 0, 0), result.getpixel((0, 199)))
        fitted = result.getpixel((100, 100))
        self.assertEqual((200, 100, 50), fitted)

    def test_fit_contain_custom_background_visible_when_letterboxed(self):
        src = Image.new("RGB", (200, 100), color=(255, 255, 255))
        result = _fit_contain_image(src, (100, 100), bg=(128, 0, 128))
        self.assertEqual((100, 100), result.size)
        self.assertEqual((128, 0, 128), result.getpixel((0, 0)))
        self.assertEqual((255, 255, 255), result.getpixel((50, 50)))


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
                self.assertEqual((0, 0, 0), sheet.getpixel((0, 0)))
                self.assertEqual((0, 0, 0), sheet.getpixel((399, 0)))

    def test_compose_scene_sheet_ceil_sqrt_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imgs = [
                self._make_image(100, 100, (255, 0, 0), tmp),
                self._make_image(100, 100, (0, 255, 0), tmp),
                self._make_image(100, 100, (0, 0, 255), tmp),
            ]
            out = tmp / "sqrt_layout.png"
            compose_scene_reference_sheet(imgs, out, size=(600, 400))
            with Image.open(out) as sheet:
                self.assertEqual((600, 400), sheet.size)
                cols = math.ceil(math.sqrt(3))
                self.assertEqual(2, cols)

    def test_compose_scene_sheet_five_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imgs = [
                self._make_image(100, 100, (255, 0, 0), tmp),
                self._make_image(100, 100, (0, 255, 0), tmp),
                self._make_image(100, 100, (0, 0, 255), tmp),
                self._make_image(100, 100, (255, 255, 0), tmp),
                self._make_image(100, 100, (255, 0, 255), tmp),
            ]
            out = tmp / "five_imgs.png"
            compose_scene_reference_sheet(imgs, out, size=(600, 400))
            with Image.open(out) as sheet:
                self.assertEqual((600, 400), sheet.size)
                cols = math.ceil(math.sqrt(5))
                self.assertEqual(3, cols)

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

    def test_compose_scene_sheet_black_background(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img = self._make_image(50, 50, (255, 255, 255), tmp)
            out = tmp / "black_bg.png"
            compose_scene_reference_sheet([img], out, size=(400, 300))
            with Image.open(out) as sheet:
                self.assertEqual((0, 0, 0), sheet.getpixel((0, 0)))
                self.assertEqual((0, 0, 0), sheet.getpixel((399, 299)))

    def test_compose_scene_sheet_gap_between_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imgs = [
                self._make_image(100, 100, (255, 0, 0), tmp),
                self._make_image(100, 100, (0, 255, 0), tmp),
            ]
            out = tmp / "gap_test.png"
            compose_scene_reference_sheet(imgs, out, size=(600, 400))
            with Image.open(out) as sheet:
                self.assertEqual((600, 400), sheet.size)
                gap = 16
                cols = math.ceil(math.sqrt(2))
                cell_w = (600 - gap * (cols + 1)) // cols
                mid = gap + cell_w + gap // 2
                mid_pixel = sheet.getpixel((mid, 200))
                self.assertEqual((0, 0, 0), mid_pixel)


class IngredientsSheetBuilderTests(unittest.TestCase):

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
                "actors": [{"id": "actor_1", "name": "Alice", "sheet_path": actor_sheet.relative_to(tmp).as_posix()}],
                "locations": [{"id": "loc_1", "name": "Room", "sheet_path": loc_sheet.relative_to(tmp).as_posix()}],
            }
            shot = {"shot_id": "shot_001", "scene": 1, "reference_ids": {"actors": ["actor_1"], "location": "loc_1"}}

            builder = IngredientsSceneSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertEqual(2, result["image_count"])
            self.assertEqual("movie/ingredients_sheets/shot_001_ingredients.png", result["sheet_path"])
            self.assertTrue((tmp / "movie" / "ingredients_sheets" / "shot_001_ingredients.png").exists())
            self.assertEqual(2, len(result["images"]))

    def test_builder_no_references_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            manifest = {"actors": [], "locations": []}
            shot = {"shot_id": "shot_002", "scene": 1}

            builder = IngredientsSceneSheetBuilder(project_dir=tmp, manifest=manifest)
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
                "actors": [{"id": "actor_1", "name": "Alice", "sheet_path": actor_sheet.relative_to(tmp).as_posix()}],
                "locations": [{"id": "loc_1", "name": "Room", "sheet_path": "movie/references/locations/loc_1/nonexistent.png"}],
            }
            shot = {"shot_id": "shot_003", "scene": 1, "reference_ids": {"actors": ["actor_1"], "location": "loc_1"}}

            builder = IngredientsSceneSheetBuilder(project_dir=tmp, manifest=manifest)
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
                "actors": [{"id": "actor_1", "name": "Alice", "sheet_path": f"movie/references/{short_name}"}],
                "locations": [],
            }
            shot = {"shot_id": "shot_004", "scene": 1, "reference_ids": {"actors": ["actor_1"], "location": ""}}

            builder = IngredientsSceneSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertEqual(1, result["image_count"])
            self.assertTrue(result["images"][0]["path"].startswith("movie/references/"))

    def test_builder_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            actor_sheet = self._make_image(200, 200, (0, 255, 0), tmp / "movie" / "references" / "actors" / "actor_1")

            manifest = {
                "actors": [{"id": "actor_1", "name": "Alice", "sheet_path": actor_sheet.relative_to(tmp).as_posix()}],
                "locations": [],
            }
            ingredients_sheets = tmp / "movie" / "ingredients_sheets"
            self.assertFalse(ingredients_sheets.exists())
            shot = {"shot_id": "shot_005", "scene": 1, "reference_ids": {"actors": ["actor_1"]}}

            builder = IngredientsSceneSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertTrue(ingredients_sheets.exists())
            self.assertEqual(1, result["image_count"])

    def test_builder_returns_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            actor_sheet = self._make_image(200, 200, (255, 255, 0), tmp / "movie" / "references" / "actors" / "actor_1")

            manifest = {
                "actors": [{"id": "actor_1", "name": "Alice", "sheet_path": actor_sheet.relative_to(tmp).as_posix()}],
                "locations": [],
            }
            shot = {"shot_id": "shot_006", "scene": 2, "reference_ids": {"actors": ["actor_1"]}}

            builder = IngredientsSceneSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertFalse(result["sheet_path"].startswith("/"))
            self.assertNotIn("\\", result["sheet_path"])
            self.assertTrue(result["sheet_path"].startswith("movie/ingredients_sheets/"))

    def test_builder_uses_sheet_path_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            actor_sheet = self._make_image(200, 200, (255, 0, 0), tmp / "movie" / "references" / "actors" / "actor_1")

            manifest = {
                "actors": [{
                    "id": "actor_1",
                    "name": "Alice",
                    "msr_sheet_path": "movie/references/actors/actor_1/nonexistent_msr.png",
                    "sheet_path": actor_sheet.relative_to(tmp).as_posix(),
                }],
                "locations": [],
            }
            shot = {"shot_id": "shot_007", "scene": 1, "reference_ids": {"actors": ["actor_1"]}}

            builder = IngredientsSceneSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertEqual(1, result["image_count"])
            self.assertEqual(
                actor_sheet.relative_to(tmp).as_posix(),
                result["images"][0]["path"],
            )

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
                    "sheet_path": actor_sheet.relative_to(tmp).as_posix(),
                }],
                "locations": [{
                    "id": "loc_1",
                    "name": "Garden",
                    "visual_description": "a bright sunlit garden with tall trees",
                    "sheet_path": loc_sheet.relative_to(tmp).as_posix(),
                }],
            }
            shot = {"shot_id": "shot_008", "scene": 1, "reference_ids": {"actors": ["actor_1"], "location": "loc_1"}}

            builder = IngredientsSceneSheetBuilder(project_dir=tmp, manifest=manifest)
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
            shot = {"shot_id": "shot_009", "scene": 1}

            builder = IngredientsSceneSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertIn("scene_reference_sheet_description", result)
            self.assertEqual("", result["scene_reference_sheet_description"])

    def test_builder_ceil_sqrt_layout_for_three_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            actors_dir = tmp / "movie" / "references" / "actors"
            imgs = []
            for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)], 1):
                d = actors_dir / f"actor_{i}"
                d.mkdir(parents=True, exist_ok=True)
                p = self._make_image(200, 200, color, d)
                imgs.append(p.relative_to(tmp).as_posix())

            manifest = {
                "actors": [
                    {"id": f"actor_{i}", "name": f"A{i}", "sheet_path": p}
                    for i, p in enumerate(imgs, 1)
                ],
                "locations": [],
            }
            shot = {
                "shot_id": "shot_010",
                "scene": 1,
                "reference_ids": {"actors": ["actor_1", "actor_2", "actor_3"]},
            }

            builder = IngredientsSceneSheetBuilder(project_dir=tmp, manifest=manifest)
            result = builder.build(shot)

            self.assertEqual(3, result["image_count"])
            with Image.open(tmp / "movie" / "ingredients_sheets" / "shot_010_ingredients.png") as _sheet:
                cols = math.ceil(math.sqrt(3))
                self.assertEqual(2, cols)


class IngredientsEnrichmentWiringTests(unittest.TestCase):

    def _make_project(self, tmp, with_bible=True, shot_fields=None):
        movie = tmp / "movie"
        refs = movie / "references"
        refs.mkdir(parents=True, exist_ok=True)
        if shot_fields is None:
            shot_fields = {}
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
                        **shot_fields,
                    }
                ],
            }),
            encoding="utf-8",
        )
        (refs / "manifest.json").write_text(
            json.dumps({
                "actors": [{"id": "actor_1", "name": "Alice", "visual_description": "a woman with brown hair"}],
                "locations": [{"id": "loc_1", "name": "Room", "visual_description": "a quiet room"}],
            }),
            encoding="utf-8",
        )
        if with_bible:
            (movie / "bible.json").write_text(
                json.dumps({
                    "title": "Test",
                    "actors": [{"id": "actor_1", "name": "Alice"}],
                    "locations": [{"id": "loc_1", "name": "Room"}],
                    "runtime_constraints": {"fps": 24},
                }),
                encoding="utf-8",
            )
        return tmp

    def test_enrichment_writes_render_plan_ingredients(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            out = enrich_movie_render_plan_with_ingredients_sheets(project_dir=tmp)

            self.assertEqual(tmp / "movie" / "render_plan_ingredients.json", out)
            self.assertTrue(out.exists())

            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(data["ingredients_enriched"])
            self.assertIn("movie_bible_path", data)
            self.assertIn("reference_manifest_path", data)
            self.assertEqual(1, len(data["shots"]))
            self.assertIn("ingredients_scene_sheet", data["shots"][0])
            self.assertIn("ingredients_scene_sheet_description", data["shots"][0])
            self.assertIn("ltx", data["shots"][0])

    def test_enrichment_shot_has_ltx_with_ingredients_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp)
            out = enrich_movie_render_plan_with_ingredients_sheets(project_dir=tmp)

            data = json.loads(out.read_text(encoding="utf-8"))
            shot = data["shots"][0]
            ltx = shot["ltx"]
            self.assertIn("ingredients_scene_sheet_description", ltx)
            self.assertTrue(ltx["native_audio"])

    def test_enrichment_no_references_still_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            movie = tmp / "movie"
            refs = movie / "references"
            refs.mkdir(parents=True, exist_ok=True)
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

            out = enrich_movie_render_plan_with_ingredients_sheets(project_dir=tmp)
            data = json.loads(out.read_text(encoding="utf-8"))

            shot = data["shots"][0]
            self.assertEqual("", shot["ingredients_scene_sheet"])
            self.assertEqual("", shot["ingredients_scene_sheet_description"])

    def test_enrichment_shot_has_ingredients_target_prompt_at_both_levels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp, shot_fields={
                "camera": "slow dolly",
                "acting": "controlled fear",
                "action": "opens the ledger",
                "dialogue": "Alice: It remembers me.",
            })
            out = enrich_movie_render_plan_with_ingredients_sheets(project_dir=tmp)
            data = json.loads(out.read_text(encoding="utf-8"))
            shot = data["shots"][0]

            self.assertIn("ingredients_target_prompt", shot)
            self.assertIn("ingredients_target_prompt", shot["ltx"])
            self.assertEqual(shot["ingredients_target_prompt"], shot["ltx"]["ingredients_target_prompt"])
            prompt = shot["ingredients_target_prompt"]
            self.assertTrue(prompt.startswith("### Target Description\n"))
            self.assertIn("slow dolly", prompt)
            self.assertIn("controlled fear", prompt)
            self.assertNotIn("Full-body cinematic character reference sheet", prompt)
            self.assertNotIn("Four vertical panels", prompt)

    def test_enrichment_target_prompt_includes_actor_names_and_dialogue_language(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            movie = tmp / "movie"
            refs = movie / "references"
            refs.mkdir(parents=True, exist_ok=True)
            (movie / "render_plan.json").write_text(
                json.dumps({
                    "resolution": {"width": 1280, "height": 720},
                    "shots": [
                        {
                            "shot_id": "shot_001",
                            "scene": 1,
                            "duration_seconds": 2,
                            "description": "Mara speaks",
                            "dialogue": "MARA: Hola mundo.",
                            "reference_ids": {"actors": ["mara"], "location": "loc_1"},
                        }
                    ],
                }),
                encoding="utf-8",
            )
            (refs / "manifest.json").write_text(
                json.dumps({
                    "actors": [{"id": "mara", "name": "Mara", "visual_description": "stern archivist"}],
                    "locations": [{"id": "loc_1", "name": "Room", "visual_description": "a quiet room"}],
                }),
                encoding="utf-8",
            )
            (movie / "bible.json").write_text(
                json.dumps({
                    "title": "Test",
                    "actors": [{"id": "mara", "name": "Mara"}],
                    "locations": [{"id": "loc_1", "name": "Room"}],
                    "runtime_constraints": {"fps": 24, "dialogue_language": "Spanish"},
                }),
                encoding="utf-8",
            )

            out = enrich_movie_render_plan_with_ingredients_sheets(project_dir=tmp)
            data = json.loads(out.read_text(encoding="utf-8"))
            prompt = data["shots"][0]["ingredients_target_prompt"]

            self.assertIn("Mara", prompt)
            self.assertIn("Spanish", prompt)
            self.assertIn("Hola mundo", prompt)

    def test_enrichment_target_prompt_graceful_without_bible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_project(tmp, with_bible=False)
            out = enrich_movie_render_plan_with_ingredients_sheets(project_dir=tmp)
            data = json.loads(out.read_text(encoding="utf-8"))
            prompt = data["shots"][0]["ingredients_target_prompt"]

            self.assertIsInstance(prompt, str)
            self.assertIn("A test scene", prompt)


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
        self.assertEqual("", generate_scene_sheet_description([], 1, (1280, 704)))

    def test_single_actor(self):
        images = [{"type": "actor", "id": "leo", "visual_description": "a young woman with long brown hair"}]
        num_cols = math.ceil(math.sqrt(len(images)))
        desc = generate_scene_sheet_description(images, num_cols, (1280, 704))
        self.assertIn("### Reference Sheet Description", desc)
        self.assertIn("**Full (Character, leo):** a young woman with long brown hair", desc)

    def test_two_images_left_right(self):
        images = [
            {"type": "actor", "id": "leo", "visual_description": "a woman with brown hair"},
            {"type": "location", "id": "forest", "visual_description": "a cobblestone alley"},
        ]
        num_cols = math.ceil(math.sqrt(len(images)))
        desc = generate_scene_sheet_description(images, num_cols, (1280, 704))
        lines = desc.split("\n")
        self.assertIn("**Left (Character, leo):** a woman with brown hair", lines)
        self.assertIn("**Right (Setting, forest):** a cobblestone alley", lines)

    def test_three_images_last_row_single(self):
        images = [
            {"type": "actor", "id": "actor_one", "visual_description": "actor one"},
            {"type": "actor", "id": "actor_two", "visual_description": "actor two"},
            {"type": "location", "id": "garden", "visual_description": "a garden"},
        ]
        num_cols = math.ceil(math.sqrt(len(images)))
        self.assertEqual(2, num_cols)
        desc = generate_scene_sheet_description(images, num_cols, (1280, 704))
        lines = desc.split("\n")
        self.assertIn("**Top Row Left (Character, actor_one):** actor one", lines)
        self.assertIn("**Top Row Right (Character, actor_two):** actor two", lines)
        self.assertIn("**Bottom Row (Setting, garden):** a garden", lines)

    def test_fallback_to_name_when_no_visual_description(self):
        images = [{"type": "actor", "id": "alice", "name": "Alice"}]
        num_cols = math.ceil(math.sqrt(len(images)))
        desc = generate_scene_sheet_description(images, num_cols, (1280, 704))
        self.assertIn("Alice", desc)
        self.assertIn("alice", desc)

    def test_description_starts_with_header(self):
        images = [{"type": "actor", "id": "test_id", "visual_description": "test"}]
        num_cols = math.ceil(math.sqrt(len(images)))
        desc = generate_scene_sheet_description(images, num_cols, (1280, 704))
        self.assertTrue(desc.startswith("### Reference Sheet Description"))

    def test_no_anchor_omits_id(self):
        images = [{"type": "actor", "visual_description": "a mysterious figure"}]
        num_cols = math.ceil(math.sqrt(len(images)))
        desc = generate_scene_sheet_description(images, num_cols, (1280, 704))
        self.assertIn("**Full (Character):** a mysterious figure", desc)
