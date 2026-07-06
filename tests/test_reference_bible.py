import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from feverslop.application.reference_bible import (
    ReferenceBibleGenerator,
    ReferenceLocation,
    ReferenceSubject,
    compose_msr_reference_sheet,
    compose_reference_sheet,
)


class FakeImageBackend:
    def __init__(self):
        self.requests = []

    def render_image(self, request):
        self.requests.append(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        output = request.output_dir / f"scene_{request.scene_number:04}.png"
        Image.new("RGB", (32, 24), color=(request.scene_number * 20, 0, 0)).save(output)
        return output


class ReferenceBibleTests(unittest.TestCase):
    def test_generator_writes_manifest_views_and_sheet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "project"
            output_dir = project_dir / "output" / "references"
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
            self.assertEqual("output/references/actors/singer/msr_sheet.png", manifest["msr_input_path"])
            self.assertEqual("output/references/actors/singer/sheet.png", manifest["sheet_path"])
            self.assertEqual("output/references/actors/singer/views/hero.png", manifest["views"][0]["path"])
            self.assertEqual(["hero", "front", "left", "right", "closeup"], [view["name"] for view in manifest["views"]])

    def test_generator_uses_hero_as_reference_for_edit_views(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            hero_backend = FakeImageBackend()
            edit_backend = FakeImageBackend()
            generator = ReferenceBibleGenerator(
                backend=hero_backend,
                edit_backend=edit_backend,
                output_dir=output_dir,
            )

            generator.generate_subject_bible(
                ReferenceSubject(
                    id="singer",
                    name="Mara",
                    image_prompt="portrait of Mara",
                )
            )

            self.assertEqual(1, len(hero_backend.requests))
            self.assertEqual(4, len(edit_backend.requests))
            self.assertTrue(all(request.reference_image.name == "hero.png" for request in edit_backend.requests))

    def test_generator_uses_short_reference_based_prompts_for_actor_edit_views(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            hero_backend = FakeImageBackend()
            edit_backend = FakeImageBackend()
            generator = ReferenceBibleGenerator(
                backend=hero_backend,
                edit_backend=edit_backend,
                output_dir=output_dir,
                actor_view_names=("hero_closeup", "front", "left", "back"),
            )

            generator.generate_subject_bible(
                ReferenceSubject(
                    id="singer",
                    name="Mara",
                    image_prompt="portrait of Mara in a forest with emerald robes and silver hair",
                )
            )

            self.assertIn("portrait of Mara in a forest", hero_backend.requests[0].prompt)
            edit_prompts = [request.prompt for request in edit_backend.requests]
            self.assertTrue(edit_prompts[0].startswith("Create a straight front view of the character from the reference image."))
            self.assertTrue(edit_prompts[1].startswith("Create a left-side view of the character from the reference image."))
            self.assertTrue(edit_prompts[2].startswith("Create a full-body back view of the character from the reference image."))
            for prompt in edit_prompts:
                self.assertNotIn("portrait of Mara in a forest", prompt)
                self.assertIn("plain white seamless studio background", prompt)
                self.assertIn("no environment", prompt)
                self.assertIn("no scenery", prompt)
                self.assertIn("no extra characters", prompt)

    def test_generator_requests_portrait_actor_hero_for_msr_and_leaves_edit_views_reference_sized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            hero_backend = FakeImageBackend()
            edit_backend = FakeImageBackend()
            generator = ReferenceBibleGenerator(
                backend=hero_backend,
                edit_backend=edit_backend,
                output_dir=output_dir,
            )

            generator.generate_subject_bible(
                ReferenceSubject(
                    id="singer",
                    name="Mara",
                    image_prompt="portrait of Mara",
                )
            )

            self.assertEqual((1088, 1920), (hero_backend.requests[0].width, hero_backend.requests[0].height))
            self.assertTrue(all(request.width is None for request in edit_backend.requests))
            self.assertTrue(all(request.height is None for request in edit_backend.requests))

    def test_generator_can_render_only_msr_hero_view(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            hero_backend = FakeImageBackend()
            edit_backend = FakeImageBackend()
            generator = ReferenceBibleGenerator(
                backend=hero_backend,
                edit_backend=edit_backend,
                output_dir=output_dir,
                view_names=("hero",),
            )

            manifest_path = generator.generate_subject_bible(
                ReferenceSubject(id="singer", name="Mara", image_prompt="portrait of Mara")
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(["hero"], [view["name"] for view in manifest["views"]])
            self.assertEqual(1, len(hero_backend.requests))
            self.assertEqual(0, len(edit_backend.requests))
            self.assertTrue((manifest_path.parent / "sheet.png").exists())

    def test_generator_writes_msr_actor_sheet_as_reference_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "project"
            output_dir = project_dir / "output" / "references"
            hero_backend = FakeImageBackend()
            edit_backend = FakeImageBackend()
            generator = ReferenceBibleGenerator(
                backend=hero_backend,
                edit_backend=edit_backend,
                output_dir=output_dir,
                actor_view_names=("hero_closeup", "front", "left", "back"),
                location_view_names=("hero",),
                msr_sheet_size=(1280, 704),
            )

            manifest_path = generator.generate_subject_bible(
                ReferenceSubject(id="warrior", name="Warrior", image_prompt="dark fantasy warrior")
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                ["hero_closeup", "front", "left", "back"],
                [view["name"] for view in manifest["views"]],
            )
            self.assertEqual("output/references/actors/warrior/msr_sheet.png", manifest["msr_input_path"])
            with Image.open(manifest_path.parent / "msr_sheet.png") as sheet:
                self.assertEqual((1280, 704), sheet.size)

    def test_generator_can_render_direct_msr_actor_sheet_without_edit_views(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "project"
            output_dir = project_dir / "output" / "references"
            hero_backend = FakeImageBackend()
            edit_backend = FakeImageBackend()
            generator = ReferenceBibleGenerator(
                backend=hero_backend,
                edit_backend=edit_backend,
                output_dir=output_dir,
                actor_view_names=("msr_sheet",),
                location_view_names=("hero",),
                msr_sheet_size=(1920, 1088),
            )

            manifest_path = generator.generate_subject_bible(
                ReferenceSubject(
                    id="warrior",
                    name="Warrior",
                    image_prompt="dark fantasy warrior with black armor",
                )
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(1, len(hero_backend.requests))
            self.assertEqual(0, len(edit_backend.requests))
            request = hero_backend.requests[0]
            self.assertEqual((1920, 1088), (request.width, request.height))
            self.assertTrue(request.prompt.startswith("vertical four panel character sheet photos."))
            self.assertIn("dark fantasy warrior with black armor", request.prompt)
            self.assertIn("1st panel is a closeup", request.prompt)
            self.assertIn("2nd panel is front view", request.prompt)
            self.assertIn("3rd panel is left view", request.prompt)
            self.assertIn("4th panel is back view", request.prompt)
            self.assertIn("the panel background is white", request.prompt)
            self.assertEqual(["msr_sheet"], [view["name"] for view in manifest["views"]])
            self.assertEqual("output/references/actors/warrior/msr_sheet.png", manifest["msr_input_path"])
            self.assertEqual("output/references/actors/warrior/sheet.png", manifest["sheet_path"])
            self.assertEqual("output/references/actors/warrior/views/msr_sheet.png", manifest["views"][0]["path"])
            self.assertTrue((manifest_path.parent / "views" / "msr_sheet.png").exists())
            self.assertTrue((manifest_path.parent / "sheet.png").exists())
            self.assertTrue((manifest_path.parent / "msr_sheet.png").exists())

    def test_generator_writes_location_manifest_views_and_sheet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "project"
            output_dir = project_dir / "output" / "references"
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
            self.assertEqual("output/references/locations/stage/views/hero.png", manifest["msr_background_path"])

    def test_generator_requests_wide_full_hd_location_hero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            hero_backend = FakeImageBackend()
            edit_backend = FakeImageBackend()
            generator = ReferenceBibleGenerator(
                backend=hero_backend,
                edit_backend=edit_backend,
                output_dir=output_dir,
                view_names=("hero",),
            )

            generator.generate_location_bible(
                ReferenceLocation(id="stage", name="Stage", image_prompt="wide stage")
            )

            self.assertEqual((1920, 1088), (hero_backend.requests[0].width, hero_backend.requests[0].height))

    def test_generator_reports_progress_for_each_subject_view(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events = []
            generator = ReferenceBibleGenerator(
                backend=FakeImageBackend(),
                output_dir=Path(temp_dir),
                on_view_complete=lambda event: events.append(event),
            )

            generator.generate_subject_bible(
                ReferenceSubject(id="singer", name="Mara", image_prompt="portrait")
            )

            self.assertEqual(["hero", "front", "left", "right", "closeup"], [event["view"] for event in events])
            self.assertEqual("actor", events[0]["kind"])
            self.assertEqual(5, events[-1]["item_total"])

    def test_actor_hero_prompt_requires_full_body_portrait_reference_frame(self):
        prompt = ReferenceBibleGenerator._view_prompt(
            ReferenceSubject(id="singer", name="Mara", image_prompt="portrait of Mara"),
            "hero",
        )

        self.assertIn("full-body", prompt)
        self.assertIn("portrait reference frame", prompt)
        self.assertIn("empty margin around the full body", prompt)
        self.assertIn("head to toe", prompt)

    def test_actor_turnaround_prompts_keep_full_body_portrait_except_closeup(self):
        front_prompt = ReferenceBibleGenerator._view_prompt(
            ReferenceSubject(id="singer", name="Mara", image_prompt="portrait of Mara"),
            "front",
        )
        closeup_prompt = ReferenceBibleGenerator._view_prompt(
            ReferenceSubject(id="singer", name="Mara", image_prompt="portrait of Mara"),
            "closeup",
        )

        self.assertIn("full-body", front_prompt)
        self.assertIn("portrait reference frame", front_prompt)
        self.assertIn("square portrait crop", closeup_prompt)
        self.assertNotIn("full-body", closeup_prompt)

    def test_msr_actor_sheet_prompts_include_closeup_front_left_and_back(self):
        closeup = ReferenceBibleGenerator._view_prompt(
            ReferenceSubject(id="singer", name="Mara", image_prompt="portrait of Mara"),
            "hero_closeup",
        )
        back = ReferenceBibleGenerator._view_prompt(
            ReferenceSubject(id="singer", name="Mara", image_prompt="portrait of Mara"),
            "back",
        )

        self.assertIn("closeup", closeup)
        self.assertIn("back view", back)
        self.assertIn("full-body", back)
        self.assertIn("portrait reference frame", back)

    def test_direct_msr_actor_sheet_prompt_keeps_music_video_default(self):
        prompt = ReferenceBibleGenerator._direct_msr_sheet_prompt(
            ReferenceSubject(id="goat", name="The Goat Demon", image_prompt="massive horned goat demon")
        )

        self.assertTrue(prompt.startswith("vertical four panel character sheet photos."))
        self.assertIn("massive horned goat demon", prompt)
        self.assertIn("1st panel is a closeup", prompt)
        self.assertIn("2nd panel is front view", prompt)
        self.assertIn("3rd panel is left view", prompt)
        self.assertIn("4th panel is back view", prompt)
        self.assertIn("the panel background is white", prompt)

    def test_direct_msr_actor_sheet_prompt_can_be_injected_by_use_case(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FakeImageBackend()
            generator = ReferenceBibleGenerator(
                backend=backend,
                output_dir=Path(temp_dir),
                actor_view_names=("msr_sheet",),
                direct_msr_sheet_prompt_builder=lambda subject: f"custom prompt for {subject.name}",
            )

            generator.generate_subject_bible(ReferenceSubject(id="goat", name="The Goat Demon"))

            self.assertEqual("custom prompt for The Goat Demon", backend.requests[0].prompt)

    def test_actor_reference_prompts_require_plain_white_background(self):
        front_prompt = ReferenceBibleGenerator._view_prompt(
            ReferenceSubject(id="singer", name="Mara", image_prompt="portrait of Mara in a forest"),
            "front",
        )
        closeup_prompt = ReferenceBibleGenerator._view_prompt(
            ReferenceSubject(id="singer", name="Mara", image_prompt="portrait of Mara in a forest"),
            "hero_closeup",
        )

        for prompt in (front_prompt, closeup_prompt):
            self.assertIn("plain white seamless studio background", prompt)
            self.assertIn("no environment", prompt)
            self.assertIn("no scenery", prompt)

    def test_wide_reference_views_are_composed_as_grid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            image_paths = []
            for index in range(5):
                image_path = temp / f"view_{index}.png"
                Image.new("RGB", (1280, 704), color=(index * 20, 0, 0)).save(image_path)
                image_paths.append(image_path)

            output_path = temp / "sheet.png"
            compose_reference_sheet(image_paths, output_path)

            with Image.open(output_path) as sheet:
                self.assertEqual((3840, 1456), sheet.size)

    def test_reference_sheet_can_be_composed_without_label_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            image_paths = []
            for index in range(4):
                image_path = temp / f"view_{index}.png"
                Image.new("RGB", (1088, 1920), color=(index * 20, 0, 0)).save(image_path)
                image_paths.append(image_path)

            output_path = temp / "sheet.png"
            compose_reference_sheet(image_paths, output_path, labels=False)

            with Image.open(output_path) as sheet:
                self.assertEqual((4352, 1920), sheet.size)

    def test_msr_reference_sheet_crops_cells_without_white_letterbox(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            image_path = temp / "portrait.png"
            Image.new("RGB", (1088, 1920), color=(20, 30, 40)).save(image_path)

            output_path = temp / "msr_sheet.png"
            compose_msr_reference_sheet([image_path], output_path, size=(1280, 704))

            with Image.open(output_path) as sheet:
                self.assertEqual((1280, 704), sheet.size)
                self.assertEqual((20, 30, 40), sheet.getpixel((0, 0)))
