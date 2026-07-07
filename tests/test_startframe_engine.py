import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image


class StartframeEngineTests(unittest.TestCase):
    def test_identity_ledger_splits_face_body_and_wardrobe_contracts(self):
        from feverslop.application.startframe_identity import build_startframe_identity_ledger

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_startframe_project(Path(temp_dir))

            output = build_startframe_identity_ledger(project_dir=project)
            data = json.loads(output.read_text(encoding="utf-8"))

        mara = data["actors"]["mara"]
        self.assertEqual("mara", mara["actor_id"])
        self.assertEqual("movie/references/actors/mara/msr_sheet.png", mara["reference_paths"]["full_body"])
        self.assertIn("gothic archivist", mara["face"]["description"])
        self.assertIn("gothic archivist", mara["body"]["description"])
        self.assertIn("gothic archivist", mara["wardrobe"]["description"])
        self.assertIn("dust", mara["wardrobe"]["may_change"])
        self.assertIn("modern jacket", mara["wardrobe"]["forbidden"])

    def test_startframe_plan_writes_actor_bboxes_and_continuity_requirements(self):
        from feverslop.application.startframe_identity import build_startframe_identity_ledger
        from feverslop.application.startframe_plan import build_startframe_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_startframe_project(Path(temp_dir))
            build_startframe_identity_ledger(project_dir=project)

            output = build_startframe_plan(project_dir=project)
            data = json.loads(output.read_text(encoding="utf-8"))

        shot = data["shots"][0]
        self.assertEqual("shot_0001", shot["shot_id"])
        self.assertEqual("Mara opens the sealed ledger.", shot["startframe_intent"]["action_moment"])
        self.assertEqual(["mara"], [actor["actor_id"] for actor in shot["actors"]])
        self.assertEqual([384, 105, 896, 668], shot["actors"][0]["bbox"])
        self.assertIn("same archive location", shot["continuity_in"]["required_carryovers"])
        self.assertIn("Use the supplied startframe as authoritative", shot["ltx_motion"]["prompt"])

    def test_director_prompts_include_ideogram_json_bboxes_and_wardrobe(self):
        from feverslop.application.startframe_director_prompts import build_startframe_director_prompts
        from feverslop.application.startframe_identity import build_startframe_identity_ledger
        from feverslop.application.startframe_plan import build_startframe_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_startframe_project(Path(temp_dir))
            build_startframe_identity_ledger(project_dir=project)
            build_startframe_plan(project_dir=project)

            output = build_startframe_director_prompts(project_dir=project, candidate_count=3, director_backend="ideogram")
            data = json.loads(output.read_text(encoding="utf-8"))

        prompt_text = data["shots"][0]["positive_prompt"]
        prompt = json.loads(prompt_text)
        self.assertEqual(3, data["shots"][0]["candidate_count"])
        self.assertEqual("Mara opens the sealed ledger.", prompt["scene_summary"])
        self.assertEqual([384, 105, 896, 668], prompt["objects"][0]["bounding_box"])
        self.assertIn("gothic archivist", prompt["objects"][0]["description"])
        self.assertNotIn("continuity_constraints", prompt)
        self.assertNotIn("required_carryovers", prompt)
        self.assertIn("readable text", data["shots"][0]["negative_prompt"])
        self.assertIn("split screen", data["shots"][0]["negative_prompt"])
        self.assertIn("contact sheet", data["shots"][0]["negative_prompt"])

    def test_krea2_director_prompts_are_single_frame_plain_cinematic_prompts(self):
        from feverslop.application.startframe_director_prompts import build_startframe_director_prompts
        from feverslop.application.startframe_identity import build_startframe_identity_ledger
        from feverslop.application.startframe_plan import build_startframe_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_two_actor_startframe_project(Path(temp_dir))
            build_startframe_identity_ledger(project_dir=project)
            build_startframe_plan(project_dir=project)

            output = build_startframe_director_prompts(project_dir=project, candidate_count=2, director_backend="krea2")
            data = json.loads(output.read_text(encoding="utf-8"))

        shot = data["shots"][0]
        self.assertEqual("krea2", shot["director_backend"])
        self.assertEqual("workflows/image_t2i_startframe_krea_v1.json", shot["workflow"])
        self.assertEqual(2, shot["candidate_count"])
        self.assertIn("single cinematic film still", shot["positive_prompt"])
        self.assertIn("Mara", shot["positive_prompt"])
        self.assertIn("charcoal coat", shot["positive_prompt"])
        self.assertIn("Ivo", shot["positive_prompt"])
        self.assertIn("copper jacket", shot["positive_prompt"])
        self.assertNotIn("{", shot["positive_prompt"])
        self.assertNotIn("required_carryovers", shot["positive_prompt"])

    def test_ideogram_director_workflow_exposes_patch_anchors(self):
        workflow = json.loads(Path("workflows/image_t2i_startframe_ideogram_director_v1.json").read_text(encoding="utf-8-sig"))
        titles = {str(node.get("_meta", {}).get("title") or "") for node in workflow.values()}

        self.assertIn("#PROMPT_POSITIVE", titles)
        self.assertIn("#PROMPT_NEGATIVE", titles)
        self.assertIn("#SAVE_IMAGE", titles)
        self.assertIn("#WIDTH", titles)
        self.assertIn("#HEIGHT", titles)

    def test_krea2_director_workflow_exposes_patch_anchors(self):
        workflow = json.loads(Path("workflows/image_t2i_startframe_krea_v1.json").read_text(encoding="utf-8-sig"))
        titles = {str(node.get("_meta", {}).get("title") or "") for node in workflow.values()}
        classes = {str(node.get("class_type") or "") for node in workflow.values()}

        self.assertIn("#PROMPT_POSITIVE", titles)
        self.assertIn("#PROMPT_NEGATIVE", titles)
        self.assertIn("#SAVE_IMAGE", titles)
        self.assertIn("#DIMENSIONS", titles)
        self.assertIn("UNETLoader", classes)
        self.assertIn("CLIPLoader", classes)
        self.assertNotIn("LoraLoaderModelOnly", classes)

    def test_startframe_comfyui_stage_workflows_expose_patch_anchors(self):
        expected = {
            "workflows/image_mask_sam3_actor_regions_v1.json": {
                "classes": {"LoadImage", "easy sam3ModelLoader", "easy sam3ImageSegmentation", "MaskToImage", "SaveImage"},
                "anchors": {"#INPUT_IMAGE", "#SAM3_MODEL", "#SEGMENT_PROMPT", "#MASK_PREVIEW", "#SAVE_MASK"},
            },
            "workflows/image_repair_sdxl_ipadapter_identity_v1.json": {
                "classes": {
                    "LoadImage",
                    "CheckpointLoaderSimple",
                    "easy ipadapterApplyADV",
                    "ImageToMask",
                    "VAEEncode",
                    "SetLatentNoiseMask",
                    "KSampler",
                    "VAEDecode",
                    "SaveImage",
                },
                "anchors": {
                    "#INPUT_IMAGE",
                    "#IDENTITY_REFERENCE",
                    "#REGION_MASK_IMAGE",
                    "#SDXL_CHECKPOINT",
                    "#IPADAPTER_FACEID",
                    "#PROMPT_POSITIVE",
                    "#PROMPT_NEGATIVE",
                    "#DENOISE",
                    "#SAVE_IMAGE",
                },
            },
            "workflows/image_detail_easyuse_startframe_v1.json": {
                "classes": {
                    "LoadImage",
                    "easy fullLoader",
                    "easy preSampling",
                    "easy fullkSampler",
                    "easy ultralyticsDetectorPipe",
                    "easy samLoaderPipe",
                    "easy preDetailerFix",
                    "easy detailerFix",
                    "SaveImage",
                },
                "anchors": {
                    "#INPUT_IMAGE",
                    "#SDXL_CHECKPOINT",
                    "#PROMPT_POSITIVE",
                    "#PROMPT_NEGATIVE",
                    "#FACE_DETECTOR",
                    "#SAM_PIPE",
                    "#DETAILER",
                    "#SAVE_IMAGE",
                },
            },
        }

        for workflow_path, contract in expected.items():
            with self.subTest(workflow=workflow_path):
                workflow = json.loads(Path(workflow_path).read_text(encoding="utf-8-sig"))
                classes = {str(node.get("class_type") or "") for node in workflow.values()}
                titles = {str(node.get("_meta", {}).get("title") or "") for node in workflow.values()}

                self.assertTrue(contract["classes"].issubset(classes))
                self.assertTrue(contract["anchors"].issubset(titles))

    def test_comfyui_startframe_director_orchestrator_runs_all_stage_workflows(self):
        from feverslop.adapters.startframe_director_comfyui import ComfyUIStartframeDirectorVisualAdapter
        from feverslop.application.startframe_director_prompts import build_startframe_director_prompts
        from feverslop.application.startframe_i2v_render_plan import write_startframe_i2v_render_plan
        from feverslop.application.startframe_identity import build_startframe_identity_ledger
        from feverslop.application.startframe_plan import build_startframe_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_two_actor_startframe_project(Path(temp_dir))
            build_startframe_identity_ledger(project_dir=project)
            build_startframe_plan(project_dir=project)
            build_startframe_director_prompts(project_dir=project, candidate_count=1, director_backend="krea2")
            render_plan_path = write_startframe_i2v_render_plan(project_dir=project)
            client = FakeComfyClient()
            validator = FakeGemmaValidator()
            video = FakeVideoUseCase()

            final_video = ComfyUIStartframeDirectorVisualAdapter(
                client=client,
                director_workflow_path=Path("workflows/image_t2i_startframe_krea_v1.json"),
                mask_workflow_path=Path("workflows/image_mask_sam3_actor_regions_v1.json"),
                identity_repair_workflow_path=Path("workflows/image_repair_sdxl_ipadapter_identity_v1.json"),
                detail_workflow_path=Path("workflows/image_detail_easyuse_startframe_v1.json"),
                video_use_case=video,
                validator=validator,
                debug_workflows_dir=project / "output" / "movie" / "startframes" / "debug_workflows",
            ).render_movie(project_dir=project, render_plan_path=render_plan_path)

            self.assertEqual(project / "output" / "movie" / "startframe-director.mp4", final_video)
            self.assertTrue((project / "output" / "movie" / "storyboard" / "final" / "scene_0001.png").exists())
            self.assertEqual(["director", "mask", "repair", "mask", "repair", "detail"], client.stage_titles)
            self.assertEqual(["scene_0001.png"], video.startframes)
            self.assertEqual(1, len(validator.calls))
            validation = json.loads((project / "movie" / "startframe_validation.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["shots"][0]["pass"])

            director = client.workflows[0]
            positive = _node_by_title(director, "#PROMPT_POSITIVE")
            self.assertIn("Mara and Ivo cross the threshold.", positive["inputs"]["text"])
            self.assertEqual(768, _node_by_title(director, "#DIMENSIONS")["inputs"]["width"])
            self.assertEqual(512, _node_by_title(director, "#DIMENSIONS")["inputs"]["height"])
            repair = client.workflows[2]
            mask = client.workflows[1]
            self.assertIn("Mara", _node_by_title(mask, "#SEGMENT_PROMPT")["inputs"]["prompt"])
            self.assertIn("charcoal coat", _node_by_title(mask, "#SEGMENT_PROMPT")["inputs"]["prompt"])
            self.assertIn("feverslop/startframe/director/", _node_by_title(repair, "#INPUT_IMAGE")["inputs"]["image"])
            self.assertIn("feverslop/startframe/masks/", _node_by_title(repair, "#REGION_MASK_IMAGE")["inputs"]["image"])
            debug_dir = project / "output" / "movie" / "startframes" / "debug_workflows"
            self.assertTrue((debug_dir / "scene_0001_director.json").exists())
            self.assertTrue((debug_dir / "scene_0001_mask_mara.json").exists())
            self.assertTrue((debug_dir / "scene_0001_repair_ivo.json").exists())
            self.assertTrue((debug_dir / "scene_0001_detail.json").exists())
            exported_director = json.loads((debug_dir / "scene_0001_director.json").read_text(encoding="utf-8"))
            self.assertIn("Mara and Ivo cross the threshold.", _node_by_title(exported_director, "#PROMPT_POSITIVE")["inputs"]["text"])

    def test_gemma4_validator_normalizes_non_json_text_fallback(self):
        from feverslop.adapters.gemma4_startframe_validator import normalize_validation_response

        result = normalize_validation_response(
            """
            Pass: False
            Score: 0.9
            Issues: ["Action moment discrepancy: ledger is held with both hands."]
            Notes: Identity, wardrobe, and location are consistent.
            """
        )

        self.assertFalse(result["pass"])
        self.assertEqual(0.9, result["score"])
        self.assertEqual(["Action moment discrepancy: ledger is held with both hands."], result["issues"])
        self.assertIn("Identity", result["notes"])


def _write_startframe_project(root: Path) -> Path:
    project = root / "movie-project"
    movie = project / "movie"
    references = movie / "references"
    references.mkdir(parents=True)
    (movie / "bible.json").write_text(
        json.dumps(
            {
                "title": "Archive",
                "actors": [
                    {
                        "id": "mara",
                        "name": "Mara",
                        "visual_description": "gothic archivist in a charcoal coat",
                    }
                ],
                "locations": [
                    {
                        "id": "archive",
                        "name": "Archive",
                        "visual_description": "dusty archive room with a sealed ledger on a table",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (movie / "render_plan.json").write_text(
        json.dumps(
            {
                "title": "Archive",
                "resolution": {"width": 1280, "height": 704},
                "shots": [
                    {
                        "scene": 1,
                        "shot_id": "shot_0001",
                        "description": "Mara opens the sealed ledger.",
                        "action": "Mara opens the sealed ledger.",
                        "camera": "medium shot",
                        "location": "Archive",
                        "location_id": "archive",
                        "duration_seconds": 4,
                        "reference_ids": {"actors": ["mara"], "location": "archive"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (movie / "continuity_plan.json").write_text(
        json.dumps(
            {
                "scene_continuity": {
                    "shot_0001": {
                        "required_carryovers": ["same archive location"],
                        "incoming": ["Mara has just entered the archive"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (references / "manifest.json").write_text(
        json.dumps(
            {
                "actors": [
                    {
                        "id": "mara",
                        "name": "Mara",
                        "visual_description": "gothic archivist in a charcoal coat",
                        "msr_sheet_path": "movie/references/actors/mara/msr_sheet.png",
                    }
                ],
                "locations": [
                    {
                        "id": "archive",
                        "name": "Archive",
                        "visual_description": "dusty archive room with a sealed ledger on a table",
                        "msr_sheet_path": "movie/references/locations/archive/views/hero.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return project


def _write_two_actor_startframe_project(root: Path) -> Path:
    project = root / "two-actor-movie"
    movie = project / "movie"
    references = movie / "references"
    references.mkdir(parents=True)
    (movie / "bible.json").write_text(
        json.dumps(
            {
                "title": "Signal Below",
                "actors": [
                    {"id": "mara", "name": "Mara", "visual_description": "silver-haired archivist in a charcoal coat"},
                    {"id": "ivo", "name": "Ivo", "visual_description": "young courier in a patched copper jacket"},
                ],
                "locations": [
                    {"id": "archive", "name": "Archive", "visual_description": "dusty municipal archive"},
                    {"id": "station", "name": "Station", "visual_description": "abandoned underground signal station"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (movie / "render_plan.json").write_text(
        json.dumps(
            {
                "title": "Signal Below",
                "resolution": {"width": 768, "height": 512},
                "shots": [
                    {
                        "scene": 1,
                        "shot_id": "shot_0001",
                        "description": "Mara and Ivo cross the threshold.",
                        "action": "Mara and Ivo cross the threshold.",
                        "camera": "wide two-shot",
                        "location_id": "archive",
                        "duration_seconds": 3,
                        "reference_ids": {"actors": ["mara", "ivo"], "location": "archive"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (movie / "continuity_plan.json").write_text(
        json.dumps({"scene_continuity": {"shot_0001": {"required_carryovers": ["Mara keeps the ledger under one arm"]}}}),
        encoding="utf-8",
    )
    (references / "manifest.json").write_text(
        json.dumps(
            {
                "actors": [
                    {
                        "id": "mara",
                        "name": "Mara",
                        "visual_description": "silver-haired archivist in a charcoal coat",
                        "msr_sheet_path": "movie/references/actors/mara/msr_sheet.png",
                    },
                    {
                        "id": "ivo",
                        "name": "Ivo",
                        "visual_description": "young courier in a patched copper jacket",
                        "msr_sheet_path": "movie/references/actors/ivo/msr_sheet.png",
                    },
                ],
                "locations": [
                    {
                        "id": "archive",
                        "name": "Archive",
                        "visual_description": "dusty municipal archive",
                        "msr_sheet_path": "movie/references/locations/archive/views/hero.png",
                    },
                    {
                        "id": "station",
                        "name": "Station",
                        "visual_description": "abandoned underground signal station",
                        "msr_sheet_path": "movie/references/locations/station/views/hero.png",
                    },
                ],
                "generator_backend": "local",
            }
        ),
        encoding="utf-8",
    )
    for rel in (
        "movie/references/actors/mara/msr_sheet.png",
        "movie/references/actors/ivo/msr_sheet.png",
        "movie/references/locations/archive/views/hero.png",
        "movie/references/locations/station/views/hero.png",
    ):
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 128), "white").save(path)
    return project


def _node_by_title(workflow: dict, title: str) -> dict:
    return next(node for node in workflow.values() if node.get("_meta", {}).get("title") == title)


class FakeComfyClient:
    def __init__(self):
        self.workflows = []
        self.stage_titles = []
        self._prompt_counter = 0

    def upload_image(self, file_path, subfolder="", file_type="input", overwrite=True, upload_name=None):
        name = upload_name or Path(file_path).name
        return {"name": name, "subfolder": subfolder, "type": file_type}

    @staticmethod
    def comfy_path_from_upload(upload):
        return f"{upload['subfolder']}/{upload['name']}" if upload.get("subfolder") else upload["name"]

    def queue_prompt(self, workflow):
        self.workflows.append(workflow)
        titles = {str(node.get("_meta", {}).get("title") or "") for node in workflow.values()}
        if "#SAVE_MASK" in titles:
            self.stage_titles.append("mask")
        elif "#IPADAPTER_FACEID" in titles:
            self.stage_titles.append("repair")
        elif "#DETAILER" in titles:
            self.stage_titles.append("detail")
        else:
            self.stage_titles.append("director")
        self._prompt_counter += 1
        return f"prompt-{self._prompt_counter}"

    def wait_for_completion(self, prompt_id):
        return {"outputs": {"1": {"images": [{"filename": f"{prompt_id}.png", "subfolder": "", "type": "output"}]}}}

    @staticmethod
    def extract_output_images(history_entry):
        return next(iter(history_entry["outputs"].values()))["images"]

    @staticmethod
    def download_view_file(filename, output_path, subfolder="", file_type="output"):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "white").save(output_path)
        return output_path


class FakeGemmaValidator:
    def __init__(self):
        self.calls = []

    def validate_startframe(self, *, image_path, shot_contract, identity_ledger):
        self.calls.append((Path(image_path), shot_contract, identity_ledger))
        return {"pass": True, "score": 0.91, "issues": [], "notes": "ok"}


class FakeVideoUseCase:
    def __init__(self):
        self.startframes = []

    def execute(self, request):
        self.startframes = sorted(path.name for path in Path(request.storyboard_dir).glob("scene_*.png"))
        output = Path(request.output_dir) / "clip_0001.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"clip")
        return [output]
