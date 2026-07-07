import json
import tempfile
import unittest
from pathlib import Path


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

            output = build_startframe_director_prompts(project_dir=project, candidate_count=3)
            data = json.loads(output.read_text(encoding="utf-8"))

        prompt_text = data["shots"][0]["positive_prompt"]
        prompt = json.loads(prompt_text)
        self.assertEqual(3, data["shots"][0]["candidate_count"])
        self.assertEqual("Mara opens the sealed ledger.", prompt["high_level_description"])
        self.assertEqual([384, 105, 896, 668], prompt["compositional_deconstruction"]["elements"][0]["bbox"])
        self.assertIn("gothic archivist", prompt["compositional_deconstruction"]["elements"][0]["desc"])
        self.assertIn("readable text", data["shots"][0]["negative_prompt"])

    def test_ideogram_director_workflow_exposes_patch_anchors(self):
        workflow = json.loads(Path("workflows/image_t2i_startframe_ideogram_director_v1.json").read_text(encoding="utf-8-sig"))
        titles = {str(node.get("_meta", {}).get("title") or "") for node in workflow.values()}

        self.assertIn("#PROMPT_POSITIVE", titles)
        self.assertIn("#PROMPT_NEGATIVE", titles)
        self.assertIn("#SAVE_IMAGE", titles)
        self.assertIn("#WIDTH", titles)
        self.assertIn("#HEIGHT", titles)

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
