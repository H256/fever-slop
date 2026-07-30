import json
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.prepared_workflow import sha256_file
from feverslop.domain.visual_consistency import ReferenceAnchor, SceneConsistencyContract
from feverslop.errors import FeverSlopValidationError
from feverslop.ports.rendering import VideoRenderRequest


class FakeClient:
    def __init__(self):
        self.uploaded = []
        self.uploaded_paths = []
        self.queued_workflow = None

    def upload_image(self, path, subfolder, file_type, overwrite, upload_name=None):
        name = upload_name or Path(path).name
        self.uploaded.append(name)
        self.uploaded_paths.append(Path(path))
        return {"name": name, "subfolder": subfolder, "type": file_type}

    def upload_file_via_image_endpoint(self, path, subfolder, file_type, overwrite, upload_name):
        return {"name": upload_name, "subfolder": subfolder, "type": file_type}

    def queue_prompt(self, workflow):
        self.queued_workflow = workflow
        return "prompt-id"

    def wait_for_completion(self, prompt_id):
        return {"outputs": {"save": {"videos": [{"filename": "scene.mp4", "type": "output"}]}}}

    def download_view_file(self, filename, subfolder, file_type, output_path):
        return Path(output_path)


class FakeRenderQueue:
    def __init__(self):
        self.calls = []

    def queue_workflow_and_download_first_video(self, workflow, scene_number, output_path):
        self.calls.append({
            "workflow": workflow,
            "scene_number": scene_number,
            "output_path": Path(output_path),
        })
        return Path(output_path)


class FakePostProcessor:
    def __init__(self):
        self.trim_specs = []

    def trim_clip(self, spec):
        self.trim_specs.append(spec)
        return spec.output_file


class FakeModelResolver:
    def resolve_workflow_models(self, workflow, workflow_path=None):
        workflow = json.loads(json.dumps(workflow))
        workflow["8"]["inputs"]["lora_name"] = "loras/LTX-2.3-Licon-MSR-V1.safetensors"
        return workflow


class LTXMSRVideoBackendTests(unittest.TestCase):
    def test_absolute_msr_reference_is_rejected_with_reference_and_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(json.dumps({
                "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
            }), encoding="utf-8")
            anchor = ReferenceAnchor(
                id="actor",
                kind="actor",
                look_id="default",
                asset_role="identity-reference",
                asset_sha256=sha256_file(actor),
                prompt_anchor="Actor wears a red jacket.",
            )
            location_anchor = ReferenceAnchor(
                id="room",
                kind="location",
                look_id="default",
                asset_role="environment-reference",
                asset_sha256=sha256_file(location),
                prompt_anchor="Room has concrete walls.",
            )
            contract = SceneConsistencyContract.create(
                scene=1,
                mode="msr",
                workflow_profile="msr-final",
                actors=(anchor,),
                location=location_anchor,
                transition_from_previous="cut",
            )
            client = FakeClient()
            backend = ComfyUIMSRVideoRenderBackend(
                client=client,
                workflow_path=workflow,
                output_dir=temp / "out",
                project_dir=temp,
                workflow_profile="msr-final",
            )
            scene = {
                "scene": 1,
                "frame_count": 17,
                "references": {
                    "actor_ids": ["actor"],
                    "location_id": "room",
                    "actor_msr_paths": [str(actor)],
                    "location_msr_path": "location.png",
                },
                "visual_consistency_sources": {
                    "actors": [{"id": "actor", "path": "actor.png"}],
                    "location": {"id": "room", "path": "location.png"},
                },
                "visual_consistency": contract.to_dict(),
            }

            with self.assertRaisesRegex(
                FeverSlopValidationError,
                r"actor.*actor.*project-relative.*actor\.png",
            ):
                backend.build_workflow(scene, prompt="prompt")

            self.assertEqual([], client.uploaded)

    def test_contract_without_location_rejects_runtime_location_before_upload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(json.dumps({
                "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
            }), encoding="utf-8")
            anchor = ReferenceAnchor(
                id="actor",
                kind="actor",
                look_id="default",
                asset_role="identity-reference",
                asset_sha256=sha256_file(actor),
                prompt_anchor="Actor wears a red jacket.",
            )
            contract = SceneConsistencyContract.create(
                scene=1,
                mode="msr",
                workflow_profile="msr-final",
                actors=(anchor,),
                location=None,
                transition_from_previous="cut",
            )
            client = FakeClient()
            backend = ComfyUIMSRVideoRenderBackend(
                client=client,
                workflow_path=workflow,
                output_dir=temp / "out",
                project_dir=temp,
                workflow_profile="msr-final",
            )
            scene = {
                "scene": 1,
                "frame_count": 17,
                "references": {
                    "actor_ids": ["actor"],
                    "location_id": "",
                    "actor_msr_paths": ["actor.png"],
                    "location_msr_path": "location.png",
                },
                "visual_consistency": contract.to_dict(),
            }

            with self.assertRaisesRegex(FeverSlopValidationError, "location"):
                backend.build_workflow(scene, prompt="prompt")

            self.assertEqual([], client.uploaded)

    def test_contract_reference_hash_disagreement_blocks_before_upload(self):
        self.assertIn(
            "workflow_profile",
            inspect.signature(ComfyUIMSRVideoRenderBackend).parameters,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"changed actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(json.dumps({
                "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
            }), encoding="utf-8")
            actor_anchor = ReferenceAnchor(
                id="actor",
                kind="actor",
                look_id="default",
                asset_role="identity-reference",
                asset_sha256="a" * 64,
                prompt_anchor="Actor wears a red jacket.",
            )
            location_anchor = ReferenceAnchor(
                id="room",
                kind="location",
                look_id="default",
                asset_role="environment-reference",
                asset_sha256=sha256_file(location),
                prompt_anchor="Room has blue concrete walls.",
            )
            contract = SceneConsistencyContract.create(
                scene=1,
                mode="msr",
                workflow_profile="msr-final",
                actors=(actor_anchor,),
                location=location_anchor,
                transition_from_previous="cut",
            )
            client = FakeClient()
            backend = ComfyUIMSRVideoRenderBackend(
                client=client,
                workflow_path=workflow,
                output_dir=temp / "out",
                project_dir=temp,
                workflow_profile="msr-final",
            )
            scene = {
                "scene": 1,
                "frame_count": 17,
                "references": {
                    "actor_ids": ["actor"],
                    "location_id": "room",
                    "actor_msr_paths": ["actor.png"],
                    "location_msr_path": "location.png",
                },
                "visual_consistency_sources": {
                    "actors": [{"id": "actor", "path": "actor.png"}],
                    "location": {"id": "room", "path": "location.png"},
                },
                "visual_consistency": contract.to_dict(),
            }

            with self.assertRaisesRegex(FeverSlopValidationError, "hash"):
                backend.build_workflow(scene, prompt="prompt")

            self.assertEqual([], client.uploaded)

    def test_render_frame_budget_rejects_above_limit_and_allows_exact_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(json.dumps({
                "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                "5": {"inputs": {"value": 0}, "_meta": {"title": "#FRAMES"}},
            }), encoding="utf-8")
            queue = FakeRenderQueue()
            backend = ComfyUIMSRVideoRenderBackend(
                client=FakeClient(), workflow_path=workflow, output_dir=temp / "out",
                project_dir=temp, preroll_frames=50, tail_loss_frames=0,
                round_render_frames_to_8n1=False, postprocess=False, render_queue=queue,
                max_render_frames=49, max_render_duration_seconds=2,
            )

            def request(frame_count):
                return VideoRenderRequest(
                    scene={"scene": 1, "fps": 24, "frame_count": frame_count,
                           "references": {"actor_msr_paths": [str(actor)],
                                          "location_msr_path": str(location)}},
                    scene_number=1, prompt="prompt", workflow_path=workflow,
                    output_dir=temp / "out", audio_file=temp / "song.mp3",
                    storyboard_dir=temp, upload_audio=False,
                )

            with self.assertRaisesRegex(FeverSlopValidationError, "Scene 1 requires 50 render frames"):
                backend.render_video(request(50))
            self.assertEqual([], queue.calls)

            backend.render_video(request(49))
            self.assertEqual(1, len(queue.calls))

    def test_backend_patches_msr_references_and_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"frame_count": 17}, "_meta": {"title": "#MSR_FRAME_COUNT"}},
                    "4": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "5": {"inputs": {"noise_seed": 0}, "_meta": {"title": "#SEED"}},
                    "6": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                    "9": {"inputs": {"value": 0}, "_meta": {"title": "#WIDTH"}},
                    "10": {"inputs": {"value": 0}, "_meta": {"title": "#HEIGHT"}},
                }),
                encoding="utf-8",
            )
            client = FakeClient()
            backend = ComfyUIMSRVideoRenderBackend(
                client=client,
                workflow_path=workflow,
                output_dir=temp / "out",
                postprocess=False,
                video_settings=VideoSettings(width=1024, height=576),
            )

            output = backend.render_video(
                VideoRenderRequest(
                    scene={
                        "scene": 7,
                        "fps": 24,
                        "width": 1280,
                        "height": 704,
                        "ltx": {"original_style_i2v_prompt": "video prompt"},
                        "references": {
                            "actor_sheet_paths": [str(actor)],
                            "location_sheet_path": str(location),
                        },
                    },
                    scene_number=7,
                    prompt="video prompt",
                    workflow_path=workflow,
                    output_dir=temp / "out",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp,
                )
            )

            self.assertEqual(temp / "out" / "raw" / "scene_0007_raw.mp4", output)
            self.assertTrue(client.queued_workflow["1"]["inputs"]["image"].startswith("feverslop/references/actor-"))
            self.assertTrue(client.queued_workflow["2"]["inputs"]["image"].startswith("feverslop/references/location-"))
            self.assertEqual(17, client.queued_workflow["3"]["inputs"]["frame_count"])
            self.assertEqual("video prompt", client.queued_workflow["4"]["inputs"]["text"])
            self.assertEqual(100007, client.queued_workflow["5"]["inputs"]["noise_seed"])
            self.assertEqual(1024, client.queued_workflow["9"]["inputs"]["value"])
            self.assertEqual(576, client.queued_workflow["10"]["inputs"]["value"])

    def test_backend_resolves_project_relative_msr_reference_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            actor = project / "output" / "references" / "actors" / "dwarf" / "msr_sheet.png"
            location = project / "output" / "references" / "locations" / "cavern" / "views" / "hero.png"
            actor.parent.mkdir(parents=True)
            location.parent.mkdir(parents=True)
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = project / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            client = FakeClient()
            backend = ComfyUIMSRVideoRenderBackend(
                client=client,
                workflow_path=workflow,
                output_dir=project / "output" / "render" / "ltx_msr",
                project_dir=project,
                postprocess=False,
            )

            backend.render_video(
                VideoRenderRequest(
                    scene={
                        "scene": 1,
                        "fps": 24,
                        "frame_count": 25,
                        "ltx": {"original_style_i2v_prompt": "video prompt"},
                        "references": {
                            "actor_msr_paths": ["output/references/actors/dwarf/msr_sheet.png"],
                            "location_msr_path": "output/references/locations/cavern/views/hero.png",
                        },
                    },
                    scene_number=1,
                    prompt="video prompt",
                    workflow_path=workflow,
                    output_dir=project / "output" / "render" / "ltx_msr",
                    audio_file=project / "input" / "song.mp3",
                    storyboard_dir=project / "output" / "storyboard",
                )
            )

            self.assertEqual([actor, location], client.uploaded_paths)

    def test_backend_randomizes_seed_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "4": {"inputs": {"noise_seed": 0}, "_meta": {"title": "#SEED"}},
                    "5": {"inputs": {"noise_seed": 0}, "class_type": "RandomNoise"},
                    "6": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            backend = ComfyUIMSRVideoRenderBackend(
                client=FakeClient(),
                workflow_path=workflow,
                output_dir=temp / "out",
                randomize_seed=True,
            )

            with patch("feverslop.adapters.comfyui_msr_video_backend.random.randint", return_value=123456789):
                patched = backend.build_workflow(
                    {
                        "scene": 7,
                        "references": {
                            "actor_msr_paths": [str(actor)],
                            "location_msr_path": str(location),
                        },
                    },
                    prompt="prompt",
                )

            self.assertEqual(123456789, patched["4"]["inputs"]["noise_seed"])
            self.assertEqual(123456789, patched["5"]["inputs"]["noise_seed"])

    def test_backend_patches_seed_value_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "4": {"inputs": {"value": 0}, "_meta": {"title": "#SEED"}},
                    "5": {"inputs": {"seed": 0}, "class_type": "SomeSampler"},
                    "6": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            backend = ComfyUIMSRVideoRenderBackend(
                client=FakeClient(),
                workflow_path=workflow,
                output_dir=temp / "out",
            )

            patched = backend.build_workflow(
                {
                    "scene": 7,
                    "references": {
                        "actor_msr_paths": [str(actor)],
                        "location_msr_path": str(location),
                    },
                },
                prompt="prompt",
            )

            self.assertEqual(100007, patched["4"]["inputs"]["value"])
            self.assertEqual(100007, patched["5"]["inputs"]["seed"])

    def test_backend_rejects_scene_without_actor_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            location = temp / "location.png"
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text("{}", encoding="utf-8")
            backend = ComfyUIMSRVideoRenderBackend(client=FakeClient(), workflow_path=workflow, output_dir=temp / "out")

            from feverslop.errors import FeverSlopValidationError

            with self.assertRaisesRegex(FeverSlopValidationError, "at least 1 actor"):
                backend.build_workflow(
                    {
                        "scene": 1,
                        "references": {
                            "actor_sheet_paths": [],
                            "location_sheet_path": str(location),
                        },
                    },
                    prompt="prompt",
                )

    def test_backend_prefers_single_msr_images_over_review_sheets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor_sheet = temp / "actor_sheet.png"
            actor_msr = temp / "actor_hero.png"
            location_sheet = temp / "location_sheet.png"
            location_msr = temp / "location_hero.png"
            for path in (actor_sheet, actor_msr, location_sheet, location_msr):
                path.write_bytes(path.name.encode("utf-8"))
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            client = FakeClient()
            backend = ComfyUIMSRVideoRenderBackend(client=client, workflow_path=workflow, output_dir=temp / "out")

            patched = backend.build_workflow(
                {
                    "scene": 3,
                    "references": {
                        "actor_sheet_paths": [str(actor_sheet)],
                        "location_sheet_path": str(location_sheet),
                        "actor_msr_paths": [str(actor_msr)],
                        "location_msr_path": str(location_msr),
                    },
                },
                prompt="prompt",
            )

            self.assertTrue(patched["1"]["inputs"]["image"].startswith("feverslop/references/actor_hero-"))
            self.assertTrue(patched["2"]["inputs"]["image"].startswith("feverslop/references/location_hero-"))

    def test_backend_uploads_same_named_actor_and_background_references_with_distinct_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor_dir = temp / "actors" / "warrior" / "views"
            location_dir = temp / "locations" / "ruins" / "views"
            actor_dir.mkdir(parents=True)
            location_dir.mkdir(parents=True)
            actor = actor_dir / "hero.png"
            location = location_dir / "hero.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            backend = ComfyUIMSRVideoRenderBackend(client=FakeClient(), workflow_path=workflow, output_dir=temp / "out")

            patched = backend.build_workflow(
                {
                    "scene": 3,
                    "references": {
                        "actor_msr_paths": [str(actor)],
                        "location_msr_path": str(location),
                    },
                },
                prompt="prompt",
            )

            actor_input = patched["1"]["inputs"]["image"]
            location_input = patched["2"]["inputs"]["image"]
            self.assertNotEqual(actor_input, location_input)
            self.assertRegex(actor_input, r"^feverslop/references/hero-[0-9a-f]{12}\.png$")
            self.assertRegex(location_input, r"^feverslop/references/hero-[0-9a-f]{12}\.png$")

    def test_backend_adds_missing_actor_reference_nodes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor_1 = temp / "actor_1.png"
            actor_2 = temp / "actor_2.png"
            location = temp / "location.png"
            actor_1.write_bytes(b"actor 1")
            actor_2.write_bytes(b"actor 2")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "class_type": "LoadImage", "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "class_type": "LoadImage", "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {
                        "inputs": {"1": ["1", 0], "background": ["2", 0], "frame_count": 17},
                        "class_type": "LiconMSR",
                        "_meta": {"title": "#MSR_FRAME_COUNT"},
                    },
                    "4": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "5": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            backend = ComfyUIMSRVideoRenderBackend(client=FakeClient(), workflow_path=workflow, output_dir=temp / "out")

            patched = backend.build_workflow(
                {
                    "scene": 5,
                    "references": {
                        "actor_msr_paths": [str(actor_1), str(actor_2)],
                        "location_msr_path": str(location),
                    },
                },
                prompt="prompt",
            )

            actor_2_id = next(
                node_id
                for node_id, node in patched.items()
                if node.get("_meta", {}).get("title") == "#MSR_ACTOR_2"
            )
            self.assertEqual("LoadImage", patched[actor_2_id]["class_type"])
            self.assertTrue(patched[actor_2_id]["inputs"]["image"].startswith("feverslop/references/actor_2-"))
            self.assertEqual([actor_2_id, 0], patched["3"]["inputs"]["2"])

    def test_backend_patches_prompt_relay_when_anchor_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {
                        "inputs": {
                            "global_prompt": "",
                            "local_prompts": "",
                            "segment_lengths": "",
                            "epsilon": 0.001,
                        },
                        "_meta": {"title": "#PROMPT_RELAY"},
                    },
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                    "5": {"inputs": {"value": 0}, "_meta": {"title": "#FRAMES"}},
                    "6": {"inputs": {"value": 0}, "_meta": {"title": "#FRAMERATE"}},
                }),
                encoding="utf-8",
            )
            backend = ComfyUIMSRVideoRenderBackend(client=FakeClient(), workflow_path=workflow, output_dir=temp / "out")

            patched = backend.build_workflow(
                {
                    "scene": 4,
                    "fps": 24,
                    "frame_count": 25,
                    "references": {
                        "actor_msr_paths": [str(actor)],
                        "location_msr_path": str(location),
                    },
                    "ltx": {
                        "base_prompt": "Reference image 1: Mara, full-body singer. Background: mirror stage.",
                    },
                },
                prompt="Mara walks toward camera, slow dolly in.",
            )

            relay_inputs = patched["3"]["inputs"]
            self.assertEqual("Reference image 1: Mara, full-body singer. Background: mirror stage.", relay_inputs["global_prompt"])
            self.assertEqual("Mara walks toward camera, slow dolly in.", relay_inputs["local_prompts"])
            self.assertEqual("24", relay_inputs["segment_lengths"])
            self.assertEqual(25, patched["5"]["inputs"]["value"])
            self.assertEqual(24, patched["6"]["inputs"]["value"])

    def test_backend_builds_msr_prompt_relay_from_reference_descriptions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {
                        "inputs": {
                            "global_prompt": "",
                            "local_prompts": "",
                            "segment_lengths": "",
                        },
                        "_meta": {"title": "#PROMPT_RELAY"},
                    },
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            backend = ComfyUIMSRVideoRenderBackend(client=FakeClient(), workflow_path=workflow, output_dir=temp / "out")

            patched = backend.build_workflow(
                {
                    "scene": 1,
                    "fps": 24,
                    "frame_count": 25,
                    "references": {
                        "actor_msr_paths": [str(actor)],
                        "location_msr_path": str(location),
                        "actor_reference_descriptions": [
                            {
                                "id": "frost_giant",
                                "name": "Thrym",
                                "role": "frost giant antagonist",
                                "visual_description": "cracked blue ice skin, glowing white eyes, runic armor",
                                "image_prompt": "full body frost giant reference with ancient runic armor and stone hammer",
                            }
                        ],
                        "location_reference_description": {
                            "id": "volcanic_mountain_pass",
                            "name": "Fire-Scarred Pass",
                            "visual_description": "fractured volcanic canyon with lava veins and ash",
                            "image_prompt": "wide volcanic mountain pass environment reference with lava veins and storm ash",
                        },
                    },
                    "ltx": {
                        "base_prompt": (
                            "Start frame: Thrym slams the frozen earth. Lock the first frame to this exact composition. "
                            "Camera motion: slow low-angle push-in. "
                            "Character motion: Thrym braces both legs and swings his stone hammer in a heavy arc. "
                            "Subject or environment motion: ash swirls around him. "
                            "Story beat: the frost giant prepares to strike."
                        ),
                        "prompt_relay": [
                            {
                                "frame_start": 0,
                                "frame_end": 24,
                                "state": "motion",
                                "prompt": (
                                    "Start frame: Thrym slams the frozen earth. Lock the first frame to this exact composition. "
                                    "Thrym raises one arm, ash swirls around him, camera pushes in slowly."
                                ),
                            }
                        ],
                    },
                },
                prompt="Start frame: stale i2v prompt",
            )

            relay_inputs = patched["3"]["inputs"]
            self.assertIn("Reference image 1: Thrym", relay_inputs["global_prompt"])
            self.assertIn("frost giant antagonist", relay_inputs["global_prompt"])
            self.assertIn("cracked blue ice skin", relay_inputs["global_prompt"])
            self.assertNotIn("full body frost giant reference", relay_inputs["global_prompt"])
            self.assertIn("Background reference: Fire-Scarred Pass", relay_inputs["global_prompt"])
            self.assertIn("fractured volcanic canyon", relay_inputs["global_prompt"])
            self.assertNotIn("wide volcanic mountain pass environment reference", relay_inputs["global_prompt"])
            self.assertNotIn("Do not duplicate reference subjects", relay_inputs["global_prompt"])
            self.assertNotIn("Start frame", relay_inputs["global_prompt"])
            self.assertIn("Camera motion: slow low-angle push-in.", relay_inputs["local_prompts"])
            self.assertIn("Character motion: Thrym braces both legs", relay_inputs["local_prompts"])
            self.assertIn("Subject or environment motion: ash swirls around him.", relay_inputs["local_prompts"])
            self.assertNotIn("Start frame", relay_inputs["local_prompts"])
            self.assertNotIn("Lock the first frame", relay_inputs["local_prompts"])

    def test_backend_prefers_render_plan_msr_prompt_relay_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {
                        "inputs": {
                            "global_prompt": "",
                            "local_prompts": "",
                            "segment_lengths": "",
                        },
                        "_meta": {"title": "#PROMPT_RELAY"},
                    },
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            backend = ComfyUIMSRVideoRenderBackend(client=FakeClient(), workflow_path=workflow, output_dir=temp / "out")

            patched = backend.build_workflow(
                {
                    "scene": 2,
                    "fps": 24,
                    "frame_count": 49,
                    "references": {
                        "actor_msr_paths": [str(actor)],
                        "location_msr_path": str(location),
                    },
                    "ltx": {
                        "base_prompt": "old global",
                        "msr_global_prompt": "Reference image 1: Spectral Wolf. Reference image 2 (scene): Megalith Circle.",
                        "prompt_relay": [
                            {
                                "frame_start": 0,
                                "frame_end": 48,
                                "state": "singing",
                                "prompt": "generic same subject sings",
                            }
                        ],
                        "msr_prompt_relay": [
                            {
                                "frame_start": 0,
                                "frame_end": 48,
                                "state": "singing",
                                "prompt": "Spectral Wolf howls toward the glowing monolith with clear lip sync.",
                            }
                        ],
                    },
                },
                prompt="fallback prompt",
            )

            relay_inputs = patched["3"]["inputs"]
            self.assertEqual(
                "Reference image 1: Spectral Wolf. Reference image 2 (scene): Megalith Circle.",
                relay_inputs["global_prompt"],
            )
            self.assertEqual(
                "Spectral Wolf howls toward the glowing monolith with clear lip sync.",
                relay_inputs["local_prompts"],
            )
            self.assertEqual("48", relay_inputs["segment_lengths"])

    def test_backend_uses_msr_preroll_and_tail_prompts_for_rolling_prompt_relay(self):
        scene = {
            "scene": 6,
            "fps": 24,
            "frame_count": 49,
            "ltx": {
                "base_prompt": "old global",
                "msr_global_prompt": "Reference image 1: Spectral Wolf. Reference image 2 (scene): Megalith Circle.",
                "msr_preroll_prompt": (
                    "Cinematic atmosphere holds around the Megalith Circle as blue mist and golden particles "
                    "coil together, keeping Spectral Wolf present before the attack begins."
                ),
                "msr_tail_prompt": (
                    "Spectral Wolf carries the howl through the Megalith Circle, blue essence clashing with "
                    "golden monolith light as the camera keeps drifting backward."
                ),
                "msr_prompt_relay": [
                    {
                        "frame_start": 0,
                        "frame_end": 48,
                        "state": "singing",
                        "prompt": "Spectral Wolf howls toward the glowing monolith with clear lip sync.",
                    }
                ],
            },
        }

        global_prompt, local_prompts, segment_lengths = ComfyUIMSRVideoRenderBackend._build_prompt_relay_payload(
            scene,
            prompt="fallback prompt",
            rolling={
                "render_frame_count": 80,
                "trim_front_frames": 6,
                "tail_loss_frames": 25,
            },
        )

        parts = local_prompts.split("\n|")
        self.assertEqual("Reference image 1: Spectral Wolf. Reference image 2 (scene): Megalith Circle.", global_prompt)
        self.assertEqual("6,48,25", segment_lengths)
        self.assertEqual(3, len(parts))
        self.assertIn("blue mist and golden particles", parts[0])
        self.assertIn("Spectral Wolf howls", parts[1])
        self.assertIn("camera keeps drifting backward", parts[2])
        self.assertNotIn("pre-roll continuity hold", local_prompts)
        self.assertNotIn("tail safety continuation", local_prompts)

    def test_backend_patches_msr_i2v_continuity_handoff_relay_and_guide(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            startframe = temp / "start.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            startframe.write_bytes(b"start")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {
                        "inputs": {"1": ["1", 0], "frame_count": 41},
                        "class_type": "LiconMSR",
                        "_meta": {"title": "#MSR_FRAME_COUNT"},
                    },
                    "4": {
                        "inputs": {
                            "global_prompt": "",
                            "local_prompts": "",
                            "segment_lengths": "",
                        },
                        "_meta": {"title": "#PROMPT_RELAY"},
                    },
                    "5": {"inputs": {"frame_idx": 0, "strength": 1}, "_meta": {"title": "#MSR_GUIDE"}},
                    "6": {"inputs": {"value": 0}, "_meta": {"title": "#FRAMES"}},
                    "7": {"inputs": {"image": ""}, "_meta": {"title": "#STARTFRAME"}},
                    "8": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            backend = ComfyUIMSRVideoRenderBackend(
                client=FakeClient(),
                workflow_path=workflow,
                output_dir=temp / "out",
            )

            patched = backend.build_workflow(
                {
                    "scene": 2,
                    "fps": 24,
                    "frame_count": 48,
                    "keyframes": {"startframe_path": str(startframe)},
                    "references": {
                        "actor_msr_paths": [str(actor)],
                        "location_msr_path": str(location),
                    },
                    "ltx": {
                        "msr_global_prompt": "Reference image 1: Mara. Background reference: Archive.",
                        "msr_continuity_handoff_prompt": "Mara remains at the archive door as fog curls around her.",
                        "msr_continuity_handoff_frames": 18,
                        "msr_continuity_msr_frame_count": 17,
                        "msr_continuity_guide_frame_idx": 18,
                        "msr_prompt_relay": [
                            {
                                "frame_start": 0,
                                "frame_end": 47,
                                "prompt": "Mara steps through the archive door.",
                            }
                        ],
                    },
                },
                prompt="fallback prompt",
                rolling={
                    "render_frame_count": 73,
                    "trim_front_frames": 0,
                    "tail_loss_frames": 25,
                    "fps": 24,
                    "scene_frame_count": 48,
                    "audio_start_seconds": 0.0,
                    "audio_duration_seconds": 3.0,
                },
            )

            relay_inputs = patched["4"]["inputs"]
            self.assertEqual("Reference image 1: Mara. Background reference: Archive.", relay_inputs["global_prompt"])
            self.assertEqual("18,54", relay_inputs["segment_lengths"])
            self.assertEqual(
                "Mara remains at the archive door as fog curls around her.\n|Mara steps through the archive door.",
                relay_inputs["local_prompts"],
            )
            self.assertEqual(18, patched["5"]["inputs"]["frame_idx"])
            self.assertEqual(17, patched["3"]["inputs"]["frame_count"])

    def test_backend_strips_global_contracts_from_handoff_and_current_relay_prompts(self):
        scene = {
            "scene": 2,
            "fps": 24,
            "frame_count": 48,
            "ltx": {
                "msr_global_prompt": "Reference image 1: Mara.",
                "msr_continuity_handoff_prompt": (
                    "Wide shot of Mara in the fog. Actors: Mara. Location: Archive. "
                    "Action: she stands at the doorway. Camera: slow push-in. Acting: frozen resolve. "
                    "Audio contract: diegetic environmental sound effects only. No spoken dialogue. "
                    "Dialogue language: German. Style: desaturated noir."
                ),
                "msr_continuity_handoff_frames": 18,
                "msr_prompt_relay": [
                    {
                        "frame_start": 0,
                        "frame_end": 47,
                        "prompt": (
                            "Close-up of Mara entering. Camera: handheld close-up. "
                            "Dialogue for native audio: Ich komme. Audio contract: scripted dialogue only. "
                            "Dialogue language: German. Style: desaturated noir."
                        ),
                    }
                ],
            },
        }

        _, local_prompts, segment_lengths = ComfyUIMSRVideoRenderBackend._build_prompt_relay_payload(
            scene,
            prompt="fallback",
            rolling={
                "render_frame_count": 73,
                "trim_front_frames": 0,
                "tail_loss_frames": 25,
            },
        )

        self.assertEqual("18,54", segment_lengths)
        self.assertIn("Wide shot of Mara in the fog", local_prompts)
        self.assertIn("Close-up of Mara entering", local_prompts)
        self.assertIn("Dialogue for native audio: Ich komme", local_prompts)
        self.assertNotIn("Audio contract:", local_prompts)
        self.assertNotIn("Dialogue language:", local_prompts)
        self.assertNotIn("Style:", local_prompts)
        self.assertNotIn("No spoken dialogue", local_prompts)

    def test_render_video_patches_audio_anchors_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            audio = temp / "song.mp3"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            audio.write_bytes(b"audio")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                    "5": {"inputs": {"audio": "", "audioUI": ""}, "_meta": {"title": "#LOAD_AUDIO"}},
                    "6": {"inputs": {"start_index": 0, "duration": 0}, "_meta": {"title": "#TRIM_AUDIO"}},
                }),
                encoding="utf-8",
            )
            client = FakeClient()
            backend = ComfyUIMSRVideoRenderBackend(
                client=client,
                workflow_path=workflow,
                output_dir=temp / "out",
                preroll_frames=0,
                tail_loss_frames=0,
                round_render_frames_to_8n1=False,
                postprocess=False,
            )

            backend.render_video(
                VideoRenderRequest(
                    scene={
                        "scene": 2,
                        "fps": 24,
                        "frame_count": 49,
                        "abs_start_seconds": 3.5,
                        "duration_seconds": 2.0,
                        "references": {
                            "actor_msr_paths": [str(actor)],
                            "location_msr_path": str(location),
                        },
                    },
                    scene_number=2,
                    prompt="prompt",
                    workflow_path=workflow,
                    output_dir=temp / "out",
                    audio_file=audio,
                    storyboard_dir=temp,
                )
            )

            audio_input = client.queued_workflow["5"]["inputs"]["audio"]
            self.assertTrue(audio_input.startswith("feverslop/audio/song-"))
            self.assertEqual(f"/api/view?filename={audio_input}&type=input", client.queued_workflow["5"]["inputs"]["audioUI"])
            self.assertEqual(3.5, client.queued_workflow["6"]["inputs"]["start_index"])
            self.assertEqual(2.0, client.queued_workflow["6"]["inputs"]["duration"])

    def test_render_video_postprocesses_raw_clip_with_rolling_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            audio = temp / "song.mp3"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            audio.write_bytes(b"audio")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                    "5": {"inputs": {"value": 0}, "_meta": {"title": "#FRAMES"}},
                    "6": {"inputs": {"audio": "", "audioUI": ""}, "_meta": {"title": "#LOAD_AUDIO"}},
                    "7": {"inputs": {"start_index": 0, "duration": 0}, "_meta": {"title": "#TRIM_AUDIO"}},
                }),
                encoding="utf-8",
            )
            render_queue = FakeRenderQueue()
            postprocessor = FakePostProcessor()
            backend = ComfyUIMSRVideoRenderBackend(
                client=FakeClient(),
                workflow_path=workflow,
                output_dir=temp / "out",
                preroll_frames=6,
                tail_loss_frames=25,
                round_render_frames_to_8n1=True,
                postprocess=True,
                render_queue=render_queue,
                postprocessor=postprocessor,
            )

            output = backend.render_video(
                VideoRenderRequest(
                    scene={
                        "scene": 2,
                        "fps": 24,
                        "frame_count": 49,
                        "abs_start_seconds": 3.5,
                        "references": {
                            "actor_msr_paths": [str(actor)],
                            "location_msr_path": str(location),
                        },
                    },
                    scene_number=2,
                    prompt="prompt",
                    workflow_path=workflow,
                    output_dir=temp / "out",
                    audio_file=audio,
                    storyboard_dir=temp,
                )
            )

            raw_output = temp / "out" / "raw" / "scene_0002_raw.mp4"
            final_output = temp / "out" / "scene_0002.mp4"
            self.assertEqual(final_output, output)
            self.assertEqual(raw_output, render_queue.calls[0]["output_path"])
            self.assertEqual(81, render_queue.calls[0]["workflow"]["5"]["inputs"]["value"])
            self.assertEqual(3.25, render_queue.calls[0]["workflow"]["7"]["inputs"]["start_index"])
            self.assertAlmostEqual(80 / 24, render_queue.calls[0]["workflow"]["7"]["inputs"]["duration"], places=4)
            self.assertEqual(1, len(postprocessor.trim_specs))
            trim_spec = postprocessor.trim_specs[0]
            self.assertEqual(raw_output, trim_spec.source_file)
            self.assertEqual(final_output, trim_spec.output_file)
            self.assertEqual(24, trim_spec.fps)
            self.assertEqual(6, trim_spec.trim_front_frames)
            self.assertEqual(49, trim_spec.keep_frames)

    def test_backend_patches_i2v_latent_length_to_render_frame_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            startframe = temp / "start.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            startframe.write_bytes(b"start")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                    "5": {"inputs": {"value": 0}, "_meta": {"title": "#FRAMES"}},
                    "6": {"inputs": {"length": 97}, "class_type": "EmptyLTXVLatentVideo"},
                    "7": {
                        "inputs": {"latent": ["6", 0], "image": ["8", 0]},
                        "class_type": "LTXVImgToVideoInplace",
                    },
                    "8": {
                        "inputs": {"image": ""},
                        "class_type": "LoadImage",
                        "_meta": {"title": "#STARTFRAME"},
                    },
                }),
                encoding="utf-8",
            )
            backend = ComfyUIMSRVideoRenderBackend(
                client=FakeClient(),
                workflow_path=workflow,
                output_dir=temp / "out",
                preroll_frames=6,
                tail_loss_frames=25,
            )

            patched = backend.build_workflow(
                {
                    "scene": 11,
                    "fps": 24,
                    "frame_count": 240,
                    "keyframes": {"startframe_path": str(startframe)},
                    "references": {
                        "actor_msr_paths": [str(actor)],
                        "location_msr_path": str(location),
                    },
                },
                prompt="prompt",
                rolling={
                    "fps": 24,
                    "scene_frame_count": 240,
                    "render_frame_count": 321,
                    "trim_front_frames": 56,
                    "tail_loss_frames": 25,
                    "audio_start_seconds": 0.0,
                    "audio_duration_seconds": 321 / 24,
                },
            )

            self.assertEqual(321, patched["5"]["inputs"]["value"])
            self.assertEqual(321, patched["6"]["inputs"]["length"])

    def test_debug_workflow_is_written_after_model_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            audio = temp / "song.mp3"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            audio.write_bytes(b"audio")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                    "8": {
                        "inputs": {"lora_name": "LTX-2.3-Licon-MSR-V1.safetensors"},
                        "class_type": "LTXICLoRALoaderModelOnly",
                        "_meta": {"title": "#MSR_LORA"},
                    },
                }),
                encoding="utf-8",
            )
            render_queue = FakeRenderQueue()
            debug_dir = temp / "debug"
            backend = ComfyUIMSRVideoRenderBackend(
                client=FakeClient(),
                workflow_path=workflow,
                output_dir=temp / "out",
                postprocess=False,
                render_queue=render_queue,
                model_resolver=FakeModelResolver(),
                debug_workflows_dir=debug_dir,
            )

            backend.render_video(
                VideoRenderRequest(
                    scene={
                        "scene": 2,
                        "fps": 24,
                        "frame_count": 49,
                        "references": {
                            "actor_msr_paths": [str(actor)],
                            "location_msr_path": str(location),
                        },
                    },
                    scene_number=2,
                    prompt="prompt",
                    workflow_path=workflow,
                    output_dir=temp / "out",
                    audio_file=audio,
                    storyboard_dir=temp,
                )
            )

            debug_workflow = json.loads((debug_dir / "scene_0002_workflow.json").read_text(encoding="utf-8"))
            self.assertEqual("loras/LTX-2.3-Licon-MSR-V1.safetensors", debug_workflow["8"]["inputs"]["lora_name"])
            self.assertEqual(
                "loras/LTX-2.3-Licon-MSR-V1.safetensors",
                render_queue.calls[0]["workflow"]["8"]["inputs"]["lora_name"],
            )
