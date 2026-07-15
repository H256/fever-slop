import json
import tempfile
import unittest
from pathlib import Path


class FakeComfyUIClient:
    def __init__(self):
        self.queued_workflow = None
        self.downloads = []

    def queue_prompt(self, workflow):
        self.queued_workflow = workflow
        return "prompt-id"

    def wait_for_completion(self, prompt_id):
        self.prompt_id = prompt_id
        return {
            "outputs": {
                "107": {
                    "audio": [
                        {
                            "filename": "Joy_Demo.mp3",
                            "subfolder": "audio",
                            "type": "output",
                        }
                    ]
                }
            }
        }

    def extract_output_files(self, history):
        files = []
        for node_output in history.get("outputs", {}).values():
            for key in ("files", "videos", "audio"):
                for item in node_output.get(key, []):
                    files.append(
                        {
                            "kind": key,
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        }
                    )
        return files

    def download_view_file(self, *, filename, subfolder, file_type, output_path):
        self.downloads.append((filename, subfolder, file_type, Path(output_path)))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"mp3")
        return Path(output_path)


class FakeModelResolver:
    def __init__(self):
        self.calls = []

    def resolve_workflow_models(self, workflow, workflow_path=None):
        self.calls.append((workflow, Path(workflow_path)))
        return workflow


class ComfyUIAceStepSongGeneratorTests(unittest.TestCase):
    def test_workflow_validation_accepts_audio_song_contract(self):
        from feverslop.adapters.comfyui_acestep_song_generator import ComfyUIAceStepSongGenerator

        generator = ComfyUIAceStepSongGenerator(
            client=FakeComfyUIClient(),
            workflow_path=Path("workflows/audio_song_v2.json"),
            model_resolver=FakeModelResolver(),
        )

        generator.validate_workflow()

    def test_generate_patches_musical_fields_synced_seed_duration_and_save_prefix(self):
        from feverslop.adapters.comfyui_acestep_song_generator import ComfyUIAceStepSongGenerator
        from feverslop.domain.full_auto import SongSpec

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "audio_song.json"
            workflow_path.write_text(Path("workflows/audio_song_v2.json").read_text(encoding="utf-8-sig"), encoding="utf-8")
            client = FakeComfyUIClient()
            resolver = FakeModelResolver()
            generator = ComfyUIAceStepSongGenerator(
                client=client,
                workflow_path=workflow_path,
                model_resolver=resolver,
            )

            result = generator.generate(
                SongSpec(
                    title="Joy Demo",
                    tags="bright pop song",
                    lyrics="[Verse]\nhello",
                    bpm=123,
                    duration_seconds=90.5,
                    language="en",
                    keyscale="D major",
                    visual_story_idea="friends",
                    visual_style="warm",
                ),
                project_slug="Joy_Demo",
                output_dir=temp / "input",
                seed=42,
            )

            workflow = client.queued_workflow
            ace = next(node for node in workflow.values() if node["_meta"]["title"] == "ACE_STEP")
            sampler = next(node for node in workflow.values() if node["_meta"]["title"] == "KSampler")
            latent = next(node for node in workflow.values() if node["_meta"]["title"] == "Empty Ace Step 1.5 Latent Audio")
            save = next(node for node in workflow.values() if node["_meta"]["title"] == "SAVE")

            self.assertEqual("bright pop song", ace["inputs"]["tags"])
            self.assertEqual("[Verse]\nhello", ace["inputs"]["lyrics"])
            self.assertEqual(123, ace["inputs"]["bpm"])
            self.assertEqual(90.5, ace["inputs"]["duration"])
            self.assertEqual(90.5, latent["inputs"]["seconds"])
            self.assertEqual("en", ace["inputs"]["language"])
            self.assertEqual("D major", ace["inputs"]["keyscale"])
            self.assertEqual("4", ace["inputs"]["timesignature"])
            self.assertEqual(42, ace["inputs"]["seed"])
            self.assertEqual(42, sampler["inputs"]["seed"])
            self.assertEqual("audio/Joy_Demo", save["inputs"]["filename_prefix"])
            self.assertEqual("V0", save["inputs"]["quality"])
            self.assertEqual(temp / "input" / "Joy_Demo.mp3", result.audio_path)
            self.assertEqual({"prompt_id": "prompt-id", "seed": 42, "workflow_path": str(workflow_path)}, result.manifest)
            self.assertEqual([("Joy_Demo.mp3", "audio", "output", temp / "input" / "Joy_Demo.mp3")], client.downloads)
            self.assertEqual(workflow_path, resolver.calls[0][1])

    def test_generate_writes_per_run_resolved_workflow_debug_json_under_project_output(self):
        from feverslop.adapters.comfyui_acestep_song_generator import ComfyUIAceStepSongGenerator
        from feverslop.domain.full_auto import SongSpec

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "audio_song.json"
            workflow_path.write_text(Path("workflows/audio_song_v2.json").read_text(encoding="utf-8-sig"), encoding="utf-8")
            client = FakeComfyUIClient()
            resolver = FakeModelResolver()
            generator = ComfyUIAceStepSongGenerator(
                client=client,
                workflow_path=workflow_path,
                model_resolver=resolver,
            )

            generator.generate(
                SongSpec(
                    title="Joy Demo",
                    tags="bright pop song",
                    lyrics="[Verse]\nhello",
                    bpm=123,
                    duration_seconds=90.5,
                    language="en",
                    keyscale="D major",
                    visual_story_idea="friends",
                    visual_style="warm",
                ),
                project_slug="Joy_Demo",
                output_dir=temp / "input",
                seed=42,
            )

            debug_files = sorted((temp / "output" / "debug" / "ace_step").glob("ace_step_*_workflow.json"))
            self.assertEqual(1, len(debug_files))
            debug_path = debug_files[0]
            debug_workflow = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertEqual(client.queued_workflow, debug_workflow)
            ace = next(node for node in debug_workflow.values() if node["_meta"]["title"] == "ACE_STEP")
            self.assertEqual("bright pop song", ace["inputs"]["tags"])
            self.assertEqual("[Verse]\nhello", ace["inputs"]["lyrics"])
            self.assertEqual(42, ace["inputs"]["seed"])
            self.assertNotIn(str(temp), debug_path.read_text(encoding="utf-8"))

    def test_validation_reports_missing_required_anchor(self):
        from feverslop.adapters.comfyui_acestep_song_generator import ComfyUIAceStepSongGenerator

        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_path = Path(temp_dir) / "bad.json"
            workflow_path.write_text(
                json.dumps(
                    {
                        "1": {
                            "inputs": {
                                "tags": "",
                                "lyrics": "",
                                "seed": 1,
                                "bpm": 120,
                                "duration": 120,
                                "timesignature": "4",
                                "language": "en",
                                "keyscale": "C major",
                            },
                            "_meta": {"title": "ACE_STEP"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            generator = ComfyUIAceStepSongGenerator(
                client=FakeComfyUIClient(),
                workflow_path=workflow_path,
                model_resolver=FakeModelResolver(),
            )

            with self.assertRaisesRegex(ValueError, "KSampler"):
                generator.validate_workflow()


if __name__ == "__main__":
    unittest.main()
