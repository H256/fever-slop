import json
import tempfile
import unittest
from pathlib import Path


class FakeBriefGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        from feverslop.domain.full_auto import SongSpec

        return SongSpec(
            title="Joy Demo",
            tags="bright pop song",
            lyrics="[Verse]\nhello",
            bpm=123,
            duration_seconds=request.duration_seconds,
            language=request.language,
            keyscale="C major",
            visual_story_idea="friends walking into sunlight",
            visual_style="warm realistic pop video",
        )


class FakeSongGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, spec, *, project_slug, output_dir, seed):
        self.calls.append((spec, project_slug, Path(output_dir), seed))
        from feverslop.domain.full_auto import GeneratedSong

        path = Path(output_dir) / f"{project_slug}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp3")
        return GeneratedSong(
            audio_path=path,
            manifest={"prompt_id": "prompt-id", "seed": seed},
        )


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, *, project_config_path, options):
        self.calls.append((Path(project_config_path), dict(options)))
        return Path(project_config_path).parent / "output" / "render" / "ltx_single_prompt" / "Joy_Demo.mp4"


class FullAutoUseCaseTests(unittest.TestCase):
    def test_full_auto_prepares_project_without_running_video_pipeline(self):
        from feverslop.application.full_auto import FullAutoRequest, FullAutoUseCase
        from feverslop.adapters.full_auto_scaffold import LocalProjectScaffold

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            use_case = FullAutoUseCase(
                brief_generator=FakeBriefGenerator(),
                song_generator=FakeSongGenerator(),
                project_scaffold=LocalProjectScaffold(),
                pipeline_runner=runner,
            )

            result = use_case.execute(
                FullAutoRequest(
                    idea="friendship and joy",
                    style="bright pop",
                    project_name="Joy Demo!",
                    projects_dir=Path(temp_dir),
                    duration_seconds=120.0,
                    language="en",
                    seed=42,
                    run_video_pipeline=False,
                )
            )

            project_dir = Path(temp_dir) / "Joy_Demo"
            self.assertEqual(project_dir / "config.json", result.project_config_path)
            self.assertEqual(project_dir / "input" / "Joy_Demo.mp3", result.audio_path)
            self.assertIsNone(result.final_video_path)
            self.assertEqual([], runner.calls)

            config = json.loads((project_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual("Joy Demo", config["project_name"])
            self.assertEqual("input/Joy_Demo.mp3", config["input_audio"])
            self.assertEqual("[Verse]\nhello", config["lyrics"])
            self.assertEqual("friends walking into sunlight", config["story_idea"])
            self.assertEqual("warm realistic pop video", config["style"])
            self.assertTrue((project_dir / "lyrics.txt").exists())
            self.assertTrue((project_dir / "full_auto_song_spec.json").exists())

    def test_full_auto_runs_video_pipeline_when_requested(self):
        from feverslop.application.full_auto import FullAutoRequest, FullAutoUseCase
        from feverslop.adapters.full_auto_scaffold import LocalProjectScaffold

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            use_case = FullAutoUseCase(
                brief_generator=FakeBriefGenerator(),
                song_generator=FakeSongGenerator(),
                project_scaffold=LocalProjectScaffold(),
                pipeline_runner=runner,
            )

            result = use_case.execute(
                FullAutoRequest(
                    idea="friendship and joy",
                    style="bright pop",
                    project_name="Joy Demo",
                    projects_dir=Path(temp_dir),
                    duration_seconds=120.0,
                    language="en",
                    seed=7,
                    run_video_pipeline=True,
                    runner_options={"smoke_only": True},
                )
            )

            self.assertEqual(1, len(runner.calls))
            self.assertEqual(Path(temp_dir) / "Joy_Demo" / "config.json", runner.calls[0][0])
            self.assertEqual({"smoke_only": True}, runner.calls[0][1])
            self.assertEqual(
                Path(temp_dir) / "Joy_Demo" / "output" / "render" / "ltx_single_prompt" / "Joy_Demo.mp4",
                result.final_video_path,
            )


if __name__ == "__main__":
    unittest.main()
