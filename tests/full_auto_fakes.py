"""Shared FullAuto song generator and pipeline runner fakes."""

from pathlib import Path


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
        return Path(project_config_path).parent / "output" / "render" / "ltx_single_prompt" / "joy-demo.mp4"
