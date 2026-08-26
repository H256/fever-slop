from __future__ import annotations

import json
import shutil
from pathlib import Path

from feverslop.config.project_config import (
    SCENE_PROMPT_WORD_COUNT_MAX,
    SCENE_PROMPT_WORD_COUNT_MIN,
)
from feverslop.domain.full_auto import GeneratedSong, ProjectScaffoldResult, SongSpec
from feverslop.domain.security import (
    guard_path_under_root,
    is_safe_identifier,
    sanitize_path_component,
)


class LocalProjectScaffold:
    def create_project(
        self,
        *,
        projects_dir: Path,
        project_slug: str,
        spec: SongSpec,
        generated_song: GeneratedSong,
        width: int = 1280,
        height: int = 704,
        fps: int = 24,
        video_pipeline: str = "ltx_i2v",
        render_profile: str = "ltx25-i2v-draft",
        silent_mode: bool = False,
    ) -> ProjectScaffoldResult:
        if not is_safe_identifier(project_slug) or sanitize_path_component(project_slug) != project_slug:
            raise ValueError(f"Unsafe project slug: {project_slug!r}")
        project_dir = guard_path_under_root(Path(projects_dir) / project_slug, projects_dir)
        input_dir = project_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        audio_path = input_dir / f"{project_slug}{Path(generated_song.audio_path).suffix or '.mp3'}"
        if Path(generated_song.audio_path).resolve() != audio_path.resolve():
            shutil.copy2(generated_song.audio_path, audio_path)

        lyrics_path = project_dir / "lyrics.txt"
        lyrics_path.write_text(spec.lyrics, encoding="utf-8")

        song_spec_path = project_dir / "full_auto_song_spec.json"
        song_spec_path.write_text(
            json.dumps(
                {
                    **spec.to_dict(),
                    "audio_manifest": generated_song.manifest,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        config_path = project_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "project_name": spec.title,
                    "input_audio": f"input/{audio_path.name}",
                    "silent_mode": bool(silent_mode),
                    "lyrics": spec.lyrics,
                    "video": {
                        "fps": int(fps),
                        "width": int(width),
                        "height": int(height),
                    },
                    "video_pipeline": video_pipeline,
                    "render_profile": render_profile,
                    "audio": {
                        "demucs_model": "htdemucs_6s",
                        "whisper_model": "large-v3",
                        "language": spec.language,
                    },
                    "scene_generation": {
                        "min_duration": 2.0,
                        "max_duration": 10.0,
                        "bias": 0.7,
                        "duration_preset": "impact_weighted",
                        "seed": -1,
                    },
                    "vocal_detection": {
                        "merge_gap": 0.5,
                        "min_vocal_duration": 0.4,
                        "min_silence_duration": 0.8,
                        "rms_low_percentile": 20,
                        "rms_high_percentile": 85,
                        "rms_ratio": 0.35,
                        "smooth_frames": 10,
                    },
                    "story_idea": spec.visual_story_idea,
                    "style": spec.visual_style,
                    "music_style": spec.music_style or spec.tags,
                    "subject": "",
                    "locations": [],
                    "steering": {
                        "global": "",
                        "story_idea": "",
                        "style": "",
                        "subject": "",
                        "locations": "",
                        "concepts": "",
                        "zimage": "",
                        "ltx": "",
                        "final_prompts": "",
                    },
                    "prompt_guidance": {
                        "character_visibility": "",
                        "shot_types": "",
                        "environments": "",
                        "lighting": "",
                        "camera_motion": "",
                        "physical_interaction": "",
                        "facial_expression": "",
                        "outfit_rules": "",
                        "prompt_structure": "",
                        "list_handling": "",
                        "word_count_min": SCENE_PROMPT_WORD_COUNT_MIN,
                        "word_count_max": SCENE_PROMPT_WORD_COUNT_MAX,
                    },
                    "lora_1": {
                        "enabled": False,
                        "name": "",
                        "strength_model": 1.0,
                        "strength_clip": 1.0,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return ProjectScaffoldResult(
            project_dir=project_dir,
            project_config_path=config_path,
            audio_path=audio_path,
            lyrics_path=lyrics_path,
            song_spec_path=song_spec_path,
        )
