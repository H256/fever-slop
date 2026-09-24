"""Tests for --resume sub-step skipping in the main pipeline.

When ``resume=True`` and a sub-step's output artifact already exists, the
pipeline must reuse the artifact instead of re-running the expensive
generator. When ``resume=False`` (or the artifact is missing), the generator
runs as before.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from feverslop.application.audio_timeline_pipeline import AudioTimelinePipeline
from feverslop.application.generate_render_plan import GenerateRenderPlanRequest
from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline
from feverslop.application.scene_timeline_pipeline import SceneTimelinePipeline


def _no_call(label: str):
    def factory(*args, **kwargs):
        raise AssertionError(f"{label} should be skipped but was called")
    return factory


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class RequestResumeFieldTests(unittest.TestCase):
    def test_resume_defaults_false(self):
        request = GenerateRenderPlanRequest(
            project_config_path=Path("project/config.json"),
            app_config_path=Path("app.json"),
        )
        self.assertFalse(request.resume)

    def test_resume_can_be_true(self):
        request = GenerateRenderPlanRequest(
            project_config_path=Path("project/config.json"),
            app_config_path=Path("app.json"),
            resume=True,
        )
        self.assertTrue(request.resume)


class AudioTimelineResumeTests(unittest.TestCase):
    def _make_pipeline(self, *, separator, analyzer, beat_analyzer):
        return AudioTimelinePipeline(
            separator_factory=separator,
            vocal_analyzer_factory=analyzer,
            beat_analyzer_factory=beat_analyzer,
            normalize_empty_vocals=lambda timeline: timeline,
            merge_same_kind_segments=lambda timeline, merge_gap: timeline,
            save_timeline_json=lambda timeline, path, **kwargs: None,
        )

    def _make_context(self, root: Path, *, request, artifact_store):
        return {
            "config": SimpleNamespace(
                input_audio=root / "song.mp3",
                lyrics="",
                audio=SimpleNamespace(),
                vocal_detection=SimpleNamespace(merge_gap=0.25),
            ),
            "paths": SimpleNamespace(stems_dir=root / "stems"),
            "timeline_json": root / "timeline.json",
            "beat_json": root / "beat.json",
            "log_step": lambda _title: None,
            "log_file": lambda _label, _path: None,
            "run_spinner": lambda _description, func: func(),
            "reporter": SimpleNamespace(message=lambda _text: None, table=lambda *args: None),
            "request": request,
            "artifact_store": artifact_store,
        }

    def _create_audio_artifacts(self, root: Path) -> None:
        stems_dir = root / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocals", "drums", "bass", "other"):
            (stems_dir / f"{name}_song.wav").write_bytes(b"audio")
        _write_json(
            root / "timeline.json",
            [{"start": 0.0, "end": 1.0, "type": "vocals", "lyrics": "hello"}],
        )
        _write_json(
            root / "beat.json",
            {"bpm": 120, "beats": [], "source_used_for_beats": "fake"},
        )

    def test_resume_skips_existing_audio_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_audio_artifacts(root)
            request = SimpleNamespace(
                skip_stem_separation=False,
                skip_whisper=False,
                skip_beat_analysis=False,
                resume=True,
            )
            artifact_store = SimpleNamespace(
                read_json=lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
            )
            pipeline = self._make_pipeline(
                separator=_no_call("separator"),
                analyzer=_no_call("analyzer"),
                beat_analyzer=_no_call("beat_analyzer"),
            )

            result = pipeline.run(self._make_context(root, request=request, artifact_store=artifact_store))

        self.assertEqual({"vocals", "drums", "bass", "other"}, set(result["stem_files"]))
        self.assertEqual(1, len(result["timeline"]))
        self.assertEqual(120, result["beat_data"]["bpm"])

    def test_resume_runs_missing_audio_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # Only the stems exist; timeline and beat are missing.
            stems_dir = root / "stems"
            stems_dir.mkdir(parents=True, exist_ok=True)
            for name in ("vocals", "drums", "bass", "other"):
                (stems_dir / f"{name}_song.wav").write_bytes(b"audio")
            request = SimpleNamespace(
                skip_stem_separation=False,
                skip_whisper=False,
                skip_beat_analysis=False,
                resume=True,
            )
            calls = []
            analyzer = SimpleNamespace(
                analyze=lambda _vocals: [
                    SimpleNamespace(kind="vocals", start=0.0, end=1.0, text="hello")
                ],
                close=lambda: None,
            )
            beat_analyzer = SimpleNamespace(
                analyze_to_json_file=lambda **kwargs: _write_json(
                    kwargs["output_json_path"],
                    {"bpm": 120, "beats": [], "source_used_for_beats": "fake"},
                )
            )
            artifact_store = SimpleNamespace(
                read_json=lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
            )
            pipeline = self._make_pipeline(
                separator=lambda _config: (calls.append("separator"), SimpleNamespace())[1],
                analyzer=lambda _config: (calls.append("analyzer"), analyzer)[1],
                beat_analyzer=lambda: (calls.append("beat_analyzer"), beat_analyzer)[1],
            )

            pipeline.run(self._make_context(root, request=request, artifact_store=artifact_store))

        # Stems exist -> separator skipped. Timeline/beat missing -> those run.
        self.assertNotIn("separator", calls)
        self.assertIn("analyzer", calls)
        self.assertIn("beat_analyzer", calls)

    def test_no_resume_runs_all_audio_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_audio_artifacts(root)
            request = SimpleNamespace(
                skip_stem_separation=False,
                skip_whisper=False,
                skip_beat_analysis=False,
                resume=False,
            )
            calls = []
            separator = SimpleNamespace(
                separate=lambda _input, output: {
                    name: output / f"{name}_song.wav"
                    for name in ("vocals", "drums", "bass", "other")
                },
                close=lambda: None,
            )
            analyzer = SimpleNamespace(
                analyze=lambda _vocals: [
                    SimpleNamespace(kind="vocals", start=0.0, end=1.0, text="hello")
                ],
                close=lambda: None,
            )
            beat_analyzer = SimpleNamespace(
                analyze_to_json_file=lambda **kwargs: _write_json(
                    kwargs["output_json_path"],
                    {"bpm": 120, "beats": [], "source_used_for_beats": "fake"},
                )
            )
            artifact_store = SimpleNamespace(
                read_json=lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
            )
            pipeline = self._make_pipeline(
                separator=lambda _config: (calls.append("separator"), separator)[1],
                analyzer=lambda _config: (calls.append("analyzer"), analyzer)[1],
                beat_analyzer=lambda: (calls.append("beat_analyzer"), beat_analyzer)[1],
            )

            pipeline.run(self._make_context(root, request=request, artifact_store=artifact_store))

        self.assertEqual(["separator", "analyzer", "beat_analyzer"], calls)


class SceneTimelineResumeTests(unittest.TestCase):
    def _make_pipeline(self, *, scene_generator, enforce, parse, validate, build_stage1, build_relay):
        return SceneTimelinePipeline(
            scene_generator_factory=scene_generator,
            enforce_scene_srt_file=enforce,
            parse_scene_srt=parse,
            validate_scene_durations=validate,
            build_stage1_segment_json=build_stage1,
            build_scene_prompt_relay=build_relay,
        )

    def _make_context(self, root: Path, *, request):
        song_id = "song"
        return {
            "config": SimpleNamespace(scene_generation=SimpleNamespace(min_duration=0.5, max_duration=10.0)),
            "video_settings": SimpleNamespace(),
            "timeline_json": root / "timeline.json",
            "beat_json": root / "beat.json",
            "scene_srt_raw": root / f"scenes_{song_id}_raw.srt",
            "scene_srt": root / f"scenes_{song_id}.srt",
            "stage1_segments_json": root / f"stage1_segments_{song_id}.json",
            "ltx_prompt_relay_json": root / f"ltx_prompt_relay_{song_id}.json",
            "scene_duration_policy": None,
            "artifact_store": SimpleNamespace(
                read_json=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
                write_json=lambda path, payload: _write_json(path, payload),
            ),
            "log_step": lambda _title: None,
            "log_file": lambda _label, _path: None,
            "reporter": SimpleNamespace(message=lambda _text: None),
            "request": request,
        }

    def _create_scene_artifacts(self, root: Path) -> None:
        song_id = "song"
        (root / f"scenes_{song_id}.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
        _write_json(root / f"stage1_segments_{song_id}.json", [{"segment_id": "001", "type": "vocals"}])
        _write_json(root / f"ltx_prompt_relay_{song_id}.json", [{"segment_id": "001"}])

    def test_resume_reuses_existing_scene_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_scene_artifacts(root)
            request = SimpleNamespace(resume=True)
            calls = []
            pipeline = self._make_pipeline(
                scene_generator=lambda _cfg: (calls.append("scene_generator"), SimpleNamespace())[1],
                enforce=lambda **kwargs: (calls.append("enforce"), None)[1],
                parse=lambda _path: [SimpleNamespace(duration=2.0)],
                validate=lambda _scenes, **kwargs: [],
                build_stage1=lambda **kwargs: (calls.append("build_stage1"), None)[1],
                build_relay=lambda **kwargs: (calls.append("build_relay"), None)[1],
            )

            result = pipeline.run(self._make_context(root, request=request))

        self.assertEqual([], calls)
        self.assertEqual(1, len(result["repaired_scenes"]))
        self.assertEqual(1, len(result["stage1_segments"]))

    def test_no_resume_runs_all_scene_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = SimpleNamespace(resume=False)
            calls = []
            scene_generator = SimpleNamespace(
                generate_from_json_file=lambda **kwargs: (
                    kwargs["output_srt_path"].write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n", encoding="utf-8"),
                    None,
                )[1]
            )
            pipeline = self._make_pipeline(
                scene_generator=lambda _cfg: (calls.append("scene_generator"), scene_generator)[1],
                enforce=lambda **kwargs: (
                    calls.append("enforce"),
                    kwargs["output_srt"].write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n", encoding="utf-8"),
                )[0],
                parse=lambda _path: [SimpleNamespace(duration=2.0)],
                validate=lambda _scenes, **kwargs: [],
                build_stage1=lambda **kwargs: (
                    calls.append("build_stage1"),
                    _write_json(kwargs["output_json_file"], [{"segment_id": "001", "type": "vocals"}]),
                )[0],
                build_relay=lambda **kwargs: (
                    calls.append("build_relay"),
                    _write_json(kwargs["output_json_file"], [{"segment_id": "001"}]),
                )[0],
            )

            pipeline.run(self._make_context(root, request=request))

        self.assertEqual(["scene_generator", "enforce", "build_stage1", "build_relay"], calls)


class PromptGenerationResumeTests(unittest.TestCase):
    def test_final_concept_gate_honors_warn_policy(self):
        pipeline = self._make_pipeline(
            llm_factory=lambda _app: None,
            prompt_pipeline_factory=lambda _llm: None,
            concept_batcher_factory=lambda _llm, _size: None,
            scene_prompt_builder_factory=lambda _llm: None,
        )
        concepts = {
            f"segment_{i:03d}": {"narrative": {"milestones": ["arrival"], "location": "room"}}
            for i in (1, 2)
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            result = pipeline._finalize_concept_prompts(
                prompt_pipeline=SimpleNamespace(save_json=lambda path, value, **kwargs: None),
                reporter=SimpleNamespace(message=lambda text: None),
                stage1_segments=[{"segment_id": key} for key in concepts],
                concept_prompts=concepts,
                global_context={"narrative_contract": {
                    "milestone_order": ["arrival"], "one_shot_milestones": ["arrival"],
                }},
                concept_prompts_json=Path(temp_dir) / "concept_prompts.json",
                artifact_store=SimpleNamespace(write_json=lambda path, value: None),
                log_file=lambda label, path: None,
                semantic_enforcement="warn",
            )

        self.assertEqual("warning", result["segment_002"]["semantic_validation"]["outcome"])

    def test_resume_repairs_legacy_contract_before_story_plan_consumes_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_prompt_artifacts(root)
            resolved_path = root / "resolved_context_song.json"
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            resolved.update({
                "story_idea": "A journey from entrance to destination.",
                "structured_locations": [{"id": "l1", "name": "Entrance"}],
                "actors": [{"id": "a1", "name": "A"}],
                "narrative_contract_source": "llm",
                "narrative_contract": {
                    "location_order": ["l1"],
                    "milestone_order": ["arrival"],
                },
            })
            _write_json(resolved_path, resolved)
            received = []
            class PromptModules:
                def create_narrative_milestone_bindings(self, **kwargs):
                    return [{"milestone_id": "arrival", "location_id": "l1", "relative_position": 0.0}]
            pipeline = self._make_pipeline(
                llm_factory=lambda _app: SimpleNamespace(model="m", client=object()),
                prompt_pipeline_factory=lambda _llm: PromptModules(),
                concept_batcher_factory=_no_call("concept_batcher"),
                scene_prompt_builder_factory=lambda _llm: SimpleNamespace(),
            )
            pipeline._resolve_story_plan = lambda **kwargs: (
                received.append(kwargs["global_context"]["narrative_contract"])
                or (None, True)
            )

            pipeline.run(self._make_context(
                root, request=SimpleNamespace(resume=True, concept_batch_size=0),
            ))

            self.assertEqual("arrival", received[0]["milestone_bindings"][0]["milestone_id"])
            persisted = json.loads(resolved_path.read_text(encoding="utf-8"))
            self.assertEqual(received[0], persisted["narrative_contract"])

    def _make_pipeline(self, *, llm_factory, prompt_pipeline_factory, concept_batcher_factory, scene_prompt_builder_factory):
        return PromptGenerationPipeline(
            llm_factory=llm_factory,
            prompt_pipeline_factory=prompt_pipeline_factory,
            concept_batcher_factory=concept_batcher_factory,
            scene_prompt_builder_factory=scene_prompt_builder_factory,
        )

    def _make_context(self, root: Path, *, request):
        song_id = "song"
        return {
            "config": SimpleNamespace(
                video_pipeline="ltx_i2v",
                trigger_word="",
                steering=SimpleNamespace(),
                prompt_guidance=SimpleNamespace(as_prompt_context=lambda: ""),
                narrative_contract={},
                audio=SimpleNamespace(),
                silent_mode=False,
                subject_mode="multi",
                max_scene_actors=4,
                story_idea="a story",
                style="a style",
                subject="a subject",
                locations=["loc"],
                cast_idea="",
                cast_policy=None,
                actors=[{
                    "id": "a1",
                    "name": "A",
                    "role": "r",
                    "gender": "m",
                    "visual_description": "v",
                    "image_prompt": "i",
                }],
                structured_locations=[{"id": "l1", "name": "loc"}],
                global_cast=None,
                global_locations=None,
                global_styles=None,
                global_props=None,
                minimax_h3_audio_refs=None,
            ),
            "app_config": SimpleNamespace(llm=SimpleNamespace(request_timeout_seconds=30)),
            "request": request,
            "stage1_segments": [{"segment_id": "001", "lyrics": "hello"}],
            "resolved_context_json": root / f"resolved_context_{song_id}.json",
            "concept_prompts_json": root / f"concept_prompts_{song_id}.json",
            "scene_details_json": root / f"scene_details_{song_id}.json",
            "scene_prompts_json": root / f"scene_prompts_{song_id}.json",
            "artifact_store": SimpleNamespace(
                read_json=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
                write_json=lambda path, payload: _write_json(path, payload),
            ),
            "log_step": lambda _title: None,
            "log_file": lambda _label, _path: None,
            "run_spinner": lambda _description, func: func(),
            "reporter": SimpleNamespace(
                message=lambda _text: None,
                panel=lambda _text, title=None: None,
                table=lambda *args: None,
            ),
        }

    def _create_prompt_artifacts(self, root: Path) -> None:
        song_id = "song"
        _write_json(root / f"resolved_context_{song_id}.json", {
            "story_idea": "a story",
            "style": "a style",
            "subject": "a subject",
            "locations": ["loc"],
            "narrative_contract": {},
        })
        _write_json(root / f"concept_prompts_{song_id}.json", {"001": "concept one"})
        _write_json(root / f"scene_details_{song_id}.json", {"001": {"camera": "static"}})
        _write_json(root / f"scene_prompts_{song_id}.json", [{"segment_id": "001"}])

    def test_resume_reuses_existing_prompt_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_prompt_artifacts(root)
            request = SimpleNamespace(resume=True, concept_batch_size=0)
            calls = []
            prompt_pipeline = SimpleNamespace(
                create_concept_prompts=lambda **kwargs: (calls.append("concepts"), {})[1],
                create_scene_details=lambda **kwargs: (calls.append("scene_details"), {})[1],
                save_json=lambda _path, _payload, **kwargs: None,
            )
            pipeline = self._make_pipeline(
                llm_factory=lambda _app: (calls.append("llm"), SimpleNamespace(model="m", client=object()))[1],
                prompt_pipeline_factory=lambda _llm: (calls.append("prompt_pipeline"), prompt_pipeline)[1],
                concept_batcher_factory=_no_call("concept_batcher"),
                scene_prompt_builder_factory=lambda _llm: (calls.append("scene_prompt_builder"), SimpleNamespace())[1],
            )

            result = pipeline.run(self._make_context(root, request=request))

        # llm + prompt_pipeline are always created; the expensive generators are skipped.
        self.assertIn("llm", calls)
        self.assertIn("prompt_pipeline", calls)
        self.assertNotIn("concepts", calls)
        self.assertNotIn("scene_details", calls)
        self.assertNotIn("scene_prompt_builder", calls)
        self.assertEqual("a story", result["global_context"]["story_idea"])
        self.assertEqual("concept one", result["concept_prompts"]["001"])

    def test_no_resume_runs_all_prompt_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = SimpleNamespace(resume=False, concept_batch_size=0)
            calls = []
            prompt_pipeline = SimpleNamespace(
                create_concept_prompts=lambda **kwargs: (calls.append("concepts"), {"001": "concept one"})[1],
                create_scene_details=lambda **kwargs: (calls.append("scene_details"), {"001": {"camera": "static"}})[1],
                save_json=lambda path, payload, **kwargs: _write_json(path, payload),
            )
            scene_prompt_builder = SimpleNamespace(
                build_scene_prompts=lambda **kwargs: _write_json(
                    kwargs["output_json_path"],
                    [{"segment_id": "001"}],
                )
            )
            pipeline = self._make_pipeline(
                llm_factory=lambda _app: (calls.append("llm"), SimpleNamespace(model="m", client=object()))[1],
                prompt_pipeline_factory=lambda _llm: (calls.append("prompt_pipeline"), prompt_pipeline)[1],
                concept_batcher_factory=_no_call("concept_batcher"),
                scene_prompt_builder_factory=lambda _llm: (calls.append("scene_prompt_builder"), scene_prompt_builder)[1],
            )

            result = pipeline.run(self._make_context(root, request=request))

        self.assertIn("concepts", calls)
        self.assertIn("scene_details", calls)
        self.assertIn("scene_prompt_builder", calls)
        self.assertEqual("concept one", result["concept_prompts"]["001"])


if __name__ == "__main__":
    unittest.main()
