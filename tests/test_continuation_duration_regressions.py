import json
import tempfile
import unittest
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.application.effective_render_plan import project_effective_plan
from feverslop.application.render_video import RenderVideoScenesRequest, RenderVideoScenesUseCase
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.continuation_segments import split_semantic_action
from feverslop.domain.duration_capability import DurationCapability
from feverslop.pipeline.render_plan_builder import build_render_plan
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.composition.stage_runners import _assemble_declared_cutless_groups
from feverslop.pipeline.continuation_render_plan import materialize_continuation_entries
from feverslop.composition.canonical_plan_regenerator import CanonicalPlanRegenerator
from feverslop.domain.canonical_render_plan import build_canonical_scene, PromptRole


class ContinuationDurationRegressions(unittest.TestCase):
    def test_h3_uses_backend_duration_contract_when_profile_has_no_limits(self):
        from feverslop.composition.continuation_capability import resolve_continuation_capability

        capability = resolve_continuation_capability("minimax-h3-r2v", None)
        self.assertEqual((24, 4, 15, 17, 5), (
            capability.fps, capability.min_seconds, capability.max_seconds,
            capability.frame_alignment, capability.frame_offset,
        ))
        self.assertIsNone(resolve_continuation_capability("ltx_i2v", None))
        custom = DurationCapability.create(fps=24, min_seconds=2, max_seconds=8, preferred_seconds=8)
        self.assertIs(custom, resolve_continuation_capability(
            "minimax-h3-r2v", SimpleNamespace(duration_capability=custom),
        ))

    def test_selected_split_preserves_human_override_and_enriched_references(self):
        semantic = {
            "scene": 7, "segment_id": "s7", "fps": 24, "abs_start_seconds": 0,
            "canonical": build_canonical_scene(
                segment_id="s7", generated_roles={PromptRole.H3_VIDEO: "orbit"},
            ),
        }
        semantic["canonical"]["roles"][PromptRole.H3_VIDEO]["override"] = {
            "value": "human approved", "provenance": {"source": "human"},
        }
        segments = split_semantic_action(
            action_id="orbit", start_seconds=0, duration_seconds=8,
            max_duration_seconds=3, fps=24,
        )
        entries = materialize_continuation_entries(
            semantic, group={"group_id": "g7", "segments": [s.__dict__ for s in segments]},
        )
        for entry in entries:
            entry["canonical"]["roles"][PromptRole.H3_VIDEO].pop("override")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "output/render/plans/base.json"
            plan.parent.mkdir(parents=True)
            plan.write_text(json.dumps([semantic]))
            reference = root / "references.json"
            reference.write_text(json.dumps([{
                **semantic, "references": {"actor_msr_paths": ["actors/actor.png"]},
            }]))
            regenerator = CanonicalPlanRegenerator(
                root, selected_scene_numbers={7}, reference_plan_path=reference,
            )
            regenerator.write(plan, entries)
            result = project_effective_plan(json.loads(plan.read_text()))
            self.assertEqual(3, len(result))
            self.assertEqual(["human approved"] * 3, [s["h3"]["prompt"] for s in result])
            self.assertTrue(all(s["references"]["actor_msr_paths"] == ["actors/actor.png"] for s in result))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_real_frame_handoff_resume_and_assembly_preserve_eight_seconds(self):
        capability = DurationCapability.create(
            fps=24, min_seconds=2, max_seconds=3, preferred_seconds=3,
        )
        segments = split_semantic_action(
            action_id="orbit", start_seconds=10, duration_seconds=8,
            max_duration_seconds=3, fps=24, capability=capability,
        )
        entries = materialize_continuation_entries(
            {"scene": 7, "segment_id": "s7", "fps": 24},
            group={"group_id": "orbit", "segments": [s.__dict__ for s in segments]},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text(json.dumps(entries))
            postprocessor = VideoPostProcessor()
            calls = []

            def render(request):
                calls.append(request.scene_number)
                clip = root / f"scene_{request.scene_number:04}" / "final.mp4"
                clip.parent.mkdir(exist_ok=True)
                frames = request.scene["frame_count"] + request.scene["anchor_frames"]
                subprocess.run([
                    "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=black:s=64x64:r=24", "-frames:v", str(frames),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
                ], check=True, capture_output=True)
                return clip

            backend = SimpleNamespace(
                pipeline_name="minimax-h3-r2v", project_dir=root,
                postprocessor=postprocessor, render_video=render,
            )
            request = RenderVideoScenesRequest(
                render_plan_path=root / "plan.json", workflow_path=root / "workflow.json",
                audio_file=root / "song.wav", storyboard_dir=root, output_dir=root,
                scene_numbers={7},
            )
            use_case = RenderVideoScenesUseCase(backend, JsonArtifactStore())
            clips = use_case.execute(request)
            calls.clear()
            self.assertEqual(clips, use_case.execute(request))
            self.assertEqual([], calls)
            [assembled] = _assemble_declared_cutless_groups(
                entries, clips, output_dir=root, postprocessor=postprocessor,
            )
            self.assertEqual(192, postprocessor._frame_count(assembled))

    def test_resume_rerenders_stale_successors_and_recovers_after_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [{
                "scene": n, "technical_segment_id": f"s{n}", "frame_count": 72,
                "continuation_predecessor_id": f"s{n-1}" if n > 1 else None,
            } for n in (1, 2, 3)]
            (root / "plan.json").write_text(json.dumps(entries))
            calls = []
            interrupted = set()

            def extract(clip, path):
                path.write_bytes(clip.read_bytes())
                return path

            def render(request):
                calls.append(request.scene_number)
                if request.scene_number in interrupted:
                    raise RuntimeError("interrupted")
                clip = root / f"scene_{request.scene_number:04}" / "final.mp4"
                clip.parent.mkdir(exist_ok=True)
                clip.write_bytes(str(len(calls)).encode())
                return clip

            backend = SimpleNamespace(
                pipeline_name="minimax-h3-r2v", project_dir=root, render_video=render,
                postprocessor=SimpleNamespace(
                    extract_last_frame=extract, last_frame_index=lambda clip: 71,
                    _frame_count=lambda clip: 0 if clip.read_bytes() == b"corrupt" else 72,
                ),
            )
            request = RenderVideoScenesRequest(
                render_plan_path=root / "plan.json", workflow_path=root / "workflow.json",
                audio_file=root / "song.wav", storyboard_dir=root, output_dir=root,
            )
            use_case = RenderVideoScenesUseCase(backend, JsonArtifactStore())
            outputs = use_case.execute(request)
            calls.clear()
            use_case.execute(request)
            self.assertEqual([], calls)
            outputs[0].write_bytes(b"changed predecessor")
            interrupted.add(2)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                use_case.execute(request)
            interrupted.clear()
            calls.clear()
            use_case.execute(replace(request, skip_existing=True))
            self.assertEqual([2, 3], calls)
            calls.clear()
            outputs[2].write_bytes(b"corrupt")
            use_case.execute(request)
            self.assertEqual([3], calls)

    def test_arbitrary_durations_have_aligned_generation_and_exact_timeline(self):
        capability = DurationCapability.create(
            fps=24, min_seconds=2, max_seconds=12, preferred_seconds=8,
            frame_alignment=17, frame_offset=5,
        )
        for frames in (289, 600, 601, 900, 1201):
            with self.subTest(frames=frames):
                segments = split_semantic_action(
                    action_id="orbit", start_seconds=10, duration_seconds=frames / 24,
                    max_duration_seconds=12, fps=24, capability=capability,
                )
                self.assertEqual(frames, sum(round(s.duration_seconds * 24) for s in segments))
                for segment in segments:
                    self.assertEqual(5, segment.render_frame_count % 17)
                    self.assertLessEqual(segment.render_frame_count, 277)
                    self.assertGreaterEqual(
                        segment.render_frame_count,
                        round(segment.duration_seconds * 24) + segment.anchor_frames,
                    )
                    self.assertEqual(int(segment.index > 1), segment.anchor_frames)

    def test_long_scenes_split_without_intent_and_keep_canonical_relay_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = [{
                "scene": number, "segment_id": f"scene-{number}", "type": "instrumental",
                "start": (number - 1) * 8, "end": number * 8, "duration": 8,
                "zimage_prompt": "z", "ltx_base_prompt": "orbit",
            } for number in (1, 2)]
            for name, payload in (
                ("prompts", prompts),
                ("relay", [{"scene": n, "prompt_relay": []} for n in (1, 2)]),
                ("h3", [{"segment_id": f"scene-{n}", "prompt": "orbit"} for n in (1, 2)]),
            ):
                (root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
            build_render_plan(
                root / "prompts.json", root / "relay.json", root / "plan.json",
                VideoSettings(), artifact_store=JsonArtifactStore(), h3_prompts_json=root / "h3.json",
                duration_capability=DurationCapability.create(
                    fps=24, min_seconds=2, max_seconds=3, preferred_seconds=3,
                    frame_alignment=8,
                ),
            )
            entries = project_effective_plan(json.loads((root / "plan.json").read_text()))
            self.assertGreater(len(entries), 2)
            self.assertEqual(len(entries), len({s["segment_id"] for s in entries}))
            self.assertEqual(384, sum(s["frame_count"] for s in entries))
            self.assertEqual(16, entries[-1]["abs_end_seconds"])
            for entry in entries:
                self.assertLessEqual(entry["render_frame_count"], 72)
                self.assertEqual(entry["frame_count"], entry["ltx"]["prompt_relay"][-1]["frame_end"])


if __name__ == "__main__":
    unittest.main()
