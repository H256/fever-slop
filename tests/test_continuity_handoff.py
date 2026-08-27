import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from feverslop.domain.visual_consistency import (
    ReferenceAnchor,
    SceneConsistencyContract,
)


def _actor(actor_id: str) -> ReferenceAnchor:
    return ReferenceAnchor(
        id=actor_id,
        kind="actor",
        look_id="default",
        asset_role="identity-reference",
        asset_sha256="a" * 64,
        prompt_anchor=actor_id,
    )


def _location(location_id: str) -> ReferenceAnchor:
    return ReferenceAnchor(
        id=location_id,
        kind="location",
        look_id="default",
        asset_role="environment-reference",
        asset_sha256="b" * 64,
        prompt_anchor=location_id,
    )


def _contract(
    scene: int,
    *,
    mode: str = "msr",
    transition: str = "cut",
    actor_id: str = "hero",
    location_id: str | None = "archive",
) -> SceneConsistencyContract:
    return SceneConsistencyContract.create(
        scene=scene,
        mode=mode,
        workflow_profile="profile",
        actors=(_actor(actor_id),),
        location=_location(location_id) if location_id else None,
        transition_from_previous=transition,
    )


class _Extractor:
    def __init__(self):
        self.calls = []

    def extract_last_frame(self, video_path: Path, output_path: Path) -> Path:
        self.calls.append((video_path, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"frame")
        return output_path


class ContinuityHandoffTests(unittest.TestCase):
    def test_explicit_technical_continuation_allows_nonsemantic_scene_numbers(self):
        from feverslop.application.continuity_handoff import ContinuityHandoffUseCase

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            clip = project / "scene_7001001.mp4"
            clip.write_bytes(b"clip")
            extractor = _Extractor()
            extractor.project_dir = project
            result = ContinuityHandoffUseCase(extractor).execute(
                _contract(7_001_001),
                _contract(7_001_002, transition="continuous"),
                clip,
                project / "handoff.png",
                {
                    "scene": 7_001_002,
                    "continuation_predecessor_id": "orbit-0001",
                },
            )

        self.assertEqual(7_001_001, result["keyframes"]["startframe_source_scene"])
        self.assertEqual(1, len(extractor.calls))

    def test_handoff_predecessors_follow_technical_segment_ids(self):
        from feverslop.composition.stage_runners import _music_handoff_predecessors

        scenes = [
            SimpleNamespace(to_dict=lambda: {
                "scene": 7_001_001,
                "technical_segment_id": "orbit-0001",
                "visual_consistency": _contract(7_001_001).to_dict(),
            }),
            SimpleNamespace(to_dict=lambda: {
                "scene": 7_001_002,
                "technical_segment_id": "orbit-0002",
                "continuation_predecessor_id": "orbit-0001",
                "visual_consistency": _contract(7_001_002, transition="continuous").to_dict(),
            }),
        ]

        predecessors = _music_handoff_predecessors(
            scenes,
            profile=SimpleNamespace(supports_start_frame=True),
        )

        self.assertEqual({7_001_002: 7_001_001}, predecessors)

    def test_rejects_out_of_order_duplicate_and_gapped_scene_sequences(self):
        from feverslop.domain.visual_consistency import validate_scene_sequence

        for scenes in (
            [{"scene": 2}, {"scene": 1}],
            [{"scene": 1}, {"scene": 1}],
            [{"scene": 1}, {"scene": 3}],
        ):
            with self.subTest(scenes=scenes), self.assertRaisesRegex(
                ValueError,
                "consecutive order",
            ):
                validate_scene_sequence(scenes)

    def test_eligible_msr_scene_uses_previous_last_frame_without_mutation(self):
        from feverslop.application.continuity_handoff import ContinuityHandoffUseCase

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            clip = project / "output" / "scene_0001.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"clip")
            frame = project / "output" / "keyframes" / "handoff.png"
            extractor = _Extractor()
            extractor.project_dir = project
            scene = {"scene": 2, "keyframes": {"kept": True}, "ltx": {"base_prompt": "next"}}

            result = ContinuityHandoffUseCase(extractor).execute(
                _contract(1),
                _contract(2, mode="i2v", transition="continuous"),
                clip.relative_to(project),
                frame,
                scene,
            )
            self.assertTrue((project / "output" / "keyframes" / "handoff.manifest.json").is_file())

        self.assertEqual({"scene": 2, "keyframes": {"kept": True}, "ltx": {"base_prompt": "next"}}, scene)
        self.assertEqual([(clip.resolve(), frame)], extractor.calls)
        self.assertEqual("last_frame_from_previous", result["keyframes"]["startframe_mode"])
        self.assertEqual(1, result["keyframes"]["startframe_source_scene"])
        self.assertEqual(frame.as_posix(), result["keyframes"]["startframe_path"])
        self.assertEqual(
            __import__("hashlib").sha256(b"frame").hexdigest(),
            result["keyframes"]["startframe_sha256"],
        )
        self.assertEqual(
            "output/scene_0001.mp4",
            result["keyframes"]["startframe_source_clip_path"],
        )
        self.assertEqual("last-frame-v1", result["keyframes"]["startframe_extractor"])
        manifest = result["keyframes"]["boundary_frame_manifest"]
        self.assertEqual("output/scene_0001.mp4", manifest["source_clip_path"])
        self.assertEqual("output/keyframes/handoff.png", manifest["frame_path"])
        self.assertEqual(0, manifest["frame_index"])
        handoff = result["keyframes"]["continuity_handoff"]
        self.assertEqual(1, handoff["source_scene"])
        self.assertEqual("continuous", handoff["transition"])
        self.assertEqual(result["keyframes"]["startframe_path"], handoff["last_frame_path"])
        self.assertEqual(64, len(result["keyframes"]["startframe_source_clip_sha256"]))
        self.assertTrue(result["keyframes"]["kept"])
        self.assertEqual(18, result["ltx"]["msr_continuity_handoff_frames"])
        self.assertEqual(17, result["ltx"]["msr_continuity_msr_frame_count"])
        self.assertEqual(18, result["ltx"]["msr_continuity_guide_frame_idx"])

    def test_rejects_ineligible_contracts_before_extraction(self):
        from feverslop.application.continuity_handoff import ContinuityHandoffUseCase

        cases = (
            (_contract(1), _contract(2, transition="cut")),
            (_contract(1), _contract(2, transition="continuous", location_id="street")),
            (_contract(1), _contract(2, transition="continuous", actor_id="villain")),
            (_contract(1), _contract(3, transition="continuous")),
            (_contract(1, mode="ingredients"), _contract(2, mode="ingredients", transition="continuous")),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            clip = project / "clip.mp4"
            clip.write_bytes(b"clip")
            extractor = _Extractor()
            use_case = ContinuityHandoffUseCase(extractor)
            for previous, current in cases:
                with self.subTest(current=current), self.assertRaisesRegex(
                    ValueError, "does not support continuity handoff",
                ):
                    use_case.execute(
                        previous,
                        current,
                        clip,
                        project / "frame.png",
                        {"scene": current.scene},
                    )
        self.assertEqual([], extractor.calls)

    def test_reuses_matching_boundary_manifest_without_reextracting(self):
        from feverslop.application.continuity_handoff import ContinuityHandoffUseCase

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            clip = project / "output" / "scene_0001.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"clip")
            frame = project / "output" / "keyframes" / "handoff.png"
            extractor = _Extractor()
            extractor.project_dir = project
            use_case = ContinuityHandoffUseCase(extractor)
            previous, current = _contract(1), _contract(2, mode="i2v", transition="continuous")

            use_case.execute(previous, current, clip, frame, {"scene": 2})
            use_case.execute(previous, current, clip, frame, {"scene": 2})

        self.assertEqual(1, len(extractor.calls))

    def test_rejects_missing_or_external_clip(self):
        from feverslop.application.continuity_handoff import ContinuityHandoffUseCase

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as outside_dir:
            project = Path(temp_dir)
            outside = Path(outside_dir) / "clip.mp4"
            outside.write_bytes(b"clip")
            postprocessor = _Extractor()
            for clip, message in (
                (project / "missing.mp4", "missing previous movie scene clip"),
                (outside, "outside project"),
            ):
                with self.subTest(clip=clip), self.assertRaisesRegex(ValueError, message):
                    from feverslop.adapters.postprocessor_frame_extractor import (
                        PostprocessorFrameExtractor,
                    )

                    ContinuityHandoffUseCase(
                        PostprocessorFrameExtractor(
                            postprocessor,
                            project_dir=project,
                            selected_rerender=True,
                        ),
                    ).execute(
                        _contract(1),
                        _contract(2, transition="continuous"),
                        clip,
                        project / "frame.png",
                        {"scene": 2},
                    )

    def test_frame_extractor_rejects_external_requested_and_returned_paths(self):
        from feverslop.adapters.postprocessor_frame_extractor import (
            PostprocessorFrameExtractor,
        )

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as outside_dir:
            project = Path(temp_dir)
            outside = Path(outside_dir)
            clip = project / "clip.mp4"
            clip.write_bytes(b"clip")
            postprocessor = _Extractor()
            extractor = PostprocessorFrameExtractor(
                postprocessor,
                project_dir=project,
            )
            with self.assertRaisesRegex(ValueError, "output frame is outside project"):
                extractor.extract_last_frame(clip, outside / "frame.png")

            postprocessor.extract_last_frame = lambda _clip, _output: outside / "returned.png"
            with self.assertRaisesRegex(ValueError, "extracted frame is outside project"):
                extractor.extract_last_frame(clip, project / "frame.png")

    def test_frame_extractor_rejects_nonexistent_returned_frame(self):
        from feverslop.adapters.postprocessor_frame_extractor import (
            PostprocessorFrameExtractor,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            clip = project / "clip.mp4"
            clip.write_bytes(b"clip")
            postprocessor = _Extractor()
            postprocessor.extract_last_frame = (
                lambda _clip, _output: project / "missing.png"
            )

            with self.assertRaisesRegex(
                ValueError,
                "unexpected output frame|did not produce output frame",
            ):
                PostprocessorFrameExtractor(
                    postprocessor,
                    project_dir=project,
                ).extract_last_frame(clip, project / "frame.png")



    def test_resolved_project_dir_computed_once(self):
        """Verify that project_dir resolution happens once, not per call."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "proj"
            project.mkdir(parents=True)
            clip = project / "output" / "scene_0001.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"clip")
            frame = project / "output" / "keyframes" / "handoff.png"

            mock_extract = _Extractor()
            mock_extract.project_dir = project

            from feverslop.application.continuity_handoff import (
                ContinuityHandoffUseCase,
            )

            result = ContinuityHandoffUseCase(mock_extract).execute(
                _contract(1),
                _contract(2, mode="i2v", transition="continuous"),
                clip.relative_to(project),
                frame,
                {"scene": 2, "keyframes": {}, "ltx": {}},
            )

            # Verify result is correct despite using resolved path
            self.assertEqual(
                "output/scene_0001.mp4",
                result["keyframes"]["startframe_source_clip_path"],
            )
            self.assertEqual("last_frame_from_previous", result["keyframes"]["startframe_mode"])

if __name__ == "__main__":
    unittest.main()

