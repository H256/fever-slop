import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from feverslop.application.sequence_reference_pipeline import (
    SequenceReferencePipeline,
    SequenceReferenceRequest,
)
from feverslop.tools.reference_bible import SEQUENCE_PHASE_LABELS


class FakeAnchorBackend:
    def __init__(self):
        self.calls = []

    def render_image(self, request):
        self.calls.append(request)
        target = Path(request.output_dir) / "anchor.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 72), "white").save(target)
        return target


class FakeSequenceBackend:
    def __init__(self):
        self.calls = []

    def build_sheet_prompt(self, description, *, kind, shots, frames):
        self.calls.append(("prompt", description, kind, shots, frames))
        return type("Prompt", (), {"prompt": f"{kind}: {description}"})()

    def render(self, *, anchor_images, prompt, output_path, seed, **kwargs):
        self.calls.append(("render", tuple(anchor_images), prompt, seed, kwargs))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"sequence")
        return output


class SequenceReferencePipelineTests(unittest.TestCase):
    def test_character_anchor_uses_identity_plan_instead_of_action_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            anchor_backend = FakeAnchorBackend()
            sequence_backend = FakeSequenceBackend()

            class Planner:
                source = "test"
                fallback_reason = None

                def plan(self, **_kwargs):
                    from feverslop.domain.reference_sheet import ReferenceSheetPlan

                    return ReferenceSheetPlan(
                        kind="character",
                        anchor_description=(
                            "A woman with long silver hair, pale skin, and dark leather armor"
                        ),
                    )

            pipeline = SequenceReferencePipeline(
                anchor_backend=anchor_backend,
                sequence_backend=sequence_backend,
                planner=Planner(),
            )
            request = SequenceReferenceRequest(
                kind="character",
                asset_id="singer",
                name="Singer",
                description="A woman with long silver hair singing intensely.",
                image_prompt=(
                    "A woman with long silver hair singing passionately on a black stone altar."
                ),
                visual_style="dark gothic cinematic fantasy",
                output_dir=Path(temp),
            )

            def fake_extract(_video, output_dir, sample_count):
                output_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for index in range(sample_count):
                    path = output_dir / f"frame_{index:04}.png"
                    Image.new("RGB", (64, 36), "white").save(path)
                    paths.append(path)
                return tuple(paths)

            with patch(
                "feverslop.application.sequence_reference_pipeline.extract_video_frames",
                fake_extract,
            ), patch(
                "feverslop.application.sequence_reference_pipeline.select_orbitsheet_frames",
                lambda paths, **_: tuple(paths[:6]),
            ):
                result = pipeline.generate(request)

            prompt = anchor_backend.calls[0].prompt
            self.assertIn("A woman with long silver hair, pale skin, and dark leather armor", prompt)
            self.assertIn("dark gothic cinematic fantasy", prompt)
            self.assertNotIn("sing", prompt.lower())
            self.assertNotIn("altar", prompt.lower())
            self.assertEqual(prompt, result.anchor_prompt)
            render_call = next(call for call in sequence_backend.calls if call[0] == "render")
            self.assertNotIn("anchor_prompt", render_call[4])

    def test_reference_phase_log_labels_are_english(self):
        self.assertEqual(
            {
                "anchor_start": "Krea anchor started",
                "anchor_complete": "Krea anchor finished",
                "sequence_start": "MiniMax sequence started",
                "sequence_complete": "MiniMax sequence finished",
            },
            SEQUENCE_PHASE_LABELS,
        )

    def test_generates_anchor_sequence_contact_and_final_sheet(self):
        with tempfile.TemporaryDirectory() as temp:
            anchor_backend = FakeAnchorBackend()
            sequence_backend = FakeSequenceBackend()
            events = []
            pipeline = SequenceReferencePipeline(
                anchor_backend=anchor_backend,
                sequence_backend=sequence_backend,
                on_phase=events.append,
            )

            def fake_extract(_video, output_dir, sample_count):
                output_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for index in range(sample_count):
                    path = output_dir / f"frame_{index:04}.png"
                    Image.new("RGB", (64, 36), (index * 10 % 255, 20, 40)).save(path)
                    paths.append(path)
                return tuple(paths)

            request = SequenceReferenceRequest(
                kind="character",
                asset_id="astronaut",
                name="Astronaut",
                description="a solitary astronaut",
                output_dir=Path(temp) / "references",
                seed=7,
                visual_style="comic, bright colors, anime style",
            )
            with patch(
                "feverslop.application.sequence_reference_pipeline.extract_video_frames",
                fake_extract,
            ), patch(
                "feverslop.application.sequence_reference_pipeline.select_orbitsheet_frames",
                lambda paths, **_: tuple(paths[:6]),
            ):
                result = pipeline.generate(request)

            self.assertEqual(6, result.selected_frames)
            self.assertTrue(Path(result.anchor_path).is_file())
            self.assertTrue(Path(result.sequence_path).is_file())
            self.assertTrue(Path(result.contact_sheet_path).is_file())
            self.assertTrue(Path(result.sheet_path).is_file())
            self.assertEqual(1, len(anchor_backend.calls))
            self.assertEqual(1, len([call for call in sequence_backend.calls if call[0] == "render"]))
            self.assertIn("comic, bright colors, anime style", anchor_backend.calls[0].prompt)
            self.assertIn("comic, bright colors, anime style", sequence_backend.calls[1][2])
            self.assertEqual(
                ["anchor_start", "anchor_complete", "sequence_start", "sequence_complete"],
                [event["phase"] for event in events],
            )

    def test_location_uses_five_views(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = SequenceReferencePipeline(
                anchor_backend=FakeAnchorBackend(),
                sequence_backend=FakeSequenceBackend(),
            )
            request = SequenceReferenceRequest(
                kind="location",
                asset_id="ship",
                name="Ship",
                description="a weathered ship",
                output_dir=Path(temp),
            )
            with patch(
                "feverslop.application.sequence_reference_pipeline.extract_video_frames",
                lambda _video, _output_dir, sample_count: (),
            ):
                with self.assertRaises(ValueError):
                    pipeline.generate(request)

    def test_rejects_unsafe_asset_ids_without_touching_filesystem(self):
        for asset_id in ("../evil", "a/b", "/abs", "..", "has space", "x\\y"):
            with self.subTest(asset_id=asset_id):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    # Sentinel: with kind="character" and id="../evil", final_dir
                    # resolves to root/"evil" -- a pre-existing sibling asset dir
                    # that the old code would have rmtree'd after validation.
                    sentinel = root / "evil" / "keep.png"
                    sentinel.parent.mkdir(parents=True)
                    sentinel.write_bytes(b"keep")
                    anchor_backend = FakeAnchorBackend()
                    sequence_backend = FakeSequenceBackend()
                    pipeline = SequenceReferencePipeline(
                        anchor_backend=anchor_backend,
                        sequence_backend=sequence_backend,
                    )
                    request = SequenceReferenceRequest(
                        kind="character",
                        asset_id=asset_id,
                        name="Sentinel",
                        description="a sentinel actor",
                        output_dir=root,
                    )
                    with self.assertRaisesRegex(ValueError, "safe path component"):
                        pipeline.generate(request)
                    self.assertEqual([], anchor_backend.calls)
                    self.assertEqual([], sequence_backend.calls)
                    self.assertEqual(["evil"], sorted(p.name for p in root.iterdir()))
                    self.assertEqual(b"keep", sentinel.read_bytes())

    def test_regenerate_replaces_previous_final_directory_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            final_dir = root / "actors" / "rogue"
            frames_dir = final_dir / "frames"
            frames_dir.mkdir(parents=True)
            old_bytes = {
                "anchor.png": b"old-anchor",
                "sequence.mp4": b"old-sequence",
                "contact-sheet.png": b"old-contact",
                "sheet.png": b"old-sheet",
            }
            for name, payload in old_bytes.items():
                (final_dir / name).write_bytes(payload)
            (final_dir / "stale-note.txt").write_bytes(b"stale")
            for index in range(3):
                (frames_dir / f"frame_{index:04}.png").write_bytes(f"old-frame-{index}".encode())

            pipeline = SequenceReferencePipeline(
                anchor_backend=FakeAnchorBackend(),
                sequence_backend=FakeSequenceBackend(),
            )
            request = SequenceReferenceRequest(
                kind="character",
                asset_id="rogue",
                name="Rogue",
                description="a rogue agent",
                output_dir=root,
            )

            def fake_extract(_video, output_dir, sample_count):
                output_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for index in range(sample_count):
                    path = output_dir / f"frame_{index:04}.png"
                    Image.new("RGB", (64, 36), "white").save(path)
                    paths.append(path)
                return tuple(paths)

            with patch(
                "feverslop.application.sequence_reference_pipeline.extract_video_frames",
                fake_extract,
            ), patch(
                "feverslop.application.sequence_reference_pipeline.select_orbitsheet_frames",
                lambda paths, **_: tuple(paths[:6]),
            ):
                result = pipeline.generate(request)

            self.assertEqual(final_dir / "anchor.png", result.anchor_path)
            self.assertEqual(final_dir / "sequence.mp4", result.sequence_path)
            self.assertEqual(final_dir / "contact-sheet.png", result.contact_sheet_path)
            self.assertEqual(final_dir / "sheet.png", result.sheet_path)
            self.assertEqual(6, result.selected_frames)
            for name, payload in old_bytes.items():
                self.assertNotEqual(payload, (final_dir / name).read_bytes())
            self.assertFalse((final_dir / "stale-note.txt").exists())
            self.assertEqual(
                [f"frame_{index:04}.png" for index in range(6)],
                sorted(p.name for p in frames_dir.iterdir()),
            )
            for path in frames_dir.iterdir():
                self.assertNotIn(b"old-frame", path.read_bytes())
            self.assertEqual(["rogue"], sorted(p.name for p in (root / "actors").iterdir()))

    def test_mid_copy_failure_preserves_previous_final_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            final_dir = root / "actors" / "rogue"
            frames_dir = final_dir / "frames"
            frames_dir.mkdir(parents=True)
            old_bytes = {
                "anchor.png": b"old-anchor",
                "sequence.mp4": b"old-sequence",
                "contact-sheet.png": b"old-contact",
                "sheet.png": b"old-sheet",
            }
            for name, payload in old_bytes.items():
                (final_dir / name).write_bytes(payload)
            for index in range(6):
                (frames_dir / f"frame_{index:04}.png").write_bytes(f"old-frame-{index}".encode())

            copied = {"count": 0}

            # Copy call 4 (dict order: anchor, sequence, contact-sheet, sheet)
            # is the sheet.png copy; faulting there leaves the staged set partial.
            def failing_copy2(source, destination, **kwargs):
                copied["count"] += 1
                if copied["count"] == 4:
                    raise OSError("simulated write failure")
                shutil.copy2(source, destination, **kwargs)

            pipeline = SequenceReferencePipeline(
                anchor_backend=FakeAnchorBackend(),
                sequence_backend=FakeSequenceBackend(),
            )
            request = SequenceReferenceRequest(
                kind="character",
                asset_id="rogue",
                name="Rogue",
                description="a rogue agent",
                output_dir=root,
            )

            def fake_extract(_video, output_dir, sample_count):
                output_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for index in range(sample_count):
                    path = output_dir / f"frame_{index:04}.png"
                    Image.new("RGB", (64, 36), "white").save(path)
                    paths.append(path)
                return tuple(paths)

            with patch(
                "feverslop.application.sequence_reference_pipeline.extract_video_frames",
                fake_extract,
            ), patch(
                "feverslop.application.sequence_reference_pipeline.select_orbitsheet_frames",
                lambda paths, **_: tuple(paths[:6]),
            ), patch("shutil.copy2", failing_copy2):
                with self.assertRaises(OSError):
                    pipeline.generate(request)

            self.assertEqual(4, copied["count"])
            for name, payload in old_bytes.items():
                self.assertEqual(payload, (final_dir / name).read_bytes())
            for index in range(6):
                self.assertEqual(
                    f"old-frame-{index}".encode(),
                    (frames_dir / f"frame_{index:04}.png").read_bytes(),
                )
            self.assertEqual(["rogue"], sorted(p.name for p in (root / "actors").iterdir()))
            self.assertEqual([], [p.name for p in root.iterdir() if p.name != "actors"])

    def test_swap_failure_restores_previous_final_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            final_dir = root / "actors" / "rogue"
            frames_dir = final_dir / "frames"
            frames_dir.mkdir(parents=True)
            (final_dir / "anchor.png").write_bytes(b"old-anchor")
            for index in range(6):
                (frames_dir / f"frame_{index:04}.png").write_bytes(f"old-frame-{index}".encode())

            real_replace = Path.replace
            replace_calls = {"count": 0}

            # Call 1 is the final->backup rename (succeeds); call 2 is the
            # commit rename (faults); call 3 is the except-branch restore.
            def flaky_replace(self, target, *args, **kwargs):
                replace_calls["count"] += 1
                if replace_calls["count"] == 2:
                    raise OSError("simulated commit rename failure")
                return real_replace(self, target, *args, **kwargs)

            pipeline = SequenceReferencePipeline(
                anchor_backend=FakeAnchorBackend(),
                sequence_backend=FakeSequenceBackend(),
            )
            request = SequenceReferenceRequest(
                kind="character",
                asset_id="rogue",
                name="Rogue",
                description="a rogue agent",
                output_dir=root,
            )

            def fake_extract(_video, output_dir, sample_count):
                output_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for index in range(sample_count):
                    path = output_dir / f"frame_{index:04}.png"
                    Image.new("RGB", (64, 36), "white").save(path)
                    paths.append(path)
                return tuple(paths)

            with patch(
                "feverslop.application.sequence_reference_pipeline.extract_video_frames",
                fake_extract,
            ), patch(
                "feverslop.application.sequence_reference_pipeline.select_orbitsheet_frames",
                lambda paths, **_: tuple(paths[:6]),
            ), patch.object(Path, "replace", flaky_replace):
                with self.assertRaises(OSError):
                    pipeline.generate(request)

            self.assertEqual(3, replace_calls["count"])
            self.assertEqual(b"old-anchor", (final_dir / "anchor.png").read_bytes())
            for index in range(6):
                self.assertEqual(
                    f"old-frame-{index}".encode(),
                    (frames_dir / f"frame_{index:04}.png").read_bytes(),
                )
            self.assertEqual(["rogue"], sorted(p.name for p in (root / "actors").iterdir()))
            self.assertEqual([], [p.name for p in root.iterdir() if p.name != "actors"])


if __name__ == "__main__":
    unittest.main()
