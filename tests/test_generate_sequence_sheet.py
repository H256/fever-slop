from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from feverslop.application.sequence_reference_pipeline import SequenceReferencePipeline, SequenceReferenceRequest
from feverslop.tools.generate_sequence_sheet import build_arg_parser, run


class _SequenceBackend:
    profile = type("Profile", (), {"workflow_filename": "test-workflow.json"})()

    def build_sheet_prompt(self, description, *, kind, shots, frames):
        return type("Prompt", (), {"prompt": description, "frames": frames})()

    def render(self, *, anchor_images, output_path, **kwargs):
        self.anchor_images = tuple(anchor_images)
        Path(output_path).write_bytes(b"video")
        return output_path


class GenerateSequenceSheetTests(unittest.TestCase):
    def test_parser_requires_source_kind_and_id(self):
        args = build_arg_parser().parse_args(["--source-image", "hero.png", "--kind", "character", "--id", "hero"])
        self.assertEqual(args.publish, "local")

    def test_dry_run_validates_source_and_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "hero.png"
            Image.new("RGB", (2, 2), "red").save(source)
            args = build_arg_parser().parse_args(["--source-image", str(source), "--kind", "character", "--id", "hero", "--dry-run", "--json"])
            payload = run(args)
            self.assertEqual(payload["status"], "planned")
            json.dumps(payload)

    def test_supplied_source_image_is_used_as_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "hero.jpg"
            Image.new("RGB", (2, 2), "red").save(source)
            frame = root / "frame.png"
            frame.write_bytes(b"frame")
            backend = _SequenceBackend()
            request = SequenceReferenceRequest(kind="character", asset_id="hero", name="Hero", description="one person", output_dir=root / "out", source_image=source)
            def create_output(_selected, output_path, **_kwargs):
                Path(output_path).write_bytes(b"sheet")

            with patch("feverslop.application.sequence_reference_pipeline.extract_video_frames", return_value=(frame,)) as extract, patch("feverslop.application.sequence_reference_pipeline.select_orbitsheet_frames", return_value=(frame,)), patch("feverslop.application.sequence_reference_pipeline.compose_contact_sheet", side_effect=create_output), patch("feverslop.application.sequence_reference_pipeline.compose_sheet_from_contact_sheet", side_effect=create_output):
                result = SequenceReferencePipeline(anchor_backend=None, sequence_backend=backend).generate(request)
            with Image.open(result.anchor_path) as image:
                self.assertEqual(image.size, (2, 2))
            extract.assert_called_once()
