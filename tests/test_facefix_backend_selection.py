"""Tests for FaceFix backend selection (issue 519, unit 1).

Covers the backend-selection boundary: the pure ``select_facefix_backend``
function, the ``run_facefix`` routing by video pipeline, the explicit
``--facefix-backend`` override, and the not-yet-implemented H3 FaceRefine
skip path.
"""
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from feverslop.domain.facefix_rendering import (
    FaceFixBackendKind,
    select_facefix_backend,
)


class TestSelectFaceFixBackend(unittest.TestCase):
    def test_h3_r2v_selects_h3_facefix(self):
        self.assertIs(
            select_facefix_backend("minimax-h3-r2v"),
            FaceFixBackendKind.H3_FACEFIX,
        )

    def test_ltx_pipelines_select_ltxv_crop(self):
        for vp in ("ltx_i2v", "ltx_msr", "ltx_ingredients"):
            with self.subTest(video_pipeline=vp):
                self.assertIs(
                    select_facefix_backend(vp),
                    FaceFixBackendKind.LTXV_CROP,
                )

    def test_other_h3_variants_stay_on_crop_path(self):
        # Only minimax-h3-r2v is in scope for the H3-native pass; other H3
        # variants keep the LTXV crop path until explicitly added.
        for vp in ("minimax-h3-t2v", "minimax-h3-i2v", "minimax-h3-fl2v", "minimax-h3-l2v"):
            with self.subTest(video_pipeline=vp):
                self.assertIs(
                    select_facefix_backend(vp),
                    FaceFixBackendKind.LTXV_CROP,
                )

    def test_unknown_pipeline_selects_ltxv_crop(self):
        self.assertIs(select_facefix_backend(""), FaceFixBackendKind.LTXV_CROP)
        self.assertIs(select_facefix_backend("unknown"), FaceFixBackendKind.LTXV_CROP)

    def test_override_wins_over_video_pipeline(self):
        # H3 R2V normally -> h3_facefix, but an explicit override wins.
        self.assertIs(
            select_facefix_backend("minimax-h3-r2v", "ltxv_crop"),
            FaceFixBackendKind.LTXV_CROP,
        )
        # LTX normally -> ltxv_crop, but an explicit override wins.
        self.assertIs(
            select_facefix_backend("ltx_i2v", "h3_facefix"),
            FaceFixBackendKind.H3_FACEFIX,
        )

    def test_invalid_override_raises(self):
        with self.assertRaises(ValueError):
            select_facefix_backend("ltx_i2v", "not_a_backend")


class TestRunFaceFixRouting(unittest.TestCase):
    def test_h3_r2v_routes_to_h3_facefix(self):
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            run_facefix,
        )

        with patch(
            "feverslop.composition.facefix_pipeline._run_h3_facefix"
        ) as mock_h3, patch(
            "feverslop.composition.facefix_pipeline._run_crop_facefix"
        ) as mock_crop, patch(
            "feverslop.composition.facefix_pipeline._run_legacy_facefix"
        ) as mock_legacy:
            mock_h3.return_value = []
            options = FaceFixCompositionOptions(
                scenes_dir="/tmp/scenes",
                video_pipeline="minimax-h3-r2v",
            )
            run_facefix(options)
            mock_h3.assert_called_once()
            mock_crop.assert_not_called()
            mock_legacy.assert_not_called()

    def test_ltx_routes_to_crop_path(self):
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            run_facefix,
        )

        with patch(
            "feverslop.composition.facefix_pipeline._run_h3_facefix"
        ) as mock_h3, patch(
            "feverslop.composition.facefix_pipeline._run_crop_facefix"
        ) as mock_crop, patch(
            "feverslop.composition.facefix_pipeline._run_legacy_facefix"
        ) as mock_legacy:
            mock_crop.return_value = [Path("/tmp/result.mp4")]
            options = FaceFixCompositionOptions(
                scenes_dir="/tmp/scenes",
                video_pipeline="ltx_i2v",
            )
            run_facefix(options)
            mock_crop.assert_called_once()
            mock_h3.assert_not_called()
            mock_legacy.assert_not_called()

    def test_override_to_h3_wins(self):
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            run_facefix,
        )

        with patch(
            "feverslop.composition.facefix_pipeline._run_h3_facefix"
        ) as mock_h3, patch(
            "feverslop.composition.facefix_pipeline._run_crop_facefix"
        ) as mock_crop:
            mock_h3.return_value = []
            options = FaceFixCompositionOptions(
                scenes_dir="/tmp/scenes",
                video_pipeline="ltx_i2v",
                facefix_backend="h3_facefix",
            )
            run_facefix(options)
            mock_h3.assert_called_once()
            mock_crop.assert_not_called()

    def test_override_to_crop_wins_over_h3(self):
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            run_facefix,
        )

        with patch(
            "feverslop.composition.facefix_pipeline._run_h3_facefix"
        ) as mock_h3, patch(
            "feverslop.composition.facefix_pipeline._run_crop_facefix"
        ) as mock_crop:
            mock_crop.return_value = [Path("/tmp/result.mp4")]
            options = FaceFixCompositionOptions(
                scenes_dir="/tmp/scenes",
                video_pipeline="minimax-h3-r2v",
                facefix_backend="ltxv_crop",
            )
            run_facefix(options)
            mock_crop.assert_called_once()
            mock_h3.assert_not_called()


class TestRunH3FaceFixSkip(unittest.TestCase):
    def test_returns_empty_and_reports_warning(self):
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            _run_h3_facefix,
        )

        options = FaceFixCompositionOptions(
            scenes_dir="/tmp/scenes",
            video_pipeline="minimax-h3-r2v",
            scene_numbers=[1, 2, 3],
        )
        with patch(
            "feverslop.composition.facefix_pipeline.ConsoleReporter"
        ) as mock_reporter_cls:
            mock_reporter = mock_reporter_cls.return_value
            result = _run_h3_facefix(options, console=Console())
            self.assertEqual(result, [])
            mock_reporter.warning.assert_called_once()
            message = mock_reporter.warning.call_args.args[0]
            self.assertIn("not implemented yet", message)
            self.assertIn("minimax-h3-r2v", message)
            self.assertIn("3 scenes", message)

    def test_no_console_still_returns_empty(self):
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            _run_h3_facefix,
        )

        options = FaceFixCompositionOptions(
            scenes_dir="/tmp/scenes",
            video_pipeline="minimax-h3-r2v",
        )
        self.assertEqual(_run_h3_facefix(options, console=None), [])


if __name__ == "__main__":
    unittest.main()
