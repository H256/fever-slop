import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feverslop.composition.render_video import (
    RenderVideoCompositionOptions,
    build_render_video_scenes_use_case,
)


class RenderVideoCompositionProfileTests(unittest.TestCase):
    def _app_config(self, directory: Path) -> Path:
        path = directory / "app_config.json"
        path.write_text(json.dumps({
            "video_workflow_profiles": [{
                "name": "ingredients-final",
                "pipeline": "ltx_ingredients",
                "workflow": "workflows/profile-final.json",
                "purpose": "final",
                "stages": 2,
                "output_scale": 1.0,
                "supports_per_pass_loras": True,
                "default": True,
            }],
        }), encoding="utf-8")
        return path

    def test_selected_final_profile_flows_into_ingredients_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = self._app_config(temp)
            with (
                patch("feverslop.composition.render_video.ComfyUIClient"),
                patch("feverslop.composition.render_video.ComfyUIIngredientsVideoRenderBackend") as backend_type,
            ):
                use_case = build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        video_pipeline="ltx_ingredients",
                        output_dir=temp / "out",
                    )
                )

        self.assertIs(use_case.backend, backend_type.return_value)
        self.assertEqual(
            Path("workflows/profile-final.json"),
            backend_type.call_args.kwargs["workflow_path"],
        )

    def test_explicit_workflow_path_overrides_selected_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = self._app_config(temp)
            explicit = temp / "explicit.json"
            with (
                patch("feverslop.composition.render_video.ComfyUIClient"),
                patch("feverslop.composition.render_video.ComfyUIIngredientsVideoRenderBackend") as backend_type,
            ):
                build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        video_pipeline="ltx_ingredients",
                        workflow_path=explicit,
                        output_dir=temp / "out",
                    )
                )

        self.assertEqual(explicit, backend_type.call_args.kwargs["workflow_path"])

    def test_named_profile_can_be_selected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = self._app_config(temp)
            with (
                patch("feverslop.composition.render_video.ComfyUIClient"),
                patch("feverslop.composition.render_video.ComfyUIIngredientsVideoRenderBackend") as backend_type,
            ):
                build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        video_pipeline="ltx_ingredients",
                        video_workflow_profile="ingredients-final",
                        output_dir=temp / "out",
                    )
                )

        self.assertEqual(
            Path("workflows/profile-final.json"),
            backend_type.call_args.kwargs["workflow_path"],
        )


if __name__ == "__main__":
    unittest.main()
