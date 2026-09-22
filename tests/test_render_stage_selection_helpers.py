"""Focused tests for the render-stage selection helpers (issue #1363)."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace

from feverslop.composition.config_loader import PipelineRunState
from feverslop.composition.stages.render_stages import (
    _selected_scene_numbers,
    _selected_workflow,
)


def _state(video_pipeline: str, *, smoke_only: bool, scenes: str) -> PipelineRunState:
    args = argparse.Namespace(
        video_pipeline=video_pipeline,
        smoke_only=smoke_only,
        scenes=scenes,
        smoke_scene=7,
    )
    context = SimpleNamespace()  # type: ignore[assignment]
    return PipelineRunState(
        args=args,
        context=context,
        app_config_path=Path("app.json"),
        storyboard_workflow=Path("storyboard.json"),
        reference_hero_workflow=Path("hero.json"),
        reference_edit_workflow=Path("edit.json"),
        msr_workflow=Path("msr.json"),
        ingredients_workflow=Path("ingredients.json"),
        relay_workflow=Path("relay.json"),
        single_prompt_workflow=Path("i2v.json"),
        facefix_workflow=Path("facefix.json"),
        plan_for_next_step=Path("plan.json"),
    )


class SelectedSceneNumbersTests(unittest.TestCase):
    def test_smoke_only_selects_single_scene(self):
        state = _state("ltx_msr", smoke_only=True, scenes="1,2")
        self.assertEqual(_selected_scene_numbers(state), {7})

    def test_explicit_scene_list_is_parsed(self):
        state = _state("ltx_msr", smoke_only=False, scenes="1, 3-5")
        self.assertEqual(_selected_scene_numbers(state), {1, 3, 4, 5})

    def test_no_selection_is_none(self):
        state = _state("ltx_ingredients", smoke_only=False, scenes="")
        self.assertIsNone(_selected_scene_numbers(state))


class SelectedWorkflowTests(unittest.TestCase):
    def test_msr_pipeline_selects_msr_workflow(self):
        state = _state("ltx_msr", smoke_only=False, scenes="")
        self.assertEqual(_selected_workflow(state), Path("msr.json"))

    def test_other_pipeline_selects_ingredients_workflow(self):
        state = _state("ltx_ingredients", smoke_only=False, scenes="")
        self.assertEqual(_selected_workflow(state), Path("ingredients.json"))


if __name__ == "__main__":
    unittest.main()
