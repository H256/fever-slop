import json
import tempfile
import unittest
from pathlib import Path


class TestMovieH3Preparation(unittest.TestCase):
    def test_minimax_stage_titles_include_prompt_preparation(self):
        from feverslop.composition.movie_pipeline import _movie_stage_titles

        titles = _movie_stage_titles({"movie_video_workflow": "minimax-h3-r2v"})
        self.assertIn("Movie MiniMax H3 prompts", titles)
        self.assertIn("Movie MiniMax H3 render", titles)

    def test_preparation_reuses_current_plan_without_building_dspy(self):
        from feverslop.adapters.movie_minimax_visual import (
            ComfyUIMiniMaxMovieVisualAdapter,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            source = project / "movie" / "render_plan.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps([{"scene": 1, "references": {"actor_msr_paths": ["actor.png"]}}]), encoding="utf-8")
            adapter = ComfyUIMiniMaxMovieVisualAdapter(
                project_dir=project,
                workflow_path="workflow.json",
                video_pipeline="minimax-h3-r2v",
            )
            prepared = adapter.output_dir / "render_plan_h3.json"
            prepared.parent.mkdir(parents=True)
            prepared.write_text(json.dumps([{"scene": 1, "h3": {"prompt": "subject_definitions:\nsummary:\nretention_analysis:\ndetailed_description:\noverall_soundscape:\nnon_diegetic_music:"}}]), encoding="utf-8")

            class FailingBuilder:
                def build_h3_prompt(self, **_kwargs):
                    raise AssertionError("DSPy should not run for a current prepared plan")

            result = adapter.prepare_render_plan(source, project, prompt_builder=FailingBuilder())

            self.assertEqual(prepared, result)
            self.assertIn("subject_definitions", result.read_text(encoding="utf-8"))

    def test_preparation_writes_h3_scene_list_when_missing(self):
        from feverslop.adapters.movie_minimax_visual import (
            ComfyUIMiniMaxMovieVisualAdapter,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            source = project / "movie" / "render_plan.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps([{"scene": 1, "description": "A witch watches.", "references": {"actor_msr_paths": ["actor.png"]}}]), encoding="utf-8")
            adapter = ComfyUIMiniMaxMovieVisualAdapter(
                project_dir=project,
                workflow_path="workflow.json",
                video_pipeline="minimax-h3-t2v",
            )

            class Builder:
                def build_h3_prompt(self, **_kwargs):
                    return {"prompt": "six sections"}

            result = adapter.prepare_render_plan(source, project, prompt_builder=Builder())
            payload = json.loads(result.read_text(encoding="utf-8"))

            self.assertIsInstance(payload, list)
            self.assertEqual("six sections", payload[0]["h3"]["prompt"])

    def test_preparation_reports_each_completed_scene(self):
        from feverslop.adapters.movie_minimax_visual import (
            ComfyUIMiniMaxMovieVisualAdapter,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            source = project / "movie" / "render_plan.json"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps([
                    {"scene": 1, "description": "A witch watches."},
                    {"scene": 2, "description": "The witch walks."},
                ]),
                encoding="utf-8",
            )
            adapter = ComfyUIMiniMaxMovieVisualAdapter(
                project_dir=project,
                workflow_path="workflow.json",
                video_pipeline="minimax-h3-t2v",
            )

            class Builder:
                def build_h3_prompt(self, **_kwargs):
                    return {"prompt": "six sections"}

            progress = []
            adapter.prepare_render_plan(
                source,
                project,
                prompt_builder=Builder(),
                on_scene_prepared=lambda completed, total, scene: progress.append((completed, total, scene)),
            )

            self.assertEqual([(1, 2, 1), (2, 2, 2)], progress)

    def test_preparation_reports_scene_before_building_prompt(self):
        from feverslop.adapters.movie_minimax_visual import (
            ComfyUIMiniMaxMovieVisualAdapter,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            source = project / "movie" / "render_plan.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps([{"scene": 1, "description": "A witch watches."}]), encoding="utf-8")
            adapter = ComfyUIMiniMaxMovieVisualAdapter(
                project_dir=project,
                workflow_path="workflow.json",
                video_pipeline="minimax-h3-t2v",
            )
            events = []

            class Builder:
                def build_h3_prompt(self, **_kwargs):
                    events.append("built")
                    return {"prompt": "six sections"}

            adapter.prepare_render_plan(
                source,
                project,
                prompt_builder=Builder(),
                on_scene_started=lambda index, total, scene: events.append((index, total, scene)),
            )

            self.assertEqual([(1, 1, 1), "built"], events)

    def test_preparation_rebuilds_legacy_hybrid_prompt(self):
        from feverslop.adapters.movie_minimax_visual import (
            ComfyUIMiniMaxMovieVisualAdapter,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            source = project / "movie" / "render_plan.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps([{"scene": 1, "description": "A witch watches.", "references": {"actor_msr_paths": ["actor.png"]}}]), encoding="utf-8")
            adapter = ComfyUIMiniMaxMovieVisualAdapter(
                project_dir=project,
                workflow_path="workflow.json",
                video_pipeline="minimax-h3-t2v",
            )
            prepared = adapter.output_dir / "render_plan_h3.json"
            prepared.parent.mkdir(parents=True)
            prepared.write_text(
                json.dumps([{"scene": 1, "h3": {"prompt": "Continuity anchors:\nReference files:"}}]),
                encoding="utf-8",
            )

            class Builder:
                def build_h3_prompt(self, **_kwargs):
                    return {"prompt": "subject_definitions:\n<Subject 1> Witch\n\nsummary: A witch watches.\n\nretention_analysis: kept\n\ndetailed_description: [Shot 1] A witch watches.\n\noverall_soundscape: wind\n\nnon_diegetic_music: N/A"}

            result = adapter.prepare_render_plan(source, project, prompt_builder=Builder())
            prompt = json.loads(result.read_text(encoding="utf-8"))[0]["h3"]["prompt"]

            self.assertIn("subject_definitions:", prompt)
            self.assertNotIn("Reference files:", prompt)

    def test_preparation_rebuilds_malformed_cached_scene(self):
        from feverslop.adapters.movie_minimax_visual import (
            ComfyUIMiniMaxMovieVisualAdapter,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            source = project / "movie" / "render_plan.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps([{"scene": 1, "description": "A witch watches."}]), encoding="utf-8")
            adapter = ComfyUIMiniMaxMovieVisualAdapter(
                project_dir=project,
                workflow_path="workflow.json",
                video_pipeline="minimax-h3-t2v",
            )
            prepared = adapter.output_dir / "render_plan_h3.json"
            prepared.parent.mkdir(parents=True)
            prepared.write_text(json.dumps([None]), encoding="utf-8")

            class Builder:
                def build_h3_prompt(self, **_kwargs):
                    return {"prompt": "six sections"}

            result = adapter.prepare_render_plan(source, project, prompt_builder=Builder())
            payload = json.loads(result.read_text(encoding="utf-8"))

            self.assertEqual("six sections", payload[0]["h3"]["prompt"])

    def test_preparation_rebuilds_cached_scene_with_malformed_h3(self):
        from feverslop.adapters.movie_minimax_visual import (
            ComfyUIMiniMaxMovieVisualAdapter,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            source = project / "movie" / "render_plan.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps([{"scene": 1, "description": "A witch watches."}]), encoding="utf-8")
            adapter = ComfyUIMiniMaxMovieVisualAdapter(
                project_dir=project,
                workflow_path="workflow.json",
                video_pipeline="minimax-h3-t2v",
            )
            prepared = adapter.output_dir / "render_plan_h3.json"
            prepared.parent.mkdir(parents=True)
            prepared.write_text(json.dumps([{"scene": 1, "h3": "legacy"}]), encoding="utf-8")

            class Builder:
                def build_h3_prompt(self, **_kwargs):
                    return {"prompt": "six sections"}

            result = adapter.prepare_render_plan(source, project, prompt_builder=Builder())
            payload = json.loads(result.read_text(encoding="utf-8"))

            self.assertEqual("six sections", payload[0]["h3"]["prompt"])
            source.write_text(json.dumps([{"scene": 1, "description": "A witch watches.", "references": {"actor_msr_paths": ["actor.png"]}}]), encoding="utf-8")
            adapter = ComfyUIMiniMaxMovieVisualAdapter(
                project_dir=project,
                workflow_path="workflow.json",
                video_pipeline="minimax-h3-r2v",
            )
            prepared = adapter.output_dir / "render_plan_h3.json"
            prepared.parent.mkdir(parents=True)
            prepared.write_text(
                json.dumps([{"scene": 1, "h3": {"prompt": "Continuity anchors:\nReference files:"}}]),
                encoding="utf-8",
            )

            class Builder:
                def build_h3_prompt(self, **_kwargs):
                    return {"prompt": "subject_definitions:\n<Subject 1> Witch\n\nsummary: A witch watches.\n\nretention_analysis: kept\n\ndetailed_description: [Shot 1] A witch watches.\n\noverall_soundscape: wind\n\nnon_diegetic_music: N/A"}

            result = adapter.prepare_render_plan(source, project, prompt_builder=Builder())
            prompt = json.loads(result.read_text(encoding="utf-8"))[0]["h3"]["prompt"]

            self.assertIn("subject_definitions:", prompt)
            self.assertNotIn("Reference files:", prompt)


if __name__ == "__main__":
    unittest.main()
