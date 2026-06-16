from __future__ import annotations

from pathlib import Path
import json

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from comfyui_client import ComfyUIClient
from workflow_patcher import WorkflowPatcher


class StoryboardRenderer:
    """
    Renders Z-Image startframes from a render_plan.json.

    This is intentionally independent from the concept/prompt pipeline.
    You can generate render_plan.json first, review/edit it, then run this renderer later.
    """

    def __init__(
        self,
        client: ComfyUIClient,
        zimage_workflow_path: str | Path,
        output_dir: str | Path,

        positive_prompt_node_title: str = "#PROMPT_POSITIVE",
        negative_prompt_node_title: str = "#PROMPT_NEGATIVE",
        save_image_node_title: str = "#SAVE_IMAGE",
        character_lora_node_title: str = "#CHARACTER_LORA",

        character_lora_strength: float = 1.0,
        negative_prompt: str = "",

        seed_node_title: str | None = None,
        seed_input_name: str = "seed",
        filename_prefix_input_name: str = "filename_prefix",
    ):
        self.client = client
        self.zimage_workflow_path = Path(zimage_workflow_path)
        self.output_dir = Path(output_dir)

        self.positive_prompt_node_title = positive_prompt_node_title
        self.negative_prompt_node_title = negative_prompt_node_title
        self.save_image_node_title = save_image_node_title
        self.character_lora_node_title = character_lora_node_title

        self.character_lora_strength = character_lora_strength
        self.negative_prompt = negative_prompt

        self.seed_node_title = seed_node_title
        self.seed_input_name = seed_input_name
        self.filename_prefix_input_name = filename_prefix_input_name

    def load_workflow(self) -> dict:
        return json.loads(self.zimage_workflow_path.read_text(encoding="utf-8"))

    def render_storyboard(
        self,
        render_plan_path: str | Path,
        limit: int | None = None,
        scene_numbers: set[int] | None = None,
        skip_existing: bool = True,
    ) -> list[Path]:
        render_plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))

        if scene_numbers is not None:
            render_plan = [
                scene
                for scene in render_plan
                if int(scene["scene"]) in scene_numbers
            ]

        if limit is not None:
            render_plan = render_plan[:limit]

        self.output_dir.mkdir(parents=True, exist_ok=True)

        rendered_files = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "Rendering storyboard",
                total=len(render_plan),
            )
            for scene in render_plan:
                scene_number = int(scene["scene"])
                output_path = self.output_dir / f"scene_{scene_number:04}.png"

                if skip_existing and output_path.exists():
                    rendered_files.append(output_path)
                    continue

                image_path = self.render_scene_startframe(scene)
                rendered_files.append(image_path)
                progress.advance(task)

        return rendered_files

    def render_scene_startframe(self, scene: dict) -> Path:
        workflow = self.load_workflow()
        patcher = WorkflowPatcher(workflow)

        scene_number = int(scene["scene"])
        prompt = scene["z_image"]["prompt"]

        patcher.set_input_by_title(
            self.positive_prompt_node_title,
            "text",
            prompt,
        )

        if self.negative_prompt_node_title:
            patcher.set_input_by_title(
                self.negative_prompt_node_title,
                "text",
                self.negative_prompt,
            )

        if self.character_lora_node_title:
            patcher.patch_lora_strength_by_title(
                self.character_lora_node_title,
                self.character_lora_strength,
            )

        if self.seed_node_title:
            patcher.set_input_by_title(
                self.seed_node_title,
                self.seed_input_name,
                scene_number,
            )

        if self.save_image_node_title:
            patcher.set_input_by_title(
                self.save_image_node_title,
                self.filename_prefix_input_name,
                f"storyboard/scene_{scene_number:04}",
            )

        prompt_id = self.client.queue_prompt(patcher.get())
        history = self.client.wait_for_completion(prompt_id)

        images = self.client.extract_output_images(history)

        if not images:
            raise RuntimeError(f"No image output for scene {scene_number}")

        first = images[0]
        output_path = self.output_dir / f"scene_{scene_number:04}.png"

        return self.client.download_view_file(
            filename=first["filename"],
            subfolder=first.get("subfolder", ""),
            file_type=first.get("type", "output"),
            output_path=output_path,
        )
