from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from feverslop.adapters.reporting import NullReporter
from feverslop.domain.facefix_rendering import FaceFixConfig, FaceFixSceneRequest
from feverslop.ports.reporting import Reporter


@dataclass(frozen=True)
class FaceFixRequest:
    scenes_dir: Path
    scene_numbers: list[int] | None
    reference_images: list[Path] = ()
    skip_existing: bool = True


class FaceFixPipelineStep:
    """Runs LTXV FaceFix postprocessing on rendered scene videos.

    Reads final.mp4 from each scene dir, writes workflow_facefix.json and
    final_facefix.mp4 back into the scene dir.
    """

    def __init__(
        self,
        *,
        backend,
        config: FaceFixConfig | None = None,
        reporter: Reporter = NullReporter(),
    ):
        self.backend = backend
        self.config = config or FaceFixConfig()
        self.reporter = reporter

    def execute(self, request: FaceFixRequest) -> list[Path]:
        scenes_dir = request.scenes_dir
        scene_numbers = request.scene_numbers
        if scene_numbers is None:
            scene_numbers = sorted(
                int(d.name.split("_")[1])
                for d in scenes_dir.iterdir()
                if d.is_dir() and d.name.startswith("scene_")
            )
        results = []
        total = len(scene_numbers)

        for idx, scene_number in enumerate(scene_numbers, 1):
            scene_dir = scenes_dir / f"scene_{scene_number:04d}"
            source = scene_dir / "final.mp4"
            if not source.exists():
                self._report(
                    f"FaceFix: final.mp4 missing for scene {scene_number}, skipping",
                    level="warn",
                )
                continue

            final = scene_dir / "final_facefix.mp4"
            if request.skip_existing and final.exists():
                results.append(final)
                self._report(
                    f"FaceFix scene {scene_number}/{total}: already exists",
                    level="info",
                )
                continue

            scene_req = FaceFixSceneRequest(
                scene_number=scene_number,
                source_video=source,
                reference_images=request.reference_images,
                output_dir=scene_dir,
            )
            scene = {"scene": scene_number}
            out_path = self.backend.render_scene(scene, request=scene_req)
            results.append(out_path)

            self._report(
                f"FaceFix scene {scene_number}/{total}: {out_path}",
                level="success",
            )

        return results

    def _report(self, message: str, *, level: str) -> None:
        if self.reporter is None:
            return
        if level == "success":
            self.reporter.message(f"[green]OK[/green] {message}")
        elif level == "warn":
            self.reporter.message(f"[yellow]WARN[/yellow] {message}")
        else:
            self.reporter.message(message)
