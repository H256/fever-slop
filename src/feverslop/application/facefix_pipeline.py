from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from feverslop.domain.facefix_rendering import FaceFixConfig, FaceFixSceneRequest
from feverslop.ports.reporting import NullReporter, Reporter


@dataclass(frozen=True)
class FaceFixRequest:
    rendered_dir: Path
    output_dir: Path
    scene_numbers: list[int]
    reference_images: list[Path] = ()
    skip_existing: bool = True


class FaceFixPipelineStep:
    """Runs LTXV FaceFix postprocessing on rendered scene videos.

    This use case discovers rendered videos, builds per-scene FaceFix requests,
    and delegates to the backend for each scene.
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
        request.output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        total = len(request.scene_numbers)

        for idx, scene_number in enumerate(request.scene_numbers, 1):
            source = request.rendered_dir / f"scene_{scene_number:04}.mp4"
            if not source.exists():
                self._report(
                    f"FaceFix: source video missing for scene {scene_number}, skipping",
                    level="warn",
                )
                continue

            final = request.output_dir / f"scene_{scene_number:04}_facefix.mp4"
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
                output_dir=request.output_dir,
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
