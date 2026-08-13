from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from feverslop.adapters.video_postprocessor import VideoPostProcessor


class PostprocessorFrameExtractor:
    def __init__(
        self,
        postprocessor: VideoPostProcessor,
        *,
        project_dir: Path,
        selected_rerender: bool = False,
    ):
        self.postprocessor = postprocessor
        self.project_dir = Path(project_dir).resolve()
        self.selected_rerender = selected_rerender

    def extract_last_frame(
        self,
        video_path: Path,
        output_path: Path,
    ) -> Path:
        video_path = self._contained(video_path, "previous clip")
        output_path = self._requested_output(output_path)
        if not self._regular_file(video_path):
            detail = " for selected re-render" if self.selected_rerender else ""
            raise ValueError(
                f"Cannot use last-frame continuity{detail}; "
                f"missing previous movie scene clip: {video_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}-",
            suffix=output_path.suffix or ".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            self._require_regular_input(video_path)
            self._require_secure_temp(temporary)
            # The project tree is trusted against concurrent hostile mutation.
            # These checks protect accidental/static escapes; path-based APIs
            # cannot eliminate replacement by a privileged concurrent process.
            cached = video_path.with_name("lastframe.png")
            if self._regular_file(cached):
                shutil.copyfile(cached, temporary)
                extracted = temporary
            else:
                extracted = self.postprocessor.extract_last_frame(
                    video_path,
                    temporary,
                )
            self._require_regular_input(video_path)
            extracted = self._contained(extracted, "extracted frame")
            if extracted != temporary.resolve():
                raise ValueError(
                    "Continuity handoff extractor returned an unexpected "
                    f"output frame: {extracted}"
                )
            self._require_secure_temp(extracted)
            self._requested_output(output_path)
            os.replace(extracted, output_path)
            final = self._contained(output_path, "output frame")
            if not self._regular_file(final):
                raise ValueError(
                    f"Continuity handoff did not produce output frame: {final}"
                )
            return final
        finally:
            temporary.unlink(missing_ok=True)

    def _contained(self, value: str | Path, label: str) -> Path:
        path = Path(value)
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (self.project_dir / path).resolve()
        )
        if not resolved.is_relative_to(self.project_dir):
            raise ValueError(
                f"Continuity handoff {label} is outside project: {resolved}"
            )
        return resolved

    def _requested_output(self, value: str | Path) -> Path:
        path = Path(value)
        absolute = path if path.is_absolute() else self.project_dir / path
        parent = absolute.parent.resolve()
        if not parent.is_relative_to(self.project_dir):
            raise ValueError(
                f"Continuity handoff output frame is outside project: {absolute}"
            )
        if absolute.is_symlink():
            raise ValueError(
                f"Continuity handoff output frame must not be a symlink: {absolute}"
            )
        return parent / absolute.name

    @staticmethod
    def _regular_file(path: Path) -> bool:
        return not path.is_symlink() and path.is_file()

    def _require_regular_input(self, path: Path) -> None:
        resolved = self._contained(path, "previous clip")
        if resolved != path or not self._regular_file(path):
            raise ValueError(
                f"Continuity handoff previous clip is not a regular file: {path}"
            )

    def _require_secure_temp(self, path: Path) -> None:
        resolved = self._contained(path, "extracted frame")
        if resolved != path or not self._regular_file(path):
            raise ValueError(
                f"Continuity handoff did not produce output frame: {path}"
            )
