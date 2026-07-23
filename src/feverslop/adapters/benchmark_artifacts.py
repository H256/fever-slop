from __future__ import annotations

from pathlib import Path
import re
import shutil
from tempfile import NamedTemporaryFile


_SAFE_CASE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class LocalBenchmarkArtifactStore:
    def __init__(self, output_directory: Path) -> None:
        self._output_directory = output_directory

    def capture(self, case_name: str, rendered_output: Path) -> Path:
        if not isinstance(case_name, str) or not _SAFE_CASE_NAME.fullmatch(case_name):
            raise ValueError("benchmark case name is not safe for an artifact filename")
        if not isinstance(rendered_output, Path) or not rendered_output.is_file():
            raise ValueError("rendered benchmark output must be an existing file")

        self._output_directory.mkdir(parents=True, exist_ok=True)
        target = self._output_directory / f"{case_name}{rendered_output.suffix}"
        if target.exists():
            raise FileExistsError(f"benchmark evidence already exists: {target}")

        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=self._output_directory,
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
            shutil.copy2(rendered_output, temporary_path)
            target.hardlink_to(temporary_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return target
