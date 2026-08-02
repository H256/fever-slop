from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QML_ROOT = ROOT / "src" / "feverslop" / "studio" / "desktop" / "qml"


def find_qmllint() -> str:
    for candidate in ("pyside6-qmllint", "qmllint", "qmllint6"):
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise RuntimeError(
        "Could not find qmllint. Install the project dependencies with `uv sync`."
    )


def main() -> int:
    qml_files = sorted(QML_ROOT.rglob("*.qml"))
    if not qml_files:
        print(f"No QML files found under {QML_ROOT}", file=sys.stderr)
        return 1

    try:
        executable = find_qmllint()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    command = [
        executable,
        "-I",
        str(QML_ROOT),
        *(str(path) for path in qml_files),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
