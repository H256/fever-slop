from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile


ARCHIVES_DIR_NAME = "archives"
STORYBOARD_RELATIVE_PARTS = ("output", "render", "storyboard")


@dataclass(frozen=True)
class ArchiveMember:
    source: Path
    arcname: str
    size: int


def resolve_project_dir(*, project: str | Path | None, project_dir: str | Path | None) -> Path:
    if bool(project) == bool(project_dir):
        raise ValueError("Pass exactly one of --project or --project-dir.")

    if project:
        config_path = Path(project).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        return config_path.parent

    directory = Path(project_dir).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    return directory


def _relative_parts(path: Path, project_dir: Path) -> tuple[str, ...]:
    return path.relative_to(project_dir).parts


def _is_inside_archives_dir(path: Path, project_dir: Path) -> bool:
    relative_parts = _relative_parts(path, project_dir)
    return bool(relative_parts) and relative_parts[0] == ARCHIVES_DIR_NAME


def _is_inside_storyboard_dir(path: Path, project_dir: Path) -> bool:
    return _relative_parts(path, project_dir)[:3] == STORYBOARD_RELATIVE_PARTS


def sanitize_file_stem(value: str | None, fallback: str) -> str:
    raw = str(value or "").strip() or fallback
    safe = "".join(char if char.isascii() and (char.isalnum() or char in "._-") else "_" for char in raw)
    safe = safe.strip("._-")
    return safe or fallback


def _is_final_muxed_video(path: Path, project_dir: Path, project_name: str | None) -> bool:
    relative_parts = _relative_parts(path, project_dir)
    if len(relative_parts) < 4:
        return False
    if relative_parts[:2] != ("output", "render"):
        return False
    if path.suffix.lower() != ".mp4":
        return False

    final_names = {"final_concat.mp4"}
    if project_name:
        final_names.add(f"{sanitize_file_stem(project_name, project_dir.name)}.mp4")
    return path.name in final_names


def is_protected_project_file(
    path: Path,
    project_dir: Path,
    *,
    project_config: Path | None,
    project_name: str | None,
) -> bool:
    resolved_path = path.resolve()
    if project_config and resolved_path == project_config.resolve():
        return True
    return (
        _is_inside_archives_dir(path, project_dir)
        or _is_inside_storyboard_dir(path, project_dir)
        or _is_final_muxed_video(path, project_dir, project_name)
    )


def collect_archive_members(
    project_dir: str | Path,
    *,
    project_config: str | Path | None = None,
    project_name: str | None = None,
) -> list[ArchiveMember]:
    project_dir = Path(project_dir).resolve()
    config_path = Path(project_config).resolve() if project_config else None
    members = []

    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        if is_protected_project_file(
            path,
            project_dir,
            project_config=config_path,
            project_name=project_name,
        ):
            continue
        arcname = path.relative_to(project_dir).as_posix()
        members.append(ArchiveMember(source=path, arcname=arcname, size=path.stat().st_size))

    return members


def build_archive_manifest(
    project_dir: str | Path,
    members: Iterable[ArchiveMember],
    *,
    created_at: str,
) -> dict:
    project_dir = Path(project_dir).resolve()
    member_list = list(members)
    return {
        "project_name": project_dir.name,
        "project_dir": str(project_dir),
        "created_at": created_at,
        "file_count": len(member_list),
        "total_bytes": sum(member.size for member in member_list),
        "files": [
            {
                "path": member.arcname,
                "bytes": member.size,
            }
            for member in member_list
        ],
    }


def default_archive_path(project_dir: str | Path, *, created_at: str | None = None) -> Path:
    project_dir = Path(project_dir).resolve()
    timestamp = created_at or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_timestamp = timestamp.replace(":", "").replace("-", "").replace("T", "_")
    return project_dir / ARCHIVES_DIR_NAME / f"{project_dir.name}_assets_{safe_timestamp}.zip"


def resolve_available_zip_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    if not path.exists():
        return path

    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find available archive path for {path}")


def create_project_archive(
    *,
    project_dir: str | Path,
    project_config: str | Path | None = None,
    project_name: str | None = None,
    output_zip: str | Path | None = None,
    created_at: str | None = None,
) -> Path:
    project_dir = Path(project_dir).resolve()
    created_at = created_at or datetime.now().replace(microsecond=0).isoformat()
    output_zip = Path(output_zip).resolve() if output_zip else default_archive_path(project_dir)
    output_zip = resolve_available_zip_path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    members = collect_archive_members(
        project_dir,
        project_config=project_config,
        project_name=project_name,
    )
    manifest = build_archive_manifest(project_dir, members, created_at=created_at)

    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("archive_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for member in members:
            archive.write(member.source, member.arcname)

    return output_zip


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a ZIP archive for a FeverSlop project directory.")
    parser.add_argument("--project", default=None, help="Path to project config.json.")
    parser.add_argument("--project-dir", default=None, help="Path to a project directory.")
    parser.add_argument("--output", default=None, help="Output ZIP path. Defaults to project archives directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected files without writing a ZIP.")
    return parser


def read_project_name(config_path: str | Path | None, fallback: str) -> str:
    if not config_path:
        return fallback
    path = Path(config_path)
    if not path.exists():
        return fallback
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return str(data.get("project_name") or fallback)


def resolve_project_config_path(*, project: str | Path | None, project_dir: Path) -> Path | None:
    if project:
        return Path(project).resolve()
    candidate = project_dir / "config.json"
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    project_dir = resolve_project_dir(project=args.project, project_dir=args.project_dir)
    project_config = resolve_project_config_path(project=args.project, project_dir=project_dir)
    project_name = read_project_name(project_config, fallback=project_dir.name)
    members = collect_archive_members(
        project_dir,
        project_config=project_config,
        project_name=project_name,
    )

    if args.dry_run:
        print(f"Project: {project_dir}")
        print(f"Files: {len(members)}")
        print(f"Bytes: {sum(member.size for member in members)}")
        for member in members:
            print(member.arcname)
        return 0

    output_zip = create_project_archive(
        project_dir=project_dir,
        project_config=project_config,
        project_name=project_name,
        output_zip=args.output,
    )
    print(f"Archive: {output_zip}")
    print(f"Files: {len(members)}")
    print(f"Bytes: {sum(member.size for member in members)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
