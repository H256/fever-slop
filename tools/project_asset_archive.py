from __future__ import annotations

from feverslop.tools.project_asset_archive import (
    ArchiveMember,
    build_archive_manifest,
    build_arg_parser,
    collect_archive_members,
    create_project_archive,
    default_archive_path,
    is_protected_project_file,
    main,
    read_project_name,
    resolve_available_zip_path,
    resolve_project_config_path,
    resolve_project_dir,
    sanitize_file_stem,
)

__all__ = [
    "ArchiveMember",
    "build_archive_manifest",
    "build_arg_parser",
    "collect_archive_members",
    "create_project_archive",
    "default_archive_path",
    "is_protected_project_file",
    "main",
    "read_project_name",
    "resolve_available_zip_path",
    "resolve_project_config_path",
    "resolve_project_dir",
    "sanitize_file_stem",
]


if __name__ == "__main__":
    raise SystemExit(main())
