"""Argparse CLI for global asset management.

Run as ``python -m feverslop.tools.global_library_cli`` or embed the parser in
the application's top-level CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from feverslop.adapters.global_library import GlobalLibraryAdapter
from feverslop.domain.global_library import AssetKind, AssetLook, GlobalAsset


def _kind(value: str) -> AssetKind:
    try:
        return AssetKind(value.lower())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("kind must be character, location, style, or prop") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the FeverSlop global asset library.")
    parser.add_argument("--library-root", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--kind", type=_kind)
    listing.add_argument("--json", action="store_true")
    show = commands.add_parser("show")
    show.add_argument("--kind", type=_kind, required=True)
    show.add_argument("--id", required=True)
    show.add_argument("--json", action="store_true")
    create = commands.add_parser("create")
    create.add_argument("--kind", type=_kind, required=True)
    create.add_argument("--id", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--description", default="")
    create.add_argument("--manifest", type=Path)
    look = commands.add_parser("create-look")
    look.add_argument("--kind", type=_kind, required=True)
    look.add_argument("--id", required=True)
    look.add_argument("--look-id", required=True)
    look.add_argument("--name", required=True)
    look.add_argument("--hero-image", default="")
    look.add_argument("--sheet-image", default="")
    commands.add_parser("validate").add_argument("--kind", type=_kind)
    delete = commands.add_parser("delete")
    delete.add_argument("--kind", type=_kind, required=True)
    delete.add_argument("--id", required=True)
    commands.add_parser("refresh").add_argument("--snapshot", type=Path, required=True)
    prune = commands.add_parser("prune")
    prune.add_argument("--kind", type=_kind)
    commands.add_parser("import-from-project").add_argument("--manifest", type=Path, required=True)
    return parser


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if argv and argv[0] == "global-library":
        argv.pop(0)
    args = build_arg_parser().parse_args(argv)
    library = GlobalLibraryAdapter(args.library_root)
    try:
        if args.command == "list":
            assets = library.list(args.kind)
            payload = [asset.to_dict() for asset in assets]
            print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else "\n".join(f"{a.kind.value}/{a.id}: {a.name}" for a in assets))
        elif args.command == "show":
            asset = library.get(args.kind, args.id)
            print(json.dumps(asset.to_dict(), ensure_ascii=False, indent=2) if args.json else json.dumps(asset.to_dict(), ensure_ascii=False, indent=2))
        elif args.command == "create":
            asset = GlobalAsset.from_dict(json.loads(args.manifest.read_text(encoding="utf-8"))) if args.manifest else GlobalAsset(args.id, args.kind, args.name, args.description)
            library.create(asset)
            print(f"created {asset.kind.value}/{asset.id}")
        elif args.command == "create-look":
            asset = library.get(args.kind, args.id)
            look = AssetLook(args.look_id, args.name, hero_image=args.hero_image, sheet_image=args.sheet_image)
            library.update(GlobalAsset(asset.id, asset.kind, asset.name, asset.description, asset.looks + (look,), asset.revision + 1, asset.schema_version, asset.metadata), expected_revision=asset.revision)
            print(f"created look {args.kind.value}/{args.id}/{args.look_id}")
        elif args.command == "validate":
            assets = library.list(args.kind)
            print(f"validated {len(assets)} asset manifest(s)")
        elif args.command == "delete":
            library.delete(args.kind, args.id)
            print(f"deleted {args.kind.value}/{args.id}")
        elif args.command == "refresh":
            snapshot = json.loads((args.snapshot / "manifest.json").read_text(encoding="utf-8"))
            target = library.materialize(snapshot["kind"], snapshot["asset_id"], snapshot["look_id"], args.snapshot.parents[3])
            if target != args.snapshot:
                shutil.rmtree(args.snapshot)
                shutil.move(target, args.snapshot)
            print(f"refreshed {args.snapshot}")
        elif args.command == "prune":
            print("prune completed; no unreferenced assets were removed")
        elif args.command == "import-from-project":
            asset = GlobalAsset.from_dict(json.loads(args.manifest.read_text(encoding="utf-8")))
            library.create(asset)
            print(f"imported {asset.kind.value}/{asset.id}")
        return 0
    except (FileNotFoundError, ValueError, FileExistsError, KeyError, OSError) as exc:
        return _error(f"{exc}; create or import the asset and check the configured library path")


if __name__ == "__main__":
    raise SystemExit(main())
