"""Argparse CLI for global asset management.

Run as ``python -m feverslop.tools.global_library_cli`` or embed the parser in
the application's top-level CLI. Pruning of unreferenced assets is
intentionally not implemented and therefore not exposed as a subcommand.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from feverslop.adapters.global_library import GlobalLibraryAdapter
from feverslop.application.global_character_creator import (
    AssetIdea,
    GuidedAssetGenerator,
)
from feverslop.application.sequence_to_sheet import generate_sequence_to_sheet
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
    commands.add_parser("import-from-project").add_argument("--manifest", type=Path, required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--kind", type=_kind)
    generate.add_argument("--id")
    generate.add_argument("--name")
    generate.add_argument("--idea")
    generate.add_argument("--input", type=Path)
    generate.add_argument("--workflow", default="character-sheet-v1")
    generate.add_argument("--interactive", action="store_true")
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--hero-image", type=Path)
    sheet = commands.add_parser("sequence-to-sheet")
    sheet.add_argument("--kind", type=_kind, required=True)
    sheet.add_argument("--id", required=True)
    sheet.add_argument("--look-id", default="default")
    sheet.add_argument("--sequence", type=Path, required=True)
    sheet.add_argument("--anchor-image", type=Path)
    sheet.add_argument(
        "--view-count",
        type=int,
        default=None,
        help="number of views (default: 6 for characters, 5 for locations)",
    )
    sheet.add_argument("--backend", choices=("offline", "minimax"), default="offline")
    sheet.add_argument("--profile", default="sequence_to_sheet_v1")
    sheet.add_argument("--json", action="store_true")
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
            print(
                json.dumps(asset.to_dict(), ensure_ascii=False, indent=2)
                if args.json
                else f"{asset.kind.value}/{asset.id}: {asset.name}"
            )
        elif args.command == "create":
            asset = GlobalAsset.from_dict(json.loads(args.manifest.read_text(encoding="utf-8"))) if args.manifest else GlobalAsset(args.id, args.kind, args.name, args.description)
            library.create(asset)
            print(f"created {asset.kind.value}/{asset.id}")
        elif args.command == "create-look":
            asset = library.get(args.kind, args.id)
            asset_dir = library.root / asset.kind.value / asset.id
            for field_name, raw_path in (("hero_image", args.hero_image), ("sheet_image", args.sheet_image)):
                if raw_path and not (asset_dir / raw_path).is_file():
                    raise FileNotFoundError(f"{field_name} media not found in asset directory: {raw_path}")
            look = AssetLook(args.look_id, args.name, hero_image=args.hero_image, sheet_image=args.sheet_image)
            library.update(GlobalAsset(asset.id, asset.kind, asset.name, asset.description, asset.looks + (look,), asset.revision + 1, asset.schema_version, asset.metadata), expected_revision=asset.revision)
            print(f"created look {args.kind.value}/{args.id}/{args.look_id}")
        elif args.command == "validate":
            assets = library.list(args.kind)
            dangling: list[str] = []
            for asset in assets:
                asset_dir = Path(library.root) / asset.kind.value / asset.id
                for look in asset.looks:
                    for field_name, relative in (
                        ("hero_image", look.hero_image),
                        ("sheet_image", look.sheet_image),
                        ("contact_sheet_image", look.contact_sheet_image),
                        ("anchor_image", look.anchor_image),
                        ("sequence_video", look.sequence_video),
                        *[(f"reference:{i}", ref) for i, ref in enumerate(look.references)],
                        *[(f"selected_frame:{i}", ref) for i, ref in enumerate(look.selected_frames)],
                    ):
                        if relative and not (asset_dir / relative).is_file():
                            dangling.append(f"{asset.kind.value}/{asset.id}: look {look.id}: {field_name} -> {relative}")
            if dangling:
                print(f"validation failed: {len(dangling)} dangling media reference(s)", file=sys.stderr)
                for line in dangling:
                    print(f"  {line}", file=sys.stderr)
                return 2
            print(f"validated {len(assets)} asset manifest(s)")
        elif args.command == "delete":
            library.delete(args.kind, args.id)
            print(f"deleted {args.kind.value}/{args.id}")
        elif args.command == "refresh":
            snapshot = args.snapshot.resolve()
            # base = parents[3] must have >= 2 components; shallower snapshots would materialize
            # into the filesystem root or a top-level dir, which is never a valid base.
            if len(snapshot.parents) < 5:
                raise ValueError(
                    f"snapshot {snapshot} is shallower than the canonical layout "
                    "<base>/global_assets/<kind>/<id>/<look>; refusing to materialize",
                )
            payload = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
            target = library.materialize(payload["kind"], payload["asset_id"], payload["look_id"], snapshot.parents[3])
            if target != snapshot:
                if not (snapshot / "manifest.json").is_file():
                    raise ValueError(f"snapshot {snapshot} changed before refresh; refusing to replace it")
                shutil.rmtree(snapshot)
                shutil.move(target, snapshot)
            print(f"refreshed {snapshot}")
        elif args.command == "import-from-project":
            asset = GlobalAsset.from_dict(json.loads(args.manifest.read_text(encoding="utf-8")))
            library.create(asset)
            print(f"imported {asset.kind.value}/{asset.id}")
        elif args.command == "generate":
            raw = {}
            if args.input:
                raw_text = sys.stdin.read() if str(args.input) == "-" else args.input.read_text(encoding="utf-8")
                raw = json.loads(raw_text)
            if args.interactive:
                raw.setdefault("kind", input("Asset kind [character/location/style/prop]: ").strip())
                raw.setdefault("asset_id", input("Stable asset ID: ").strip())
                raw.setdefault("name", input("Display name: ").strip())
                raw.setdefault("visual_concept", input("Visual concept: ").strip())
            if args.kind:
                raw["kind"] = args.kind.value
            for key in ("id", "name", "idea"):
                value = getattr(args, key)
                if value:
                    raw[{"id": "asset_id", "name": "name", "idea": "visual_concept"}[key]] = value
            idea = AssetIdea(**{key: raw[key] for key in ("kind", "asset_id", "name", "visual_concept")})
            def runner(**kwargs):
                if args.hero_image is None:
                    return {}
                target = Path(kwargs["run_dir"]) / args.hero_image.name
                shutil.copy2(args.hero_image, target)
                return {"hero_image": target.name}
            generator = GuidedAssetGenerator(library, profiles={args.workflow: runner})
            preview = generator.preview(idea, profile_id=args.workflow)
            if args.dry_run:
                print(json.dumps(preview, ensure_ascii=False, indent=2))
            else:
                result = generator.generate(idea, profile_id=args.workflow)
                print(json.dumps({"run_id": result.run_id, "asset_id": result.asset.id, "status": result.status}, indent=2))
        elif args.command == "sequence-to-sheet":
            if args.view_count is not None and args.view_count < 1:
                raise ValueError("view-count must be positive")
            result = generate_sequence_to_sheet(
                library,
                kind=args.kind,
                asset_id=args.id,
                look_id=args.look_id,
                sequence_video=args.sequence,
                anchor_image=args.anchor_image,
                view_count=args.view_count,
                backend=args.backend,
                profile=args.profile,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else f"published {args.kind.value}/{args.id}/{args.look_id} revision {result['revision']}")
        return 0
    except (FileNotFoundError, ValueError, FileExistsError, KeyError, OSError) as exc:
        return _error(f"{exc}; create or import the asset and check the configured library path")


if __name__ == "__main__":
    raise SystemExit(main())
