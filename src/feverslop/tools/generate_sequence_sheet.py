"""Generate a sequence reference sheet from an existing hero image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.global_library import GlobalLibraryAdapter
from feverslop.adapters.sequence_to_sheet_backend import ComfyUISequenceToSheetBackend
from feverslop.application.sequence_reference_pipeline import (
    SequenceReferencePipeline,
    SequenceReferenceRequest,
)
from feverslop.config.app_config import AppConfig
from feverslop.domain.global_library import AssetKind, GlobalAsset
from feverslop.utils.io import atomic_write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a MiniMax sequence-sheet reference from an existing hero image.",
    )
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--kind", choices=("character", "location"), required=True)
    parser.add_argument("--id", dest="asset_id", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--description-mode", choices=("explicit", "auto"), default="explicit")
    parser.add_argument("--workflow", type=Path, default=Path("workflows/sequence_to_sheet_minimax_h3_i2va_v1.json"))
    parser.add_argument("--profile", default="minimax_h3")
    parser.add_argument("--output-dir", type=Path, default=Path("references"))
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--publish", choices=("local", "project", "global", "both"), default="local")
    parser.add_argument("--library-root", type=Path, default=None)
    parser.add_argument("--look-id", default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--app-config", type=Path, default=Path("app_config.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _validate_id(asset_id: str) -> None:
    if not asset_id or Path(asset_id).name != asset_id or asset_id in {".", ".."}:
        raise ValueError("--id must be one safe path component")


def _validate_look_id(look_id: str) -> None:
    if not look_id or Path(look_id).name != look_id or look_id in {".", ".."}:
        raise ValueError("--look-id must be one safe path component")


def _resolve_description(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    if args.description and args.description.strip():
        return args.description.strip(), {"mode": "explicit"}
    if args.description_mode != "auto":
        raise ValueError("--description is required unless --description-mode auto is selected")
    app_config = AppConfig.load(str(args.app_config), required_keys=["llm"])
    from feverslop.adapters.llm_client import LocalOpenAIClient, describe_image_with_llm

    llm = LocalOpenAIClient(
        base_url=app_config.llm.base_url,
        api_key=app_config.llm.api_key,
        model=app_config.llm.model_for("vision"),
        temperature=app_config.llm.temperature,
        max_tokens=app_config.llm.max_tokens,
        request_timeout_seconds=app_config.llm.request_timeout_seconds,
    )
    description = describe_image_with_llm(llm, args.source_image)
    if not description.strip():
        raise ValueError("vision description was empty")
    return description.strip(), {"mode": "auto", "model": llm.model}


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    _validate_id(args.asset_id)
    _validate_look_id(args.look_id)
    if args.publish in {"project", "both"} and args.project is None:
        raise ValueError("--project is required for --publish project/both")
    if args.publish in {"global", "both"}:
        library = GlobalLibraryAdapter(args.library_root)
        try:
            asset = library.get(AssetKind(args.kind), args.asset_id)
        except FileNotFoundError:
            asset = None
        if asset is not None:
            if any(look.id == args.look_id for look in asset.looks):
                raise ValueError(f"global look already exists: {args.kind}/{args.asset_id}/{args.look_id}")
            if asset.looks:
                raise ValueError(f"global asset already has looks; requested look is not present: {args.kind}/{args.asset_id}/{args.look_id}")
    source = args.source_image.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source image not found: {source}")
    output_dir = (args.project / "references") if args.publish in {"project", "both"} and args.project else args.output_dir
    return {
        "kind": args.kind,
        "asset_id": args.asset_id,
        "name": args.name or args.asset_id,
        "source_image": str(source),
        "workflow": str(args.workflow),
        "profile": args.profile,
        "publish": args.publish,
        "output_dir": str(output_dir.resolve()),
        "look_id": args.look_id,
        "seed": args.seed,
        "frames": args.frames,
        "artifacts": ["hero.png", "anchor.png", "sequence.mp4", "frames/", "contact-sheet.png", "sheet.png", "manifest.json"],
    }


def _publish_global(args: argparse.Namespace, result: Any, description: str, hero_path: Path) -> dict[str, Any]:
    library = GlobalLibraryAdapter(args.library_root)
    kind = AssetKind(args.kind)
    created = False
    try:
        asset = library.get(kind, args.asset_id)
    except FileNotFoundError:
        asset = library.create(GlobalAsset(args.asset_id, kind, args.name or args.asset_id, description))
        created = True
    try:
        updated = library.update_look_artifacts(
            kind,
            args.asset_id,
            args.look_id,
            hero_image=hero_path,
            anchor_image=result.anchor_path,
            sequence_video=result.sequence_path,
            selected_frames=result.selected_frame_paths,
            sheet_image=result.sheet_path,
            contact_sheet_image=result.contact_sheet_path,
            provenance={"workflow": result.workflow_profile, "seed": str(result.seed)},
            expected_revision=asset.revision,
        )
    except BaseException:
        if created:
            library.delete(kind, args.asset_id)
        raise
    return {"global_manifest": str(library.root / kind.value / args.asset_id / "manifest.json"), "revision": updated.revision}


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _plan(args)
    if args.dry_run:
        return {"status": "planned", **plan}
    description, description_provenance = _resolve_description(args)
    app_config = AppConfig.load(str(args.app_config), required_keys=["comfyui"])
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    backend = ComfyUISequenceToSheetBackend(
        client=client,
        workflow_path=args.workflow,
        backend=args.profile,
    )
    output_dir = Path(plan["output_dir"])
    result = SequenceReferencePipeline(anchor_backend=None, sequence_backend=backend).generate(
        SequenceReferenceRequest(
            kind=args.kind,
            asset_id=args.asset_id,
            name=args.name or args.asset_id,
            description=description,
            output_dir=output_dir,
            seed=args.seed,
            frames=args.frames,
            source_image=args.source_image,
        ),
    )
    final_dir = result.sheet_path.parent
    with Image.open(args.source_image) as image:
        image.convert("RGB").save(final_dir / "hero.png", format="PNG")
    manifest = {
        "status": "complete",
        "kind": args.kind,
        "asset_id": args.asset_id,
        "name": args.name or args.asset_id,
        "description": description,
        "description_provenance": description_provenance,
        "hero_path": "hero.png",
        "anchor_path": "anchor.png",
        "sequence_path": "sequence.mp4",
        "selected_frames": [f"frames/frame_{index:04}.png" for index in range(result.selected_frames)],
        "contact_sheet_path": "contact-sheet.png",
        "sheet_path": "sheet.png",
        "workflow_profile": result.workflow_profile,
        "seed": result.seed,
        "frames": result.frames,
        "planning_profile": result.planning_profile,
    }
    manifest_path = final_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    published = _publish_global(args, result, description, final_dir / "hero.png") if args.publish in {"global", "both"} else {}
    return {"status": "complete", "manifest": str(manifest_path), "asset_dir": str(final_dir), **published}


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        payload = run(args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json or args.dry_run else f"generated {payload['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
