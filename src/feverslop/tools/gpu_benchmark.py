"""Bounded GPU quality/performance benchmark runner.

Renders a declared profile/scene/seed matrix over a pluggable render backend
and records timing, peak VRAM, and exact prepared-workflow/model inventories
into a comparable manifest plus a resumable state file.

The core is backend-agnostic: CI drives it with a deterministic fake backend,
production drives it with a ComfyUI-backed renderer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from feverslop.utils.sub_step_progress import SubStepProgress

SCHEMA = "feverslop.gpu-benchmark/v1"
STATE_SCHEMA = "feverslop.gpu-benchmark-state/v1"

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class Cell:
    """One profile/scene/seed combination to render."""

    profile: str
    scene: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.profile}:{self.scene}:{self.seed}"


@dataclass
class CellResult:
    """Outcome of rendering one matrix cell."""

    profile: str
    scene: int
    seed: int
    status: str
    duration_seconds: float = 0.0
    peak_vram_mb: float | None = None
    output_path: str | None = None
    prepared_workflow_sha256: str | None = None
    model_inventory: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def key(self) -> str:
        return f"{self.profile}:{self.scene}:{self.seed}"


class RenderBackend(Protocol):
    """Renders a single cell and reports its timing/VRAM/inventory."""

    def render(self, cell: Cell) -> CellResult:
        ...


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def expand_matrix(
    profiles: Sequence[str],
    scenes: Sequence[int],
    seeds: Sequence[int],
) -> list[Cell]:
    """Expand an explicit profile/scene/seed matrix into ordered cells."""
    if not profiles:
        raise ValueError("matrix requires at least one profile")
    if not scenes:
        raise ValueError("matrix requires at least one scene")
    if not seeds:
        raise ValueError("matrix requires at least one seed")
    cells: list[Cell] = []
    for profile in profiles:
        _nonblank(profile, "profile")
    for scene in scenes:
        _positive_int(scene, "scene")
    for seed in seeds:
        _positive_int(seed, "seed")
    for profile in profiles:
        for scene in scenes:
            for seed in seeds:
                cells.append(Cell(profile=profile, scene=scene, seed=seed))
    return cells


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


class BenchmarkRunner:
    """Render a declared matrix over a backend, recording timing and VRAM.

    The runner is backend-agnostic: it emits progress, decides resume skips,
    and persists a resumable state file plus a comparable manifest. The
    backend supplies per-cell timing/VRAM/inventory.
    """

    def __init__(
        self,
        backend: RenderBackend,
        cells: Sequence[Cell],
        *,
        output_dir: str | Path,
        state_path: str | Path | None = None,
        reporter: Any = None,
    ) -> None:
        self.backend = backend
        self.cells = list(cells)
        self.output_dir = Path(output_dir)
        self.state_path = Path(state_path) if state_path else self.output_dir / "state.json"
        self.reporter = reporter

    def _report(self, method: str, text: str) -> None:
        if self.reporter is not None:
            getattr(self.reporter, method)(text)

    def _load_state(self) -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        cells = payload.get("cells", {})
        return cells if isinstance(cells, dict) else {}

    def _should_skip(self, cell: Cell, state: dict[str, dict[str, Any]]) -> bool:
        entry = state.get(cell.key)
        if not isinstance(entry, dict) or entry.get("status") != STATUS_OK:
            return False
        output = entry.get("output_path")
        if not isinstance(output, str) or not output:
            return False
        return (self.output_dir / output).is_file()

    def run(self) -> dict[str, Any]:
        self._report("step", "GPU benchmark: preparing matrix")
        self._report(
            "message",
            f"benchmark matrix: {len(self.cells)} cells "
            f"across profiles/scenes/seeds",
        )
        state = self._load_state()
        progress = SubStepProgress(self.reporter, "benchmark cells", len(self.cells))
        results: list[CellResult] = []
        for index, cell in enumerate(self.cells, start=1):
            if self._should_skip(cell, state):
                entry = state[cell.key]
                result = CellResult(
                    profile=cell.profile,
                    scene=cell.scene,
                    seed=cell.seed,
                    status=STATUS_SKIPPED,
                    duration_seconds=float(entry.get("duration_seconds", 0.0)),
                    peak_vram_mb=entry.get("peak_vram_mb"),
                    output_path=entry.get("output_path"),
                    prepared_workflow_sha256=entry.get("prepared_workflow_sha256"),
                    model_inventory=dict(entry.get("model_inventory", {})),
                )
                self._report("message", f"resume skip {cell.key} (already rendered)")
            else:
                try:
                    result = self.backend.render(cell)
                except Exception as exc:  # noqa: BLE001 - record per-cell failure
                    result = CellResult(
                        profile=cell.profile,
                        scene=cell.scene,
                        seed=cell.seed,
                        status=STATUS_FAILED,
                        error=str(exc),
                    )
                if result.status == STATUS_OK:
                    state[cell.key] = {
                        "status": STATUS_OK,
                        "duration_seconds": result.duration_seconds,
                        "peak_vram_mb": result.peak_vram_mb,
                        "output_path": result.output_path,
                        "prepared_workflow_sha256": result.prepared_workflow_sha256,
                        "model_inventory": result.model_inventory,
                    }
                    _atomic_write_json(self.state_path, {
                        "schema": STATE_SCHEMA,
                        "cells": state,
                    })
            results.append(result)
            progress.update(index, detail=cell.key)
        self._report("step", "GPU benchmark: matrix complete")
        return self._build_manifest(results)

    def _build_manifest(self, results: Sequence[CellResult]) -> dict[str, Any]:
        profiles = sorted({r.profile for r in results})
        scenes = sorted({r.scene for r in results})
        seeds = sorted({r.seed for r in results})
        return {
            "schema": SCHEMA,
            "matrix": {
                "profiles": profiles,
                "scenes": scenes,
                "seeds": seeds,
            },
            "cells": [self._result_to_dict(r) for r in results],
        }

    @staticmethod
    def _result_to_dict(result: CellResult) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "key": result.key,
            "profile": result.profile,
            "scene": result.scene,
            "seed": result.seed,
            "status": result.status,
            "duration_seconds": result.duration_seconds,
        }
        if result.peak_vram_mb is not None:
            entry["peak_vram_mb"] = result.peak_vram_mb
        if result.output_path is not None:
            entry["output_path"] = result.output_path
        if result.prepared_workflow_sha256 is not None:
            entry["prepared_workflow_sha256"] = result.prepared_workflow_sha256
        if result.model_inventory:
            entry["model_inventory"] = result.model_inventory
        if result.error is not None:
            entry["error"] = result.error
        return entry


def write_review_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> Path:
    """Write a review manifest (default artifact) for human review."""
    out = Path(path)
    _atomic_write_json(out, manifest)
    return out


def build_contact_sheet(
    outputs: Sequence[Path],
    destination: Path,
    *,
    command: Sequence[str] = ("ffmpeg",),
) -> Path | None:
    """Build an optional ffmpeg contact sheet from rendered outputs.

    Returns the sheet path on success, or None when no inputs are present.
    The montage is opt-in so large renders are not committed by default.
    """
    inputs = [p for p in outputs if p.is_file()]
    if not inputs:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    args = list(command) + ["-y"]
    for path in inputs:
        args += ["-i", str(path)]
    args += ["-filter_complex", "hstack=inputs=%d" % len(inputs), str(destination)]
    import subprocess

    subprocess.run(args, check=True, capture_output=True)
    return destination


class FakeBenchmarkBackend:
    """Deterministic backend for CI and wiring tests.

    Produces a stable per-cell duration and peak VRAM derived from the cell,
    writes a small placeholder output, and reports a fixed prepared-workflow
    hash and model inventory. No GPU or ComfyUI is required.
    """

    def __init__(self, output_dir: Path, *, models: dict[str, Any] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.models = models or {"unet": "ltx-2.5-fp8.safetensors", "vae": "vae.safetensors"}
        self.calls: list[Cell] = []

    def render(self, cell: Cell) -> CellResult:
        self.calls.append(cell)
        out = self.output_dir / f"scene_{cell.scene}_{cell.profile}_seed{cell.seed}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-video")
        duration = 0.5 + 0.1 * (cell.seed % 10)
        peak_vram = 4096.0 + 128.0 * (cell.scene % 8)
        return CellResult(
            profile=cell.profile,
            scene=cell.scene,
            seed=cell.seed,
            status=STATUS_OK,
            duration_seconds=duration,
            peak_vram_mb=peak_vram,
            output_path=str(out.relative_to(self.output_dir)),
            prepared_workflow_sha256="0" * 64,
            model_inventory=dict(self.models),
        )


def _parse_list(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(part for part in value.split(",") if part.strip())
    return items


def _parse_ints(values: list[str]) -> list[int]:
    items: list[int] = []
    for value in values:
        items.extend(int(part) for part in value.split(",") if part.strip())
    return items


def main(argv: list[str] | None = None) -> int:
    import argparse

    from rich.console import Console

    from feverslop.ports.reporting import ConsoleReporter
    from feverslop.utils.cli_output import emit_cli_data

    parser = argparse.ArgumentParser(
        description="Render a bounded profile/scene/seed GPU benchmark matrix.",
    )
    parser.add_argument("--profiles", action="append", default=[], help="Comma-separated workflow profile names.")
    parser.add_argument("--scenes", action="append", default=[], help="Comma-separated scene numbers.")
    parser.add_argument("--seeds", action="append", default=[], help="Comma-separated seed integers.")
    parser.add_argument("--backend", choices=("fake", "comfyui"), default="fake")
    parser.add_argument("--output-dir", default="benchmark-output")
    parser.add_argument("--contact-sheet", action="store_true", help="Build an ffmpeg contact sheet (opt-in).")
    parser.add_argument("--json", action="store_true", help="Emit the manifest as machine-readable JSON.")
    args = parser.parse_args(argv)

    profiles = _parse_list(args.profiles)
    scenes = _parse_ints(args.scenes)
    seeds = _parse_ints(args.seeds)
    if not profiles or not scenes or not seeds:
        parser.error("--profiles, --scenes, and --seeds are all required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = expand_matrix(profiles, scenes, seeds)
    if args.backend == "fake":
        backend: RenderBackend = FakeBenchmarkBackend(output_dir)
    else:
        parser.error("comfyui backend requires a configured ComfyUI target")
    reporter = ConsoleReporter(Console())
    runner = BenchmarkRunner(backend, cells, output_dir=output_dir, reporter=reporter)
    manifest = runner.run()
    write_review_manifest(manifest, output_dir / "manifest.json")
    if args.contact_sheet:
        outputs = [
            output_dir / cell["output_path"]
            for cell in manifest["cells"]
            if cell.get("status") in (STATUS_OK, STATUS_SKIPPED) and cell.get("output_path")
        ]
        sheet = build_contact_sheet(outputs, output_dir / "contact_sheet.png")
        if sheet is not None:
            reporter.message(f"contact sheet written: {sheet}")
    if args.json:
        emit_cli_data(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        reporter.message(
            f"benchmark complete: {len(manifest['cells'])} cells -> {output_dir / 'manifest.json'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
