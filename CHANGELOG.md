# Changelog

All notable changes to FeverSlop are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-08-25

### Added

- Installable `feverslop` CLI with canonical `run`, `movie`, `render`, and
  `full-auto` command paths while retaining the legacy entry points.
- Standalone hero image sequence-sheet generation and reference-sheet benchmark
  evaluation tools.
- MiniMax H3 support for location-only scenes, explicit megapixel resolution,
  subject-bound audio reference stems, and a VRAM-optimized Turbo LoRA R2V
  workflow.
- CLI and H3 compatibility documentation, shared CLI argument definitions, and
  package-owned helper tools.

### Changed

- Timeline export now produces both Shotcut-compatible MLT and OpenShot output
  by default; `--format mlt`, `--format openshot`, and `--format both` remain
  available for explicit selection.
- Legacy root-level helper scripts are now compatibility facades over the
  installable package implementations.
- Completed MiniMax H3 renders are reused during resume, and media path
  handling is centralized across the pipeline.

### Fixed

- MiniMax H3 resolution overrides now support explicit megapixel values.
- H3 audio stems are matched to the subjects visible in each scene.
- Location-only H3 scenes no longer require an actor reference.
- CLI parser defaults and public import paths remain compatible during the CLI
  package migration.

## [0.4.1] - 2026-08-25

### Fixed

- Safe resume now reports the required LLM handoff when the configured model is
  still loading instead of failing without an actionable next step.
- Unchanged MSR reference assets and bindings are reused during resume instead
  of being rendered again when their fingerprints still match.
- Stale generated H3 prompts in `references.json` are no longer misclassified
  as manual legacy edits that block the next resume.
- Reference-generation settings and sequence-to-sheet workflow selection remain
  stable across resume and migration checks.
- Fresh derived reference plans without a legacy comparison baseline no longer
  block execution as unresolved manual edits.

## [0.4.0] - 2026-08-24

### Added

- Human-overridable canonical music-video render plans. Generator-owned and
  operator-owned values now coexist in `output/render/plans/base.json`, with
  deterministic effective-value resolution and stable scene identities.
- Safe migration tooling for existing edits in legacy and derived render plans:
  `main.py plan-migrate PROJECT` previews changes and `--apply` creates backups
  before atomically importing unambiguous overrides.
- Per-scene MiniMax H3 prompt checkpoints under
  `output/render/scenes/scene_NNNN/h3_prompt.json`. Judged scenes become
  inspectable immediately and fingerprint-valid checkpoints survive interrupted
  runs.
- Backend-specific effective prompt projections for MiniMax H3, LTX MSR,
  Ingredients, and classic I2V without flattening their distinct prompt,
  reference, audio, or timing contracts.
- Scene-local canonical dependency fingerprints and prepared-workflow manifest
  provenance for targeted invalidation of derived plans, workflows, and clips.
- Read-only canonical plan inspection commands: `plan path`, `plan validate`,
  `plan show`, `plan overrides`, and `status`.
- Explainable `main.py run PROJECT --dry-run` and `--resume` execution plans with
  `RUN`, `REUSE`, `BLOCKED`, and `NOT_SELECTED` decisions per phase and scene.
- Cross-pipeline semantic override regression corpus and complete operator
  workflows for migration, correction, H3 interruption, stale artifacts, and
  full regeneration.

### Changed

- `base.json` is now the sole human-editable music-video render plan.
  `compact.json`, `anchored.json`, `references.json`, `ingredients.json`, H3
  aggregates/checkpoints, manifests, and prepared workflows remain derived
  inspection or runtime caches.
- Pipeline stage terminology is renderer-neutral while legacy stage names and
  direct CLI entry points remain compatible.
- Headless project, job, timeline, persistence, rebuild, and workspace services
  now live in their canonical application/domain/adapter layers instead of the
  former Studio package.
- CI separates unit checks from local integration and end-to-end suites, and
  test doubles isolate Movie and prompt tests from unavailable external LLM,
  FFmpeg, and ComfyUI services.
- Ruff cleanup and import-boundary coverage were expanded across the codebase.

### Fixed

- Windows global-library deletion no longer races on lock files stored inside
  the asset directory; delete and recreate remain serialized through stable
  external locks.
- MiniMax H3 shared picture-reference validation accepts valid reused picture
  references without weakening unknown-reference checks.
- Stale derived plans, changed reference bindings/assets, workflow templates,
  and prepared scene workflows now fail closed instead of being reused
  silently.
- Human overrides are preserved during full and selected-scene plan
  regeneration, H3 synchronization, and anchor correction.

### Removed

- The deprecated Studio/QML package after its reusable headless services were
  moved to their canonical layers. The CLI pipeline remains supported.

## [0.3.0] - 2026-08-23

- Previous tagged release.

[0.5.0]: https://forgejo.elysium.lan/H256/fever-slop/compare/v0.4.1...v0.5.0
[0.4.1]: https://forgejo.elysium.lan/H256/fever-slop/compare/v0.4.0...v0.4.1
[0.4.0]: https://forgejo.elysium.lan/H256/fever-slop/compare/v0.3.0...v0.4.0
[0.3.0]: https://forgejo.elysium.lan/H256/fever-slop/releases/tag/v0.3.0
