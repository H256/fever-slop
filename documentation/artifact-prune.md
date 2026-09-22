# Artifact prune

`feverslop artifact prune` is a manual command that reclaims disk space from
regenerable cache artifacts in a project. Every deletion decision is governed
by the project's scene and plan manifests, and every deletion is
archive-first: eligible files are packaged into a recoverable ZIP before they
are removed.

The command is manual-only. It never runs automatically at pipeline
completion.

## Command

```bash
uv run feverslop artifact prune PROJECT [--safe | --apply] [--archive PATH]
```

- `PROJECT` — path to the project directory (required).
- `--safe` — scan and report only; never write (default).
- `--apply` — archive the eligible candidates, then delete them.
- `--archive PATH` — archive path for `--apply` (required with `--apply`).

`--safe` and `--apply` are mutually exclusive.

## Safe mode

Safe mode (no flag, or `--safe`) scans the project, classifies every
canonical artifact, and prints a machine-readable JSON report to stdout. It
writes nothing to the project tree.

Each candidate entry in the report shows:

- `path` — project-relative path of the artifact,
- `class` — its lifecycle class (`resume_cache` for candidates),
- `size_bytes` — byte estimate,
- `sha256` — content snapshot used to reject a candidate that changed before
  apply,
- `eligible` — whether it may be deleted,
- `reasons` — every ineligibility reason (all reasons are collected, not
  first-failure), so the report explains every blocker.

Each protected entry shows `path`, `class`, `size_bytes`, and a human-readable
`reason` explaining why it is never pruned.

Exit codes:

- `0` — OK, no eligible candidates.
- `2` — eligible candidates found but not applied; re-run with
  `--apply --archive PATH`.

## Apply mode

`--apply` requires `--archive PATH` (exit `1` without it). Apply mode:

1. rejects a report from another project, then scans the project again,
2. retains only files that are still eligible and byte-identical to the safe
   scan snapshot,
3. packages those candidates into a ZIP at the archive path, writing
   `archive_manifest.json` into the ZIP first,
4. reopens and verifies ZIP integrity, member names, and member hashes,
5. verifies every source file still matches its archived hash immediately
   before deletion,
6. deletes exactly the verified archived files,
7. writes a machine-readable report to
   `output/prune_report_<timestamp>.json` in the project, and prints the
   final report to stdout.

The archive is created through a temporary ZIP and atomically promoted only
after verification. If archive creation or verification fails, nothing is
deleted. Exit code is `0` on success.

## What is pruned

Only three artifact classes are candidates, and only with valid matching
manifests:

| Class | Paths |
|---|---|
| Expanded scene workflows | `scene_NNNN/workflow.json` |
| Raw clips | `scene_NNNN/raw.mp4` |
| Derived prompt/render plans | `output/render/plans/compact.json`, `anchored.json`, `references.json`, `ingredients.json` |

Other files — facefix/upscale intermediates, `h3_prompt.json`, and anything
else — are not candidates for this command.

## What is never pruned

These classes are protected and survive every run:

| Class | Paths |
|---|---|
| Project configuration | `config.json` |
| Canonical plans (user overrides live inside it) | `output/render/plans/base.json`, `output/prompts/story_plan_*.json` |
| Reference assets and manifests | `output/references/**`, `scene_NNNN/manifest.json` |
| Final scene clips | `scene_NNNN/final.mp4`, `final_facefix.mp4`, `upscale_final.mp4` |
| Assembled final videos | `output/render/final/**` |
| Review exports | `output/**/*.md`, `output/**/*.html` |

## Eligibility rules

A candidate is eligible only if it has zero ineligibility reasons.

For scene-scoped candidates (`workflow.json`, `raw.mp4`), "valid matching
manifest" means the scene's `manifest.json` is readable (schema v1-v3),
verifies cleanly (no hash mismatches or missing referenced files), and its
workflow/reference fingerprints match the base plan's canonical dependencies:

| Reason | Meaning |
|---|---|
| `manifest_missing` | `scene_NNNN/manifest.json` is absent |
| `manifest_invalid` | manifest unreadable, unsupported schema, or verification mismatch |
| `fingerprint_mismatch` | workflow/reference fingerprint drift vs the base plan |
| `no_final_successor` | `scene_NNNN/final.mp4` is absent |
| `facefix_pending` | facefix started but incomplete (`workflow_facefix.json` or `facefix/` intermediates present without `final_facefix.mp4`) |
| `upscale_pending` | `config.upscale.enabled` is true, or `upscale_pass_*` intermediates exist, without `upscale_final.mp4` |
| `referenced_by_unfinished` | another scene's manifest references this file and that scene's final chain is incomplete or its manifest is invalid |

Note that pending detection is file- and config-based: the facefix CLI flag is
not visible to the tool, so a facefix run is considered pending when its
intermediate artifacts exist without the final output.

For derived plan candidates, "valid matching manifest" means the plan carries
per-scene canonical dependency provenance that matches the current base plan:

| Reason | Meaning |
|---|---|
| `manifest_missing` | the plan has no canonical dependency provenance |
| `fingerprint_mismatch` | provenance drift vs the base plan |
| `dependencies_missing` | `base.json` is absent |

## Report schema

The machine-readable report uses `schema_version`
`feverslop.artifact-prune/v1`:

```json
{
  "schema_version": "feverslop.artifact-prune/v1",
  "created_at": "2026-09-21T12:00:00",
  "project": "/path/to/project",
  "mode": "safe",
  "archive_path": null,
  "candidates": [
    {
      "path": "output/render/scenes/scene_0001/raw.mp4",
      "class": "resume_cache",
      "size_bytes": 12345678,
      "eligible": true,
      "reasons": []
    }
  ],
  "protected": [
    {
      "path": "output/render/plans/base.json",
      "class": "authoritative",
      "size_bytes": 45678,
      "reason": "canonical plan (base.json / story plan) is authoritative"
    }
  ],
  "deleted": [],
  "errors": []
}
```

In apply mode, `mode` is `"apply"`, `archive_path` is the created ZIP path,
and `deleted` lists every removed file with its size.

## Recovery

The ZIP archive (with its `archive_manifest.json`) is the recoverable record.
It is fully verified before any deletion, and it contains exactly the files
that were deleted. There is no separate restore command; to recover, extract
the archived files back into the project at their original paths.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | OK — no eligible candidates (safe mode), or apply completed |
| `1` | error — bad project, I/O failure, `--apply` without `--archive`, or archive creation failure |
| `2` | eligible candidates found but not applied (safe mode) |
