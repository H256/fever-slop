# AGENTS.md

This file gives working instructions for AI coding agents in this repository.

## Scope

- These instructions apply to the entire repository.
- Keep this file in the repository root unless a subdirectory needs different rules.
- If a nested `AGENTS.md` is added later, its instructions override this file only for files below that directory.

## Project Context

- This is a Python 3.12 project for the FeverSlop music video pipeline.
- Use `uv` for dependency management and command execution.
- Source packages live under `src/`.
- Tests live under `tests/`.
- User/project data normally lives under `projects/`; avoid editing generated outputs unless the task explicitly targets them.
- ComfyUI workflow JSON files live under `workflows/`.
- Configuration examples live in the repository root and `config/`.

## Working Rules

- Read the relevant code before editing; do not make broad assumptions from file names alone.
- Keep changes narrowly scoped to the requested behavior.
- Preserve existing public CLI behavior unless the request explicitly changes it.
- Do not commit secrets, local credentials, generated media, model outputs, or machine-specific config.
- Prefer structured parsing for JSON, TOML, SRT, and workflow files instead of ad hoc string replacement.
- Avoid unrelated formatting churn, especially in large workflow JSON files.

## Planning Workflow

Historical internal tracker reference removed for the public history.
- Put rough, unsorted ideas in `docs/ideas/inbox.md`.
- When an idea is ready to refine, create a dedicated plan file in `docs/ideas/planned/`.
- Name plan files with an ISO date and short slug, for example `2026-06-19-better-storyboard-review.md`.
- After creating a plan for an inbox idea, mark that idea as checked in `docs/ideas/inbox.md`.
- Treat indented checkbox items in `docs/ideas/inbox.md` as dependent sub-ideas. When planning or implementing a sub-idea, account for the planned or completed changes from its parent idea and any relevant ancestor ideas.
- When a feature idea is picked from `docs/ideas/inbox.md`, do not jump straight to a plan. First use the `/grill me` workflow with the user to challenge assumptions, clarify scope, define non-goals, identify risks, and capture concrete acceptance criteria.
- Before writing a plan for an extension or behavior change, inspect the affected code paths and summarize the relevant files, interfaces, and constraints. Skip this only for pure documentation or clearly code-independent planning.
- After the feature has been specified through `/grill me` and the affected code has been inspected, use `/writing plans` to turn the captured information into the plan document under `docs/ideas/planned/`.
- Develop each planned feature on its own git branch. Use the `codex/` prefix by default unless the user requests another branch name.
- Record the intended branch name in the plan document before implementation starts, so paused work can be resumed later from the correct branch.
- After implementation, move the plan file to `docs/ideas/done/`.
- After completing an implementation task from a planned idea, commit the finished changes before handing the work back, unless the user explicitly asks not to commit or unresolved verification failures make a commit misleading.
- If an idea is explicitly abandoned, move it to `docs/ideas/rejected/`.
- Do not create extra status folders unless the workflow clearly needs them.
- Keep planning files in `docs/ideas/`, not in the repository root or `.agent/`.

Plan files should use this structure when practical:

```markdown
# Feature Name

Status: Planned
Created: YYYY-MM-DD
Branch: codex/YYYY-MM-DD-feature-name

## Raw Idea

Short description of the original idea.

## Problem

What is broken, missing, confusing, slow, risky, or otherwise worth changing?

## Proposed Change

What should change at a user-visible or system-behavior level?

## Non-Goals

What is intentionally out of scope?

## Implementation Plan

- [ ] Step 1
- [ ] Step 2
- [ ] Step 3

## Verification

How to check that the change works.

## Notes

Open questions, alternatives, and follow-up ideas.
```

## Commands

Install dependencies:

```powershell
uv sync
```

Run linting with Ruff:

```powershell
ruff check .
```

Run the test suite:

```powershell
uv run python -m unittest discover -s tests
```

Run the default project helper:

```powershell
uv run python run_pipeline.py .\projects\my_song
```

## Verification

- For linting, use `ruff check .`.
- For Python code changes, run the focused tests first, then the full `unittest` suite when practical.
- For CLI/script changes, verify the affected command path directly.
- For render workflow changes, validate JSON structure and avoid requiring a full render unless explicitly requested.
- If a change depends on local ComfyUI, FFmpeg, model files, GPU, or audio assets that are not available, say exactly what could not be verified.

## Style

- Match the existing style in nearby code.
- Keep comments short and only where they clarify non-obvious behavior.
- Prefer small, explicit functions over hidden side effects.
- Use ASCII in new text unless the surrounding file already uses non-ASCII or the content requires it.
