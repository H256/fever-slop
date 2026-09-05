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
- Configuration examples live in the repository root.

## Working Rules

- Read the relevant code before editing; do not make broad assumptions from file names alone.
- Keep changes narrowly scoped to the requested behavior.
- Preserve existing public CLI behavior unless the request explicitly changes it.
- Do not commit secrets, local credentials, generated media, model outputs, or machine-specific config.
- Prefer structured parsing for JSON, TOML, SRT, and workflow files instead of ad hoc string replacement.
- Avoid unrelated formatting churn, especially in large workflow JSON files.

### Progress and observability

- Every long-running pipeline action must provide visible Rich output from the
  beginning of execution. The user must always be able to tell that the tool is
  still working; do not leave expensive network, render, model, or file loops
  silent.
- Report both levels of progress: major pipeline stages and meaningful
  intermediate work within a stage. Use the `Reporter` protocol (normally
  `ConsoleReporter`) for stage boundaries and status messages, and use
  `SubStepProgress` for repeated scene, shot, frame, request, or item work.
  Emit an initial status before the expensive operation and a completion/status
  update afterward; long loops must also emit throttled intermediate updates.
- For determinate Rich progress bars, use the shared
  `feverslop.utils.rich_progress.build_progress` factory. Keep the existing
  reporter state machines (`RenderProgressReporter` and
  `MovieStageProgressReporter`) separate when their semantics differ, but do
  not duplicate the Rich column/presentation setup.
- Thread the reporter/progress callback through new application and pipeline
  layers instead of creating hidden consoles or printing directly from deep
  implementation code. CLI and Studio adapters may translate the same events
  into console output, streamed job logs, or UI progress.
- Progress must reflect the actual selected stage set, including skipped
  stages, and must advance on deterministic stage events rather than relying
  on incidental log text matching. Do not log secrets, prompts, response
  bodies, image bytes, or other sensitive payloads.

## Planning Workflow

- When user-facing documentation is requested, create or update it under `documentation/`.
- Store project planning state outside the repository in the platform-specific repository state directory listed below.
- Use `<repo-state>/ideas/` for lightweight project planning and keep the corresponding issue-tracker item linked from the plan.
- Put rough, unsorted ideas in `<repo-state>/ideas/inbox.md`.
- When an idea is ready to refine, create a dedicated plan file in `<repo-state>/ideas/planned/`.
- Name plan files with an ISO date and short slug, for example `2026-06-19-better-storyboard-review.md`.
- After creating a plan for an inbox idea, mark that idea as checked in `<repo-state>/ideas/inbox.md`.
- Treat indented checkbox items in `<repo-state>/ideas/inbox.md` as dependent sub-ideas. When planning or implementing a sub-idea, account for the planned or completed changes from its parent idea and any relevant ancestor ideas.
- When a feature idea is picked from `<repo-state>/ideas/inbox.md`, do not jump straight to a plan. First use the `/grill me` workflow with the user to challenge assumptions, clarify scope, define non-goals, identify risks, and capture concrete acceptance criteria.
- Before writing a plan for an extension or behavior change, inspect the affected code paths and summarize the relevant files, interfaces, and constraints. Skip this only for pure documentation or clearly code-independent planning.
- After the feature has been specified through `/grill me` and the affected code has been inspected, use `/writing plans` to turn the captured information into the plan document under `<repo-state>/ideas/planned/`.
- Develop each planned feature on its own git branch. Use the `agent/` prefix by default unless the user requests another branch name.
- Record the intended branch name in the plan document before implementation starts, so paused work can be resumed later from the correct branch.
- After implementation, move the plan file to `<repo-state>/ideas/done/`.
- After completing an implementation task from a planned idea, commit the finished changes before handing the work back, unless the user explicitly asks not to commit or unresolved verification failures make a commit misleading.
- If an idea is explicitly abandoned, move it to `<repo-state>/ideas/rejected/`.
- Do not create extra status folders unless the workflow clearly needs them.
- Keep planning files in `<repo-state>/ideas/`, not in the repository.

### Repository state locations

Use a stable `<repo-id>` for the repository, independent of its local checkout
path:

- Linux: `~/.local/state/repos/<repo-id>/`
- macOS: `~/Library/Application Support/repos/<repo-id>/`
- Windows: `%LOCALAPPDATA%\repos\<repo-id>\`

In the planning rules above, `<repo-state>` means the matching platform path
for the current repository.

Every plan file should also have a corresponding issue-tracker item:

- Create the issue-tracker item when the plan file is written. Link its URL in the plan file's header (`Issue: <url>`).
- On completion, update both the plan file (move to `<repo-state>/ideas/done/`) and the issue-tracker item.
- Link the implementation commits in the issue-tracker item before closing it.

Plan files should use this structure when practical:

```markdown
# Feature Name

Status: Planned
Created: YYYY-MM-DD
Branch: agent/YYYY-MM-DD-feature-name

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
uv run python run_pipeline.py ./projects/my_song
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


## Library Documentations
Use Context7 MCP to fetch current documentation whenever the user asks about a library, framework, SDK, API, CLI tool, or cloud service — even well-known ones like React, Next.js, Prisma, Express, Tailwind, Django, or Spring Boot. This includes API syntax, configuration, version migration, library-specific debugging, setup instructions, and CLI tool usage. Use even when you think you know the answer — your training data may not reflect recent changes. Prefer this over web search for library docs.
Do not use for: refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

### Steps

1. Always start with `resolve-library-id` using the library name and what to look up in the library's documentation, unless the user provides an exact library ID in `/org/project` format
2. Pick the best match (ID format: `/org/project`) by: exact name match, description relevance, code snippet count, source reputation (High/Medium preferred), and benchmark score (higher is better). If results don't look right, try alternate names or queries (e.g., "next.js" not "nextjs", or rephrase the question). Use version-specific IDs when the user mentions a version
3. `query-docs` with the selected library ID and what to look up in the library's documentation (not single words), scoped to a single concept. If the question spans multiple distinct concepts (e.g. routing and auth and caching), make a separate `query-docs` call per concept with the same library ID, unless the question is about how the concepts interact — combined queries dilute ranking and return shallow results for each topic
4. Answer using the fetched docs

