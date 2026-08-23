# Studio Ownership Map

This document defines the target package ownership for the remaining
`feverslop.studio` modules. It is the sequencing contract for umbrella issue
[#600](https://forgejo.elysium.lan/H256/fever-slop/issues/600) and its child
issues. It describes Python package ownership only; it does not change any
project files or runtime behavior.

## Target boundaries

- `feverslop.application` owns use cases and application orchestration. It
  depends on domain types and ports, not concrete storage or rendering
  adapters.
- `feverslop.adapters` owns filesystem, persistence, locking, and external
  service implementations.
- `feverslop.composition` owns pipeline wiring and movie job construction.
- `feverslop.config` owns configuration validation and configuration policy.
- `feverslop.utils` owns transport-neutral utility behavior, including Rich
  log rendering.
- `feverslop.studio` is a temporary compatibility namespace for headless
  services. It must not become a new catch-all package.

## Module ownership map

Every current module has one target owner. A destination marked `new` is a
planned module name, not permission to move multiple responsibilities into one
generic service file.

| Current module | Target owner | Destination | Migration slice |
| --- | --- | --- | --- |
| `studio/__init__.py` | Compatibility boundary | `studio/` temporary shim package | #605 |
| `studio/artifact_catalog.py` | Adapter | `adapters/artifact_catalog.py` | #602, complete target |
| `studio/artifact_locking.py` | Adapter | `adapters/artifact_locking.py` | #602, complete target |
| `studio/job_service.py` | Application contracts + composition | `application/job_contracts.py` (contract in #613); runtime split in #603 | #613, #603 |
| `studio/jobs.py` | Runtime adapter | `adapters/job_runtime.py` (planned); no concrete runtime in `application/` | #613, #603 |
| `studio/logging.py` | Utility | `utils/logging.py` (new) | #603 |
| `studio/media_store.py` | Persistence adapter | `adapters/media_store.py` (new) | #604 |
| `studio/movie_pipeline_jobs.py` | Composition | `composition/movie_pipeline_jobs.py` | #602, complete target |
| `studio/pipeline_actions.py` | Application service | `application/pipeline_actions.py` (new) | #603 |
| `studio/pipeline_state_store.py` | Persistence adapter | `adapters/pipeline_state_store.py` (new) | #604 |
| `studio/project_repository.py` | Persistence adapter | `adapters/project_repository.py` (new) | #604 |
| `studio/project_requests.py` | Application request mapping | `application/project_requests.py` (new) | #603 |
| `studio/project_validation.py` | Configuration validation | `config/project_validation.py` (new) | #604 |
| `studio/projects.py` | Project application service | `application/project_service.py` (new) | #604 |
| `studio/rebuild_service.py` | Application service | `application/rebuild_service.py` (new) | #603 |
| `studio/reference_workspace_service.py` | Application service | `application/reference_workspace.py` | #603, extend existing owner |
| `studio/scene_workspace_service.py` | Application service | `application/scene_workspace.py` | #603, extend existing owner |
| `studio/timeline_service.py` | Application service | `application/timeline_app.py` | #603, extend existing owner |

The map deliberately separates project orchestration from persistence. A
`ProjectStore` replacement belongs in the application layer, while repository,
media, artifact, and pipeline-state file access belongs in adapters. No new
`studio2`, `services`, or similarly broad package is part of this migration.

## Job lifecycle split

Issue #613 defines the transport-neutral application contracts in
`application/job_contracts.py`: immutable submissions, snapshots, status values,
log events, and the `JobRuntime` protocol. Concrete executors remain outside
the application layer because they own subprocesses, thread pools, filesystem
access, or Rich integration. The follow-up #603 move must adapt those runtime
implementations behind the protocol before the legacy `studio` modules can be
removed.

## Compatibility policy

The current `feverslop.studio.*` imports are legacy Python import contracts.
During the migration:

1. Internal production code and tests use the target modules listed above.
2. A legacy module may remain only as a documented, tested re-export or
   compatibility shim; it must not retain a second implementation.
3. New code must not add imports from `feverslop.studio`.
4. The compatibility window remains open until child issues #602, #603, and
   #604 are complete, #605 has removed all internal consumers, and a release
   note announces the breaking import cleanup.
5. Removal of the remaining shims happens only in the #605 cleanup change,
   after the import-boundary tests and full test suite pass. External callers
   must migrate to the target modules before that change.

The compatibility window does not permit changing behavior, public request
shapes, job lifecycle semantics, or error handling while moving ownership.

## On-disk project contracts

The following files are data-format contracts independent of the Python package
name:

- `.studio/project.json` stores project metadata used by existing project and
  movie workflows.
- `.studio/pipeline_state.json` stores resumable pipeline state and completed
  stage information.

The migration keeps both paths, JSON shapes, read/write behavior, atomic-write
guarantees, and path validation unchanged. A Python import move is not a data
migration. Any future change to these files requires a separate migration plan,
fixtures for existing projects, and explicit compatibility coverage.

## Verification before each move

- Check the import graph and update all internal callers to the target owner.
- Keep a focused compatibility test for every retained legacy shim.
- Run import-boundary tests to prove canonical callers do not load the old
  package for the moved capability.
- Run persistence, path-security, atomic-write, and job-lifecycle tests for
  the affected slice.
- Run `ruff check src tests` and the full unittest suite before removing a
  compatibility shim.
