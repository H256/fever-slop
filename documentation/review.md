# FeverSlop Code Review

Status: Historical review; findings describe the repository state on 2026-07-27 and are not a current implementation specification.

**Scope**: `application/movie.py`, `adapters/movie_planning.py`, `__init__.py`, `main.py` plus supporting ports, domain, composition, and memory modules.
**Date**: 2026-07-27
**Architecture**: Ports & Adapters (hexagonal) — Python 3.12

---

## Code Review Summary

Three structural problems dominate the review:

1. **Reflection-based capability checks replace Protocol contracts** — the entire port/adapter boundary is short-circuited by `getattr` + `callable()` in the application layer, making Protocol declarations in `ports/movie.py` documentation only.
2. **Three god modules** (>600 lines each) each violate SRP by mixing orchestration, domain logic, serialization, and utility code.
3. **Systematic duplication of low-level utilities** (`_safe_id`, `_string_list`, `_transition_from_previous`, `_configured_actors`, `_configured_locations`, `_clean_visual_description`) across `movie.py`, `movie_planning.py`, and `movie_memory.py`.

---

## Architecture Issues (P0–P1)

### A1 — Protocol contracts are dead code (P0)

**Files**: `ports/movie.py`, `application/movie.py`, `application/movie_memory.py`

`ScenePlanningPort` declares typed methods:

```python
class ScenePlanningPort(Protocol):
    def generate_movie_bible(...) -> MovieBible: ...
    def generate_movie_continuity_plan(...) -> MovieContinuityPlan | dict: ...
    def generate_movie_screenplay(...) -> MovieScreenplayArtifact | dict: ...
    # ... etc.
```

But the application layer never trusts these types. Instead, every call site uses reflection-based dispatch:

```python
# movie.py:367-379
generator = getattr(planner, "generate_movie_bible", None)
if callable(generator):
    bible = generator(...)
    if isinstance(bible, MovieBible):
        return _normalize_movie_bible(...)
return _movie_bible_from_config(...)
```

Same pattern in `movie_memory.py:29-46`, `movie_memory.py:50-66`, `movie_memory.py:148-163`.

**Impact**: The Protocol types serve no runtime guarantee. Union return types (`MovieContinuityPlan | dict`) are an admission that the adapter may return anything. This is not hexagonal — this is duck typing with annotations.

**Fix**: Either (a) enforce the Protocol contract by removing the `getattr` fallbacks and requiring adapters to return domain types, or (b) abandon Protocol and document the duck-typed interface. Mixing both is confusing.

**Effort**: M

---

### A2 — Domain logic leaks into application and adapters (P1)

**Files**: `application/movie.py`, `adapters/movie_planning.py`

The application layer directly constructs domain objects and encodes business rules that belong in the domain layer:

| Application function | Business rule | Should be in domain |
|---|---|---|
| `constrain_movie_shots_to_bible` | Validate actor_ids/location_ids against bible, clamp max actors | `MovieBible.constrain(shots)` |
| `augment_movie_bible_from_shot_references` | Auto-discover actors/locations from shot data | `MovieBible.augment_from_shots()` |
| `build_movie_continuity_fallback` | Build entire continuity plan with style bible, character states, narrative chain | `MovieContinuityPlan.fallback(bible, shots)` |
| `normalize_movie_continuity_plan` | Merge partial plan with fallback | `MovieContinuityPlan.normalize()` |

Similarly, `movie_planning.py` builds `CinematicShot`, `MovieBible`, `MovieActor`, and `MovieLocation` objects directly in adapter methods, encoding construction logic that should be domain factories.

**Impact**: Domain models are anemic. Business rules scatter across three modules instead of living in `domain/`.

**Fix**: Move construction logic into domain methods or dedicated domain factory modules (`domain/movie_factories.py`).

**Effort**: L

---

### A3 — Hexagonal boundary violation: `composition/` bypasses ports (P1)

**Files**: `main.py`, `composition/movie_pipeline.py`

`main.py` imports from `feverslop.composition.*` which is not part of the four-layer hexagonal structure (`domain`, `ports`, `application`, `adapters`). `composition/movie_pipeline.py` (953 lines) directly wires adapters, application use cases, config loaders, and stage runners — it is a second CLI entry point that bypasses the clean architecture.

The existing `main.py` only supports the old render-plan pipeline (`generate_render_plan`), not the movie pipeline. The movie pipeline is invoked from `composition/movie_pipeline.py` instead.

**Impact**: Two competing entry points with different wiring. The `composition/` package is a god module that should be the composition root but instead contains business logic, argparse definitions, and adapter instantiation.

**Fix**: Consolidate into a single composition root. Move CLI argument parsing out of `composition/movie_pipeline.py` into a dedicated `cli/` module. Keep `composition/` for dependency wiring only.

**Effort**: L

---

### A4 — God modules (P1)

| Module | Lines | Responsibilities |
|---|---|---|
| `application/movie.py` | 1,291 | Use-case orchestration (2), domain construction (continuity, bible, shots), render-plan serialization, reference manifests, actor/location utility functions, prompt sanitization, screenplay detection |
| `adapters/movie_planning.py` | 1,275 | Two adapter implementations (LLM + Deterministic), 7 prompt template functions, screenplay parsing utilities, actor/location extraction, shot normalization, visual description sanitization |
| `application/movie_memory.py` | 623 | Story design generation, screenplay generation, narrative plan generation, serialization round-trips (dict↔domain), fallback builders, scene/shot card builders |

**Impact**: Each file is hard to review, test, or refactor. Functions with 1–letter prefix (`_safe_id`, `_string_list`) are hidden utilities that span the module.

**Fix**: Split along responsibility boundaries (see Recommendations).

**Effort**: L

---

## Module-Specific Findings

### movie.py (application layer, 1,291 lines)

#### SRP Violations

| Lines | Responsibility | Functions |
|---|---|---|
| 83–272 | **Use case orchestration** | `ScaffoldMovieUseCase.execute`, `AutoProduceMovieUseCase.execute` |
| 366–831 | **Bible construction & normalization** | `generate_movie_bible`, `plan_movie_shots_from_bible`, `generate_movie_continuity_plan`, `constrain_movie_shots_to_bible`, `augment_movie_bible_from_shot_references`, `_movie_bible_from_config`, `_normalize_movie_bible` |
| 430–550 | **Continuity domain logic** | `apply_movie_continuity_to_shots`, `build_movie_continuity_fallback`, `_safe_continuity_facts`, `_split_continuity_text` |
| 586–720 | **Continuity serialization** | `normalize_movie_continuity_plan`, `movie_continuity_plan_to_dict`, `movie_continuity_plan_from_dict`, `_narrative_for_shot`, `_character_state_from_dict`, `_location_state_from_dict`, `_scene_packet_from_dict`, `_narrative_from_dict` |
| 792–849 | **Bible serialization** | `movie_bible_from_dict`, `_story_arch_from_dict`, `_actor_from_dict`, `_location_from_dict`, `_bible_dict` |
| 275–291 | **Input validation** | `validate_movie_input`, `_looks_like_screenplay` |
| 299–362 | **Render plan generation** | `_render_plan`, `_render_plan_shot`, `_reference_manifest` |
| 1175–1290 | **Actor prompt engineering** | `build_movie_actor_visual_description`, `build_movie_actor_reference_prompt`, `_sanitize_actor_cues`, `_sanitize_actor_cue_fragment`, `_strip_actor_prompt_boilerplate`, `_actor_static_cues`, `_shot_cues` |
| 1023–1159 | **Actor/location reference helpers** | `_movie_actor_refs`, `_movie_location_refs`, `_default_actor_id`, `_default_location_id`, `_movie_actor_name`, `_movie_location_name`, `_shots_for_actor`, `_configured_actor_ref` |

#### Coupling Issues

- **Line 84**: `ScaffoldMovieUseCase.__init__` accepts `console: Any | None` — the application layer depends on a specific console abstraction (Rich) via `ConsoleReporter`. This should accept `Reporter` only.
- **Line 202–203**: `execute()` writes `.studio/project.json` directly — file I/O in a use case. This belongs in an adapter.
- **Line 216–228**: `execute()` writes 11 JSON files — the use case is also the file persistence adapter.
- **Line 1006–1013**: `_story_arch_from_dict` does a lazy import (`from feverslop.domain.movie import StoryArch`) inside a function — indicates a circular import workaround, which is a structural smell.

#### Reflection-based dispatch

- Lines 367–379: `getattr(planner, "generate_movie_bible", None)` + `callable()` check
- Lines 382–405: `getattr(planner, "plan_shots_from_bible", None)` + fallback to `planner.plan_shots()`
- Lines 408–427: `getattr(planner, "generate_movie_continuity_plan", None)` + try/except pass

#### Duplicate utilities (also in `movie_planning.py` and `movie_memory.py`)

- `_safe_id` (line 1127) — identical regex logic, only differs by having `fallback` parameter
- `_string_list` (line 1132) — nearly identical to `movie_memory.py:618`
- `_configured_movie_actors` (line 873) — duplicates `_configured_actors` in `movie_planning.py:410`
- `_configured_movie_locations` (line 883) — duplicates `_configured_locations` in `movie_planning.py:427
- `_clean_movie_location_visual_description` (line 963) — same generic token stripping logic as `_clean_visual_description` in `movie_planning.py:663

---

### movie_planning.py (adapters layer, 1,275 lines)

#### Conflation of Concerns

| Lines | Concern | Assessment |
|---|---|---|
| 13–252 | `LLMMoviePlanner` | Adapter — OK, but methods are too broad |
| 254–366 | `DeterministicMoviePlanner` | Second adapter in same file — should be separate module |
| 369–398 | `_movie_bible_from_data` | Shared helper — mixes both adapters |
| 401–558 | Screenplay parsing / extraction | Domain logic masquerading as adapter utility |
| 563–692 | Location/actor text processing | Domain logic (visual description sanitization) |
| 695–705 | Config helpers | Infrastructure concerns |
| 708–733 | `_shots_from_data` | Shared conversion helper |
| 736–766 | Screenplay beat parsing | Domain logic |
| 769–1103 | **Seven prompt template functions** | ~340 lines of f-strings — should be externalized or in a dedicated module |
| 1142–1160 | `asdict_like_*` helpers | Serialization utilities |
| 1163–1273 | Shot normalization utilities | Shared with movie.py |

#### Complexity Hotspots

- `_shot_plan_from_bible_prompt` (line 916): 52-line function building a massive f-string that serializes the entire bible. The prompt itself is harder to maintain as Python code than as a template file.
- `_movie_continuity_plan_prompt` (line 975): 62-line function with embedded JSON schema and rules.
- `_merge_screenplay_references` (line 528): 31 lines of fuzzy matching logic with `getattr` duck-typing — fragile and untestable.
- `_is_character_action` (line 635): 26 lines of regex-based NLP for screenplay parsing — brittle and unmaintainable.

#### Duplicate utilities

- `_safe_id` (line 1187) — identical to `movie.py:1127` (minus fallback)
- `_string_list` (line 1169) — variant with `_safe_id` normalization, still duplicate structure
- `_configured_actors` (line 410) — duplicates `movie.py:873`
- `_configured_locations` (line 427) — duplicates `movie.py:883`
- `_clean_visual_description` (line 663) — same generic token filter as `movie.py:963`
- `_transition_from_previous` (line 970) — identical to `movie_memory.py:237`
- `_normalize_movie_shots` (line 1229) — shot splitting + duration normalization logic not found elsewhere but >30 lines
- `_display_name` (line 570) — identical to `movie.py:1140`
- `_dialogue_actor_ids` (line 1177) — similar regex pattern to `movie_memory.py:490`

---

### movie_memory.py (application layer, 623 lines)

Not a primary target but relevant for the duplicate analysis:

- `_safe_id` (line 613) — third copy
- `_string_list` (line 618) — third copy
- `_transition_from_previous` (line 237) — second copy (identical to `movie_planning.py:970`)
- Contains all the `getattr(planner, "generate_movie_*", None)` reflection pattern

---

## Code Quality Issues (P1–P2)

### Q1 — Systematic utility duplication (P1)

Six utility functions are duplicated across 2–3 files with minor variations:

| Function | movie.py | movie_planning.py | movie_memory.py | Variation |
|---|---|---|---|---|
| `_safe_id` | 1127 | 1187 | 613 | movie.py has `fallback` param |
| `_string_list` | 1132 | 1169 | 618 | movie_planning applies `_safe_id` to items |
| `_configured_actors` | 873 | 410 | — | Identical logic |
| `_configured_locations` | 883 | 427 | — | Identical logic |
| `_transition_from_previous` | — | 970 | 237 | Identical |
| `_display_name` | 1140 | 570 | — | Identical |
| `_clean_visual_description` | 963 | 663 | — | Near-identical (different generic token lists) |

**Fix**: Extract to `domain/shared_utils.py` or `domain/movie_utils.py`. Make `_safe_id` accept an optional `fallback` to unify all three versions.

**Effort**: S

---

### Q2 — Reflection-based capability checks everywhere (P1)

Already covered in A1. Every planner call in `movie.py` and `movie_memory.py` uses the `getattr` + `callable` + `isinstance` pattern. This is 8+ locations across 2 files.

**Effort**: M (to fix)

---

### Q3 — Prompt templates embedded as Python f-strings (P2)

Seven prompt functions in `movie_planning.py` (lines 769–1103, ~340 lines) are massive f-strings. They are:

- Hard to read (nested JSON escaping, curly brace escaping)
- Impossible to test in isolation without calling the function
- Not i18n-friendly
- Not versionable as separate documents

**Fix**: Move prompts to `prompts/` directory as `.jinja` or `.prompt` files. Load at runtime with `importlib.resources` or `pathlib`. This also reduces `movie_planning.py` by ~25%.

**Effort**: M

---

### Q4 — `__init__.py` is useless (P2)

```python
__all__ = ["application", "adapters", "domain", "ports"]
```

This exposes package namespaces, not public API. Consumers must know the internal structure to import anything. The actual use cases (`ScaffoldMovieUseCase`, `AutoProduceMovieUseCase`), domain models, and port protocols are not re-exported.

**Fix**: Re-export the public surface:

```python
from feverslop.application.movie import ScaffoldMovieUseCase, AutoProduceMovieUseCase, MovieInput
from feverslop.ports.movie import StoryGenerationPort, ScenePlanningPort, VisualGenerationPort, ReferenceGenerationPort
from feverslop.ports.reporting import Reporter, NullReporter, ConsoleReporter
# ...
__all__ = [...]
```

**Effort**: S

---

### Q5 — `main.py` supports only the old pipeline (P2)

`main.py` imports from `composition/generate_render_plan` and `composition/arg_parser`. It has zero movie pipeline support. The movie pipeline lives in `composition/movie_pipeline.py` with its own argparse setup.

Two entry points, two different CLI interfaces, no coordination.

**Fix**: Either consolidate into one `main.py` with subcommands (`main.py movie`, `main.py render`) or rename the movie CLI to `main_movie.py`.

**Effort**: S

---

### Q6 — ConsoleReporter uses reflection instead of typing (P2)

```python
# reporting.py:50-83
class ConsoleReporter:
    def __init__(self, console: object):  # <-- should be Rich Console type
        self.console = console

    def step(self, title: str) -> None:
        print_method = getattr(self.console, "print", None)  # <-- reflection
```

The adapter checks for `print` and `rule` methods at runtime. If someone passes a wrong object, the reporter silently does nothing.

**Fix**: Type `console` as `Console` from `rich.console`. Use `self.console.print()` directly.

**Effort**: S

---

### Q7 — File I/O in use cases (P1)

`ScaffoldMovieUseCase.execute()` writes 11 JSON files directly (lines 202–229). This violates the hexagonal principle: use cases should emit domain events or DTOs, and adapters should handle persistence.

**Fix**: Separate persistence into `MovieArtifactWriter` adapter. Use case returns `MovieScaffoldResult`; adapter writes files.

**Effort**: M

---

## Recommendations by Priority

### P0 — Critical

| # | Issue | Location | Fix | Effort |
|---|---|---|---|---|
| R1 | Reflection replaces Protocol contracts | `movie.py:367-427`, `movie_memory.py:29-163` | Remove `getattr` fallbacks. Require adapters to implement full Protocol. Return domain types, not `dict \| DomainType` unions. | M |
| R2 | File I/O inside use cases | `movie.py:202-229` | Extract `MovieArtifactWriter` adapter. Use case returns result; adapter persists. | M |
| R3 | Three god modules | `movie.py` (1291), `movie_planning.py` (1275), `movie_memory.py` (623) | Split `movie.py` into: (a) `use_cases.py` — orchestration classes, (b) `continuity.py` — continuity domain logic + serialization, (c) `references.py` — actor/location prompt builders. Split `movie_planning.py` into: (a) `llm_planner.py`, (b) `deterministic_planner.py`, (c) `prompts/` directory. | L |

### P1 — Important

| # | Issue | Location | Fix | Effort |
|---|---|---|---|---|
| R4 | Domain logic in application/adapters | `movie.py:430-720,723-849`, `movie_planning.py:369-733` | Move `constrain_movie_shots_to_bible`, `augment_movie_bible_from_shot_references`, `build_movie_continuity_fallback` into domain methods or `domain/movie_factories.py`. | L |
| R5 | Six utilities duplicated across 3 files | All three files | Extract `_safe_id`, `_string_list`, `_configured_actors`, `_configured_locations`, `_transition_from_previous`, `_display_name`, `_clean_visual_description` to `domain/movie_utils.py`. | S |
| R6 | `composition/` bypasses hexagonal architecture | `composition/movie_pipeline.py`, `main.py` | Consolidate entry points. `main.py` with argparse subcommands or a `cli/` package. Keep `composition/` as wiring only. | L |
| R7 | `ConsoleReporter` uses reflection | `ports/reporting.py:50-83` | Type `console` parameter as `Console`. Call methods directly. | S |
| R8 | `__init__.py` exposes nothing useful | `__init__.py` | Re-export public API surface (use cases, ports, domain models). | S |

### P2 — Nice to have

| # | Issue | Location | Fix | Effort |
|---|---|---|---|---|
| R9 | Prompt templates as f-strings | `movie_planning.py:769-1103` | Extract to `prompts/` directory as template files. | M |
| R10 | `main.py` only supports old pipeline | `main.py` | Add movie subcommand or create separate CLI entry point. | S |
| R11 | Lazy import in `_story_arch_from_dict` | `movie.py:1006-1013` | Fix circular import at module level or restructure imports. | S |
| R12 | `_looks_like_screenplay` thin wrapper | `movie.py:294-295` | Call `looks_like_screenplay` directly; remove wrapper. | S |
| R13 | `_split_screenplay_dialogue` thin wrapper | `movie_memory.py:482-483` | Import and call directly. | S |
| R14 | `_is_screenplay_character_cue` thin wrapper | `movie_memory.py:486-487` | Import and call directly. | S |

---

## Summary of Effort

| Priority | Count | Total effort |
|---|---|---|
| P0 | 3 | M + M + L |
| P1 | 5 | L + S + L + S + S |
| P2 | 6 | M + S + S + S + S + S |

**Highest ROI**: R5 (deduplicate utilities, S effort, affects 3 files), R7 (type ConsoleReporter, S effort), R8 (fix __init__.py, S effort), R1 (remove reflection pattern, M effort but unlocks clean architecture).
