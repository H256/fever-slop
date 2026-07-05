from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from feverslop.domain.movie import (
    CinematicShot,
    MovieActor,
    MovieBible,
    MovieContinuityCharacterState,
    MovieContinuityLedger,
    MovieContinuityLocationState,
    MovieContinuityPlan,
    MovieContinuityRule,
    MovieContinuityStyleBible,
    MovieLocation,
    MovieNarrativeBeat,
    MovieNarrativePlan,
    MovieProject,
    MovieSceneCard,
    MovieSceneContinuityPacket,
    MovieScreenplayArtifact,
    MovieScreenplayScene,
    MovieShotCard,
)
from feverslop.ports.movie import ReferenceGenerationPort, ScenePlanningPort, StoryGenerationPort, VisualGenerationPort


_SCREENPLAY_HEADING_RE = re.compile(r"\b(?:INT|EXT|INT/EXT)\.\s+", re.IGNORECASE)
_DIALOGUE_CUE_RE = re.compile(r"\b[A-Z][A-Z0-9 _'-]{1,30}:\s+\S")


@dataclass(frozen=True)
class MovieInput:
    name: str
    source_type: str
    story_text: str
    desired_length: float
    width: int = 1280
    height: int = 704
    mode: str = "scaffold"
    min_scene_duration: float = 4.0
    max_scene_duration: float = 20.0
    config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MovieScaffoldResult:
    project_slug: str
    project_dir: Path
    bible_path: Path
    story_arch_path: Path
    render_plan_path: Path
    reference_manifest_path: Path
    screenplay_path: Path | None = None
    narrative_plan_path: Path | None = None
    scene_cards_path: Path | None = None
    shot_cards_path: Path | None = None


@dataclass(frozen=True)
class MovieProductionResult(MovieScaffoldResult):
    final_video_path: Path | None = None


class ScaffoldMovieUseCase:
    def __init__(self, *, planner: StoryGenerationPort & ScenePlanningPort, projects_root: Path):
        self.planner = planner
        self.projects_root = Path(projects_root)

    def execute(self, request: MovieInput) -> MovieScaffoldResult:
        validate_movie_input(request)
        slug = slugify_project_name(request.name)
        project_dir = self.projects_root / slug
        movie_dir = project_dir / "movie"
        movie_dir.mkdir(parents=True, exist_ok=False)

        config = dict(request.config or {})
        story_arch = self.planner.generate_story_arch(
            title=request.name,
            source_type=request.source_type,
            story_text=_planner_source_text(request, config),
            desired_length=float(request.desired_length),
        )
        bible = generate_movie_bible(
            planner=self.planner,
            request=request,
            story_arch=story_arch,
            config=config,
        )
        screenplay = generate_movie_screenplay(
            planner=self.planner,
            request=request,
            bible=bible,
            story_arch=story_arch,
            config=config,
        )
        narrative_plan = generate_movie_narrative_plan(
            planner=self.planner,
            request=request,
            bible=bible,
            screenplay=screenplay,
            config=config,
        )
        shots = plan_movie_shots_from_bible(
            planner=self.planner,
            bible=bible,
            desired_length=float(request.desired_length),
            width=int(request.width),
            height=int(request.height),
            min_duration=float(request.min_scene_duration),
            max_duration=float(request.max_scene_duration),
        )
        bible = augment_movie_bible_from_shot_references(bible, shots, config=config)
        shots = constrain_movie_shots_to_bible(shots, bible)
        continuity_plan = generate_movie_continuity_plan(planner=self.planner, request=request, bible=bible, shots=shots, config=config)
        shots = apply_movie_continuity_to_shots(shots, continuity_plan)
        scene_cards = build_movie_scene_cards(screenplay=screenplay, shots=shots)
        shot_cards = build_movie_shot_cards(shots=shots, scene_cards=scene_cards)
        movie = MovieProject(
            slug=slug,
            name=request.name,
            bible=bible,
            story_arch=story_arch,
            shots=shots,
            duration_seconds=float(request.desired_length),
            width=int(request.width),
            height=int(request.height),
            mode=request.mode,
            config=config,
        )

        metadata = {
            "project_type": "movie",
            "display_name": request.name,
            "slug": slug,
            "movie": {
                "source_type": request.source_type,
                "story_text": request.story_text,
                "desired_length": float(request.desired_length),
                "width": int(request.width),
                "height": int(request.height),
                "mode": request.mode,
            },
        }
        (project_dir / ".studio").mkdir(parents=True, exist_ok=True)
        (project_dir / ".studio" / "project.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        story_arch_path = movie_dir / "story_arch.json"
        bible_path = movie_dir / "bible.json"
        screenplay_path = movie_dir / "screenplay.json"
        screenplay_md_path = movie_dir / "screenplay.md"
        narrative_plan_path = movie_dir / "narrative_plan.json"
        scene_cards_path = movie_dir / "scene_cards.json"
        continuity_plan_path = movie_dir / "continuity_plan.json"
        shot_cards_path = movie_dir / "shot_cards.json"
        render_plan_path = movie_dir / "render_plan.json"
        reference_manifest_path = movie_dir / "references" / "manifest.json"
        story_arch_path.write_text(json.dumps(asdict(movie.story_arch), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        bible_path.write_text(json.dumps(_bible_dict(movie.bible), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        screenplay_path.write_text(json.dumps(movie_screenplay_to_dict(screenplay), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        screenplay_md_path.write_text(movie_screenplay_to_markdown(screenplay), encoding="utf-8")
        narrative_plan_path.write_text(json.dumps(movie_narrative_plan_to_dict(narrative_plan), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        scene_cards_path.write_text(json.dumps(movie_scene_cards_to_dict(scene_cards), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        continuity_plan_path.write_text(json.dumps(movie_continuity_plan_to_dict(continuity_plan), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        shot_cards_path.write_text(json.dumps(movie_shot_cards_to_dict(shot_cards), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if config:
            (project_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        render_plan_path.write_text(json.dumps(_render_plan(movie, shot_cards=shot_cards), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        reference_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        reference_manifest_path.write_text(json.dumps(_reference_manifest(movie), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return MovieScaffoldResult(
            slug,
            project_dir,
            bible_path,
            story_arch_path,
            render_plan_path,
            reference_manifest_path,
            screenplay_path,
            narrative_plan_path,
            scene_cards_path,
            shot_cards_path,
        )


class AutoProduceMovieUseCase:
    def __init__(self, *, scaffold: ScaffoldMovieUseCase, visual_backend: VisualGenerationPort, reference_generator: ReferenceGenerationPort | None = None):
        self.scaffold = scaffold
        self.visual_backend = visual_backend
        self.reference_generator = reference_generator

    def execute(self, request: MovieInput) -> MovieProductionResult:
        scaffolded = self.scaffold.execute(request)
        if self.reference_generator is not None:
            self.reference_generator.generate(project_dir=scaffolded.project_dir)
        final_video = self.visual_backend.render_movie(
            project_dir=scaffolded.project_dir,
            render_plan_path=scaffolded.render_plan_path,
        )
        return MovieProductionResult(
            project_slug=scaffolded.project_slug,
            project_dir=scaffolded.project_dir,
            bible_path=scaffolded.bible_path,
            story_arch_path=scaffolded.story_arch_path,
            render_plan_path=scaffolded.render_plan_path,
            reference_manifest_path=scaffolded.reference_manifest_path,
            screenplay_path=scaffolded.screenplay_path,
            narrative_plan_path=scaffolded.narrative_plan_path,
            scene_cards_path=scaffolded.scene_cards_path,
            shot_cards_path=scaffolded.shot_cards_path,
            final_video_path=final_video,
        )


def validate_movie_input(request: MovieInput) -> None:
    if request.source_type not in {"short_story", "screenplay"}:
        raise ValueError("source_type must be short_story or screenplay")
    if not request.name.strip():
        raise ValueError("Movie project name is required")
    if not slugify_project_name(request.name):
        raise ValueError("Movie project slug is empty after slugifying the name")
    if len(request.story_text.strip()) < 20:
        raise ValueError("Movie story input is too short")
    if float(request.desired_length) <= 0:
        raise ValueError("desired_length must be positive")
    if int(request.width) <= 0 or int(request.height) <= 0:
        raise ValueError("resolution width and height must be positive")
    if request.mode not in {"scaffold", "full_auto"}:
        raise ValueError("movie mode must be scaffold or full_auto")
    if request.source_type == "screenplay" and not _looks_like_screenplay(request.story_text):
        raise ValueError("screenplay input must contain scene headings such as INT. or EXT.")


def _looks_like_screenplay(text: str) -> bool:
    upper = text.upper()
    return "INT." in upper or "EXT." in upper


def _screenplay_scenes_from_text(text: str, *, bible: MovieBible) -> tuple[MovieScreenplayScene, ...]:
    scenes: list[MovieScreenplayScene] = []
    heading = ""
    body: list[str] = []
    start_line = 1
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if _SCREENPLAY_HEADING_RE.match(line):
            if heading:
                scenes.append(_screenplay_scene_from_parts(len(scenes) + 1, heading, body, start_line=start_line, end_line=line_number - 1, bible=bible))
            heading = line
            body = []
            start_line = line_number
        elif heading:
            body.append(line)
    if heading:
        scenes.append(_screenplay_scene_from_parts(len(scenes) + 1, heading, body, start_line=start_line, end_line=len(text.splitlines()), bible=bible))
    return tuple(scenes) or _screenplay_scenes_from_beats((text,), bible=bible)


def _screenplay_scene_from_parts(index: int, heading: str, body: list[str], *, start_line: int, end_line: int, bible: MovieBible) -> MovieScreenplayScene:
    dialogue, actions = _split_screenplay_dialogue(body)
    action = " ".join(actions).strip()
    actor_ids = tuple(_valid_actor_ids(_dialogue_actor_ids(dialogue), bible)) or _default_bible_actor_ids(bible)
    location_id = _location_id_from_heading(heading, bible)
    summary = action or dialogue or heading
    return MovieScreenplayScene(
        scene_id=f"scene_{index:04}",
        heading=heading,
        summary=summary,
        action=action,
        dialogue=dialogue,
        actor_ids=actor_ids,
        location_id=location_id,
        source_span=f"lines:{start_line}-{end_line}",
    )


def _screenplay_scenes_from_beats(beats: tuple[str, ...], *, bible: MovieBible) -> tuple[MovieScreenplayScene, ...]:
    actor_ids = _default_bible_actor_ids(bible)
    location_id = bible.locations[0].id if bible.locations else "primary_location"
    return tuple(
        MovieScreenplayScene(
            scene_id=f"scene_{index:04}",
            heading=f"SCENE {index}",
            summary=str(beat).strip(),
            action=str(beat).strip(),
            actor_ids=actor_ids,
            location_id=location_id,
            source_span=f"beat:{index}",
        )
        for index, beat in enumerate(beats, start=1)
        if str(beat).strip()
    )


def _split_screenplay_dialogue(lines: list[str]) -> tuple[str, list[str]]:
    dialogue: list[str] = []
    actions: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_screenplay_character_cue(line) and index + 1 < len(lines):
            dialogue.append(f"{line}: {lines[index + 1]}")
            index += 2
            continue
        if ":" in line and line.split(":", 1)[0].strip().isupper():
            dialogue.append(line)
        else:
            actions.append(line)
        index += 1
    return " ".join(dialogue).strip(), actions


def _is_screenplay_character_cue(line: str) -> bool:
    words = line.split()
    return bool(words) and len(words) <= 4 and line.upper() == line and not _SCREENPLAY_HEADING_RE.match(line)


def _dialogue_actor_ids(dialogue: str) -> list[str]:
    ids = []
    for match in re.finditer(r"\b([A-Z][A-Z0-9 _'-]{1,30}):", dialogue):
        actor_id = _safe_id(match.group(1), "")
        if actor_id and actor_id not in ids:
            ids.append(actor_id)
    return ids


def _valid_actor_ids(raw_ids: object, bible: MovieBible) -> tuple[str, ...]:
    valid = {actor.id for actor in bible.actors}
    ids = []
    for raw_id in _string_list(raw_ids):
        actor_id = _safe_id(raw_id, "")
        if actor_id in valid and actor_id not in ids:
            ids.append(actor_id)
    return tuple(ids)


def _valid_location_id(location_id: str, bible: MovieBible) -> str:
    valid = {location.id for location in bible.locations}
    safe = _safe_id(location_id, "")
    if safe in valid:
        return safe
    return bible.locations[0].id if bible.locations else "primary_location"


def _default_bible_actor_ids(bible: MovieBible) -> tuple[str, ...]:
    return (bible.actors[0].id,) if bible.actors else ("main_character",)


def _location_id_from_heading(heading: str, bible: MovieBible) -> str:
    lower = heading.lower()
    for location in bible.locations:
        if location.id.lower() in lower or location.name.lower() in lower:
            return location.id
    return bible.locations[0].id if bible.locations else "primary_location"


def _start_frame_brief(shot: CinematicShot) -> str:
    return f"Opening frame: {shot.action or shot.description}; actors {', '.join(shot.actor_ids) or 'none'} in {shot.location or shot.location_id}."


def _end_frame_brief(shot: CinematicShot) -> str:
    return f"Ending frame: {shot.story_state_after or shot.action or shot.description}; preserve actor identity and location geography."


def slugify_project_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return re.sub(r"-+", "-", slug)


def _render_plan(movie: MovieProject, *, shot_cards: tuple[MovieShotCard, ...] = ()) -> dict:
    cards_by_id = {card.shot_id: card for card in shot_cards}
    return {
        "project_type": "movie",
        "title": movie.name,
        "duration_seconds": movie.duration_seconds,
        "resolution": {"width": movie.width, "height": movie.height},
        "audio_policy": "ltx_native",
        "visual_backends": ["krea2", "ltx_msr"],
        "movie_screenplay_path": "movie/screenplay.json",
        "movie_narrative_plan_path": "movie/narrative_plan.json",
        "movie_scene_cards_path": "movie/scene_cards.json",
        "movie_shot_cards_path": "movie/shot_cards.json",
        "shots": [_render_plan_shot(shot, movie_config(movie), shot_card=cards_by_id.get(shot.shot_id)) for shot in movie.shots],
    }


def _render_plan_shot(shot, config: dict, *, shot_card: MovieShotCard | None = None) -> dict:
    data = asdict(shot)
    data["acting"] = data.get("expression", "")
    data["continuity_notes"] = data.get("continuity_notes", "")
    data["reference_ids"] = {
        "actors": list(shot.actor_ids) or [_default_actor_id(config)],
        "location": shot.location_id or _default_location_id(config),
    }
    if shot_card:
        data["shot_card"] = asdict(shot_card)
        data["start_frame_brief"] = shot_card.start_frame_brief
        data["end_frame_brief"] = shot_card.end_frame_brief
    return data


def _reference_manifest(movie: MovieProject) -> dict:
    return {
        "project_type": "movie",
        "actors": [
            {
                "id": actor.id,
                "name": actor.name,
                "role": actor.role,
                "visual_description": actor.visual_description,
                "image_prompt": build_movie_actor_reference_prompt(actor.name, actor.visual_description),
                "prompt": build_movie_actor_reference_prompt(actor.name, actor.visual_description),
                "status": "required",
                "msr_sheet_path": "",
            }
            for actor in movie.bible.actors
        ],
        "locations": [
            {
                "id": location.id,
                "name": location.name,
                "visual_description": location.visual_description,
                "image_prompt": location.visual_description,
                "prompt": location.visual_description,
                "status": "required",
                "msr_sheet_path": "",
            }
            for location in movie.bible.locations
        ],
    }


def generate_movie_screenplay(*, planner, request: MovieInput, bible: MovieBible, story_arch, config: dict) -> MovieScreenplayArtifact:
    generator = getattr(planner, "generate_movie_screenplay", None)
    if callable(generator):
        raw = generator(
            title=request.name,
            source_type=request.source_type,
            story_text=_planner_source_text(request, config),
            desired_length=float(request.desired_length),
            bible=bible,
            story_arch=story_arch,
            config=config,
        )
        if isinstance(raw, MovieScreenplayArtifact):
            return raw
        if isinstance(raw, dict) and raw.get("scenes"):
            return movie_screenplay_from_dict(raw, fallback_title=request.name, source_type=request.source_type, bible=bible)
    return build_movie_screenplay_fallback(request=request, bible=bible, story_arch=story_arch, config=config)


def build_movie_screenplay_fallback(*, request: MovieInput, bible: MovieBible, story_arch, config: dict) -> MovieScreenplayArtifact:
    dialogue_language = str(config.get("dialogue_language") or (bible.runtime_constraints or {}).get("dialogue_language") or "").strip()
    if request.source_type == "screenplay":
        scenes = _screenplay_scenes_from_text(request.story_text, bible=bible)
    else:
        scenes = _screenplay_scenes_from_beats(story_arch.beats or (story_arch.premise,), bible=bible)
    return MovieScreenplayArtifact(
        title=request.name,
        source_type=request.source_type,
        dialogue_language=dialogue_language,
        scenes=scenes,
    )


def generate_movie_narrative_plan(*, planner, request: MovieInput, bible: MovieBible, screenplay: MovieScreenplayArtifact, config: dict) -> MovieNarrativePlan:
    generator = getattr(planner, "generate_movie_narrative_plan", None)
    if callable(generator):
        raw = generator(
            title=request.name,
            source_type=request.source_type,
            desired_length=float(request.desired_length),
            bible=bible,
            screenplay=screenplay,
            config=config,
        )
        if isinstance(raw, MovieNarrativePlan):
            return raw
        if isinstance(raw, dict) and (raw.get("sequences") or raw.get("causal_chain")):
            return movie_narrative_plan_from_dict(raw, fallback_title=request.name)
    return build_movie_narrative_plan_fallback(screenplay=screenplay)


def build_movie_narrative_plan_fallback(*, screenplay: MovieScreenplayArtifact) -> MovieNarrativePlan:
    scene_ids = tuple(scene.scene_id for scene in screenplay.scenes)
    sequences = (
        {
            "sequence_id": "sequence_0001",
            "title": screenplay.title,
            "scene_ids": list(scene_ids),
            "dramatic_function": "Carry the movie premise through a causal beginning, middle, and resolution.",
        },
    )
    causal_chain = []
    previous = "The movie begins from the stated premise."
    for index, scene in enumerate(screenplay.scenes, start=1):
        after = scene.action or scene.summary or scene.heading
        causal_chain.append(
            {
                "scene_id": scene.scene_id,
                "story_state_before": previous,
                "story_state_after": after,
                "cause_from_previous": "Opening scene establishes the premise." if index == 1 else f"Previous scene leaves: {previous}",
                "sets_up_next": screenplay.scenes[index].summary if index < len(screenplay.scenes) else "The final scene resolves the current arc.",
            }
        )
        previous = after
    return MovieNarrativePlan(title=screenplay.title, sequences=sequences, causal_chain=tuple(causal_chain), open_threads=())


def build_movie_scene_cards(*, screenplay: MovieScreenplayArtifact, shots: tuple[CinematicShot, ...]) -> tuple[MovieSceneCard, ...]:
    cards = []
    for index, scene in enumerate(screenplay.scenes):
        shot = shots[min(index, len(shots) - 1)] if shots else None
        shot_ids = (shot.shot_id,) if shot else ()
        cards.append(
            MovieSceneCard(
                scene_id=scene.scene_id,
                shot_ids=shot_ids,
                dramatic_purpose=scene.summary or scene.action or scene.heading,
                story_state_before=shot.story_state_before if shot else "",
                story_state_after=shot.story_state_after if shot else scene.action,
                active_actor_ids=scene.actor_ids or (shot.actor_ids if shot else ()),
                location_id=scene.location_id or (shot.location_id if shot else ""),
                dialogue=scene.dialogue,
            )
        )
    return tuple(cards)


def build_movie_shot_cards(*, shots: tuple[CinematicShot, ...], scene_cards: tuple[MovieSceneCard, ...]) -> tuple[MovieShotCard, ...]:
    scene_by_shot = {shot_id: card for card in scene_cards for shot_id in card.shot_ids}
    cards = []
    for shot in shots:
        scene_card = scene_by_shot.get(shot.shot_id)
        scene_id = scene_card.scene_id if scene_card else shot.shot_id.replace("shot", "scene", 1)
        action = shot.action or shot.description
        cards.append(
            MovieShotCard(
                shot_id=shot.shot_id,
                scene_id=scene_id,
                action=action,
                camera=shot.camera,
                acting=shot.expression,
                dialogue=shot.dialogue,
                start_frame_brief=_start_frame_brief(shot),
                end_frame_brief=_end_frame_brief(shot),
            )
        )
    return tuple(cards)


def movie_screenplay_to_dict(screenplay: MovieScreenplayArtifact) -> dict:
    return {
        "title": screenplay.title,
        "source_type": screenplay.source_type,
        "dialogue_language": screenplay.dialogue_language,
        "scenes": [asdict(scene) for scene in screenplay.scenes],
    }


def movie_screenplay_from_dict(data: dict, *, fallback_title: str, source_type: str, bible: MovieBible) -> MovieScreenplayArtifact:
    scenes = []
    for index, raw in enumerate(data.get("scenes") or [], start=1):
        if not isinstance(raw, dict):
            continue
        scenes.append(
            MovieScreenplayScene(
                scene_id=str(raw.get("scene_id") or f"scene_{index:04}"),
                heading=str(raw.get("heading") or f"Scene {index}").strip(),
                summary=str(raw.get("summary") or raw.get("action") or "").strip(),
                action=str(raw.get("action") or raw.get("summary") or "").strip(),
                dialogue=str(raw.get("dialogue") or "").strip(),
                actor_ids=tuple(_valid_actor_ids(raw.get("actor_ids") or [], bible)),
                location_id=_valid_location_id(str(raw.get("location_id") or ""), bible),
                source_span=str(raw.get("source_span") or "").strip(),
            )
        )
    if not scenes:
        scenes = list(_screenplay_scenes_from_beats((str(data.get("premise") or fallback_title),), bible=bible))
    return MovieScreenplayArtifact(
        title=str(data.get("title") or fallback_title),
        source_type=str(data.get("source_type") or source_type),
        dialogue_language=str(data.get("dialogue_language") or (bible.runtime_constraints or {}).get("dialogue_language") or "").strip(),
        scenes=tuple(scenes),
    )


def movie_screenplay_to_markdown(screenplay: MovieScreenplayArtifact) -> str:
    parts = [f"# {screenplay.title}", ""]
    for scene in screenplay.scenes:
        parts.extend([f"## {scene.heading}", "", scene.action or scene.summary])
        if scene.dialogue:
            parts.extend(["", scene.dialogue])
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def movie_narrative_plan_to_dict(plan: MovieNarrativePlan) -> dict:
    return {
        "title": plan.title,
        "sequences": list(plan.sequences),
        "causal_chain": list(plan.causal_chain),
        "open_threads": list(plan.open_threads),
    }


def movie_narrative_plan_from_dict(data: dict, *, fallback_title: str) -> MovieNarrativePlan:
    return MovieNarrativePlan(
        title=str(data.get("title") or fallback_title),
        sequences=tuple(item for item in data.get("sequences") or [] if isinstance(item, dict)),
        causal_chain=tuple(item for item in data.get("causal_chain") or [] if isinstance(item, dict)),
        open_threads=tuple(str(item).strip() for item in data.get("open_threads") or [] if str(item).strip()),
    )


def movie_scene_cards_to_dict(cards: tuple[MovieSceneCard, ...]) -> dict:
    return {"scene_cards": [asdict(card) for card in cards]}


def movie_shot_cards_to_dict(cards: tuple[MovieShotCard, ...]) -> dict:
    memory_pack = {}
    if cards:
        first = cards[0]
        memory_pack = {
            "current_shot": {
                "shot_id": first.shot_id,
                "description": first.action,
                "scene_id": first.scene_id,
            }
        }
    return {"shot_cards": [asdict(card) for card in cards], "memory_pack": memory_pack}


def generate_movie_bible(*, planner, request: MovieInput, story_arch, config: dict) -> MovieBible:
    generator = getattr(planner, "generate_movie_bible", None)
    if callable(generator):
        bible = generator(
            title=request.name,
            source_type=request.source_type,
            story_text=_planner_source_text(request, config),
            desired_length=float(request.desired_length),
            story_arch=story_arch,
            config=config,
        )
        if isinstance(bible, MovieBible):
            return _normalize_movie_bible(bible, story_arch=story_arch, config=config, request=request)
    return _movie_bible_from_config(request=request, story_arch=story_arch, config=config)


def plan_movie_shots_from_bible(*, planner, bible: MovieBible, desired_length: float, width: int, height: int, min_duration: float, max_duration: float) -> tuple[CinematicShot, ...]:
    planner_from_bible = getattr(planner, "plan_shots_from_bible", None)
    if callable(planner_from_bible):
        return tuple(
            planner_from_bible(
                bible=bible,
                desired_length=desired_length,
                width=width,
                height=height,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        )
    return tuple(
        planner.plan_shots(
            story_arch=bible.story_arch,
            desired_length=desired_length,
            width=width,
            height=height,
            min_duration=min_duration,
            max_duration=max_duration,
        )
    )


def generate_movie_continuity_plan(*, planner, request: MovieInput, bible: MovieBible, shots: tuple[CinematicShot, ...], config: dict) -> MovieContinuityPlan:
    generator = getattr(planner, "generate_movie_continuity_plan", None)
    if callable(generator):
        try:
            raw = generator(
                title=request.name,
                source_type=request.source_type,
                story_text=_planner_source_text(request, config),
                desired_length=float(request.desired_length),
                bible=bible,
                shots=shots,
                config=config,
            )
            if isinstance(raw, MovieContinuityPlan):
                return normalize_movie_continuity_plan(raw, bible=bible, shots=shots)
            if isinstance(raw, dict):
                return movie_continuity_plan_from_dict(raw, bible=bible, shots=shots)
        except Exception:
            pass
    return build_movie_continuity_fallback(bible=bible, shots=shots)


def apply_movie_continuity_to_shots(shots: tuple[CinematicShot, ...], continuity_plan: MovieContinuityPlan) -> tuple[CinematicShot, ...]:
    narrative_by_id = {beat.shot_id: beat for beat in continuity_plan.narrative_chain}
    packet_by_id = continuity_plan.scene_continuity
    updated = []
    for shot in shots:
        beat = narrative_by_id.get(shot.shot_id)
        packet = packet_by_id.get(shot.shot_id)
        continuity_notes = "; ".join(
            part
            for part in [
                "; ".join(_safe_continuity_facts(packet.incoming)) if packet else "",
                "; ".join(_safe_continuity_facts(packet.required_carryovers)) if packet else "",
                "; ".join(_safe_continuity_facts(packet.outgoing)) if packet else "",
            ]
            if part
        )
        updated.append(
            replace(
                shot,
                continuity_notes=shot.continuity_notes or continuity_notes,
                story_state_before=shot.story_state_before or (beat.story_state_before if beat else ""),
                story_state_after=shot.story_state_after or (beat.story_state_after if beat else ""),
                cause_from_previous=shot.cause_from_previous or (beat.cause_from_previous if beat else ""),
                narrative_purpose=shot.narrative_purpose or (beat.narrative_purpose if beat else ""),
                conflict_or_tension=shot.conflict_or_tension or (beat.conflict_or_tension if beat else ""),
                turning_point=shot.turning_point or (beat.turning_point if beat else ""),
                sets_up_next=shot.sets_up_next or (beat.sets_up_next if beat else ""),
            )
        )
    return tuple(updated)


def build_movie_continuity_fallback(*, bible: MovieBible, shots: tuple[CinematicShot, ...]) -> MovieContinuityPlan:
    style_bible = MovieContinuityStyleBible(
        visual_style="; ".join(bible.style_constraints),
        negative_constraints=("visible text", "unmotivated wardrobe changes", "unexplained prop changes"),
    )
    characters = {
        actor.id: MovieContinuityCharacterState(
            character_id=actor.id,
            base_identity=actor.visual_description or actor.name,
            wardrobe=actor.visual_description or actor.name,
            emotional_state=actor.role,
        )
        for actor in bible.actors
    }
    locations = {
        location.id: MovieContinuityLocationState(
            location_id=location.id,
            name=location.name,
            environmental_state=location.visual_description,
        )
        for location in bible.locations
    }
    scene_continuity: dict[str, MovieSceneContinuityPacket] = {}
    narrative_chain: list[MovieNarrativeBeat] = []
    previous_outgoing: tuple[str, ...] = ()
    previous_after = bible.premise or bible.story_arch.premise
    for index, shot in enumerate(shots):
        shot_characters = {
            actor_id: replace(
                characters.get(actor_id)
                or MovieContinuityCharacterState(character_id=actor_id, base_identity=actor_id.replace("_", " ").title()),
                last_location=shot.location_id or shot.location,
                last_action=shot.action or shot.description,
                emotional_state=shot.expression,
            )
            for actor_id in shot.actor_ids
        }
        location = locations.get(shot.location_id) or MovieContinuityLocationState(
            location_id=shot.location_id or "primary_location",
            name=shot.location,
            environmental_state=shot.location,
        )
        carryovers = tuple(
            item
            for item in [
                *_safe_continuity_facts(rule.description for rule in bible.continuity if rule.description),
                *(f"{actor_id} identity and wardrobe remain consistent" for actor_id in shot.actor_ids),
                f"location remains {location.name or shot.location}" if location.name or shot.location else "",
            ]
            if item
        )
        outgoing = tuple(item for item in [shot.action or shot.description, shot.dialogue, shot.expression] if item)
        scene_continuity[shot.shot_id] = MovieSceneContinuityPacket(
            shot_id=shot.shot_id,
            location_id=shot.location_id,
            incoming=previous_outgoing,
            required_carryovers=carryovers,
            allowed_changes=tuple(item for item in [shot.action, shot.camera] if item),
            outgoing=outgoing,
            characters=shot_characters,
            location=location,
        )
        next_shot = shots[index + 1] if index + 1 < len(shots) else None
        state_before = previous_after or f"Before {shot.shot_id}, the story is ready for the next beat."
        state_after = shot.action or shot.description or f"{shot.shot_id} completes its story beat."
        narrative_chain.append(
            MovieNarrativeBeat(
                shot_id=shot.shot_id,
                story_state_before=state_before,
                story_state_after=state_after,
                cause_from_previous="Opening beat establishes the premise." if index == 0 else f"Previous beat leaves: {'; '.join(previous_outgoing) or previous_after}",
                narrative_purpose=shot.description,
                conflict_or_tension=shot.expression or "story tension continues",
                turning_point=shot.action or shot.description,
                sets_up_next=(next_shot.description if next_shot else "Final beat resolves the current movie arc."),
            )
        )
        previous_outgoing = outgoing
        previous_after = state_after
    return MovieContinuityPlan(
        continuity_ledger=MovieContinuityLedger(
            style_bible=style_bible,
            characters=characters,
            locations=locations,
            scene_order=tuple(shot.shot_id for shot in shots),
        ),
        scene_continuity=scene_continuity,
        narrative_chain=tuple(narrative_chain),
    )


def _safe_continuity_facts(value: object) -> tuple[str, ...]:
    candidates = _split_continuity_text(value)
    facts: list[str] = []
    for candidate in candidates:
        fact = " ".join(str(candidate or "").split()).strip(" .")
        if not fact or _looks_like_screenplay_dump(fact):
            continue
        if fact not in facts:
            facts.append(fact)
    return tuple(facts)


def _split_continuity_text(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[;\n]+", value) if part.strip()]
    if hasattr(value, "__iter__"):
        parts: list[str] = []
        for item in value:
            parts.extend(_split_continuity_text(item))
        return parts
    return [str(value).strip()] if str(value).strip() else []


def _looks_like_screenplay_dump(text: str) -> bool:
    if len(text) > 300:
        return True
    if _SCREENPLAY_HEADING_RE.search(text):
        return True
    return bool(_DIALOGUE_CUE_RE.search(text))


def normalize_movie_continuity_plan(plan: MovieContinuityPlan, *, bible: MovieBible, shots: tuple[CinematicShot, ...]) -> MovieContinuityPlan:
    fallback = build_movie_continuity_fallback(bible=bible, shots=shots)
    shot_ids = [shot.shot_id for shot in shots]
    return MovieContinuityPlan(
        continuity_ledger=plan.continuity_ledger or fallback.continuity_ledger,
        scene_continuity={shot_id: plan.scene_continuity.get(shot_id) or fallback.scene_continuity[shot_id] for shot_id in shot_ids},
        narrative_chain=tuple(_narrative_for_shot(shot_id, plan.narrative_chain, fallback.narrative_chain) for shot_id in shot_ids),
    )


def movie_continuity_plan_to_dict(plan: MovieContinuityPlan) -> dict:
    return asdict(plan)


def movie_continuity_plan_from_dict(data: dict, *, bible: MovieBible, shots: tuple[CinematicShot, ...]) -> MovieContinuityPlan:
    fallback = build_movie_continuity_fallback(bible=bible, shots=shots)
    ledger_data = data.get("continuity_ledger") or {}
    style_data = ledger_data.get("style_bible") or {}
    style_bible = MovieContinuityStyleBible(
        visual_style=str(style_data.get("visual_style") or fallback.continuity_ledger.style_bible.visual_style),
        palette=str(style_data.get("palette") or ""),
        lighting=str(style_data.get("lighting") or ""),
        camera=str(style_data.get("camera") or ""),
        negative_constraints=tuple(_string_list(style_data.get("negative_constraints"))),
    )
    characters = {
        character_id: _character_state_from_dict(character_id, value)
        for character_id, value in (ledger_data.get("characters") or {}).items()
        if isinstance(value, dict)
    } or fallback.continuity_ledger.characters
    locations = {
        location_id: _location_state_from_dict(location_id, value)
        for location_id, value in (ledger_data.get("locations") or {}).items()
        if isinstance(value, dict)
    } or fallback.continuity_ledger.locations
    scene_data = data.get("scene_continuity") or {}
    scene_continuity = {
        shot.shot_id: _scene_packet_from_dict(shot.shot_id, scene_data.get(shot.shot_id) or {}, fallback.scene_continuity[shot.shot_id])
        for shot in shots
    }
    narrative_data = data.get("narrative_chain") or []
    narrative_chain = tuple(
        _narrative_for_shot(
            shot.shot_id,
            tuple(_narrative_from_dict(item) for item in narrative_data if isinstance(item, dict)),
            fallback.narrative_chain,
        )
        for shot in shots
    )
    return MovieContinuityPlan(
        continuity_ledger=MovieContinuityLedger(
            style_bible=style_bible,
            characters=characters,
            locations=locations,
            scene_order=tuple(str(item) for item in ledger_data.get("scene_order") or fallback.continuity_ledger.scene_order),
        ),
        scene_continuity=scene_continuity,
        narrative_chain=narrative_chain,
    )


def _narrative_for_shot(shot_id: str, planned: tuple[MovieNarrativeBeat, ...], fallback: tuple[MovieNarrativeBeat, ...]) -> MovieNarrativeBeat:
    by_id = {beat.shot_id: beat for beat in planned}
    fallback_by_id = {beat.shot_id: beat for beat in fallback}
    candidate = by_id.get(shot_id)
    base = fallback_by_id[shot_id]
    if candidate is None:
        return base
    return MovieNarrativeBeat(
        shot_id=shot_id,
        story_state_before=candidate.story_state_before or base.story_state_before,
        story_state_after=candidate.story_state_after or base.story_state_after,
        cause_from_previous=candidate.cause_from_previous or base.cause_from_previous,
        narrative_purpose=candidate.narrative_purpose or base.narrative_purpose,
        conflict_or_tension=candidate.conflict_or_tension or base.conflict_or_tension,
        turning_point=candidate.turning_point or base.turning_point,
        sets_up_next=candidate.sets_up_next or base.sets_up_next,
    )


def _character_state_from_dict(character_id: str, data: dict) -> MovieContinuityCharacterState:
    return MovieContinuityCharacterState(
        character_id=str(data.get("character_id") or character_id),
        base_identity=str(data.get("base_identity") or ""),
        wardrobe=str(data.get("wardrobe") or data.get("base_identity") or ""),
        carried_props=tuple(_string_list(data.get("carried_props"))),
        physical_state=str(data.get("physical_state") or ""),
        emotional_state=str(data.get("emotional_state") or ""),
        last_location=str(data.get("last_location") or ""),
        last_action=str(data.get("last_action") or ""),
    )


def _location_state_from_dict(location_id: str, data: dict) -> MovieContinuityLocationState:
    return MovieContinuityLocationState(
        location_id=str(data.get("location_id") or location_id),
        name=str(data.get("name") or location_id),
        time_of_day=str(data.get("time_of_day") or ""),
        lighting=str(data.get("lighting") or ""),
        props=tuple(_string_list(data.get("props"))),
        environmental_state=str(data.get("environmental_state") or data.get("state") or ""),
    )


def _scene_packet_from_dict(shot_id: str, data: dict, fallback: MovieSceneContinuityPacket) -> MovieSceneContinuityPacket:
    location_data = data.get("location") if isinstance(data.get("location"), dict) else {}
    characters_data = data.get("characters") if isinstance(data.get("characters"), dict) else {}
    return MovieSceneContinuityPacket(
        shot_id=str(data.get("shot_id") or shot_id),
        location_id=str(data.get("location_id") or fallback.location_id),
        incoming=tuple(_string_list(data.get("incoming"))) or fallback.incoming,
        required_carryovers=tuple(_string_list(data.get("required_carryovers"))) or fallback.required_carryovers,
        allowed_changes=tuple(_string_list(data.get("allowed_changes"))) or fallback.allowed_changes,
        outgoing=tuple(_string_list(data.get("outgoing"))) or fallback.outgoing,
        characters={
            character_id: _character_state_from_dict(character_id, value)
            for character_id, value in characters_data.items()
            if isinstance(value, dict)
        }
        or fallback.characters,
        location=_location_state_from_dict(fallback.location_id, location_data) if location_data else fallback.location,
    )


def _narrative_from_dict(data: dict) -> MovieNarrativeBeat:
    return MovieNarrativeBeat(
        shot_id=str(data.get("shot_id") or ""),
        story_state_before=str(data.get("story_state_before") or ""),
        story_state_after=str(data.get("story_state_after") or ""),
        cause_from_previous=str(data.get("cause_from_previous") or ""),
        narrative_purpose=str(data.get("narrative_purpose") or ""),
        conflict_or_tension=str(data.get("conflict_or_tension") or ""),
        turning_point=str(data.get("turning_point") or ""),
        sets_up_next=str(data.get("sets_up_next") or ""),
    )


def constrain_movie_shots_to_bible(shots: tuple[CinematicShot, ...], bible: MovieBible) -> tuple[CinematicShot, ...]:
    actor_ids = [actor.id for actor in bible.actors]
    location_ids = [location.id for location in bible.locations]
    default_actor = actor_ids[0] if actor_ids else "main_character"
    default_location = location_ids[0] if location_ids else "primary_location"
    max_scene_actors = min(4, max(1, int(bible.runtime_constraints.get("max_scene_actors") or 4)))
    constrained = []
    for shot in shots:
        valid_actors = [actor_id for actor_id in shot.actor_ids if actor_id in actor_ids]
        if not valid_actors:
            valid_actors = [default_actor]
        location_id = shot.location_id if shot.location_id in location_ids else default_location
        constrained.append(
            replace(
                shot,
                actor_ids=tuple(dict.fromkeys(valid_actors[:max_scene_actors])),
                location_id=location_id,
                location=_location_name(bible, location_id),
            )
        )
    return tuple(constrained)


def augment_movie_bible_from_shot_references(bible: MovieBible, shots: tuple[CinematicShot, ...], *, config: dict) -> MovieBible:
    configured_actors = bool(_configured_movie_actors(config))
    configured_locations = bool(_configured_movie_locations(config))
    actors = list(bible.actors)
    locations = list(bible.locations)
    if not configured_actors:
        shot_actor_ids = []
        for shot in shots:
            for actor_id in shot.actor_ids:
                if actor_id and actor_id not in shot_actor_ids:
                    shot_actor_ids.append(actor_id)
        if shot_actor_ids and (len(actors) == 1 and actors[0].id == "main_character"):
            actors = []
        known_actor_ids = {actor.id for actor in actors}
        for index, actor_id in enumerate(shot_actor_ids, start=1):
            if actor_id not in known_actor_ids:
                actors.append(_generic_actor_from_id(actor_id, index))
                known_actor_ids.add(actor_id)
    if not configured_locations:
        shot_locations: dict[str, str] = {}
        for shot in shots:
            if shot.location_id and shot.location_id not in shot_locations:
                shot_locations[shot.location_id] = shot.location or shot.location_id.replace("_", " ").title()
        if shot_locations and (len(locations) == 1 and locations[0].id == "primary_location"):
            locations = []
        known_location_ids = {location.id for location in locations}
        for index, (location_id, name) in enumerate(shot_locations.items(), start=1):
            if location_id not in known_location_ids:
                locations.append(_generic_location_from_id(location_id, name, index))
                known_location_ids.add(location_id)
    return replace(bible, actors=tuple(actors), locations=tuple(locations))


def _bible_dict(bible: MovieBible) -> dict:
    return {
        "title": bible.title,
        "premise": bible.premise,
        "story_arch": asdict(bible.story_arch),
        "actors": [asdict(actor) for actor in bible.actors],
        "locations": [asdict(location) for location in bible.locations],
        "continuity": [asdict(rule) for rule in bible.continuity],
        "style_constraints": list(bible.style_constraints),
        "runtime_constraints": dict(bible.runtime_constraints),
    }


def movie_bible_from_dict(data: dict) -> MovieBible:
    story_data = data.get("story_arch") or {}
    story_arch = _story_arch_from_dict(story_data, title=str(data.get("title") or "Movie"), premise=str(data.get("premise") or ""))
    return MovieBible(
        title=str(data.get("title") or story_arch.title),
        premise=str(data.get("premise") or story_arch.premise),
        story_arch=story_arch,
        actors=tuple(_actor_from_dict(actor, index) for index, actor in enumerate(data.get("actors") or [], start=1) if isinstance(actor, dict)),
        locations=tuple(_location_from_dict(location, index) for index, location in enumerate(data.get("locations") or [], start=1) if isinstance(location, dict)),
        continuity=tuple(
            MovieContinuityRule(
                id=_safe_id(rule.get("id") or rule.get("description"), f"continuity_{index}"),
                description=str(rule.get("description") or "").strip(),
            )
            for index, rule in enumerate(data.get("continuity") or [], start=1)
            if isinstance(rule, dict)
        ),
        style_constraints=tuple(str(item).strip() for item in data.get("style_constraints") or [] if str(item).strip()),
        runtime_constraints=dict(data.get("runtime_constraints") or {}),
    )


def _normalize_movie_bible(bible: MovieBible, *, story_arch, config: dict, request: MovieInput) -> MovieBible:
    configured_actors = _configured_movie_actors(config)
    configured_locations = _configured_movie_locations(config)
    actors = tuple(configured_actors) if configured_actors else tuple(bible.actors) or (_default_movie_actor(request, 1),)
    locations = tuple(configured_locations) if configured_locations else tuple(bible.locations) or (_default_movie_location(request, 1),)
    runtime_constraints = _runtime_constraints(request, config)
    runtime_constraints.update(dict(bible.runtime_constraints or {}))
    if "max_scene_actors" in config:
        runtime_constraints["max_scene_actors"] = min(4, max(1, int(config.get("max_scene_actors") or 4)))
    return replace(
        bible,
        title=bible.title or request.name,
        premise=bible.premise or story_arch.premise,
        story_arch=story_arch,
        actors=actors,
        locations=locations,
        runtime_constraints=runtime_constraints,
    )


def _movie_bible_from_config(*, request: MovieInput, story_arch, config: dict) -> MovieBible:
    actors = tuple(_configured_movie_actors(config)) or (_default_movie_actor(request, 1),)
    locations = tuple(_configured_movie_locations(config)) or (_default_movie_location(request, 1),)
    continuity = (
        MovieContinuityRule(id="visual_continuity", description="Keep actor wardrobe, locations, lighting logic, and story geography consistent across shots."),
    )
    return MovieBible(
        title=story_arch.title,
        premise=story_arch.premise,
        story_arch=story_arch,
        actors=actors,
        locations=locations,
        continuity=continuity,
        style_constraints=_style_constraints(config),
        runtime_constraints=_runtime_constraints(request, config),
    )


def movie_config(movie: MovieProject) -> dict:
    return dict(movie.config or {})


def _planner_source_text(request: MovieInput, config: dict) -> str:
    parts = [request.story_text]
    for label, value in [
        ("story_idea", config.get("story_idea")),
        ("style", config.get("style")),
        ("subject", config.get("subject")),
        ("steering", config.get("steering")),
        ("prompt_guidance", config.get("prompt_guidance")),
        ("dialogue_language", config.get("dialogue_language")),
        ("actors", config.get("actors")),
        ("locations", config.get("locations")),
    ]:
        if value:
            parts.append(f"\n{label}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(parts).strip()


def _configured_movie_actors(config: dict) -> list[MovieActor]:
    configured = config.get("actors") if isinstance(config.get("actors"), list) else []
    actors = []
    for index, actor in enumerate(configured, start=1):
        if not isinstance(actor, dict):
            continue
        actors.append(_actor_from_dict(actor, index))
    return actors


def _configured_movie_locations(config: dict) -> list[MovieLocation]:
    raw_locations = config.get("structured_locations")
    if not isinstance(raw_locations, list) or not raw_locations:
        raw_locations = config.get("locations") if isinstance(config.get("locations"), list) else []
    locations = []
    for index, location in enumerate(raw_locations, start=1):
        if isinstance(location, dict):
            locations.append(_location_from_dict(location, index))
        elif str(location or "").strip():
            name = str(location).strip()
            locations.append(MovieLocation(id=_safe_id(name, f"location_{index}"), name=name, visual_description=name))
    return locations


def _actor_from_dict(actor: dict, index: int) -> MovieActor:
    actor_id = _safe_id(actor.get("id") or actor.get("name"), f"actor_{index}")
    name = str(actor.get("name") or actor.get("id") or f"Actor {index}").strip()
    visual_description = build_movie_actor_visual_description(
        str(actor.get("visual_description") or actor.get("image_prompt") or actor.get("prompt") or name).strip()
    )
    return MovieActor(
        id=actor_id,
        name=name,
        role=str(actor.get("role") or "").strip(),
        visual_description=visual_description,
    )


def _location_from_dict(location: dict, index: int) -> MovieLocation:
    location_id = _safe_id(location.get("id") or location.get("name"), f"location_{index}")
    name = str(location.get("name") or location.get("id") or f"Location {index}").strip()
    return MovieLocation(
        id=location_id,
        name=name,
        visual_description=str(location.get("visual_description") or location.get("image_prompt") or location.get("prompt") or name).strip(),
    )


def _default_movie_actor(request: MovieInput, index: int) -> MovieActor:
    subject = request.config.get("subject") if isinstance(request.config, dict) else None
    name = str(subject or "Main Character").strip()
    return MovieActor(
        id=_safe_id(name, "main_character"),
        name=name,
        role="lead",
        visual_description=f"{name}, story-defined cinematic character with consistent face, body shape, hair, wardrobe, and posture",
    )


def _default_movie_location(request: MovieInput, index: int) -> MovieLocation:
    location_name = "Primary Location"
    return MovieLocation(
        id=_safe_id(location_name, "primary_location"),
        name=location_name,
        visual_description="story-defined cinematic location with consistent production design, lighting, geography, and atmosphere",
    )


def _generic_actor_from_id(actor_id: str, index: int) -> MovieActor:
    name = actor_id.replace("_", " ").title()
    return MovieActor(
        id=_safe_id(actor_id, f"actor_{index}"),
        name=name,
        role="character",
        visual_description=f"{name}, story-defined cinematic character with consistent face, hair, body shape, wardrobe, and posture",
    )


def _generic_location_from_id(location_id: str, name: str, index: int) -> MovieLocation:
    display_name = str(name or location_id.replace("_", " ")).strip()
    return MovieLocation(
        id=_safe_id(location_id, f"location_{index}"),
        name=display_name,
        visual_description=f"{display_name}, story-defined cinematic location with consistent production design, geography, lighting, and atmosphere",
    )


def _style_constraints(config: dict) -> tuple[str, ...]:
    values = []
    for key in ("style", "prompt_guidance", "subject"):
        value = config.get(key)
        if value:
            values.append(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
    return tuple(values)


def _runtime_constraints(request: MovieInput, config: dict) -> dict:
    constraints = {
        "desired_length": float(request.desired_length),
        "width": int(request.width),
        "height": int(request.height),
        "max_scene_actors": min(4, max(1, int(config.get("max_scene_actors") or 4))),
    }
    dialogue_language = str(config.get("dialogue_language") or "").strip()
    if dialogue_language:
        constraints["dialogue_language"] = dialogue_language
    for key in ("fps", "width", "height"):
        if key in config:
            constraints[key] = int(config[key])
    return constraints


def _story_arch_from_dict(data: dict, *, title: str, premise: str):
    from feverslop.domain.movie import StoryArch

    return StoryArch(
        title=str(data.get("title") or title),
        premise=str(data.get("premise") or premise),
        beats=tuple(str(beat).strip() for beat in data.get("beats") or [] if str(beat).strip()),
    )


def _location_name(bible: MovieBible, location_id: str) -> str:
    for location in bible.locations:
        if location.id == location_id:
            return location.name
    return location_id.replace("_", " ").title()


def _movie_actor_refs(movie: MovieProject, config: dict) -> list[dict[str, str]]:
    configured = config.get("actors") if isinstance(config.get("actors"), list) else []
    actors = [
        _configured_actor_ref(actor, index)
        for index, actor in enumerate(configured, start=1)
        if isinstance(actor, dict)
    ]
    if actors:
        return actors
    names = []
    for shot in movie.shots:
        for actor_id in shot.actor_ids:
            if actor_id and actor_id not in names:
                names.append(actor_id)
        dialogue = str(shot.dialogue or "")
        speaker = dialogue.split(":", 1)[0].strip() if ":" in dialogue else ""
        if speaker and speaker not in names:
            names.append(speaker)
    if not names:
        names = [_movie_actor_name(movie)]
    actors = []
    for index, name in enumerate(names, start=1):
        display_name = str(name).replace("_", " ").title()
        actor_id = _safe_id(name, f"actor_{index}")
        visual_description = _actor_visual_description(_shots_for_actor(movie, actor_id))
        actors.append({
            "id": actor_id,
            "name": display_name,
            "visual_description": visual_description,
            "prompt": build_movie_actor_reference_prompt(display_name, visual_description),
        })
    return actors


def _configured_actor_ref(actor: dict, index: int) -> dict[str, str]:
    actor_id = _safe_id(actor.get("id") or actor.get("name"), f"actor_{index}")
    name = str(actor.get("name") or actor.get("id") or f"Actor {index}").strip()
    visual_description = build_movie_actor_visual_description(
        str(actor.get("visual_description") or actor.get("image_prompt") or actor.get("prompt") or name).strip()
    )
    return {
        "id": actor_id,
        "name": name,
        "visual_description": visual_description,
        "prompt": build_movie_actor_reference_prompt(name, visual_description),
    }


def _movie_location_refs(movie: MovieProject, config: dict) -> list[dict[str, str]]:
    configured = config.get("locations") if isinstance(config.get("locations"), list) else []
    locations = []
    for index, location in enumerate(configured, start=1):
        if isinstance(location, dict):
            name = str(location.get("name") or location.get("id") or f"Location {index}").strip()
            locations.append({
                "id": _safe_id(location.get("id") or name, f"location_{index}"),
                "name": name,
                "prompt": str(location.get("image_prompt") or location.get("visual_description") or name).strip(),
            })
        elif str(location or "").strip():
            name = str(location).strip()
            locations.append({"id": _safe_id(name, f"location_{index}"), "name": name, "prompt": name})
    if locations:
        return locations
    shot_locations: dict[str, str] = {}
    for shot in movie.shots:
        location_id = str(getattr(shot, "location_id", "") or "").strip()
        location_name = _display_name(str(getattr(shot, "location", "") or "").strip())
        if location_id and location_id not in shot_locations:
            shot_locations[location_id] = location_name or location_id.replace("_", " ").title()
    if shot_locations:
        return [
            {
                "id": _safe_id(location_id, f"location_{index}"),
                "name": name,
                "prompt": f"{name}, story-consistent cinematic environment, production design, lighting, and atmosphere",
            }
            for index, (location_id, name) in enumerate(shot_locations.items(), start=1)
        ]
    name = _movie_location_name(movie)
    return [{
        "id": _safe_id(name, "primary_location"),
        "name": name,
        "prompt": f"{name}, story-consistent cinematic environment, production design, lighting, and atmosphere",
    }]


def _default_actor_id(config: dict) -> str:
    actors = config.get("actors") if isinstance(config.get("actors"), list) else []
    if actors and isinstance(actors[0], dict):
        return _safe_id(actors[0].get("id") or actors[0].get("name"), "actor_1")
    return "main_character"


def _default_location_id(config: dict) -> str:
    locations = config.get("locations") if isinstance(config.get("locations"), list) else []
    if locations:
        first = locations[0]
        if isinstance(first, dict):
            return _safe_id(first.get("id") or first.get("name"), "location_1")
        return _safe_id(str(first), "location_1")
    return "primary_location"


def _safe_id(value: object, fallback: str) -> str:
    raw = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return raw or fallback


def _string_list(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _display_name(value: str) -> str:
    return value.title() if value.isupper() else value


def _movie_actor_name(movie: MovieProject) -> str:
    for shot in movie.shots:
        dialogue = str(getattr(shot, "dialogue", "") or "")
        speaker = dialogue.split(":", 1)[0].strip()
        if speaker:
            return speaker.title()
    return "Main Character"


def _movie_location_name(movie: MovieProject) -> str:
    for shot in movie.shots:
        location = str(getattr(shot, "location", "") or "").strip()
        if location and location != "story-consistent cinematic location":
            return location.title()
    return "Primary Location"


def _shots_for_actor(movie: MovieProject, actor_id: str) -> list:
    shots = [shot for shot in movie.shots if actor_id in getattr(shot, "actor_ids", ())]
    solo_shots = [shot for shot in shots if len(getattr(shot, "actor_ids", ())) == 1]
    return solo_shots or shots


def _actor_reference_prompt(name: str, shots: list) -> str:
    return build_movie_actor_reference_prompt(name, _actor_visual_description(shots))


def _actor_visual_description(shots: list) -> str:
    return build_movie_actor_visual_description(_actor_static_cues(shots))


def build_movie_actor_visual_description(cues: str) -> str:
    return _sanitize_actor_cues(cues)


def build_movie_actor_reference_prompt(name: str, cues: str = "") -> str:
    cue_text = _sanitize_actor_cues(cues)
    description = f" {cue_text}." if cue_text else ""
    return (
        f"Full-body cinematic character reference sheet for {name}.{description} "
        "Four vertical panels in one image: 1st panel head-and-shoulders closeup, "
        "2nd panel straight full-body front view, 3rd panel clean full-body left view, "
        "4th panel clean full-body back view. Consistent face, hair, body shape, wardrobe, "
        "posture, neutral expression, plain white seamless studio background, even reference-sheet lighting, "
        "no environment, no scenery, no props, no text, no extra characters."
    )


def _actor_static_cues(shots: list) -> str:
    parts: list[str] = []
    for shot in shots[:4]:
        for value in (getattr(shot, "description", ""), getattr(shot, "expression", "")):
            text = _sanitize_actor_cue_fragment(str(value or ""))
            if text and text not in parts:
                parts.append(text)
    return "; ".join(parts)[:700]


def _sanitize_actor_cues(cues: str) -> str:
    parts: list[str] = []
    cues = _strip_actor_prompt_boilerplate(str(cues or ""))
    for raw in re.split(r";|\.", cues):
        text = _sanitize_actor_cue_fragment(raw)
        if text and text not in parts:
            parts.append(text)
    return "; ".join(parts).strip(" ;.")


def _strip_actor_prompt_boilerplate(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"(?is)^.*?Full-body cinematic character reference sheet for [^.]+[.]\s*", "", text)
    text = re.sub(r"(?is)\bFour vertical panels in one image\b.*$", "", text)
    text = re.sub(r"(?is)\bConsistent face\b.*$", "", text)
    return text


def _sanitize_actor_cue_fragment(value: str) -> str:
    text = " ".join(str(value or "").split()).strip(" .;,")
    if not text:
        return ""
    lower = text.lower()
    if lower.endswith("'s") or lower in {"the man", "the woman", "the character"}:
        return ""
    if any(token in lower for token in ("jump cut", "shot", "close-up", "closeup", "camera", "tracking", "split-screen")):
        text = re.sub(r"(?i)^a\s+(?:sudden,\s+violent\s+)?jump cut to\s+", "", text)
        text = re.sub(r"(?i)^an?\s+[^.;,]*\bshot of\s+", "", text)
        text = re.sub(r"(?i)^extreme close-up of\s+", "", text)
        text = re.sub(r"(?i)^close-up of\s+", "", text)
        text = re.sub(r"(?i)^medium shot of\s+", "", text)
        text = re.sub(r"(?i)^wide shot of\s+", "", text)
    lower = text.lower()
    if any(
        token in lower
        for token in (
            "lunges",
            "bellows",
            "glides",
            "walks",
            "stumbles",
            "recoiling",
            "falls",
            "stands",
            "tearing through",
            "shaking",
            "appearing from",
            "eye fluttering",
            "eyes roll back",
            "screen fades",
            "enters a trance",
            "gaze",
            "mesmerized",
            "reaches",
            "leans",
            "breathing",
        )
    ):
        if any(token in lower for token in ("gaze", "mesmerized", "reaches")):
            return ""
        if "," not in text and " with " not in lower:
            return ""
        text = re.sub(r"(?i)\btearing through\b.*$", "", text)
        text = re.sub(r"(?i)\bappearing from\b.*$", "", text)
        text = re.sub(r"(?i)\bglides\b.*$", "", text)
        text = re.sub(r"(?i)\bbellows\b.*$", "", text)
        text = re.sub(r"(?i)\blunges\b.*$", "", text)
        text = re.sub(r"(?i)\beye fluttering\b.*$", "", text)
        text = re.sub(r"(?i)\beyes roll back\b.*$", "", text)
        text = re.sub(r"(?i)\bscreen fades\b.*$", "", text)
    text = text.strip(" .;,")
    if text.lower().endswith("'s") or text.lower() in {"the man", "the woman", "the character"}:
        return ""
    return text


def _shot_cues(shots: list, *, include_location: bool) -> str:
    parts: list[str] = []
    for shot in shots[:4]:
        for value in (
            getattr(shot, "description", ""),
            getattr(shot, "action", ""),
            getattr(shot, "expression", ""),
            getattr(shot, "location", "") if include_location else "",
        ):
            text = str(value or "").strip()
            if text and text not in parts:
                parts.append(text)
    return "; ".join(parts)[:700]
