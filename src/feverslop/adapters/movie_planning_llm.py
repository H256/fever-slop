from __future__ import annotations

from dataclasses import replace

from feverslop.errors import FeverSlopLMLError
from feverslop.domain.movie import CinematicShot, MovieActor, MovieBible, MovieLocation, MovieScreenplayArtifact, StoryArch
from feverslop.adapters.movie_planning_helpers import (
    _beat_text,
    _ensure_minimum_actors,
    _normalize_movie_shots,
    _safe_id,
    _shots_from_data,
    _string_list,
    _transition_from_previous,
)
from feverslop.adapters.movie_planning_bible import _movie_bible_from_data
from feverslop.adapters.movie_planning_prompts import (
    _krea_reference_guides,
    _sanitize_location_image_prompt,
)
from feverslop.prompting.movie_planning_modules import MoviePlanningModules
import logging


logger = logging.getLogger(__name__)


def _module_data(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Movie planning predictor returned unsupported output: {type(value).__name__}")


class LLMMoviePlanner:
    def __init__(self, llm, *, reference_hero_workflow: str | None = None):
        self.llm = llm
        self.reference_hero_workflow = reference_hero_workflow
        self._modules = MoviePlanningModules(llm)

    def generate_story_arch(self, *, title: str, source_type: str, story_text: str, desired_length: float) -> StoryArch:
        data = _module_data(self._modules.story_arch({
            "title": title, "source_type": source_type, "story_text": story_text, "desired_length": desired_length,
        }))
        beats = data.get("beats") or []
        return StoryArch(
            title=str(data.get("title") or title),
            premise=str(data.get("premise") or story_text).strip(),
            beats=tuple(_beat_text(beat) for beat in beats if _beat_text(beat)),
        )

    def generate_movie_bible(self, *, title: str, source_type: str, story_text: str, desired_length: float, story_arch: StoryArch, config: dict) -> MovieBible:
        data = _module_data(self._modules.movie_bible({
            "title": title, "source_type": source_type, "story_text": story_text, "desired_length": desired_length,
            "story_arch": story_arch, "config": config,
        }))
        bible = _movie_bible_from_data(
            data,
            title=title,
            source_type=source_type,
            source_text=story_text,
            story_arch=story_arch,
            config=config,
            desired_length=desired_length,
        )
        if config.get("refine_location_prompts"):
            refined = self.refine_locations(bible.locations, source_text=story_text)
            bible = replace(bible, locations=tuple(refined))
        if config.get("refine_actor_prompts"):
            refined = self.refine_actors(bible.actors, source_text=story_text, premise=bible.premise)
            bible = replace(bible, actors=tuple(refined))
        return bible

    def refine_locations(self, locations: tuple[MovieLocation, ...], *, source_text: str) -> list[MovieLocation]:
        location_guide, _ = _krea_reference_guides(self.reference_hero_workflow)
        try:
            data = _module_data(self._modules.refine_locations({
                "locations": locations, "source_text": source_text, "guide": location_guide,
            }))
        except (FeverSlopLMLError, ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Location prompt refinement failed; using original locations", exc_info=exc)
            return list(locations)
        refined_by_id: dict[str, dict] = {}
        for item in data.get("locations") or []:
            if isinstance(item, dict) and item.get("id"):
                refined_by_id[str(item["id"])] = item
        result: list[MovieLocation] = []
        for loc in locations:
            refined = refined_by_id.get(loc.id)
            if refined:
                result.append(
                    MovieLocation(
                        id=loc.id,
                        name=loc.name,
                        visual_description=str(refined.get("visual_description") or loc.visual_description).strip(),
                        image_prompt=_sanitize_location_image_prompt(
                            str(refined.get("image_prompt") or loc.visual_description)
                        ),
                    )
                )
            else:
                result.append(loc)
        return result

    def refine_actors(self, actors: tuple[MovieActor, ...], *, source_text: str, premise: str) -> list[MovieActor]:
        _, actor_guide = _krea_reference_guides(self.reference_hero_workflow)
        try:
            data = _module_data(self._modules.refine_actors({
                "actors": actors, "source_text": source_text, "premise": premise, "guide": actor_guide,
            }))
        except (FeverSlopLMLError, ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Actor prompt refinement failed; using original actors", exc_info=exc)
            return list(actors)
        refined_by_id: dict[str, dict] = {}
        for item in data.get("actors") or []:
            if isinstance(item, dict) and item.get("id"):
                refined_by_id[str(item["id"])] = item
        result: list[MovieActor] = []
        for actor in actors:
            refined = refined_by_id.get(actor.id)
            if refined:
                result.append(
                    MovieActor(
                        id=actor.id,
                        name=actor.name,
                        role=actor.role,
                        visual_description=str(refined.get("visual_description") or actor.visual_description).strip(),
                    )
                )
            else:
                result.append(actor)
        return result

    def generate_movie_continuity_plan(self, *, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, shots: tuple[CinematicShot, ...], config: dict) -> dict:
        return _module_data(self._modules.continuity_plan({
            "title": title, "source_type": source_type, "story_text": story_text, "desired_length": desired_length,
            "bible": bible, "shots": shots, "config": config,
        }))

    def generate_movie_story_design(self, *, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, story_arch: StoryArch, config: dict) -> dict:
        return _module_data(self._modules.story_design({
            "title": title, "source_type": source_type, "story_text": story_text, "desired_length": desired_length,
            "bible": bible, "story_arch": story_arch, "config": config,
        }))

    def generate_movie_screenplay(self, *, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, story_arch: StoryArch, story_design, config: dict) -> dict:
        return _module_data(self._modules.screenplay({
            "title": title, "source_type": source_type, "story_text": story_text, "desired_length": desired_length,
            "bible": bible, "story_arch": story_arch, "story_design": story_design, "config": config,
        }))

    def generate_movie_narrative_plan(self, *, title: str, source_type: str, desired_length: float, bible: MovieBible, screenplay, config: dict) -> dict:
        return _module_data(self._modules.narrative_plan({
            "title": title, "source_type": source_type, "desired_length": desired_length,
            "bible": bible, "screenplay": screenplay, "config": config,
        }))

    def plan_shots_from_bible(
        self,
        *,
        bible: MovieBible,
        screenplay: MovieScreenplayArtifact | None = None,
        desired_length: float,
        width: int,
        height: int,
        min_duration: float = 4.0,
        max_duration: float = 20.0,
    ) -> tuple[CinematicShot, ...]:
        data = _module_data(self._modules.shot_plan_from_bible({
            "bible": bible, "screenplay": screenplay, "desired_length": desired_length,
            "width": width, "height": height, "min_duration": min_duration, "max_duration": max_duration,
        }))
        shots = data.get("shots") or []
        if not isinstance(shots, list) or not shots:
            from feverslop.adapters.movie_planning_deterministic import DeterministicMoviePlanner
            return DeterministicMoviePlanner().plan_shots_from_bible(
                bible=bible,
                screenplay=screenplay,
                desired_length=desired_length,
                width=width,
                height=height,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        return _shots_from_data(
            shots,
            desired_length=desired_length,
            min_duration=min_duration,
            max_duration=max_duration,
        )

    def plan_shots(
        self,
        *,
        story_arch: StoryArch,
        desired_length: float,
        width: int,
        height: int,
        min_duration: float = 4.0,
        max_duration: float = 20.0,
    ) -> tuple[CinematicShot, ...]:
        data = _module_data(self._modules.shot_plan({
            "story_arch": story_arch, "desired_length": desired_length, "width": width, "height": height,
            "min_duration": min_duration, "max_duration": max_duration,
        }))
        shots = data.get("shots") or []
        if not isinstance(shots, list) or not shots:
            from feverslop.adapters.movie_planning_deterministic import DeterministicMoviePlanner
            return DeterministicMoviePlanner().plan_shots(
                story_arch=story_arch,
                desired_length=desired_length,
                width=width,
                height=height,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        duration = max(1.0, float(desired_length) / len(shots))
        planned = []
        for index, raw_shot in enumerate(shots, start=1):
            shot = raw_shot if isinstance(raw_shot, dict) else {"description": str(raw_shot)}
            planned.append(
                CinematicShot(
                    shot_id=str(shot.get("shot_id") or f"shot_{index:04}"),
                    description=str(shot.get("description") or shot.get("action") or f"Shot {index}").strip(),
                    duration_seconds=float(shot.get("duration_seconds") or duration),
                    camera=str(shot.get("camera") or "motivated cinematic camera movement").strip(),
                    action=str(shot.get("action") or shot.get("description") or "").strip(),
                    expression=str(shot.get("expression") or "emotionally grounded performance").strip(),
                    location=str(shot.get("location") or "story-consistent cinematic location").strip(),
                    dialogue=str(shot.get("dialogue") or "").strip(),
                    actor_ids=tuple(_string_list(shot.get("actor_ids") or shot.get("actors"))),
                    location_id=_safe_id(shot.get("location_id") or shot.get("location")),
                    transition_from_previous=_transition_from_previous(shot.get("transition_from_previous")),
                )
            )
        planned = _ensure_minimum_actors(planned, story_arch)
        return _normalize_movie_shots(
            planned,
            desired_length=float(desired_length),
            min_duration=float(min_duration),
            max_duration=float(max_duration),
        )
