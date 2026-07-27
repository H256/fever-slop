from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoryArch:
    title: str
    premise: str
    beats: tuple[str, ...]


@dataclass(frozen=True)
class MovieActor:
    id: str
    name: str
    role: str = ""
    visual_description: str = ""


@dataclass(frozen=True)
class MovieLocation:
    id: str
    name: str
    visual_description: str = ""
    image_prompt: str = ""


@dataclass(frozen=True)
class MovieContinuityRule:
    id: str
    description: str


@dataclass(frozen=True)
class MovieBible:
    title: str
    premise: str
    story_arch: StoryArch
    actors: tuple[MovieActor, ...]
    locations: tuple[MovieLocation, ...]
    continuity: tuple[MovieContinuityRule, ...]
    style_constraints: tuple[str, ...]
    runtime_constraints: dict

    def constrain(self, shots: tuple) -> tuple:
        """Validate and constrain shot actor_ids / location_ids against this bible.

        Returns a new tuple of shots with actor_ids filtered to known actors,
        location_ids clamped to known locations, max actors enforced, and
        location names resolved.
        """
        from dataclasses import replace

        actor_ids = [actor.id for actor in self.actors]
        location_ids = [location.id for location in self.locations]
        default_actor = actor_ids[0] if actor_ids else "main_character"
        default_location = location_ids[0] if location_ids else "primary_location"
        max_scene_actors = min(4, max(1, int(self.runtime_constraints.get("max_scene_actors") or 4)))
        constrained = []
        for shot in shots:
            valid_actors = [aid for aid in shot.actor_ids if aid in actor_ids]
            if not valid_actors:
                valid_actors = [default_actor]
            loc_id = shot.location_id if shot.location_id in location_ids else default_location
            loc_name = next((loc.name for loc in self.locations if loc.id == loc_id), loc_id.replace("_", " ").title())
            constrained.append(
                replace(
                    shot,
                    actor_ids=tuple(dict.fromkeys(valid_actors[:max_scene_actors])),
                    location_id=loc_id,
                    location=loc_name,
                )
            )
        return tuple(constrained)

    def augment_from_shots(self, shots: tuple, *, actors: bool = True, locations: bool = True) -> MovieBible:
        """Return a new bible with actors/locations discovered from shots.

        Actors are only added when *actors* is True and the bible contains only
        a single placeholder ("main_character"). Same logic applies for locations
        with the single placeholder "primary_location".
        """
        from dataclasses import replace

        new_actors = list(self.actors)
        new_locations = list(self.locations)

        if actors:
            shot_actor_ids = []
            for shot in shots:
                for actor_id in shot.actor_ids:
                    if actor_id and actor_id not in shot_actor_ids:
                        shot_actor_ids.append(actor_id)
            if shot_actor_ids and len(new_actors) == 1 and new_actors[0].id == "main_character":
                new_actors = []
            known_actor_ids = {actor.id for actor in new_actors}
            for index, actor_id in enumerate(shot_actor_ids, start=1):
                if actor_id not in known_actor_ids:
                    name = actor_id.replace("_", " ").title()
                    new_actors.append(MovieActor(id=actor_id, name=name, role="character", visual_description=name))
                    known_actor_ids.add(actor_id)

        if locations:
            shot_locations: dict[str, str] = {}
            for shot in shots:
                if shot.location_id and shot.location_id not in shot_locations:
                    shot_locations[shot.location_id] = shot.location or shot.location_id.replace("_", " ").title()
            if shot_locations and len(new_locations) == 1 and new_locations[0].id == "primary_location":
                new_locations = []
            known_location_ids = {location.id for location in new_locations}
            for location_id, name in shot_locations.items():
                if location_id not in known_location_ids:
                    new_locations.append(MovieLocation(id=location_id, name=name, visual_description=name))
                    known_location_ids.add(location_id)

        return replace(self, actors=tuple(new_actors), locations=tuple(new_locations))
