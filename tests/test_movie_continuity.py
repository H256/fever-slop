"""Tests for MovieContinuityPlan.normalize() ledger fallback merge (issue #1328)."""
import unittest

from feverslop.domain.movie import CinematicShot
from feverslop.domain.movie_continuity import (
    MovieContinuityLedger,
    MovieContinuityPlan,
    MovieContinuityStyleBible,
)
from feverslop.domain.movie_references import (
    MovieActor,
    MovieBible,
    MovieLocation,
    StoryArch,
)


def _bible() -> MovieBible:
    return MovieBible(
        title="Test",
        premise="A test premise.",
        story_arch=StoryArch(title="Test", premise="A test premise.", beats=("beat",)),
        actors=(
            MovieActor(id="hero", name="Hero", role="protagonist", visual_description="tall hero"),
            MovieActor(id="villain", name="Villain", role="antagonist", visual_description="dark villain"),
        ),
        locations=(
            MovieLocation(id="loc_a", name="Location A", visual_description="bright room"),
        ),
        continuity=(),
        style_constraints=("no visible text",),
        runtime_constraints={},
    )


def _shots() -> tuple[CinematicShot, ...]:
    return (
        CinematicShot(
            shot_id="s1",
            description="Hero enters.",
            duration_seconds=5.0,
            camera="static",
            action="enter",
            expression="neutral",
            location="Location A",
            actor_ids=("hero",),
            location_id="loc_a",
        ),
        CinematicShot(
            shot_id="s2",
            description="Villain arrives.",
            duration_seconds=5.0,
            camera="static",
            action="arrive",
            expression="tense",
            location="Location A",
            actor_ids=("villain",),
            location_id="loc_a",
        ),
    )


def _empty_plan() -> MovieContinuityPlan:
    """A partial plan whose ledger sections are all empty (the bug scenario)."""
    return MovieContinuityPlan(
        continuity_ledger=MovieContinuityLedger(
            style_bible=MovieContinuityStyleBible(),
            characters={},
            locations={},
            scene_order=(),
        ),
        scene_continuity={},
        narrative_chain=(),
    )


class NormalizeLedgerFallbackTest(unittest.TestCase):
    def test_empty_sections_are_filled_from_fallback(self):
        plan = _empty_plan()
        result = plan.normalize(bible=_bible(), shots=_shots())

        self.assertIn("hero", result.continuity_ledger.characters)
        self.assertIn("villain", result.continuity_ledger.characters)
        self.assertIn("loc_a", result.continuity_ledger.locations)
        self.assertEqual(result.continuity_ledger.scene_order, ("s1", "s2"))
        # Fallback-derived character state is present.
        self.assertEqual(result.continuity_ledger.characters["hero"].wardrobe, "tall hero")
        # Style bible is filled from the fallback (bible style constraints).
        self.assertEqual(result.continuity_ledger.style_bible.visual_style, "no visible text")

    def test_present_values_take_priority(self):
        from feverslop.domain.movie_continuity import MovieContinuityCharacterState

        custom = MovieContinuityCharacterState(
            character_id="hero", base_identity="custom identity", wardrobe="custom wardrobe"
        )
        plan = MovieContinuityPlan(
            continuity_ledger=MovieContinuityLedger(
                style_bible=MovieContinuityStyleBible(visual_style="planned style"),
                characters={"hero": custom},
                locations={},
                scene_order=("s1",),
            ),
            scene_continuity={},
            narrative_chain=(),
        )
        result = plan.normalize(bible=_bible(), shots=_shots())

        # Present character value wins over the fallback.
        self.assertEqual(result.continuity_ledger.characters["hero"].wardrobe, "custom wardrobe")
        # Present style value wins over the fallback.
        self.assertEqual(result.continuity_ledger.style_bible.visual_style, "planned style")
        # Empty location section is still filled from the fallback.
        self.assertIn("loc_a", result.continuity_ledger.locations)
        # Present scene_order is kept (not overwritten by the fallback's longer order).
        self.assertEqual(result.continuity_ledger.scene_order, ("s1",))


if __name__ == "__main__":
    unittest.main()
