"""Tests for frozen dataclass immutability enforcement (issue #258)."""
import types
import unittest

from feverslop.domain.movie import (
    CinematicShot,
    MovieBible,
    MovieProject,
    StoryArch,
)
from feverslop.domain.movie_continuity import (
    MovieContinuityCharacterState,
    MovieContinuityLedger,
    MovieContinuityLocationState,
    MovieContinuityStyleBible,
    MovieSceneContinuityPacket,
)
from feverslop.domain.render_plan import RenderPlan, RenderScene
from feverslop.domain.visual_consistency import (
    SCHEMA,
    ReferenceAnchor,
    SceneConsistencyContract,
    validate_scene_sequence,
)

ACTOR_HASH = "a" * 64


def make_reference_anchor() -> ReferenceAnchor:
    return ReferenceAnchor(
        id="hero",
        kind="actor",
        look_id="hero-look",
        asset_role="identity-reference",
        asset_sha256=ACTOR_HASH,
        prompt_anchor="hero wears a red jacket",
    )


class MovieSceneContinuityPacketImmutabilityTest(unittest.TestCase):
    def test_characters_dict_is_immutable_after_construction(self):
        packet = MovieSceneContinuityPacket(
            shot_id="s1",
            characters={
                "hero": MovieContinuityCharacterState(character_id="hero"),
            },
        )
        # Access should work
        self.assertIn("hero", packet.characters)
        # In-place mutation should fail
        with self.assertRaises(TypeError):
            packet.characters["villain"] = MovieContinuityCharacterState(character_id="villain")

    def test_characters_is_mapping_proxy_type(self):
        packet = MovieSceneContinuityPacket(
            shot_id="s1",
            characters={
                "hero": MovieContinuityCharacterState(character_id="hero"),
            },
        )
        self.assertIsInstance(packet.characters, types.MappingProxyType)

    def test_characters_none_is_allowed(self):
        packet = MovieSceneContinuityPacket(shot_id="s1")
        self.assertIsNone(packet.characters)


class MovieContinuityLedgerImmutabilityTest(unittest.TestCase):
    def test_characters_and_locations_dicts_are_immutable(self):
        ledger = MovieContinuityLedger(
            style_bible=MovieContinuityStyleBible(),
            characters={
                "hero": MovieContinuityCharacterState(character_id="hero"),
            },
            locations={
                "loc": MovieContinuityLocationState(location_id="loc"),
            },
            scene_order=("s1",),
        )
        with self.assertRaises(TypeError):
            ledger.characters["new"] = MovieContinuityCharacterState(character_id="new")
        with self.assertRaises(TypeError):
            ledger.locations["new"] = MovieContinuityLocationState(location_id="new")

    def test_ledger_dicts_are_mapping_proxy_type(self):
        ledger = MovieContinuityLedger(
            style_bible=MovieContinuityStyleBible(),
            characters={"hero": MovieContinuityCharacterState(character_id="hero")},
            locations={"loc": MovieContinuityLocationState(location_id="loc")},
            scene_order=("s1",),
        )
        self.assertIsInstance(ledger.characters, types.MappingProxyType)
        self.assertIsInstance(ledger.locations, types.MappingProxyType)


class MovieProjectConfigImmutabilityTest(unittest.TestCase):
    def _make_project(self, config=None):
        story_arch = StoryArch(title="Test", premise="Test", beats=("beat",))
        bible = MovieBible(
            title="Test",
            premise="Test",
            story_arch=story_arch,
            actors=(),
            locations=(),
            continuity=(),
            style_constraints=(),
            runtime_constraints={},
        )
        return MovieProject(
            slug="test",
            name="Test",
            bible=bible,
            story_arch=story_arch,
            shots=(
                CinematicShot(
                    shot_id="s1",
                    description="Test shot",
                    duration_seconds=5.0,
                    camera="static",
                    action="test",
                    expression="neutral",
                    location="Test location",
                ),
            ),
            duration_seconds=12.0,
            width=1280,
            height=704,
            mode="scaffold",
            config=config,
        )

    def test_config_is_mapping_proxy_type(self):
        project = self._make_project(config={"a": 1})
        self.assertIsInstance(project.config, types.MappingProxyType)

    def test_config_none_is_allowed(self):
        project = self._make_project()
        self.assertIsNone(project.config)

    def test_config_inplace_mutation_fails(self):
        project = self._make_project(config={"a": 1})
        with self.assertRaises(TypeError):
            project.config["x"] = 1

    def test_movie_config_helper_returns_plain_dict_copy(self):
        from feverslop.application.movie_bible import movie_config

        project = self._make_project(config={"a": 1})
        result = movie_config(project)
        self.assertIsInstance(result, dict)
        result["x"] = 2
        self.assertNotIn("x", project.config)


class RenderPlanImmutabilityTest(unittest.TestCase):
    def test_scenes_is_tuple(self):
        plan = RenderPlan.from_dicts([{"scene": 1}, {"scene": 2}])
        self.assertIsInstance(plan.scenes, tuple)
        self.assertEqual(len(plan.scenes), 2)

    def test_scenes_cannot_be_mutated(self):
        plan = RenderPlan.from_dicts([{"scene": 1}])
        with self.assertRaises(AttributeError):
            plan.scenes.append(RenderScene(data={"scene": 2}))

    def test_select_returns_new_plan_with_tuple(self):
        plan = RenderPlan.from_dicts([{"scene": 1}, {"scene": 2}, {"scene": 3}])
        selected = plan.select(scene_numbers={2})
        self.assertIsInstance(selected.scenes, tuple)
        self.assertEqual([s.scene_number for s in selected.scenes], [2])


class SceneConsistencyContractConstructorTest(unittest.TestCase):
    def test_direct_construction_from_list_actors_succeeds(self):
        anchor = make_reference_anchor()
        # Compute the fingerprint that create() would produce
        created = SceneConsistencyContract.create(
            scene=1,
            mode="msr",
            workflow_profile="test-profile",
            actors=(anchor,),
            location=None,
            transition_from_previous="cut",
        )
        # Direct construction with a list (not a tuple) should work
        direct = SceneConsistencyContract(
            schema=SCHEMA,
            scene=1,
            mode="msr",
            workflow_profile="test-profile",
            actors=[anchor],  # list, not tuple
            location=None,
            transition_from_previous="cut",
            fingerprint=created.fingerprint,
        )
        self.assertIsInstance(direct.actors, tuple)
        self.assertEqual(direct, created)

    def test_direct_construction_from_tuple_still_works(self):
        anchor = make_reference_anchor()
        created = SceneConsistencyContract.create(
            scene=1,
            mode="msr",
            workflow_profile="test-profile",
            actors=(anchor,),
            location=None,
            transition_from_previous="cut",
        )
        direct = SceneConsistencyContract(
            schema=SCHEMA,
            scene=1,
            mode="msr",
            workflow_profile="test-profile",
            actors=(anchor,),  # tuple
            location=None,
            transition_from_previous="cut",
            fingerprint=created.fingerprint,
        )
        self.assertEqual(direct, created)


class ValidateSceneSequenceDocTest(unittest.TestCase):
    def test_rejects_non_positive_with_clear_message(self):
        with self.assertRaisesRegex(ValueError, "positive integers in consecutive order"):
            validate_scene_sequence([{"scene": -1}, {"scene": 2}])

    def test_rejects_gaps_with_clear_message(self):
        with self.assertRaisesRegex(ValueError, "consecutive order without duplicates or gaps"):
            validate_scene_sequence([{"scene": 1}, {"scene": 3}])

    def test_accepts_subset_sequence(self):
        """Subsets are valid — function does not enforce 1-based start."""
        result = validate_scene_sequence([{"scene": 5}, {"scene": 6}, {"scene": 7}])
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
