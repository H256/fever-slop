"""Regression tests for movie scene/shot card building (issue #1181).

When the shot planner produces fewer shots than screenplay scenes, scene
cards beyond the last shot must stay visible but unbound (empty shot_ids)
instead of being silently coupled to the last shot.
"""

import unittest

from feverslop.application.movie_memory import (
    build_movie_scene_cards,
    build_movie_shot_cards,
    movie_scene_cards_to_dict,
)
from feverslop.domain.movie import CinematicShot, MovieScreenplayArtifact, MovieScreenplayScene


def _scene(index: int) -> MovieScreenplayScene:
    return MovieScreenplayScene(
        scene_id=f"scene_{index:04d}",
        heading=f"Scene {index} heading",
        summary=f"Scene {index} summary",
        action=f"Scene {index} action",
        dialogue="",
        actor_ids=(),
        location_id="",
    )


def _shot(index: int) -> CinematicShot:
    return CinematicShot(
        shot_id=f"shot_{index:04d}",
        description=f"Shot {index} description",
        duration_seconds=5.0,
        camera="wide",
        action=f"Shot {index} action",
        expression="neutral",
        location="stage",
    )


def _screenplay(scene_count: int) -> MovieScreenplayArtifact:
    return MovieScreenplayArtifact(
        title="Test Movie",
        source_type="short_story",
        dialogue_language="en",
        scenes=tuple(_scene(index) for index in range(1, scene_count + 1)),
    )


class MovieSceneCardCouplingTests(unittest.TestCase):
    def test_fewer_shots_than_scenes_leaves_tail_scenes_unbound(self):
        """The reported failure: 10 scenes / 3 shots.

        Pre-fix, scenes 4-10 all claimed shot_0003; post-fix they carry an
        empty shot_ids tuple so the missing assignment is visible, not wrong.
        """
        cards = build_movie_scene_cards(screenplay=_screenplay(10), shots=tuple(_shot(i) for i in range(1, 4)))

        self.assertEqual(10, len(cards))
        self.assertEqual(("shot_0001",), cards[0].shot_ids)
        self.assertEqual(("shot_0002",), cards[1].shot_ids)
        self.assertEqual(("shot_0003",), cards[2].shot_ids)
        for index in range(3, 10):
            self.assertEqual((), cards[index].shot_ids, f"scene {index + 1} must not be bound to a wrong shot")
        # Exactly one scene card per shot, and no shot id appears twice.
        self.assertEqual(3, sum(1 for card in cards if card.shot_ids))
        all_shot_ids = tuple(shot_id for card in cards for shot_id in card.shot_ids)
        self.assertEqual(len(all_shot_ids), len(set(all_shot_ids)))

    def test_shot_cards_do_not_reuse_the_last_scene_for_unbound_shots(self):
        """shot_cards scene lookup must resolve each shot to its own scene."""
        cards = build_movie_scene_cards(screenplay=_screenplay(10), shots=tuple(_shot(i) for i in range(1, 4)))
        shot_cards = build_movie_shot_cards(shots=tuple(_shot(i) for i in range(1, 4)), scene_cards=cards)

        self.assertEqual("scene_0001", shot_cards[0].scene_id)
        self.assertEqual("scene_0002", shot_cards[1].scene_id)
        self.assertEqual("scene_0003", shot_cards[2].scene_id)

    def test_shot_ids_roundtrip_through_scene_cards_dict(self):
        """The on-disk scene_cards.json shape must preserve empty shot_ids."""
        cards = build_movie_scene_cards(screenplay=_screenplay(3), shots=tuple(_shot(i) for i in range(1, 2)))
        data = movie_scene_cards_to_dict(cards)

        self.assertEqual((("shot_0001",), (), ()), tuple(entry["shot_ids"] for entry in data["scene_cards"]))

    def test_one_shot_per_scene_still_binds_positionally(self):
        cards = build_movie_scene_cards(screenplay=_screenplay(4), shots=tuple(_shot(i) for i in range(1, 5)))

        for index, card in enumerate(cards, start=1):
            self.assertEqual((f"shot_{index:04d}",), card.shot_ids)

    def test_more_shots_than_scenes_binds_only_available_scenes(self):
        cards = build_movie_scene_cards(screenplay=_screenplay(2), shots=tuple(_shot(i) for i in range(1, 6)))

        self.assertEqual(2, len(cards))
        self.assertEqual(("shot_0001",), cards[0].shot_ids)
        self.assertEqual(("shot_0002",), cards[1].shot_ids)

    def test_no_shots_keeps_scene_cards_unbound(self):
        cards = build_movie_scene_cards(screenplay=_screenplay(2), shots=())

        for card in cards:
            self.assertEqual((), card.shot_ids)
        self.assertEqual("Scene 1 action", cards[0].story_state_after)


if __name__ == "__main__":
    unittest.main()
