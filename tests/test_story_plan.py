"""Tests for the strict canonical StoryPlan contract (issue #1384)."""

import copy
import hashlib
import unittest

from pydantic import ValidationError

from feverslop.domain.story_plan import (
    STORY_PLAN_SCHEMA_VERSION,
    CreativeOverride,
    PlanProvenance,
    SegmentBrief,
    StoryMode,
    StoryPlan,
    VocalPresentation,
    apply_creative_override,
    resolve_singer_conflict,
)
from feverslop.errors import FeverSlopDataError


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _music_video_plan() -> dict:
    return {
        "mode": "music_video",
        "source_fingerprint": _sha("source"),
        "provenance": {"producer": "test-planner", "source_refs": ["song:demo"]},
        "characters": [
            {"id": "char-singer", "name": "Singer", "is_singer": True},
            {"id": "char-dancer", "name": "Dancer"},
        ],
        "locations": [{"id": "loc-stage", "name": "Stage"}],
        "props": [{"id": "prop-mic", "name": "Microphone"}],
        "beats": [
            {
                "id": "beat-open",
                "phase": "opening",
                "description": "Cold open",
                "character_ids": ["char-singer"],
                "location_id": "loc-stage",
            },
            {"id": "beat-dev", "phase": "development", "description": "Build-up"},
            {"id": "beat-res", "phase": "resolution", "description": "Final chorus"},
        ],
        "arcs": [
            {
                "id": "arc-singer",
                "character_id": "char-singer",
                "beat_ids": ["beat-open", "beat-res"],
            }
        ],
        "segments": [
            {
                "id": "brief-1",
                "target": "seg-1",
                "beat_id": "beat-open",
                "vocal_presentation": "on_screen",
                "audio_ref": {"segment_id": "seg-1", "fingerprint": _sha("seg-1")},
            },
            {
                "id": "brief-2",
                "target": "seg-2",
                "beat_id": "beat-res",
                "vocal_presentation": "offscreen",
                "audio_ref": {"segment_id": "seg-2", "fingerprint": _sha("seg-2")},
            },
        ],
    }


def _build(payload: dict) -> StoryPlan:
    return StoryPlan.model_validate(payload, strict=False)


class StoryPlanConstructionTests(unittest.TestCase):
    def test_valid_music_video_plan(self):
        plan = _build(_music_video_plan())
        self.assertEqual(plan.mode, StoryMode.music_video)
        self.assertEqual(plan.schema_version, STORY_PLAN_SCHEMA_VERSION)
        self.assertEqual(len(plan.characters), 2)
        self.assertEqual(plan.characters[0].is_singer, True)
        self.assertEqual(len(plan.segments), 2)

    def test_valid_narrative_film_plan(self):
        payload = _music_video_plan()
        payload["mode"] = "narrative_film"
        payload["segments"] = [
            {"id": "shot-1", "target": "shot-1", "beat_id": "beat-open"},
            {"id": "shot-2", "target": "shot-2", "beat_id": "beat-res"},
        ]
        plan = _build(payload)
        self.assertEqual(plan.mode, StoryMode.narrative_film)
        self.assertIsNone(plan.segments[0].audio_ref)

    def test_rejects_unsupported_schema_version(self):
        payload = _music_video_plan()
        payload["schema_version"] = "story-plan/v999"
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_non_sha256_source_fingerprint(self):
        payload = _music_video_plan()
        payload["source_fingerprint"] = "xyz"
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_extra_fields(self):
        payload = _music_video_plan()
        payload["surprise"] = "value"
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_non_strict_types(self):
        with self.assertRaises(ValidationError):
            StoryPlan(
                mode=123,
                source_fingerprint=_sha("source"),
                provenance=PlanProvenance(producer="test-planner"),
            )
        with self.assertRaises(ValidationError):
            StoryPlan(
                mode="music_video",
                source_fingerprint=_sha("source"),
                provenance=PlanProvenance(producer="test-planner"),
            )

    def test_validation_does_not_mutate_input(self):
        payload = _music_video_plan()
        snapshot = copy.deepcopy(payload)
        _build(payload)
        self.assertEqual(payload, snapshot)


class StoryPlanReferenceTests(unittest.TestCase):
    def test_rejects_duplicate_character_ids(self):
        payload = _music_video_plan()
        payload["characters"].append({"id": "char-singer", "name": "Twin"})
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_duplicate_beat_ids(self):
        payload = _music_video_plan()
        payload["beats"].append(
            {"id": "beat-open", "phase": "development", "description": "Extra"}
        )
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_duplicate_segment_ids(self):
        payload = _music_video_plan()
        payload["segments"].append(
            {
                "id": "brief-1",
                "target": "seg-3",
                "audio_ref": {"segment_id": "seg-3", "fingerprint": _sha("seg-3")},
            }
        )
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_dangling_beat_character_reference(self):
        payload = _music_video_plan()
        payload["beats"][1]["character_ids"] = ["ghost"]
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_dangling_beat_location_reference(self):
        payload = _music_video_plan()
        payload["beats"][1]["location_id"] = "ghost"
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_dangling_arc_character_reference(self):
        payload = _music_video_plan()
        payload["arcs"][0]["character_id"] = "ghost"
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_dangling_brief_beat_reference(self):
        payload = _music_video_plan()
        payload["segments"][0]["beat_id"] = "ghost"
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_duplicate_brief_targets(self):
        payload = _music_video_plan()
        payload["segments"][1]["target"] = "seg-1"
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_rejects_early_terminal_beat(self):
        payload = _music_video_plan()
        payload["beats"] = [
            {"id": "beat-open", "phase": "resolution", "description": "Early end"},
            {"id": "beat-res", "phase": "development", "description": "After the end"},
        ]
        with self.assertRaises(ValidationError) as ctx:
            _build(payload)
        self.assertIn("ends the story before the last beat", str(ctx.exception))
        payload["beats"] = [
            {"id": "beat-open", "phase": "development", "description": "Build"},
            {"id": "beat-res", "phase": "resolution", "description": "End"},
        ]
        _build(payload)

    def test_rejects_overlapping_exclusive_beat_allocations(self):
        payload = _music_video_plan()
        payload["segments"][0]["exclusive"] = True
        payload["segments"][1]["exclusive"] = True
        payload["segments"][1]["beat_id"] = "beat-open"
        with self.assertRaises(ValidationError):
            _build(payload)


class MusicVideoAuthorityTests(unittest.TestCase):
    def test_music_video_brief_requires_audio_reference(self):
        payload = _music_video_plan()
        del payload["segments"][0]["audio_ref"]
        with self.assertRaises(ValidationError):
            _build(payload)

    def test_music_video_brief_rejects_carried_lyrics(self):
        payload = _music_video_plan()
        payload["segments"][0]["lyrics"] = ["hello"]
        with self.assertRaises(ValidationError) as ctx:
            _build(payload)
        self.assertIn("must not carry authoritative audio data", str(ctx.exception))

    def test_music_video_brief_rejects_carried_timestamps(self):
        payload = _music_video_plan()
        payload["segments"][0]["start_seconds"] = 1.0
        payload["segments"][0]["end_seconds"] = 2.0
        with self.assertRaises(ValidationError) as ctx:
            _build(payload)
        self.assertIn("must not carry authoritative audio data", str(ctx.exception))

    def test_music_video_brief_rejects_carried_vocal_evidence(self):
        with self.assertRaises(ValidationError) as ctx:
            SegmentBrief.model_validate(
                {"id": "b1", "target": "seg-1", "vocal_evidence": "singing"},
                strict=False,
            )
        self.assertIn("must not carry authoritative audio data", str(ctx.exception))


class StoryPlanSerializationTests(unittest.TestCase):
    def test_stable_json_round_trip(self):
        plan = _build(_music_video_plan())
        first = plan.to_json()
        again = StoryPlan.from_json(first).to_json()
        self.assertEqual(first, again)
        self.assertEqual(StoryPlan.from_dict(plan.to_dict()), plan)

    def test_round_trip_preserves_fields(self):
        plan = _build(_music_video_plan())
        restored = StoryPlan.from_json(plan.to_json())
        self.assertEqual(restored.model_dump(), plan.model_dump())

    def test_from_json_rejects_malformed_json(self):
        with self.assertRaises(FeverSlopDataError):
            StoryPlan.from_json("{not json")

    def test_from_json_rejects_non_object(self):
        with self.assertRaises(FeverSlopDataError):
            StoryPlan.from_json("[1]")


class CreativeOverrideTests(unittest.TestCase):
    def _override(self, field_path: str, value: str, override_id: str = "o1") -> CreativeOverride:
        return CreativeOverride(
            id=override_id,
            source="director",
            provenance="note-1",
            field_path=field_path,
            value=value,
        )

    def test_override_requires_source_and_provenance(self):
        with self.assertRaises(ValidationError):
            CreativeOverride(
                id="o1", source="", provenance="note-1", field_path="mode", value="narrative_film"
            )
        with self.assertRaises(ValidationError):
            CreativeOverride(
                id="o1", source="director", provenance="", field_path="mode", value="narrative_film"
            )

    def test_override_wins_when_result_validates(self):
        plan = _build(_music_video_plan())
        result = apply_creative_override(
            plan, self._override("segments.brief-1.vocal_presentation", "offscreen")
        )
        self.assertTrue(result.applied)
        self.assertEqual(result.diagnostics, ())
        self.assertEqual(result.plan.segments[0].vocal_presentation, VocalPresentation.offscreen)
        self.assertEqual(result.plan.segments[0].audio_ref.segment_id, "seg-1")

    def test_override_rejected_when_result_invalid(self):
        plan = _build(_music_video_plan())
        result = apply_creative_override(plan, self._override("segments.brief-2.target", "seg-1"))
        self.assertFalse(result.applied)
        self.assertEqual(result.plan, plan)
        self.assertEqual(result.diagnostics[0].code, "override_rejected")

    def test_override_unknown_target_rejected(self):
        plan = _build(_music_video_plan())
        result = apply_creative_override(
            plan, self._override("segments.nope.vocal_presentation", "offscreen")
        )
        self.assertFalse(result.applied)
        self.assertEqual(result.diagnostics[0].code, "override_unknown_target")

    def test_override_precedence_sequence(self):
        plan = _build(_music_video_plan())
        applied = apply_creative_override(
            plan, self._override("segments.brief-2.visual_direction", "dark corridor")
        )
        self.assertTrue(applied.applied)
        rejected = apply_creative_override(
            applied.plan, self._override("segments.brief-2.target", "seg-1", "o2")
        )
        self.assertFalse(rejected.applied)
        self.assertEqual(rejected.plan.segments[1].visual_direction, "dark corridor")
        self.assertEqual(rejected.diagnostics[0].code, "override_rejected")

    def test_offscreen_override_wins_over_on_screen(self):
        plan = _build(_music_video_plan())
        self.assertEqual(plan.segments[0].vocal_presentation, VocalPresentation.on_screen)
        result = apply_creative_override(
            plan, self._override("segments.brief-1.vocal_presentation", "offscreen")
        )
        self.assertTrue(result.applied)
        self.assertEqual(result.plan.segments[0].vocal_presentation, VocalPresentation.offscreen)


class SingerConflictTests(unittest.TestCase):
    def test_singer_conflict_resolves_to_offscreen(self):
        self.assertEqual(
            resolve_singer_conflict(VocalPresentation.on_screen, has_audio_binding=False),
            VocalPresentation.offscreen,
        )

    def test_singer_conflict_keeps_on_screen_with_binding(self):
        self.assertEqual(
            resolve_singer_conflict(VocalPresentation.on_screen, has_audio_binding=True),
            VocalPresentation.on_screen,
        )

    def test_singer_conflict_non_on_screen_unchanged(self):
        self.assertEqual(
            resolve_singer_conflict(VocalPresentation.offscreen, has_audio_binding=False),
            VocalPresentation.offscreen,
        )
        self.assertEqual(
            resolve_singer_conflict(VocalPresentation.instrumental, has_audio_binding=False),
            VocalPresentation.instrumental,
        )


if __name__ == "__main__":
    unittest.main()
