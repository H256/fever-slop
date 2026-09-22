import unittest
from pathlib import Path

from feverslop.application.prompt_generation_pipeline import (
    PromptGenerationPipeline,
    _contract_entry_id,
)
from feverslop.config.project_config import ProjectConfig
from feverslop.prompting.music_video_signatures import MusicVideoNarrativeContract


class NarrativeContractPromptPipeline:
    """Fake prompt pipeline with a controllable narrative-contract derivation."""

    def __init__(self, *, contract=None, contract_error=None, bindings=None):
        self.contract = contract
        self.contract_error = contract_error
        self.bindings = bindings
        self.narrative_calls = 0
        self.binding_calls = []

    def create_story_idea(self, lyrics, notes=""):
        return "a singer descends into a cave and reaches a hidden fountain"

    def create_style_block(self, lyrics, notes=""):
        return "cinematic gothic film"

    def create_subject_and_locations(self, story_idea, notes="", cast_idea=""):
        return {
            "subject": "a singer in a cave",
            "actors": [
                {
                    "id": "singer",
                    "name": "Singer",
                    "role": "singer",
                    "gender": "female",
                    "visual_description": "A person",
                    "image_prompt": "A person singing",
                },
            ],
            "locations": [
                {"id": "cave_entrance", "name": "Cave Entrance"},
                {"id": "fountain", "name": "Hidden Fountain"},
            ],
        }

    def create_narrative_contract(self, story_idea, locations, actors, notes=""):
        self.narrative_calls += 1
        if self.contract_error is not None:
            raise self.contract_error
        return self.contract

    def create_narrative_milestone_bindings(
        self, story_idea, location_order, milestone_order, missing_milestone_ids,
    ):
        self.binding_calls.append(list(missing_milestone_ids))
        return self.bindings


class NoContractPipeline:
    """A prompt pipeline that does not expose create_narrative_contract."""

    def create_story_idea(self, lyrics, notes=""):
        return "story"

    def create_style_block(self, lyrics, notes=""):
        return "style"

    def create_subject_and_locations(self, story_idea, notes="", cast_idea=""):
        return {
            "subject": "s",
            "actors": [{"id": "a", "name": "A", "role": "r", "gender": "none"}],
            "locations": [{"id": "l", "name": "L"}],
        }


def build_pipeline():
    return PromptGenerationPipeline(
        llm_factory=lambda app_config: None,
        prompt_pipeline_factory=lambda llm: None,
        concept_batcher_factory=lambda llm, size: None,
        scene_prompt_builder_factory=lambda llm: None,
    )


def _config(**kwargs):
    base = {
        "project_dir": Path(),
        "project_name": "test",
        "input_audio": Path("song.mp3"),
    }
    base.update(kwargs)
    return ProjectConfig(**base)


def _resolve(prompt_pipeline, config):
    return build_pipeline().build_resolved_global_context(
        config=config,
        prompt_pipeline=prompt_pipeline,
        all_lyrics="",
        run_spinner=lambda _description, func: func(),
        console=None,
    )


class NarrativeContractDerivationTests(unittest.TestCase):
    def test_derived_contract_repairs_only_missing_milestone_placements(self):
        pipeline = NarrativeContractPromptPipeline(
            contract={
                "location_order": ["cave_entrance", "fountain"],
                "milestone_order": ["arrival", "decision"],
                "milestone_bindings": [
                    {"milestone_id": "arrival", "location_id": "cave_entrance", "relative_position": 0.0},
                ],
            },
            bindings=[
                {"milestone_id": "decision", "location_id": "fountain", "relative_position": 1.0},
            ],
        )

        context = _resolve(pipeline, _config())

        self.assertEqual(["decision"], pipeline.binding_calls[0])
        self.assertEqual(2, len(context["narrative_contract"]["milestone_bindings"]))
        self.assertEqual("llm", context["narrative_contract_source"])

    def test_config_contract_wins_and_llm_not_called(self):
        configured = {
            "location_order": [{"id": "cave_entrance", "source": "enter the cave"}],
            "milestone_order": [{"id": "cave_entered", "source": "enter the cave"}],
        }
        config = _config(narrative_contract=configured)
        pipeline = NarrativeContractPromptPipeline(
            contract={"location_order": [{"id": "fountain", "source": "reach"}]},
        )

        context = _resolve(pipeline, config)

        self.assertEqual(0, pipeline.narrative_calls)
        self.assertEqual(configured, context["narrative_contract"])
        self.assertEqual("config", context["narrative_contract_source"])

    def test_llm_derivation_used_when_config_empty(self):
        config = _config()
        derived = {
            "location_order": [
                {"id": "cave_entrance", "source": "enter the cave"},
                {"id": "fountain", "source": "reach the fountain"},
            ],
            "milestone_order": [
                {"id": "cave_entered", "source": "enter the cave"},
                {"id": "fountain_reached", "source": "reach the fountain"},
            ],
            "milestone_bindings": [
                {
                    "milestone_id": "cave_entered",
                    "location_id": "cave_entrance",
                    "relative_position": 0.0,
                },
                {
                    "milestone_id": "fountain_reached",
                    "location_id": "fountain",
                    "relative_position": 1.0,
                },
            ],
        }
        pipeline = NarrativeContractPromptPipeline(contract=derived)

        context = _resolve(pipeline, config)

        self.assertEqual(1, pipeline.narrative_calls)
        self.assertEqual(derived, context["narrative_contract"])
        self.assertEqual("llm", context["narrative_contract_source"])

    def test_invalid_contract_falls_back_to_empty(self):
        config = _config()
        invalid = {"location_order": [{"id": "unknown_place", "source": "somewhere"}]}
        pipeline = NarrativeContractPromptPipeline(contract=invalid)

        context = _resolve(pipeline, config)

        self.assertEqual(1, pipeline.narrative_calls)
        self.assertEqual({}, context["narrative_contract"])
        self.assertEqual("empty", context["narrative_contract_source"])

    def test_llm_failure_falls_back_to_empty(self):
        config = _config()
        pipeline = NarrativeContractPromptPipeline(contract_error=RuntimeError("llm down"))

        context = _resolve(pipeline, config)

        self.assertEqual(1, pipeline.narrative_calls)
        self.assertEqual({}, context["narrative_contract"])
        self.assertEqual("empty", context["narrative_contract_source"])

    def test_empty_llm_result_falls_back_to_empty(self):
        config = _config()
        pipeline = NarrativeContractPromptPipeline(contract={})

        context = _resolve(pipeline, config)

        self.assertEqual(1, pipeline.narrative_calls)
        self.assertEqual({}, context["narrative_contract"])
        self.assertEqual("empty", context["narrative_contract_source"])

    def test_missing_create_narrative_contract_falls_back_to_empty(self):
        config = _config()

        context = _resolve(NoContractPipeline(), config)

        self.assertEqual({}, context["narrative_contract"])
        self.assertEqual("empty", context["narrative_contract_source"])


class NarrativeContractValidationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = build_pipeline()
        self.actors = [{"id": "singer", "name": "Singer"}]
        self.locations = [
            {"id": "cave_entrance", "name": "Cave Entrance"},
            {"id": "fountain", "name": "Hidden Fountain"},
        ]

    def test_valid_contract_has_no_warnings(self):
        contract = {
            "location_order": ["cave_entrance", "fountain"],
            "milestone_order": [{"id": "cave_entered", "source": "enter"}],
            "milestone_bindings": [
                {
                    "milestone_id": "cave_entered",
                    "location_id": "cave_entrance",
                    "relative_position": 0.0,
                }
            ],
            "terminal_states": {"singer": {"state": "ascended"}},
            "actor_allowed_locations": {"singer": ["fountain"]},
        }
        self.assertEqual([], self.pipeline._validate_narrative_contract(
            contract, self.actors, self.locations))

    def test_missing_milestone_binding_is_reported(self):
        warnings = self.pipeline._validate_narrative_contract(
            {
                "location_order": ["cave_entrance", "fountain"],
                "milestone_order": ["cave_entered", "fountain_reached"],
                "milestone_bindings": [
                    {
                        "milestone_id": "cave_entered",
                        "location_id": "cave_entrance",
                        "relative_position": 0.0,
                    }
                ],
            },
            self.actors,
            self.locations,
        )
        self.assertTrue(any("milestone_bindings" in warning for warning in warnings))

    def test_milestone_binding_is_a_typed_contract_value(self):
        contract = MusicVideoNarrativeContract.model_validate(
            {
                "milestone_bindings": [
                    {
                        "milestone_id": "departure",
                        "location_id": "cave_entrance",
                        "relative_position": 0.25,
                    }
                ]
            }
        )

        binding = contract.milestone_bindings[0]
        self.assertEqual("departure", binding.milestone_id)
        self.assertEqual("cave_entrance", binding.location_id)
        self.assertEqual(0.25, binding.relative_position)

    def test_unknown_location_in_location_order(self):
        warnings = self.pipeline._validate_narrative_contract(
            {"location_order": ["unknown_place"]}, self.actors, self.locations)
        self.assertTrue(any("unknown location" in warning for warning in warnings))

    def test_unknown_actor_in_terminal_states(self):
        warnings = self.pipeline._validate_narrative_contract(
            {"terminal_states": {"ghost": {"state": "x"}}}, self.actors, self.locations)
        self.assertTrue(any("unknown actor" in warning for warning in warnings))

    def test_unknown_location_in_actor_allowed_locations(self):
        warnings = self.pipeline._validate_narrative_contract(
            {"actor_allowed_locations": {"singer": ["nowhere"]}}, self.actors, self.locations)
        self.assertTrue(any("unknown location" in warning for warning in warnings))

    def test_unknown_actor_in_actor_allowed_locations(self):
        warnings = self.pipeline._validate_narrative_contract(
            {"actor_allowed_locations": {"ghost": ["fountain"]}}, self.actors, self.locations)
        self.assertTrue(any("unknown actor" in warning for warning in warnings))

    def test_empty_contract_has_no_warnings(self):
        self.assertEqual([], self.pipeline._validate_narrative_contract(
            {}, self.actors, self.locations))


class ContractEntryIdTests(unittest.TestCase):
    def test_dict_entry_uses_id(self):
        self.assertEqual("loc_a", _contract_entry_id({"id": "loc_a", "source": "x"}))

    def test_string_entry_is_stripped(self):
        self.assertEqual("loc_b", _contract_entry_id("  loc_b  "))

    def test_missing_id_is_empty(self):
        self.assertEqual("", _contract_entry_id({}))
        self.assertEqual("", _contract_entry_id(None))


if __name__ == "__main__":
    unittest.main()
