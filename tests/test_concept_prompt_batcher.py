import json
import unittest

from feverslop.prompting.concept_prompt_batcher import (
    ConceptPromptBatcher,
    _coerce_disallowed_actor_locations,
    _coerce_terminal_states,
    _dedup_one_shot_milestones,
    _reorder_out_of_order_milestones,
    validate_and_annotate_concept_chronology,
)


class FakeConceptModules:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def concepts(self, payload, *, batch=False, silent_mode=False, timeout=None):
        self.calls.append(("concepts", payload, timeout))
        return next(self.responses)

    def repair_concepts(self, payload, *, timeout=None):
        self.calls.append(("repair_concepts", payload, timeout))
        return next(self.responses)

    def summary(self, payload, *, timeout=None):
        self.calls.append(("summary", payload, timeout))
        return next(self.responses)


def semantic_concept(
    concept: str,
    *,
    story_beat: str,
    action: str,
    action_phase: str,
    milestone: str,
    prop_state: str,
    objective: str = "complete the fountain rite",
    reset_events: list[str] | None = None,
) -> dict:
    return {
        "concept": concept,
        "references": {
            "actor_ids": ["ravena"],
            "location_id": "fountain_grotto",
        },
        "narrative": {
            "story_beat": story_beat,
            "objective": objective,
            "action": action,
            "action_phase": action_phase,
            "milestones": [milestone],
            "location": "fountain_grotto",
            "cast_states": {"ravena": "corporeal"},
            "props": {"silver_cup": prop_state},
            "reset_events": reset_events or [],
        },
    }


class ConceptPromptBatcherTests(unittest.TestCase):
    def test_story_complete_is_only_valid_on_final_segment_without_contract(self):
        concepts = {
            "segment_001": {
                "concept": "The screen fades to black.",
                "narrative": {"milestones": ["story_complete"]},
            },
            "segment_002": {
                "concept": "The ritual continues.",
                "narrative": {"milestones": ["ritual_continues"]},
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            "story_complete.*only valid on final segment 'segment_002'",
        ):
            validate_and_annotate_concept_chronology(concepts, {})

    def test_planning_omits_replicated_evidence_in_batches_and_repairs(self):
        from copy import deepcopy
        segment = dict(segment_id="s1", start=1, end=3, duration=2, type="vocals", lyrics="Keep these words",
                       performance_intervals=[{"vocal_sources": [{"alignment": "x" * 800000}]}],
                       word_timestamps=[{"word": "Keep", "start": 1, "end": 2}],
                       reason_codes=["uncertain_vocal_evidence"])
        original = deepcopy(segment)
        modules = FakeConceptModules([{}, {"s1": "A scene"}, "summary"])
        result = ConceptPromptBatcher(object(), prompt_modules=modules).create_concept_prompts_batched(
            stage1_segments=[segment], story_idea="Story", global_context={})
        self.assertEqual({"s1": "A scene"}, result)
        for _, payload, _ in modules.calls:
            self.assertLess(len(json.dumps(payload)), 1000)
            self.assertNotIn("performance_intervals", json.dumps(payload))
        sent = modules.calls[0][1]["CURRENT_BATCH_SEGMENTS"][0]
        self.assertEqual("Keep these words", sent["lyrics"])
        self.assertEqual((1, 3), (sent["start"], sent["end"]))
        self.assertEqual(segment, original)

    def test_reports_batch_progress(self):
        modules = FakeConceptModules([
            json.dumps({"seg_1": "concept 1"}),
            "summary",
            json.dumps({"seg_2": "concept 2"}),
            "summary",
        ])
        progress = []
        batcher = ConceptPromptBatcher(
            llm=object(),
            prompt_modules=modules,
            batch_size=1,
            progress_callback=progress.append,
        )

        batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1"}, {"segment_id": "seg_2"}],
            story_idea="idea",
            global_context={},
        )

        self.assertEqual(
            [
                "Concept batch 1/2: generating scenes 1-1",
                "Concept batch 1/2: response received, validating keys",
                "Concept batch 1/2: complete (1 scenes total)",
                "Concept batch 2/2: generating scenes 2-2",
                "Concept batch 2/2: response received, validating keys",
                "Concept batch 2/2: complete (2 scenes total)",
            ],
            progress,
        )

    def test_passes_timeout_to_llm(self):
        modules = FakeConceptModules([json.dumps({"seg_1": "concept 1"}), "summary"])

        batcher = ConceptPromptBatcher(
            llm=object(),
            prompt_modules=modules,
            batch_size=1,
            request_timeout_seconds=42.0,
        )

        batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1"}],
            story_idea="idea",
            global_context={"silent_mode": False},
            notes="notes",
        )

        # Check _generate_batch call
        found_batch_call = False
        found_summary_call = False

        for name, payload, timeout in modules.calls:
            if name == "concepts" and "CURRENT_BATCH_SEGMENTS" in payload:
                found_batch_call = True
                self.assertEqual(timeout, 42.0)
            if name == "summary" and "RECENT_CONCEPTS" in payload:
                found_summary_call = True
                self.assertEqual(timeout, 42.0)

        self.assertTrue(found_batch_call, "_generate_batch was not called")
        self.assertTrue(found_summary_call, "_summarize_progress was not called")

    def test_passes_timeout_to_repair_call(self):
        modules = FakeConceptModules([
            json.dumps({}),  # batch result (missing seg_1)
            json.dumps({"seg_1": "repaired concept"}),  # repair result
            "summary",  # summarize progress result
        ])

        batcher = ConceptPromptBatcher(
            llm=object(),
            prompt_modules=modules,
            batch_size=1,
            request_timeout_seconds=99.0,
        )

        batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1"}],
            story_idea="idea",
            global_context={},
        )

        found_repair_call = False
        for name, payload, timeout in modules.calls:
            if name == "repair_concepts" and "MISSING_SEGMENTS" in payload:
                found_repair_call = True
                self.assertEqual(timeout, 99.0)

        self.assertTrue(found_repair_call, "_repair_missing_or_extra_keys was not called")

    def test_reports_ids_of_missing_scene_keys_before_repair(self):
        modules = FakeConceptModules([
            json.dumps({"seg_1": "concept 1"}),
            json.dumps({"seg_2": "repaired concept 2", "seg_3": "repaired concept 3"}),
            "summary",
        ])
        progress = []
        batcher = ConceptPromptBatcher(
            llm=object(),
            prompt_modules=modules,
            batch_size=3,
            progress_callback=progress.append,
        )

        batcher.create_concept_prompts_batched(
            stage1_segments=[
                {"segment_id": "seg_1"},
                {"segment_id": "seg_2"},
                {"segment_id": "seg_3"},
            ],
            story_idea="idea",
            global_context={},
        )

        self.assertIn(
            "Concept batch: repairing 2 scene keys (2 missing, 0 invalid): seg_2, seg_3 [missing: seg_2, seg_3]",
            progress,
        )

    def test_repair_source_message_separates_missing_and_invalid(self):
        message = ConceptPromptBatcher._repair_source_message(
            total=3,
            missing=["seg_1"],
            invalid=[
                {"segment_id": "seg_2", "reason": "actor not named"},
                {"segment_id": "seg_3", "reason": "milestone order"},
            ],
            repair_ids=["seg_1", "seg_2", "seg_3"],
        )
        self.assertIn("repairing 3 scene keys (1 missing, 2 invalid)", message)
        self.assertIn("[missing: seg_1]", message)
        self.assertIn("[invalid: seg_2 (actor not named); seg_3 (milestone order)]", message)

    def test_repair_source_message_single_key_and_empty_sources(self):
        message = ConceptPromptBatcher._repair_source_message(
            total=1,
            missing=["seg_1"],
            invalid=[],
            repair_ids=["seg_1"],
        )
        self.assertIn("repairing 1 scene key (1 missing, 0 invalid)", message)
        self.assertIn("[missing: seg_1]", message)
        self.assertNotIn("[invalid:", message)

    def test_reports_invalid_scene_keys_with_reasons_before_repair(self):
        modules = FakeConceptModules([
            json.dumps({
                "seg_1": {
                    "concept": "The singer performs on stage.",
                    "references": {"actor_ids": ["singer", "bass"], "location_id": "stage"},
                },
            }),
            json.dumps({
                "seg_1": {
                    "concept": "Goth Singer and Bass Player perform together on stage.",
                    "references": {"actor_ids": ["singer", "bass"], "location_id": "stage"},
                },
            }),
            "summary",
        ])
        progress = []
        batcher = ConceptPromptBatcher(
            llm=object(),
            prompt_modules=modules,
            batch_size=1,
            progress_callback=progress.append,
        )

        batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1", "type": "vocals"}],
            story_idea="idea",
            global_context={
                "actors": [
                    {"id": "singer", "name": "Goth Singer"},
                    {"id": "bass", "name": "Bass Player"},
                ],
            },
        )

        repair_messages = [
            message for message in progress
            if message.startswith("Concept batch: repairing 1 scene key (0 missing, 1 invalid): seg_1")
        ]
        self.assertTrue(repair_messages, "invalid repair message was not reported")
        self.assertIn(
            "[invalid: seg_1 (",
            repair_messages[0],
        )
        self.assertTrue(
            any("bass player" in message for message in progress),
            "invalid reason was not reported",
        )

    def test_repairs_concept_when_selected_actor_is_not_named(self):
        modules = FakeConceptModules([
            json.dumps({
                "seg_1": {
                    "concept": "The singer performs on stage.",
                    "references": {"actor_ids": ["singer", "bass"], "location_id": "stage"},
                },
            }),
            json.dumps({
                "seg_1": {
                    "concept": "Goth Singer and Bass Player perform together on stage; the Bass Player holds the bass.",
                    "references": {"actor_ids": ["singer", "bass"], "location_id": "stage"},
                },
            }),
            "summary",
        ])
        batcher = ConceptPromptBatcher(llm=object(), prompt_modules=modules, batch_size=1)

        result = batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1", "type": "vocals"}],
            story_idea="idea",
            global_context={
                "actors": [
                    {"id": "singer", "name": "Goth Singer"},
                    {"id": "bass", "name": "Bass Player"},
                ],
            },
        )

        repair_calls = [call for call in modules.calls if call[0] == "repair_concepts"]
        self.assertEqual(1, len(repair_calls))
        self.assertEqual("seg_1", repair_calls[0][1]["INVALID_SEGMENTS"][0]["segment_id"])
        self.assertIn("bass player", repair_calls[0][1]["INVALID_SEGMENTS"][0]["reason"])
        self.assertIn("Bass Player", result["seg_1"]["concept"])

    def test_repairs_accidental_duplicate_scene_before_accepting_batch(self):
        first = semantic_concept(
            "Ravena raises the silver cup at the fountain.",
            story_beat="raise_the_cup",
            action="raise_silver_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        duplicate = semantic_concept(
            "At the fountain Ravena lifts the silver chalice upward.",
            story_beat="raise the cup",
            action="raise silver cup",
            action_phase="completed",
            milestone="cup raised",
            prop_state="raised",
        )
        replacement = semantic_concept(
            "Ravena drinks once from the raised silver cup.",
            story_beat="drink_from_the_cup",
            action="drink_from_silver_cup",
            action_phase="completed",
            milestone="drink_completed",
            prop_state="consumed",
        )
        modules = FakeConceptModules([
            {"seg_1": first, "seg_2": duplicate},
            {"seg_2": replacement},
            "summary",
        ])

        result = ConceptPromptBatcher(
            object(), prompt_modules=modules, batch_size=2,
        ).create_concept_prompts_batched(
            stage1_segments=[
                {"segment_id": "seg_1", "scene": 10},
                {"segment_id": "seg_2", "scene": 11},
            ],
            story_idea="Ravena drinks once from the well.",
            global_context={
                "actors": [{"id": "ravena", "name": "Ravena"}],
                "narrative_contract": {"one_shot_milestones": ["cup_raised"]},
            },
        )

        repair = next(call for call in modules.calls if call[0] == "repair_concepts")
        diagnostic = repair[1]["INVALID_SEGMENTS"][0]
        self.assertEqual("seg_2", diagnostic["segment_id"])
        self.assertEqual("seg_1", diagnostic["prior_segment_id"])
        self.assertIn("semantic scene duplicates seg_1", diagnostic["reason"])
        self.assertIn("milestone 'cup_raised' repeats seg_1", diagnostic["reason"])
        self.assertEqual("drink_completed", result["seg_2"]["narrative"]["milestones"][0])
        self.assertEqual("accepted", result["seg_2"]["semantic_validation"]["outcome"])

    def test_semantic_signature_ignores_prose_and_camera_wording(self):
        first = semantic_concept(
            "A static close shot shows Ravena raise the cup.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        paraphrase = semantic_concept(
            "The camera circles as Ravena hoists the chalice.",
            story_beat="raise cup",
            action="raise cup",
            action_phase="completed",
            milestone="cup raised",
            prop_state="raised",
        )

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"seg_2": paraphrase},
            {"actors": [{"id": "ravena", "name": "Ravena"}]},
            previous_concepts={"seg_1": first},
            expected_ids=["seg_2"],
        )

        self.assertEqual(1, len(invalid))
        self.assertEqual("seg_1", invalid[0]["prior_segment_id"])
        self.assertIn("semantic scene duplicates seg_1", invalid[0]["reason"])

    def test_semantic_validation_tolerates_malformed_narrative_list_items(self):
        concept = semantic_concept(
            "Ravena raises the cup.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        concept["narrative"]["milestones"].extend([None, {"unexpected": "value"}])
        concept["narrative"]["reset_events"] = [None, {"unexpected": "value"}]

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"seg_1": concept},
            {"actors": [{"id": "ravena", "name": "Ravena"}]},
            expected_ids=["seg_1"],
        )

        self.assertEqual([], invalid)

    def test_repairs_non_object_narrative_props_before_accepting_batch(self):
        malformed = semantic_concept(
            "Ravena raises the cup.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        malformed["narrative"]["props"] = ["silver_cup", "raised"]
        repaired = semantic_concept(
            "Ravena raises the silver cup.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        modules = FakeConceptModules([
            {"seg_1": malformed},
            {"seg_1": repaired},
            "summary",
        ])

        result = ConceptPromptBatcher(
            object(), prompt_modules=modules, batch_size=1,
        ).create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1", "scene": 1}],
            story_idea="Ravena completes the fountain rite.",
            global_context={"actors": [{"id": "ravena", "name": "Ravena"}]},
        )

        repair = next(call for call in modules.calls if call[0] == "repair_concepts")
        self.assertIn(
            "narrative.props must be an object",
            repair[1]["INVALID_SEGMENTS"][0]["reason"],
        )
        self.assertEqual({"silver_cup": "raised"}, result["seg_1"]["narrative"]["props"])

    def test_non_object_narrative_props_are_schema_invalid(self):
        for malformed_props in (["silver_cup", "raised"], "silver_cup=raised"):
            with self.subTest(props=malformed_props):
                concept = semantic_concept(
                    "Ravena raises the cup.",
                    story_beat="raise_cup",
                    action="raise_cup",
                    action_phase="completed",
                    milestone="cup_raised",
                    prop_state="raised",
                )
                concept["narrative"]["props"] = malformed_props

                invalid = ConceptPromptBatcher._invalid_concepts(
                    {"seg_1": concept},
                    {"actors": [{"id": "ravena", "name": "Ravena"}]},
                    expected_ids=["seg_1"],
                )

                self.assertEqual(1, len(invalid))
                self.assertIn("narrative.props must be an object", invalid[0]["reason"])

    def test_malformed_prior_props_do_not_crash_prop_regression_validation(self):
        current = semantic_concept(
            "Ravena raises the silver cup.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        for malformed_props in (["silver_cup", "acquired"], "silver_cup=acquired"):
            with self.subTest(props=malformed_props):
                prior = semantic_concept(
                    "Ravena acquires the cup.",
                    story_beat="acquire_cup",
                    action="acquire_cup",
                    action_phase="completed",
                    milestone="cup_acquired",
                    prop_state="acquired",
                )
                prior["narrative"]["props"] = malformed_props

                invalid = ConceptPromptBatcher._invalid_concepts(
                    {"seg_2": current},
                    {
                        "actors": [{"id": "ravena", "name": "Ravena"}],
                        "narrative_contract": {
                            "prop_state_order": {
                                "silver_cup": ["unseen", "acquired", "raised"],
                            },
                        },
                    },
                    previous_concepts={"seg_1": prior},
                    expected_ids=["seg_2"],
                )

                self.assertEqual([], invalid)

    def test_recurring_cast_location_and_motif_are_valid_when_action_advances(self):
        acquired = semantic_concept(
            "Ravena receives the silver cup beneath the recurring halo motif.",
            story_beat="acquire_cup",
            action="receive_cup",
            action_phase="completed",
            milestone="cup_acquired",
            prop_state="acquired",
        )
        raised = semantic_concept(
            "Ravena raises the silver cup beneath the recurring halo motif.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"seg_2": raised},
            {"actors": [{"id": "ravena", "name": "Ravena"}]},
            previous_concepts={"seg_1": acquired},
            expected_ids=["seg_2"],
        )

        self.assertEqual([], invalid)

    def test_repeated_non_one_shot_milestone_is_valid_when_action_advances(self):
        first = semantic_concept(
            "Ravena sings the first chorus beside the fountain.",
            story_beat="first_chorus",
            action="sing_first_chorus",
            action_phase="completed",
            milestone="chorus_refrain",
            prop_state="acquired",
        )
        later = semantic_concept(
            "Ravena sings the final chorus while lowering the cup.",
            story_beat="final_chorus",
            action="sing_final_chorus",
            action_phase="completed",
            milestone="chorus_refrain",
            prop_state="lowered",
        )

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"seg_2": later},
            {"actors": [{"id": "ravena", "name": "Ravena"}]},
            previous_concepts={"seg_1": first},
            expected_ids=["seg_2"],
        )

        self.assertEqual([], invalid)

    def test_prop_state_regression_normalizes_contract_and_narrative_keys(self):
        raised = semantic_concept(
            "Ravena holds the raised silver cup.",
            story_beat="hold_raised_cup",
            action="hold_cup",
            action_phase="sustained",
            milestone="cup_raised",
            prop_state="raised",
        )
        raised["narrative"]["props"] = {"silver cup": "raised"}
        regressed = semantic_concept(
            "Ravena approaches while the cup is absent.",
            story_beat="approach_fountain",
            action="approach_fountain",
            action_phase="completed",
            milestone="fountain_approached",
            prop_state="unseen",
        )
        regressed["narrative"]["props"] = {"Silver-Cup": "unseen"}

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"seg_2": regressed},
            {
                "actors": [{"id": "ravena", "name": "Ravena"}],
                "narrative_contract": {
                    "prop_state_order": {
                        "silver_cup": ["unseen", "acquired", "raised"],
                    },
                },
            },
            previous_concepts={"seg_1": raised},
            expected_ids=["seg_2"],
        )

        self.assertEqual(1, len(invalid))
        self.assertIn("prop 'silver_cup' state 'unseen' regresses", invalid[0]["reason"])

    def test_explicit_matching_reset_event_authorizes_reprise(self):
        original = semantic_concept(
            "Ravena raises the silver cup.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        reprise = semantic_concept(
            "Ravena raises the silver cup in the musical reprise.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
            reset_events=["cup_raised"],
        )

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"seg_2": reprise},
            {
                "actors": [{"id": "ravena", "name": "Ravena"}],
                "narrative_contract": {"one_shot_milestones": ["cup_raised"]},
            },
            previous_concepts={"seg_1": original},
            expected_ids=["seg_2"],
        )
        accepted = ConceptPromptBatcher._annotate_semantic_validation(
            {"seg_2": reprise},
            previous_concepts={"seg_1": original},
            contract={"one_shot_milestones": ["cup_raised"]},
        )

        self.assertEqual([], invalid)
        self.assertTrue(accepted["seg_2"]["semantic_validation"]["authorized_reprise"])
        self.assertEqual(
            ["cup_raised"],
            accepted["seg_2"]["semantic_validation"]["authorized_reset_events"],
        )
        self.assertEqual(
            "seg_1", accepted["seg_2"]["semantic_validation"]["reprise_of"],
        )

    def test_reprise_authorization_only_requires_reset_for_one_shot_milestones(self):
        original = semantic_concept(
            "Ravena raises the cup during the chorus.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        original["narrative"]["milestones"].append("chorus_refrain")
        reprise = semantic_concept(
            "Ravena repeats the cup raise during the final chorus.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
            reset_events=["cup_raised"],
        )
        reprise["narrative"]["milestones"].append("chorus_refrain")
        contract = {"one_shot_milestones": ["cup_raised"]}

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"seg_2": reprise},
            {
                "actors": [{"id": "ravena", "name": "Ravena"}],
                "narrative_contract": contract,
            },
            previous_concepts={"seg_1": original},
            expected_ids=["seg_2"],
        )
        accepted = ConceptPromptBatcher._annotate_semantic_validation(
            {"seg_2": reprise},
            previous_concepts={"seg_1": original},
            contract=contract,
        )

        self.assertEqual([], invalid)
        self.assertTrue(accepted["seg_2"]["semantic_validation"]["authorized_reprise"])
        self.assertEqual(
            ["cup_raised"],
            accepted["seg_2"]["semantic_validation"]["authorized_reset_events"],
        )

    def test_unrelated_reset_event_does_not_authorize_reprise(self):
        original = semantic_concept(
            "Ravena raises the silver cup.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        reprise = semantic_concept(
            "Ravena raises the silver cup again.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
            reset_events=["unrelated_event"],
        )

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"seg_2": reprise},
            {
                "actors": [{"id": "ravena", "name": "Ravena"}],
                "narrative_contract": {"one_shot_milestones": ["cup_raised"]},
            },
            previous_concepts={"seg_1": original},
            expected_ids=["seg_2"],
        )

        self.assertEqual(1, len(invalid))
        self.assertIn("without reset_events authorization", invalid[0]["reason"])

    def test_prop_state_regression_requires_matching_reset_event(self):
        raised = semantic_concept(
            "Ravena holds the raised silver cup.",
            story_beat="hold_raised_cup",
            action="hold_cup",
            action_phase="sustained",
            milestone="cup_raised",
            prop_state="raised",
        )
        regressed = semantic_concept(
            "Ravena approaches while the cup is absent.",
            story_beat="approach_fountain",
            action="approach_fountain",
            action_phase="completed",
            milestone="fountain_approached",
            prop_state="unseen",
        )
        context = {
            "actors": [{"id": "ravena", "name": "Ravena"}],
            "narrative_contract": {
                "prop_state_order": {
                    "silver_cup": [
                        "unseen", "acquired", "raised", "at_lips", "consumed", "lowered",
                    ],
                },
            },
        }

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"seg_2": regressed},
            context,
            previous_concepts={"seg_1": raised},
            expected_ids=["seg_2"],
        )
        authorized = semantic_concept(
            "A causal reset removes the cup before Ravena approaches.",
            story_beat="approach_fountain",
            action="approach_fountain",
            action_phase="completed",
            milestone="fountain_approached",
            prop_state="unseen",
            reset_events=["silver_cup"],
        )

        self.assertEqual(1, len(invalid))
        self.assertEqual("seg_1", invalid[0]["prior_segment_id"])
        self.assertIn(
            "prop 'silver_cup' state 'unseen' regresses from 'raised' in seg_1",
            invalid[0]["reason"],
        )
        self.assertEqual(
            [],
            ConceptPromptBatcher._invalid_concepts(
                {"seg_2": authorized},
                context,
                previous_concepts={"seg_1": raised},
                expected_ids=["seg_2"],
            ),
        )

    def test_duplicate_in_later_batch_is_compared_with_all_prior_concepts(self):
        first = semantic_concept(
            "Ravena raises the cup.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        duplicate = semantic_concept(
            "Ravena hoists the chalice.",
            story_beat="raise cup",
            action="raise cup",
            action_phase="completed",
            milestone="cup raised",
            prop_state="raised",
        )
        replacement = semantic_concept(
            "Ravena lowers the cup.",
            story_beat="lower_cup",
            action="lower_cup",
            action_phase="completed",
            milestone="cup_lowered",
            prop_state="lowered",
        )
        modules = FakeConceptModules([
            {"seg_1": first}, "first summary",
            {"seg_2": duplicate}, {"seg_2": replacement}, "second summary",
        ])

        result = ConceptPromptBatcher(
            object(), prompt_modules=modules, batch_size=1, max_previous_concepts=0,
        ).create_concept_prompts_batched(
            stage1_segments=[
                {"segment_id": "seg_1", "scene": 10},
                {"segment_id": "seg_2", "scene": 11},
            ],
            story_idea="Ravena completes the rite.",
            global_context={
                "actors": [{"id": "ravena", "name": "Ravena"}],
            },
        )

        repair = next(call for call in modules.calls if call[0] == "repair_concepts")
        self.assertEqual("seg_1", repair[1]["INVALID_SEGMENTS"][0]["prior_segment_id"])
        self.assertEqual("cup_lowered", result["seg_2"]["narrative"]["milestones"][0])

    def test_rejects_semantically_invalid_repair_instead_of_propagating_it(self):
        first = semantic_concept(
            "Ravena raises the cup.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        duplicate = semantic_concept(
            "Ravena raises the cup again.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone="cup_raised",
            prop_state="raised",
        )
        modules = FakeConceptModules([
            {"seg_1": first, "seg_2": duplicate},
            {"seg_2": duplicate},
        ])

        with self.assertRaisesRegex(
            ValueError,
            r"seg_2.*semantic scene duplicates seg_1.*milestone 'cup_raised' repeats seg_1",
        ):
            ConceptPromptBatcher(
                object(), prompt_modules=modules, batch_size=2,
                semantic_enforcement="block",
            ).create_concept_prompts_batched(
                stage1_segments=[
                    {"segment_id": "seg_1", "scene": 10},
                    {"segment_id": "seg_2", "scene": 11},
                ],
                story_idea="Ravena drinks once from the well.",
                global_context={
                    "actors": [{"id": "ravena", "name": "Ravena"}],
                    "narrative_contract": {"one_shot_milestones": ["cup_raised"]},
                },
            )

    def test_rejects_milestone_before_required_predecessors(self):
        cave = semantic_concept(
            "Ravena descends into the Weeping Caves.",
            story_beat="cave_descent",
            action="descend",
            action_phase="completed",
            milestone="caves_entered",
            prop_state="unseen",
        )
        fountain = semantic_concept(
            "Ravena reaches the fountain before confronting its guardians.",
            story_beat="fountain_arrival",
            action="reach_fountain",
            action_phase="completed",
            milestone="fountain_arrival",
            prop_state="unseen",
        )
        contract = {
            "milestone_order": [
                {"id": "caves_entered", "source": "story_idea: enter the Weeping Caves"},
                {"id": "lich_encounter", "source": "story_idea: confront the Lich"},
                {"id": "dragon_encounter", "source": "story_idea: pass the Dragon"},
                {"id": "fountain_arrival", "source": "story_idea: reach the fountain"},
            ],
        }

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"segment_002": fountain},
            {"narrative_contract": contract},
            previous_concepts={"segment_001": cave},
            expected_ids=["segment_002"],
        )

        self.assertEqual(1, len(invalid))
        self.assertEqual("segment_001", invalid[0]["prior_segment_id"])
        self.assertIn("milestone 'fountain_arrival'", invalid[0]["reason"])
        self.assertIn("required predecessor 'lich_encounter' is unresolved", invalid[0]["reason"])
        self.assertIn("story_idea: confront the Lich", invalid[0]["reason"])

    def test_named_causal_event_authorizes_explicit_flashback(self):
        fountain = semantic_concept(
            "Ravena stands at the fountain.",
            story_beat="fountain_arrival",
            action="reach_fountain",
            action_phase="completed",
            milestone="fountain_arrival",
            prop_state="consumed",
        )
        fountain["narrative"]["location"] = "fountain_grotto"
        flashback = semantic_concept(
            "An explicit flashback returns to Ravena entering the caves.",
            story_beat="cave_flashback",
            action="remember_descent",
            action_phase="completed",
            milestone="caves_entered",
            prop_state="unseen",
        )
        flashback["narrative"]["location"] = "weeping_caves"
        flashback["narrative"]["causal_events"] = ["flashback_to_caves"]
        contract = {
            "milestone_order": ["caves_entered", "fountain_arrival"],
            "location_order": ["weeping_caves", "fountain_grotto"],
            "chronology_exceptions": {
                "flashback_to_caves": {
                    "allows": ["milestone_order", "location_order", "prop_state_order"],
                    "source": "story_idea: explicit memory of the descent",
                },
            },
            "prop_state_order": {"silver_cup": ["unseen", "consumed"]},
        }

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"segment_007": flashback},
            {"narrative_contract": contract},
            previous_concepts={"segment_006": fountain},
            expected_ids=["segment_007"],
        )
        accepted = ConceptPromptBatcher._annotate_semantic_validation(
            {"segment_007": flashback},
            previous_concepts={"segment_006": fountain},
            contract=contract,
        )

        self.assertEqual([], invalid)
        chronology = accepted["segment_007"]["semantic_validation"]["chronology"]
        self.assertEqual("accepted", chronology["validation_result"])
        self.assertEqual("flashback_to_caves", chronology["approved_exception"])

    def test_terminal_state_cannot_silently_revert(self):
        ascent = semantic_concept(
            "Ravena completes her ascent and disappears.",
            story_beat="ascent",
            action="ascend",
            action_phase="completed",
            milestone="ascent_complete",
            prop_state="consumed",
        )
        ascent["narrative"]["cast_states"] = {"ravena": "ascended_absent"}
        returned = semantic_concept(
            "Ravena stands corporeal in the caves again.",
            story_beat="cave_return",
            action="stand_in_caves",
            action_phase="completed",
            milestone="caves_entered",
            prop_state="consumed",
        )
        returned["narrative"]["cast_states"] = {"ravena": "corporeal"}
        contract = {
            "terminal_states": {
                "ravena": {
                    "milestone": "ascent_complete",
                    "state": "ascended_absent",
                    "reset_event": "ravena_returns",
                    "source": "story_idea: Ravena ascends and disappears",
                },
            },
        }

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"segment_007": returned},
            {"narrative_contract": contract},
            previous_concepts={"segment_006": ascent},
            expected_ids=["segment_007"],
        )

        self.assertEqual(1, len(invalid))
        self.assertIn("terminal state 'ascended_absent'", invalid[0]["reason"])
        self.assertIn("observed 'corporeal'", invalid[0]["reason"])
        self.assertIn("story_idea: Ravena ascends and disappears", invalid[0]["reason"])

    def test_flashback_exception_does_not_authorize_skipped_forward_milestones(self):
        fountain = semantic_concept(
            "A flashback label cannot move Ravena directly to the fountain.",
            story_beat="fountain_arrival",
            action="reach_fountain",
            action_phase="completed",
            milestone="fountain_arrival",
            prop_state="unseen",
        )
        fountain["narrative"]["causal_events"] = ["flashback_to_caves"]
        contract = {
            "milestone_order": ["caves_entered", "lich_encounter", "fountain_arrival"],
            "chronology_exceptions": {
                "flashback_to_caves": {"allows": ["milestone_order"]},
            },
        }

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"segment_001": fountain},
            {"narrative_contract": contract},
            expected_ids=["segment_001"],
        )

        self.assertEqual(1, len(invalid))
        self.assertIn("required predecessor 'caves_entered' is unresolved", invalid[0]["reason"])

    def test_terminal_milestone_requires_terminal_state_in_same_scene(self):
        ascent = semantic_concept(
            "Ravena ascends but incorrectly remains corporeal.",
            story_beat="ascent",
            action="ascend",
            action_phase="completed",
            milestone="ascent_complete",
            prop_state="consumed",
        )
        ascent["narrative"]["cast_states"] = {"ravena": "corporeal"}
        contract = {
            "terminal_states": {
                "ravena": {
                    "milestone": "ascent_complete",
                    "state": "ascended_absent",
                    "reset_event": "ravena_returns",
                },
            },
        }

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"segment_006": ascent},
            {"narrative_contract": contract},
            expected_ids=["segment_006"],
        )

        self.assertEqual(1, len(invalid))
        self.assertIn("terminal milestone 'ascent_complete'", invalid[0]["reason"])
        self.assertIn("observed 'corporeal'", invalid[0]["reason"])

    def test_complete_sequence_requires_every_ordered_milestone(self):
        cave = semantic_concept(
            "Ravena enters the caves.",
            story_beat="cave_descent",
            action="descend",
            action_phase="completed",
            milestone="caves_entered",
            prop_state="unseen",
        )
        contract = {
            "milestone_order": [
                {"id": "caves_entered", "source": "story_idea: cave descent"},
                {"id": "lich_encounter", "source": "story_idea: Lich encounter"},
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "required milestone 'lich_encounter'.*story_idea: Lich encounter",
        ):
            validate_and_annotate_concept_chronology(
                {"segment_001": cave},
                contract,
            )

    def test_named_terminal_reset_remains_effective_in_following_scenes(self):
        ascent = semantic_concept(
            "Ravena ascends and disappears.",
            story_beat="ascent",
            action="ascend",
            action_phase="completed",
            milestone="ascent_complete",
            prop_state="consumed",
        )
        ascent["narrative"]["cast_states"] = {"ravena": "ascended_absent"}
        returned = semantic_concept(
            "The authored resurrection returns Ravena.",
            story_beat="resurrection",
            action="return",
            action_phase="completed",
            milestone="ravena_returned",
            prop_state="consumed",
        )
        returned["narrative"]["cast_states"] = {"ravena": "corporeal"}
        returned["narrative"]["causal_events"] = ["ravena_returns"]
        later = semantic_concept(
            "Ravena remains corporeal after the authored resurrection.",
            story_beat="return_aftermath",
            action="walk_forward",
            action_phase="completed",
            milestone="return_aftermath",
            prop_state="consumed",
        )
        later["narrative"]["cast_states"] = {"ravena": "corporeal"}
        contract = {
            "terminal_states": {
                "ravena": {
                    "milestone": "ascent_complete",
                    "state": "ascended_absent",
                    "reset_event": "ravena_returns",
                },
            },
        }

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"segment_003": later},
            {"narrative_contract": contract},
            previous_concepts={"segment_001": ascent, "segment_002": returned},
            expected_ids=["segment_003"],
        )

        self.assertEqual([], invalid)

    def test_terminal_milestone_requires_actor_state_to_be_explicit(self):
        ascent = semantic_concept(
            "Ravena completes the ascent.",
            story_beat="ascent",
            action="ascend",
            action_phase="completed",
            milestone="ascent_complete",
            prop_state="consumed",
        )
        ascent["narrative"]["cast_states"] = {}
        contract = {
            "terminal_states": {
                "ravena": {
                    "milestone": "ascent_complete",
                    "state": "ascended_absent",
                    "reset_event": "ravena_returns",
                },
            },
        }

        invalid = ConceptPromptBatcher._invalid_concepts(
            {"segment_006": ascent},
            {"narrative_contract": contract},
            expected_ids=["segment_006"],
        )

        self.assertEqual(1, len(invalid))
        self.assertIn("requires explicit cast state 'ascended_absent'", invalid[0]["reason"])

def boundary_concept(
    concept: str,
    *,
    story_beat: str,
    action: str,
    action_phase: str,
    location: str,
    cast_states: dict[str, str],
    incoming: dict | None = None,
    transition_from_previous: str = "cut",
) -> dict:
    value = {
        "concept": concept,
        "references": {"actor_ids": [], "location_id": location},
        "narrative": {
            "story_beat": story_beat,
            "objective": "confront_the_lair_guardian",
            "action": action,
            "action_phase": action_phase,
            "milestones": [],
            "location": location,
            "cast_states": cast_states,
            "props": {},
            "transition_events": [],
            "transition_from_previous": transition_from_previous,
        },
    }
    if incoming is not None:
        value["narrative"]["incoming"] = incoming
    return value


class CollateralContinuityRepairTests(unittest.TestCase):
    """Repaired predecessors must not strand accepted successors (#1175/#1176 shape).

    A dropped key (segment_005/008) is regenerated by the repair call, while the
    continuous successor (segment_006/009) was already generated with boundary
    states that paraphrase the same meaning. The repair must be told the exact
    accepted boundary vocabulary and get one bounded follow-up pass.
    """

    LICH = {
        "predecessor": "segment_004",
        "missing": "segment_005",
        "successor": "segment_006",
        "location": "lich_s_lair",
        "paraphrased_location": "center_of_lich_s_lair",
        "repaired_action": "turning_and_observing_the_lich",
        "paraphrased_action": "facing_the_lich",
        "repaired_cast": {
            "ravena": "turning_tense",
            "varen": "gripping_sword",
            "silas": "holding_lantern",
        },
        "paraphrased_cast": {
            "ravena": "facing_lich",
            "varen": "facing_lich",
            "silas": "facing_lich",
        },
    }
    DRAGON = {
        "predecessor": "segment_007",
        "missing": "segment_008",
        "successor": "segment_009",
        "location": "dragon_s_lair",
        "paraphrased_location": "center_of_dragon_s_lair",
        "repaired_action": "gesturing_toward_the_dragon",
        "paraphrased_action": "gesturing_to_dragon",
        "repaired_cast": {"ravena": "gesturing_soft_expression"},
        "paraphrased_cast": {"ravena": "gesturing"},
    }

    def _scenes(self, shape):
        predecessor = boundary_concept(
            "The trio crosses the chamber floor toward the guardian.",
            story_beat="cross_chamber",
            action="cross_chamber_floor",
            action_phase="completed",
            location=shape["location"],
            cast_states={actor: "advancing" for actor in shape["repaired_cast"]},
        )
        repaired = boundary_concept(
            "The guardian is observed from the centre of the lair.",
            story_beat="observe_guardian",
            action=shape["repaired_action"],
            action_phase="ongoing",
            location=shape["location"],
            cast_states=dict(shape["repaired_cast"]),
        )
        paraphrased = boundary_concept(
            "The trio faces the guardian at the heart of the lair.",
            story_beat="face_guardian",
            action=shape["paraphrased_action"],
            action_phase="ongoing",
            location=shape["location"],
            cast_states=dict(shape["paraphrased_cast"]),
            incoming={
                "location": shape["paraphrased_location"],
                "action": shape["paraphrased_action"],
                "action_phase": "ongoing",
                "cast_states": dict(shape["paraphrased_cast"]),
            },
            transition_from_previous="continuous",
        )
        return predecessor, repaired, paraphrased

    def _run(self, shape, final_successor, *, semantic_enforcement="block"):
        predecessor, repaired, paraphrased = self._scenes(shape)
        modules = FakeConceptModules([
            {shape["predecessor"]: predecessor, shape["successor"]: paraphrased},
            {shape["missing"]: repaired},
            {shape["successor"]: final_successor},
            "summary",
        ])
        progress = []
        batcher = ConceptPromptBatcher(
            object(),
            prompt_modules=modules,
            batch_size=3,
            progress_callback=progress.append,
            semantic_enforcement=semantic_enforcement,
        )
        result = batcher.create_concept_prompts_batched(
            stage1_segments=[
                {"segment_id": shape["predecessor"], "scene": 4},
                {"segment_id": shape["missing"], "scene": 5},
                {"segment_id": shape["successor"], "scene": 6},
            ],
            story_idea="The trio confronts the lair guardian.",
            global_context={},
        )
        return result, modules, progress

    def test_lich_boundary_paraphrase_is_repaired_not_rejected(self):
        shape = self.LICH
        aligned = boundary_concept(
            "The trio faces the guardian at the heart of the lair.",
            story_beat="face_guardian",
            action=shape["paraphrased_action"],
            action_phase="ongoing",
            location=shape["location"],
            cast_states=dict(shape["paraphrased_cast"]),
            incoming={},
            transition_from_previous="continuous",
        )

        result, modules, progress = self._run(shape, aligned)

        self.assertIn(
            f"Concept batch: repairing 1 scene key broken by adjacent repairs: {shape['successor']}",
            progress,
        )
        self.assertEqual("accepted", result[shape["successor"]]["semantic_validation"]["outcome"])
        self.assertEqual(
            aligned["narrative"], result[shape["successor"]]["narrative"],
            "the collateral-repaired successor must be the one actually used",
        )
        continuity = result[shape["successor"]]["semantic_validation"]["continuity"]
        self.assertEqual(shape["location"], continuity["incoming"]["location"])
        self.assertEqual(
            shape["repaired_cast"], continuity["incoming"]["cast_states"],
        )

    def test_dragon_boundary_paraphrase_is_repaired_not_rejected(self):
        shape = self.DRAGON
        aligned = boundary_concept(
            "Ravena addresses the dragon beside the hoard.",
            story_beat="address_dragon",
            action=shape["paraphrased_action"],
            action_phase="ongoing",
            location=shape["location"],
            cast_states=dict(shape["paraphrased_cast"]),
            incoming={},
            transition_from_previous="continuous",
        )

        result, modules, progress = self._run(shape, aligned)

        self.assertIn(
            f"Concept batch: repairing 1 scene key broken by adjacent repairs: {shape['successor']}",
            progress,
        )
        self.assertEqual("accepted", result[shape["successor"]]["semantic_validation"]["outcome"])
        continuity = result[shape["successor"]]["semantic_validation"]["continuity"]
        self.assertEqual(shape["location"], continuity["incoming"]["location"])
        self.assertEqual(
            shape["repaired_cast"], continuity["incoming"]["cast_states"],
        )

    def test_collateral_repair_receives_exact_predecessor_boundary_state(self):
        shape = self.LICH
        aligned = boundary_concept(
            "The trio faces the guardian at the heart of the lair.",
            story_beat="face_guardian",
            action=shape["paraphrased_action"],
            action_phase="ongoing",
            location=shape["location"],
            cast_states=dict(shape["paraphrased_cast"]),
            incoming={},
            transition_from_previous="continuous",
        )

        _, modules, _ = self._run(shape, aligned)

        repair_calls = [call for call in modules.calls if call[0] == "repair_concepts"]
        self.assertEqual(2, len(repair_calls))
        first, second = (call[1] for call in repair_calls)
        self.assertEqual([shape["missing"]], first["EXPECTED_KEYS"])
        self.assertEqual([shape["successor"]], second["EXPECTED_KEYS"])
        self.assertIn("is incompatible", second["INVALID_SEGMENTS"][0]["reason"])
        self.assertEqual(
            {
                "scene_id": shape["missing"],
                "outgoing": {
                    "location": shape["location"],
                    "action": shape["repaired_action"],
                    "action_phase": "ongoing",
                    "cast_states": dict(shape["repaired_cast"]),
                    "props": {},
                },
            },
            second["BOUNDARY_CONTEXT"][shape["successor"]]["predecessor"],
        )

    def test_genuinely_incompatible_collateral_transition_still_fails(self):
        shape = self.LICH
        worse = boundary_concept(
            "The trio suddenly stands in the dragon's lair.",
            story_beat="face_guardian",
            action=shape["paraphrased_action"],
            action_phase="ongoing",
            location=shape["location"],
            cast_states=dict(shape["paraphrased_cast"]),
            incoming={"location": "dragon_s_lair"},
            transition_from_previous="continuous",
        )

        with self.assertRaisesRegex(
            ValueError,
            r"segment_006\.incoming\.location: 'dragon_s_lair' is incompatible",
        ):
            self._run(shape, worse)

    def test_repair_boundary_context_follows_explicit_outgoing_block(self):
        # The smoke-run shape: the predecessor's explicit `outgoing` block
        # overrides its own narrative action. The repair prompt must be told
        # the outgoing-block value, and a successor repaired to that value
        # must pass both the batch gate and the chronology gate.
        predecessor = boundary_concept(
            "Ravena stands on the cave path watching the throne.",
            story_beat="observe_from_cave_path",
            action="standing_and_observing_the_cave_path",
            action_phase="ongoing",
            location="lich_s_lair",
            cast_states={"ravena": "still"},
        )
        predecessor["narrative"]["outgoing"] = {
            "location": "lich_s_lair",
            "action": "standing_and_observing",
            "action_phase": "ongoing",
            "cast_states": {"ravena": "still"},
        }
        repaired = boundary_concept(
            "Ravena steps toward the lich throne.",
            story_beat="approach_throne",
            action="step_toward_throne",
            action_phase="ongoing",
            location="lich_s_lair",
            cast_states={"ravena": "advancing"},
            incoming={
                "location": "lich_s_lair",
                "action": "standing_and_observing",
                "action_phase": "ongoing",
                "cast_states": {"ravena": "still"},
            },
            transition_from_previous="continuous",
        )
        modules = FakeConceptModules([
            {"segment_001": predecessor},
            {"segment_002": repaired},
            "summary",
        ])

        batcher = ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=2)
        result = batcher.create_concept_prompts_batched(
            stage1_segments=[
                {"segment_id": "segment_001", "scene": 1},
                {"segment_id": "segment_002", "scene": 2},
            ],
            story_idea="Ravena confronts the Lich.",
            global_context={},
        )

        repair = next(call for call in modules.calls if call[0] == "repair_concepts")
        self.assertEqual(
            "standing_and_observing",
            repair[1]["BOUNDARY_CONTEXT"]["segment_002"]["predecessor"]["outgoing"]["action"],
        )
        self.assertEqual(
            "accepted", result["segment_002"]["semantic_validation"]["outcome"],
        )
        annotated = validate_and_annotate_concept_chronology(result, {})
        self.assertEqual(
            "standing_and_observing",
            annotated["segment_002"]["semantic_validation"]["continuity"]["incoming"]["action"],
        )

    def test_genuinely_invalid_own_repair_never_gets_a_second_attempt(self):
        # The missing scene's own repair is invalid against its predecessor's
        # monotonic prop state; the batch must fail immediately without ever
        # running the collateral follow-up pass.
        predecessor = boundary_concept(
            "Ravena holds the raised silver cup.",
            story_beat="hold_raised_cup",
            action="hold_cup",
            action_phase="sustained",
            location="fountain_grotto",
            cast_states={"ravena": "corporeal"},
        )
        predecessor["narrative"]["props"] = {"silver_cup": "raised"}
        successor = boundary_concept(
            "Ravena drinks from the raised silver cup.",
            story_beat="drink_cup",
            action="drink_cup",
            action_phase="completed",
            location="fountain_grotto",
            cast_states={"ravena": "corporeal"},
        )
        successor["narrative"]["props"] = {"silver_cup": "raised"}
        bad_repair = boundary_concept(
            "Ravena approaches while the cup is absent.",
            story_beat="approach_fountain",
            action="approach_fountain",
            action_phase="completed",
            location="fountain_grotto",
            cast_states={"ravena": "corporeal"},
        )
        bad_repair["narrative"]["props"] = {"silver_cup": "unseen"}
        modules = FakeConceptModules([
            {"segment_001": predecessor, "segment_003": successor},
            {"segment_002": bad_repair},
            "summary",
        ])
        progress = []

        with self.assertRaisesRegex(ValueError, "prop 'silver_cup' state 'unseen' regresses"):
            ConceptPromptBatcher(
                object(),
                prompt_modules=modules,
                batch_size=3,
                progress_callback=progress.append,
                semantic_enforcement="block",
            ).create_concept_prompts_batched(
                stage1_segments=[
                    {"segment_id": "segment_001", "scene": 1},
                    {"segment_id": "segment_002", "scene": 2},
                    {"segment_id": "segment_003", "scene": 3},
                ],
                story_idea="Ravena completes the fountain rite.",
                global_context={"narrative_contract": {
                    "prop_state_order": {"silver_cup": ["unseen", "raised"]},
                }},
            )

        self.assertFalse(
            any("broken by adjacent repairs" in message for message in progress),
        )
        self.assertEqual(1, len([call for call in modules.calls if call[0] == "repair_concepts"]))


class RepairContextCompletenessTests(unittest.TestCase):
    """Contiguous repair gaps must not blind the repair model (#1174/#1176 shape).

    When a generation response loses a contiguous suffix, every neighbor of
    every repair target is itself a repair target. The repair payload must
    still carry the nearest accepted boundary state (across the gap), the full
    state of any duplicated prior scene, and a compact ledger of all accepted
    semantic states. Strict duplicate validation must stay intact.
    """

    @staticmethod
    def _concept(beat: str, *, cast: str = "advancing") -> dict:
        return boundary_concept(
            f"Ravena performs {beat.replace('_', ' ')}.",
            story_beat=beat,
            action=beat,
            action_phase="ongoing",
            location="cave_system",
            cast_states={"ravena": cast},
        )

    def test_repair_boundary_context_falls_back_across_repair_gaps(self):
        c1 = self._concept("enter_caves", cast="walking")
        c2 = self._concept("cross_chamber", cast="advancing")
        modules = FakeConceptModules([
            {"s1": c1, "s2": c2},
            {"s3": self._concept("observe_throne"), "s4": self._concept("approach_throne")},
            {"s5": self._concept("face_guardian")},
            "summary",
        ])
        batcher = ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=5)

        batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": f"s{i}", "scene": i} for i in range(1, 6)],
            story_idea="Ravena crosses the lair.",
            global_context={},
        )

        repairs = [call[1] for call in modules.calls if call[0] == "repair_concepts"]
        self.assertEqual(2, len(repairs))
        first = repairs[0]
        self.assertEqual(["s3", "s4"], first["EXPECTED_KEYS"])
        # s3 has an adjacent accepted predecessor: unchanged shape, no distance key.
        s3_boundary = first["BOUNDARY_CONTEXT"]["s3"]["predecessor"]
        self.assertEqual("s2", s3_boundary["scene_id"])
        self.assertNotIn("neighbor_distance", s3_boundary)
        self.assertNotIn("successor", first["BOUNDARY_CONTEXT"]["s3"])
        # s4's immediate predecessor is itself being repaired: the boundary
        # entry must fall back to the nearest accepted scene across the gap.
        s4_boundary = first["BOUNDARY_CONTEXT"]["s4"]["predecessor"]
        self.assertEqual("s2", s4_boundary["scene_id"])
        self.assertEqual(2, s4_boundary["neighbor_distance"])
        self.assertEqual("cross_chamber", s4_boundary["outgoing"]["action"])
        self.assertNotIn("successor", first["BOUNDARY_CONTEXT"]["s4"])

    def test_gap_fallback_repair_that_duplicates_boundary_neighbor_still_fails(self):
        # Negative guard: enriching the gap context must not soften the gate.
        # A repair that reproduces the neighbor's full semantic state must
        # still fail immediately with exactly one repair attempt.
        c1 = self._concept("enter_caves", cast="walking")
        c2 = self._concept("cross_chamber", cast="advancing")
        from copy import deepcopy
        modules = FakeConceptModules([
            {"s1": c1, "s2": c2},
            {"s3": deepcopy(c2)},
            "summary",
        ])
        with self.assertRaisesRegex(ValueError, "semantic scene duplicates s2"):
            ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=3, semantic_enforcement="block").create_concept_prompts_batched(
                stage1_segments=[{"segment_id": f"s{i}", "scene": i} for i in range(1, 4)],
                story_idea="Ravena crosses the lair.",
                global_context={},
            )
        self.assertEqual(1, len([call for call in modules.calls if call[0] == "repair_concepts"]))

    def test_invalid_repair_payload_carries_prior_segment_state(self):
        first = semantic_concept(
            "Ravena raises the silver cup at the fountain.",
            story_beat="raise_the_cup",
            action="raise_silver_cup",
            action_phase="ongoing",
            milestone="cup_raised",
            prop_state="unseen",
        )
        from copy import deepcopy
        modules = FakeConceptModules([
            {"s1": first, "s2": deepcopy(first)},
            {"s2": semantic_concept(
                "Ravena kneels before the fountain.",
                story_beat="kneel_before_fountain",
                action="kneel",
                action_phase="ongoing",
                milestone="kneeled",
                prop_state="unseen",
            )},
            "summary",
        ])

        ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=2).create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "s1"}, {"segment_id": "s2"}],
            story_idea="Ravena completes the fountain rite.",
            global_context={},
        )

        repair = next(call for call in modules.calls if call[0] == "repair_concepts")
        invalid = repair[1]["INVALID_SEGMENTS"][0]
        self.assertEqual("s2", invalid["segment_id"])
        self.assertEqual("s1", invalid["prior_segment_id"])
        state = invalid["prior_segment_state"]
        self.assertEqual(
            {
                "story_beat", "objective", "action", "action_phase",
                "milestones", "location", "cast_states", "props",
            },
            set(state),
        )
        self.assertEqual("raise_the_cup", state["story_beat"])
        self.assertEqual({"ravena": "corporeal"}, state["cast_states"])

    def test_generation_payload_includes_accepted_state_ledger(self):
        first = semantic_concept(
            "Ravena raises the silver cup at the fountain.",
            story_beat="raise_the_cup",
            action="raise_silver_cup",
            action_phase="ongoing",
            milestone="cup_raised",
            prop_state="unseen",
        )
        modules = FakeConceptModules([
            {"s1": first},
            "summary",
            {"s2": semantic_concept(
                "Ravena kneels before the fountain.",
                story_beat="kneel_before_fountain",
                action="kneel",
                action_phase="ongoing",
                milestone="kneeled",
                prop_state="unseen",
            )},
            "summary",
        ])
        batcher = ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=1)

        batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "s1"}, {"segment_id": "s2"}],
            story_idea="Ravena completes the fountain rite.",
            global_context={},
        )

        generations = [call[1] for call in modules.calls if call[0] == "concepts"]
        self.assertEqual([], generations[0]["ACCEPTED_STATE_LEDGER"])
        self.assertEqual(
            [{
                "scene_id": "s1",
                "story_beat": "raise_the_cup",
                "action": "raise_silver_cup",
                "action_phase": "ongoing",
                "location": "fountain_grotto",
            }],
            generations[1]["ACCEPTED_STATE_LEDGER"],
        )

    def test_repair_payload_includes_accepted_state_ledger(self):
        first = semantic_concept(
            "Ravena raises the silver cup at the fountain.",
            story_beat="raise_the_cup",
            action="raise_silver_cup",
            action_phase="ongoing",
            milestone="cup_raised",
            prop_state="unseen",
        )
        modules = FakeConceptModules([
            {"s1": first},
            {"s2": semantic_concept(
                "Ravena kneels before the fountain.",
                story_beat="kneel_before_fountain",
                action="kneel",
                action_phase="ongoing",
                milestone="kneeled",
                prop_state="unseen",
            )},
            "summary",
        ])

        ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=2).create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "s1"}, {"segment_id": "s2"}],
            story_idea="Ravena completes the fountain rite.",
            global_context={},
        )

        repair = next(call for call in modules.calls if call[0] == "repair_concepts")
        ledger = repair[1]["ACCEPTED_STATE_LEDGER"]
        self.assertEqual(["s1"], [entry["scene_id"] for entry in ledger])
        self.assertEqual("raise_the_cup", ledger[0]["story_beat"])

    def test_incomplete_repair_response_is_reported(self):
        modules = FakeConceptModules([
            {},  # generation lost every key
            {"s1": "concept one"},  # chunk (s1, s2) answered only s1
            {"s3": "concept three"},  # chunk (s3) answered
            "summary",
        ])
        progress = []
        batcher = ConceptPromptBatcher(
            object(), prompt_modules=modules, batch_size=3, progress_callback=progress.append,
        )

        result = batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": f"s{i}"} for i in range(1, 4)],
            story_idea="idea",
            global_context={},
        )

        self.assertTrue(
            any("repair response incomplete for s2" in message for message in progress),
            progress,
        )
        # Existing fallback semantics are preserved (documented one-shot rule),
        # but the gap is now visible instead of silent.
        self.assertIn("s2", result["s2"])


class ConceptCheckpointTests(unittest.TestCase):
    """Concept work must survive a mid-stage crash without stale reuse."""

    class CrashingModules(FakeConceptModules):
        def __init__(self, responses):
            super().__init__(responses)
            self.generation_calls = 0

        def concepts(self, payload, *, batch=False, silent_mode=False, timeout=None):
            self.generation_calls += 1
            if self.generation_calls == 2:
                raise RuntimeError("simulated batch 2 failure")
            return super().concepts(
                payload, batch=batch, silent_mode=silent_mode, timeout=timeout,
            )

    @staticmethod
    def _store():
        from feverslop.adapters.local_artifacts import JsonArtifactStore

        return JsonArtifactStore()

    @staticmethod
    def _segments():
        return [{"segment_id": f"s{i}", "scene": i} for i in range(1, 3)]

    def _batcher(self, modules, temp, name="concept_checkpoint.json"):
        from pathlib import Path

        batcher = ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=1)
        batcher.enable_checkpoint(
            path=Path(temp) / name, artifact_store=self._store(),
        )
        return batcher

    def test_checkpoint_skips_completed_batches_and_is_cleared_on_success(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "concept_checkpoint.json"
            crashing = self.CrashingModules([{"s1": "concept one"}, "summary"])
            with self.assertRaises(RuntimeError):
                self._batcher(crashing, temp).create_concept_prompts_batched(
                    stage1_segments=self._segments(), story_idea="idea", global_context={},
                )
            self.assertTrue(checkpoint.is_file())
            saved = self._store().read_json(checkpoint)
            self.assertEqual(["s1"], sorted(saved["concepts"]))

            modules = FakeConceptModules(["summary", {"s2": "concept two"}, "summary"])
            progress = []
            batcher = ConceptPromptBatcher(
                object(), prompt_modules=modules, batch_size=1, progress_callback=progress.append,
            )
            batcher.enable_checkpoint(path=checkpoint, artifact_store=self._store())

            result = batcher.create_concept_prompts_batched(
                stage1_segments=self._segments(), story_idea="idea", global_context={},
            )

            self.assertEqual({"s1", "s2"}, set(result))
            self.assertEqual(
                [2],
                [call[1]["BATCH_INDEX"] for call in modules.calls if call[0] == "concepts"],
                "batch 1 must not be regenerated after a checkpoint restore",
            )
            self.assertTrue(
                any("Resuming concept generation from checkpoint" in message for message in progress),
                progress,
            )
            self.assertFalse(checkpoint.exists(), "checkpoint must be cleared after success")

    def test_stale_checkpoint_is_ignored_and_inputs_regenerated(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            from pathlib import Path

            checkpoint = Path(temp) / "concept_checkpoint.json"
            crashing = self.CrashingModules([{"s1": "concept one"}, "summary"])
            with self.assertRaises(RuntimeError):
                self._batcher(crashing, temp).create_concept_prompts_batched(
                    stage1_segments=self._segments(), story_idea="idea", global_context={},
                )
            data = self._store().read_json(checkpoint)
            data["identity"] = "stale-identity"
            self._store().write_json(checkpoint, data)

            modules = FakeConceptModules(
                [{"s1": "concept one"}, "summary", {"s2": "concept two"}, "summary"],
            )
            progress = []
            batcher = ConceptPromptBatcher(
                object(), prompt_modules=modules, batch_size=1, progress_callback=progress.append,
            )
            batcher.enable_checkpoint(path=checkpoint, artifact_store=self._store())

            result = batcher.create_concept_prompts_batched(
                stage1_segments=self._segments(), story_idea="changed idea", global_context={},
            )

            self.assertEqual({"s1", "s2"}, set(result))
            self.assertEqual(
                [1, 2],
                [call[1]["BATCH_INDEX"] for call in modules.calls if call[0] == "concepts"],
            )
            self.assertTrue(
                any("Ignoring stale concept checkpoint" in message for message in progress),
                progress,
            )


class BoundaryVocabularyFrontLoadTests(unittest.TestCase):
    """#1247: the initial batch must front-load the exact boundary vocabulary
    (Fix 1) and a closed-vocabulary block (Fix 2), and guard the request
    token budget so the added context cannot grow the request unbounded."""

    def test_initial_batch_front_loads_predecessor_boundary_anchor(self):
        # Batch 1 is accepted; batch 2's first scene (s2) must anchor on s1's
        # exact accepted outgoing state, not infer it.
        first = semantic_concept(
            "Ravena raises the silver cup at the fountain.",
            story_beat="raise_the_cup",
            action="raise_silver_cup",
            action_phase="ongoing",
            milestone="cup_raised",
            prop_state="unseen",
        )
        modules = FakeConceptModules([
            {"s1": first},
            "summary",
            {"s2": semantic_concept(
                "Ravena kneels before the fountain.",
                story_beat="kneel_before_fountain",
                action="kneel",
                action_phase="ongoing",
                milestone="kneeled",
                prop_state="unseen",
            )},
            "summary",
        ])
        ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=1).create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "s1"}, {"segment_id": "s2"}],
            story_idea="Ravena completes the fountain rite.",
            global_context={},
        )
        generations = [call[1] for call in modules.calls if call[0] == "concepts"]
        # First batch has no predecessor: no BOUNDARY_CONTEXT anchor.
        self.assertNotIn("BOUNDARY_CONTEXT", generations[0])
        # Second batch front-loads the accepted predecessor's outgoing snapshot.
        anchor = generations[1]["BOUNDARY_CONTEXT"]["s2"]
        self.assertEqual("s1", anchor["scene_id"])
        self.assertEqual("raise_silver_cup", anchor["outgoing"]["action"])
        self.assertEqual("fountain_grotto", anchor["outgoing"]["location"])
        self.assertEqual({"ravena": "corporeal"}, anchor["outgoing"]["cast_states"])

    def test_closed_vocabulary_block_is_derived_and_capped(self):
        from feverslop.prompting.concept_prompt_batcher import _closed_vocabulary
        global_context = {
            "actors": [{"id": "a1"}, {"id": "a2"}, {"id": "a1"}],
            "structured_locations": [{"id": "loc1"}, {"id": "loc2"}],
            "props": [{"id": "prop1"}],
            "narrative_contract": {
                "milestone_order": ["m1", "m2"],
                "location_order": ["loc1", "loc2"],
                "allowed_actions": ["act1", "act2"],
                "allowed_cast_states": ["state1"],
            },
        }
        vocab = _closed_vocabulary(
            global_context, global_context["narrative_contract"],
        )
        self.assertEqual(["a1", "a2"], vocab["actor_ids"])
        self.assertEqual(["loc1", "loc2"], vocab["location_ids"])
        self.assertEqual(["prop1"], vocab["prop_ids"])
        self.assertEqual(["m1", "m2"], vocab["milestones"])
        self.assertEqual(["loc1", "loc2"], vocab["locations"])
        self.assertEqual(["act1", "act2"], vocab["allowed_actions"])
        self.assertEqual(["state1"], vocab["allowed_cast_states"])

    def test_closed_vocabulary_total_cap_bounds_the_block(self):
        from feverslop.prompting.concept_prompt_batcher import _closed_vocabulary
        # Far more structured ids than the total cap allows.
        global_context = {
            "actors": [{"id": f"a{i}"} for i in range(150)],
            "structured_locations": [{"id": f"loc{i}"} for i in range(150)],
            "props": [{"id": f"p{i}"} for i in range(150)],
        }
        vocab = _closed_vocabulary(global_context, {})
        self.assertLessEqual(sum(len(v) for v in vocab.values()), 200)

    def test_request_token_budget_guard_raises_when_exceeded(self):
        from feverslop.prompting import concept_prompt_batcher as cpb
        original = cpb._REQUEST_TOKEN_CEILING
        try:
            # Shrink the ceiling so a normal-sized request trips the guard.
            cpb._REQUEST_TOKEN_CEILING = 1
            modules = FakeConceptModules([{"s1": "concept"}, "summary"])
            with self.assertRaisesRegex(ValueError, "exceeds token budget"):
                ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=1).create_concept_prompts_batched(
                    stage1_segments=[{"segment_id": "s1"}],
                    story_idea="A story idea.",
                    global_context={},
                )
        finally:
            cpb._REQUEST_TOKEN_CEILING = original

    def test_closed_vocabulary_present_in_batch_payload(self):
        modules = FakeConceptModules([{"s1": "concept"}, "summary"])
        ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=1).create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "s1"}],
            story_idea="A story idea.",
            global_context={
                "actors": [{"id": "a1"}],
                "narrative_contract": {"milestone_order": ["m1"]},
            },
        )
        payload = next(call[1] for call in modules.calls if call[0] == "concepts")
        self.assertEqual(["a1"], payload["CLOSED_VOCABULARY"]["actor_ids"])
        self.assertEqual(["m1"], payload["CLOSED_VOCABULARY"]["milestones"])

    def test_narrative_constraints_block_is_derived(self):
        from feverslop.prompting.concept_prompt_batcher import _narrative_constraints
        contract = {
            "one_shot_milestones": ["m1", "m2"],
            "terminal_states": {
                "a1": {
                    "milestone": "m1",
                    "state": "ascended",
                    "reset_event": "return_rite",
                },
            },
        }
        constraints = _narrative_constraints(contract)
        self.assertEqual(["m1", "m2"], constraints["one_shot_milestones"])
        self.assertEqual(
            {
                "a1": {
                    "milestone": "m1",
                    "state": "ascended",
                    "reset_event": "return_rite",
                },
            },
            constraints["terminal_states"],
        )

    def test_narrative_constraints_empty_contract_returns_empty(self):
        from feverslop.prompting.concept_prompt_batcher import _narrative_constraints
        self.assertEqual({}, _narrative_constraints({}))

    def test_narrative_constraints_present_in_batch_payload(self):
        modules = FakeConceptModules([{"s1": "concept"}, "summary"])
        ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=1).create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "s1"}],
            story_idea="A story idea.",
            global_context={
                "actors": [{"id": "a1"}],
                "narrative_contract": {
                    "milestone_order": ["m1", "m2"],
                    "one_shot_milestones": ["m1"],
                    "terminal_states": {
                        "a1": {
                            "milestone": "m2",
                            "state": "ascended",
                            "reset_event": "return_rite",
                        },
                    },
                },
            },
        )
        payload = next(call[1] for call in modules.calls if call[0] == "concepts")
        self.assertEqual(["m1"], payload["NARRATIVE_CONSTRAINTS"]["one_shot_milestones"])
        self.assertEqual(
            "m2",
            payload["NARRATIVE_CONSTRAINTS"]["terminal_states"]["a1"]["milestone"],
        )

    def test_narrative_constraints_absent_without_contract(self):
        modules = FakeConceptModules([{"s1": "concept"}, "summary"])
        ConceptPromptBatcher(object(), prompt_modules=modules, batch_size=1).create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "s1"}],
            story_idea="A story idea.",
            global_context={"actors": [{"id": "a1"}]},
        )
        payload = next(call[1] for call in modules.calls if call[0] == "concepts")
        self.assertNotIn("NARRATIVE_CONSTRAINTS", payload)

    def test_derive_fix_instructions_one_shot_milestone(self):
        from feverslop.prompting.concept_prompt_batcher import _derive_fix_instructions
        reasons = [
            "milestone 'drink_silver_water' repeats segment_031 without "
            "reset_events authorization",
        ]
        instructions = _derive_fix_instructions(reasons)
        self.assertEqual(1, len(instructions))
        self.assertIn("drink_silver_water", instructions[0])
        self.assertIn("reset_events", instructions[0])

    def test_derive_fix_instructions_terminal_state(self):
        from feverslop.prompting.concept_prompt_batcher import _derive_fix_instructions
        reasons = [
            "terminal milestone 'transfiguration_and_ascension' requires "
            "explicit cast state 'ascended_soul_in_a_shaft_of_light' for "
            "actor 'ravena' (source: narrative_contract)",
        ]
        instructions = _derive_fix_instructions(reasons)
        self.assertEqual(1, len(instructions))
        self.assertIn("ascended_soul_in_a_shaft_of_light", instructions[0])
        self.assertIn("ravena", instructions[0])

    def test_derive_fix_instructions_fallback(self):
        from feverslop.prompting.concept_prompt_batcher import _derive_fix_instructions
        reasons = ["some unknown reason"]
        instructions = _derive_fix_instructions(reasons)
        self.assertEqual(1, len(instructions))
        self.assertIn("some unknown reason", instructions[0])

    def test_repair_payload_contains_narrative_constraints_and_fix_instructions(self):
        """End-to-end: a one-shot violation produces NARRATIVE_CONSTRAINTS and
        actionable fix_instructions in the actual repair payload before the
        batcher raises on the unrepaired violation."""
        first = semantic_concept(
            "Ravena drinks from the silver cup.",
            story_beat="drink_silver_water",
            action="drink",
            action_phase="completed",
            milestone="drink_silver_water",
            prop_state="drunk",
        )
        duplicate = semantic_concept(
            "Ravena drinks from the silver cup again.",
            story_beat="drink_silver_water",
            action="drink",
            action_phase="completed",
            milestone="drink_silver_water",
            prop_state="drunk",
        )
        modules = FakeConceptModules([
            {"seg_1": first, "seg_2": duplicate},
            {"seg_2": duplicate},  # repair returns the same invalid concept
            "summary",
        ])
        with self.assertRaisesRegex(
            ValueError,
            "Concept semantic validation failed after repair",
        ):
            ConceptPromptBatcher(
                object(),
                prompt_modules=modules,
                batch_size=2,
                semantic_enforcement="block",
            ).create_concept_prompts_batched(
                stage1_segments=[
                    {"segment_id": "seg_1", "scene": 10},
                    {"segment_id": "seg_2", "scene": 11},
                ],
                story_idea="Ravena drinks once from the well.",
                global_context={
                    "actors": [{"id": "ravena", "name": "Ravena"}],
                    "narrative_contract": {
                        "one_shot_milestones": ["drink_silver_water"],
                        "terminal_states": {
                            "ravena": {
                                "milestone": "drink_silver_water",
                                "state": "transfigured",
                                "reset_event": "return_rite",
                            },
                        },
                    },
                },
            )
        # The repair payload must contain NARRATIVE_CONSTRAINTS.
        repair_calls = [c for c in modules.calls if c[0] == "repair_concepts"]
        self.assertTrue(repair_calls, "expected a repair call")
        repair_payload = repair_calls[0][1]
        self.assertIn("NARRATIVE_CONSTRAINTS", repair_payload)
        self.assertEqual(
            ["drink_silver_water"],
            repair_payload["NARRATIVE_CONSTRAINTS"]["one_shot_milestones"],
        )
        # The repair payload must contain fix_instructions in INVALID_SEGMENTS.
        invalid = repair_payload["INVALID_SEGMENTS"]
        self.assertTrue(invalid, "expected invalid segments in repair payload")
        self.assertTrue(
            all("fix_instructions" in item for item in invalid),
            "expected fix_instructions in every invalid segment",
        )

    def test_narrative_constraints_omits_empty_reset_event(self):
        from feverslop.prompting.concept_prompt_batcher import _narrative_constraints
        contract = {
            "terminal_states": {
                "a1": {"milestone": "m1", "state": "ascended"},
            },
        }
        constraints = _narrative_constraints(contract)
        self.assertNotIn("reset_event", constraints["terminal_states"]["a1"])

    def test_kind_aliases_map_to_supported_taxonomy(self):
        from feverslop.prompting.semantic_intent_extraction import _KIND_ALIASES
        self.assertEqual("scene_obligation", _KIND_ALIASES["action"])
        self.assertEqual("scene_obligation", _KIND_ALIASES["sequence"])
        self.assertEqual("identity", _KIND_ALIASES["attribute"])

    def test_warn_mode_returns_concepts_with_warning_annotation(self):
        """Warn mode preserves structurally valid concepts and marks them
        with a warning outcome instead of raising."""
        first = semantic_concept(
            "Ravena drinks from the silver cup.",
            story_beat="drink_silver_water",
            action="drink",
            action_phase="completed",
            milestone="drink_silver_water",
            prop_state="drunk",
        )
        duplicate = semantic_concept(
            "Ravena drinks from the silver cup again.",
            story_beat="drink_silver_water",
            action="drink",
            action_phase="completed",
            milestone="drink_silver_water",
            prop_state="drunk",
        )
        modules = FakeConceptModules([
            {"seg_1": first, "seg_2": duplicate},
            {"seg_2": duplicate},  # repair returns the same invalid concept
            "summary",
        ])
        warnings: list[str] = []
        result = ConceptPromptBatcher(
            object(),
            prompt_modules=modules,
            batch_size=2,
            semantic_enforcement="warn",
            progress_callback=warnings.append,
        ).create_concept_prompts_batched(
            stage1_segments=[
                {"segment_id": "seg_1", "scene": 10},
                {"segment_id": "seg_2", "scene": 11},
            ],
            story_idea="Ravena drinks once from the well.",
            global_context={
                "actors": [{"id": "ravena", "name": "Ravena"}],
                "narrative_contract": {"one_shot_milestones": ["drink_silver_water"]},
            },
        )
        # Both concepts are returned (not raised).
        self.assertIn("seg_1", result)
        self.assertIn("seg_2", result)
        # The invalid segment is marked with a warning outcome.
        self.assertEqual("warning", result["seg_2"]["semantic_validation"]["outcome"])
        self.assertIsNotNone(result["seg_2"]["semantic_validation"]["unresolved_diagnostic"])
        # The valid segment is marked as accepted.
        self.assertEqual("accepted", result["seg_1"]["semantic_validation"]["outcome"])
        # A warning was reported.
        self.assertTrue(
            any("semantic validation issue" in w for w in warnings),
            "expected a semantic validation warning",
        )

    def test_block_mode_raises_on_unrepaired_violation(self):
        """Block mode raises ValueError on unrepaired semantic violations."""
        first = semantic_concept(
            "Ravena drinks from the silver cup.",
            story_beat="drink_silver_water",
            action="drink",
            action_phase="completed",
            milestone="drink_silver_water",
            prop_state="drunk",
        )
        duplicate = semantic_concept(
            "Ravena drinks from the silver cup again.",
            story_beat="drink_silver_water",
            action="drink",
            action_phase="completed",
            milestone="drink_silver_water",
            prop_state="drunk",
        )
        modules = FakeConceptModules([
            {"seg_1": first, "seg_2": duplicate},
            {"seg_2": duplicate},  # repair returns the same invalid concept
            "summary",
        ])
        with self.assertRaisesRegex(
            ValueError,
            "Concept semantic validation failed after repair",
        ):
            ConceptPromptBatcher(
                object(),
                prompt_modules=modules,
                batch_size=2,
                semantic_enforcement="block",
            ).create_concept_prompts_batched(
                stage1_segments=[
                    {"segment_id": "seg_1", "scene": 10},
                    {"segment_id": "seg_2", "scene": 11},
                ],
                story_idea="Ravena drinks once from the well.",
                global_context={
                    "actors": [{"id": "ravena", "name": "Ravena"}],
                    "narrative_contract": {"one_shot_milestones": ["drink_silver_water"]},
                },
            )

    def test_invalid_enforcement_value_raises(self):
        """An invalid enforcement value raises ValueError at construction."""
        with self.assertRaisesRegex(
            ValueError,
            "semantic_enforcement must be 'warn' or 'block'",
        ):
            ConceptPromptBatcher(
                object(),
                batch_size=2,
                semantic_enforcement="invalid",
            )

    def test_sequence_aware_repair_expands_window(self):
        """When a conflict cites a prior segment, the repair window expands
        to include the contiguous sequence between them."""
        from feverslop.prompting.concept_prompt_batcher import (
            _sequence_aware_repair_ids,
        )
        invalid = [
            {"segment_id": "seg_3", "prior_segment_id": "seg_1", "reason": "x"},
        ]
        expected_ids = ["seg_1", "seg_2", "seg_3", "seg_4"]
        result = _sequence_aware_repair_ids([], invalid, expected_ids)
        # seg_1 is the cited predecessor and is not itself invalid, so it
        # is not included. The window is just seg_3.
        self.assertEqual(["seg_3"], result)

    def test_sequence_aware_repair_includes_invalid_predecessor(self):
        """When the cited predecessor is also invalid, it is included in
        the window."""
        from feverslop.prompting.concept_prompt_batcher import (
            _sequence_aware_repair_ids,
        )
        invalid = [
            {"segment_id": "seg_1", "reason": "x"},
            {"segment_id": "seg_3", "prior_segment_id": "seg_1", "reason": "y"},
        ]
        expected_ids = ["seg_1", "seg_2", "seg_3", "seg_4"]
        result = _sequence_aware_repair_ids([], invalid, expected_ids)
        # seg_1 is invalid, so it's included. The window spans seg_1 to seg_3.
        self.assertEqual(["seg_1", "seg_2", "seg_3"], result)

    def test_sequence_allocation_includes_window_structure(self):
        """The sequence allocation describes preparation, irreversible event,
        completion, and aftermath for the repair prompt."""
        from feverslop.prompting.concept_prompt_batcher import (
            _sequence_allocation,
        )
        invalid = [
            {
                "segment_id": "seg_2",
                "prior_segment_id": "seg_1",
                "reason": "milestone 'drink' repeats seg_1 without reset_events authorization",
            },
            {
                "segment_id": "seg_1",
                "reason": "x",
            },
        ]
        expected_ids = ["seg_1", "seg_2", "seg_3"]
        contract = {"one_shot_milestones": ["drink"]}
        allocation = _sequence_allocation(invalid, expected_ids, contract)
        self.assertIsNotNone(allocation)
        # The window spans the contiguous range of affected segments.
        self.assertEqual(["seg_1", "seg_2"], allocation["window"])
        self.assertEqual(["drink"], allocation["one_shot_milestones"])

    def test_sequence_allocation_prefers_original_over_duplicate(self):
        """Regression: when segment_033 repeats a milestone first accepted
        in segment_031, the allocation window includes both, and the
        irreversible event is segment_031 (the original), not segment_033
        (the duplicate)."""
        from feverslop.prompting.concept_prompt_batcher import (
            _sequence_allocation,
        )
        # Only segment_033 is invalid; segment_031 is valid.
        invalid = [
            {
                "segment_id": "segment_033",
                "prior_segment_id": "segment_031",
                "reason": "milestone 'drink' repeats segment_031 without reset_events authorization",
            },
        ]
        expected_ids = ["segment_031", "segment_032", "segment_033"]
        contract = {"one_shot_milestones": ["drink"]}
        allocation = _sequence_allocation(invalid, expected_ids, contract)
        self.assertIsNotNone(allocation)
        # The allocation window includes the valid cited predecessor.
        self.assertEqual(["segment_031", "segment_032", "segment_033"], allocation["window"])
        # The irreversible event is the original accepted occurrence.
        self.assertEqual(["segment_031"], allocation["irreversible_event"])
        # The duplicate is in the aftermath, not the event.
        self.assertIn("segment_033", allocation.get("aftermath", []))
        self.assertEqual(["drink"], allocation["one_shot_milestones"])

    def test_sequence_allocation_cross_batch_predecessor(self):
        """Regression: when a scene in batch N repeats a milestone first
        accepted in batch N-1, the allocation window uses the full ordered
        sequence (accepted + current batch) so the predecessor is found
        and classified as the irreversible event."""
        from feverslop.prompting.concept_prompt_batcher import (
            _sequence_allocation,
        )
        # segment_010 is from a previous batch (accepted);
        # segment_015 is in the current batch and repeats the milestone.
        invalid = [
            {
                "segment_id": "segment_015",
                "prior_segment_id": "segment_010",
                "reason": "milestone 'drink' repeats segment_010 without reset_events authorization",
            },
        ]
        # Full ordered sequence: accepted IDs followed by current batch IDs.
        full_sequence_ids = [
            "segment_010",  # accepted (previous batch)
            "segment_011",
            "segment_012",
            "segment_015",  # current batch (invalid)
            "segment_016",
        ]
        contract = {"one_shot_milestones": ["drink"]}
        allocation = _sequence_allocation(invalid, full_sequence_ids, contract)
        self.assertIsNotNone(allocation)
        # The window spans from the accepted predecessor through the duplicate.
        self.assertEqual(
            ["segment_010", "segment_011", "segment_012", "segment_015"],
            allocation["window"],
        )
        # The irreversible event is the original accepted occurrence.
        self.assertEqual(["segment_010"], allocation["irreversible_event"])
        # The duplicate is in the aftermath.
        self.assertIn("segment_015", allocation.get("aftermath", []))
        self.assertEqual(["drink"], allocation["one_shot_milestones"])


def _loc_concept(
    concept: str,
    *,
    location: str,
    cast_states: dict,
    story_beat: str = "beat_a",
    action: str = "act_a",
    action_phase: str = "started",
) -> dict:
    return {
        "concept": concept,
        "references": {"actor_ids": [], "location_id": location},
        "narrative": {
            "story_beat": story_beat,
            "objective": "objective",
            "action": action,
            "action_phase": action_phase,
            "milestones": [],
            "location": location,
            "cast_states": dict(cast_states),
            "props": {},
            "reset_events": [],
        },
    }


class ActorLocationCoercionTests(unittest.TestCase):
    def test_coerce_forces_disallowed_actor_to_absent(self):
        concepts = {
            "seg_1": _loc_concept(
                "Stranger at the mirror.",
                location="mirror_threshold",
                cast_states={"stranger_reflection": "present"},
            ),
            "seg_2": _loc_concept(
                "Stranger drifts into the fog.",
                location="the_void_fog",
                cast_states={"stranger_reflection": "present"},
                story_beat="beat_b",
                action="act_b",
            ),
        }
        contract = {
            "actor_allowed_locations": {"stranger_reflection": ["mirror_threshold"]},
        }
        coerced = _coerce_disallowed_actor_locations(concepts, contract)
        self.assertEqual(
            [{"segment_id": "seg_2", "actor": "stranger_reflection"}],
            coerced,
        )
        self.assertEqual(
            "absent",
            concepts["seg_2"]["narrative"]["cast_states"]["stranger_reflection"],
        )
        # The allowed occurrence is untouched.
        self.assertEqual(
            "present",
            concepts["seg_1"]["narrative"]["cast_states"]["stranger_reflection"],
        )

    def test_coerce_leaves_allowed_actor_untouched(self):
        concepts = {
            "seg_1": _loc_concept(
                "Stranger at the mirror.",
                location="mirror_threshold",
                cast_states={"stranger_reflection": "present"},
            ),
        }
        contract = {
            "actor_allowed_locations": {"stranger_reflection": ["mirror_threshold"]},
        }
        self.assertEqual([], _coerce_disallowed_actor_locations(concepts, contract))
        self.assertEqual(
            "present",
            concepts["seg_1"]["narrative"]["cast_states"]["stranger_reflection"],
        )

    def test_coerce_ignores_already_absent_actor(self):
        concepts = {
            "seg_1": _loc_concept(
                "Stranger gone from the fog.",
                location="the_void_fog",
                cast_states={"stranger_reflection": "absent"},
            ),
        }
        contract = {
            "actor_allowed_locations": {"stranger_reflection": ["mirror_threshold"]},
        }
        self.assertEqual([], _coerce_disallowed_actor_locations(concepts, contract))
        self.assertEqual(
            "absent",
            concepts["seg_1"]["narrative"]["cast_states"]["stranger_reflection"],
        )

    def test_coerce_no_contract_is_noop(self):
        concepts = {
            "seg_1": _loc_concept(
                "Stranger in the fog.",
                location="the_void_fog",
                cast_states={"stranger_reflection": "present"},
            ),
        }
        self.assertEqual([], _coerce_disallowed_actor_locations(concepts, {}))
        self.assertEqual(
            "present",
            concepts["seg_1"]["narrative"]["cast_states"]["stranger_reflection"],
        )

    def test_coerce_terminal_state_sets_required_state(self):
        concepts = {
            "seg_1": _loc_concept(
                "The end.",
                location="the_void_fog",
                cast_states={"lead_subject": "dissolving_silhouette"},
            ),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["spiritual_erasure_drift"]
        contract = {
            "terminal_states": {
                "lead_subject": {
                    "milestone": "spiritual_erasure_drift",
                    "state": "drifting_endlessly",
                },
            },
        }
        coerced = _coerce_terminal_states(concepts, contract)
        self.assertEqual(
            [{"segment_id": "seg_1", "actor": "lead_subject"}],
            coerced,
        )
        self.assertEqual(
            "drifting_endlessly",
            concepts["seg_1"]["narrative"]["cast_states"]["lead_subject"],
        )

    def test_coerce_terminal_state_leaves_reset_authorized_untouched(self):
        concepts = {
            "seg_1": _loc_concept(
                "The return.",
                location="the_void_fog",
                cast_states={"lead_subject": "corporeal"},
            ),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["spiritual_erasure_drift"]
        concepts["seg_1"]["narrative"]["causal_events"] = ["lead_subject_returns"]
        contract = {
            "terminal_states": {
                "lead_subject": {
                    "milestone": "spiritual_erasure_drift",
                    "state": "drifting_endlessly",
                    "reset_event": "lead_subject_returns",
                },
            },
        }
        self.assertEqual([], _coerce_terminal_states(concepts, contract))
        self.assertEqual(
            "corporeal",
            concepts["seg_1"]["narrative"]["cast_states"]["lead_subject"],
        )

    def test_coerce_terminal_state_no_contract_is_noop(self):
        concepts = {
            "seg_1": _loc_concept(
                "The end.",
                location="the_void_fog",
                cast_states={"lead_subject": "dissolving_silhouette"},
            ),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["spiritual_erasure_drift"]
        self.assertEqual([], _coerce_terminal_states(concepts, {}))
        self.assertEqual(
            "dissolving_silhouette",
            concepts["seg_1"]["narrative"]["cast_states"]["lead_subject"],
        )

    def test_reorder_moves_out_of_order_milestone(self):
        concepts = {
            "seg_1": _loc_concept(
                "Early beat.",
                location="dissolving_apartment",
                cast_states={},
            ),
            "seg_2": _loc_concept(
                "Jumping ahead.",
                location="dissolving_apartment",
                cast_states={},
            ),
            "seg_3": _loc_concept(
                "The predecessor.",
                location="mirror_threshold",
                cast_states={},
            ),
        }
        # seg_2 has the later milestone (rank 1) before seg_3 has the
        # predecessor (rank 0).
        concepts["seg_2"]["narrative"]["milestones"] = ["internal_void_discovery"]
        concepts["seg_3"]["narrative"]["milestones"] = ["reflection_stranger_encounter"]
        contract = {
            "milestone_order": [
                "reflection_stranger_encounter",
                "internal_void_discovery",
            ],
        }
        moved = _reorder_out_of_order_milestones(concepts, contract)
        self.assertEqual(
            1,
            len(moved),
        )
        # The out-of-order milestone is moved to after the predecessor.
        self.assertNotIn(
            "internal_void_discovery",
            concepts["seg_2"]["narrative"]["milestones"],
        )
        self.assertIn(
            "internal_void_discovery",
            concepts["seg_3"]["narrative"]["milestones"],
        )

    def test_reorder_leaves_in_order_milestones_untouched(self):
        concepts = {
            "seg_1": _loc_concept(
                "The predecessor.",
                location="mirror_threshold",
                cast_states={},
            ),
            "seg_2": _loc_concept(
                "The successor.",
                location="the_void_fog",
                cast_states={},
            ),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["reflection_stranger_encounter"]
        concepts["seg_2"]["narrative"]["milestones"] = ["internal_void_discovery"]
        contract = {
            "milestone_order": [
                "reflection_stranger_encounter",
                "internal_void_discovery",
            ],
        }
        self.assertEqual([], _reorder_out_of_order_milestones(concepts, contract))

    def test_reorder_no_contract_is_noop(self):
        concepts = {
            "seg_1": _loc_concept(
                "The successor.",
                location="the_void_fog",
                cast_states={},
            ),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["internal_void_discovery"]
        self.assertEqual([], _reorder_out_of_order_milestones(concepts, {}))

    def test_final_validation_coerces_disallowed_actor(self):
        concepts = {
            "seg_1": _loc_concept(
                "Stranger at the mirror.",
                location="mirror_threshold",
                cast_states={"stranger_reflection": "present"},
            ),
            "seg_2": _loc_concept(
                "Stranger drifts into the fog.",
                location="the_void_fog",
                cast_states={"stranger_reflection": "present"},
                story_beat="beat_b",
                action="act_b",
            ),
        }
        contract = {
            "actor_allowed_locations": {"stranger_reflection": ["mirror_threshold"]},
        }
        annotated = validate_and_annotate_concept_chronology(concepts, contract)
        self.assertEqual(
            "absent",
            concepts["seg_2"]["narrative"]["cast_states"]["stranger_reflection"],
        )
        self.assertEqual("accepted", annotated["seg_2"]["semantic_validation"]["outcome"])


class OneShotMilestoneDedupTests(unittest.TestCase):
    def _concepts(self, first_milestones, second_milestones, reset_events=None):
        first = semantic_concept(
            "Ravena raises the silver cup.",
            story_beat="raise_cup",
            action="raise_cup",
            action_phase="completed",
            milestone=first_milestones[0],
            prop_state="raised",
        )
        first["narrative"]["milestones"] = list(first_milestones)
        second = semantic_concept(
            "Ravena watches the cup drain.",
            story_beat="cup_drains",
            action="watch_cup_drain",
            action_phase="completed",
            milestone=second_milestones[0],
            prop_state="raised",
        )
        second["narrative"]["milestones"] = list(second_milestones)
        if reset_events is not None:
            second["narrative"]["reset_events"] = reset_events
        return {"seg_1": first, "seg_2": second}

    def test_dedup_removes_unauthorized_repeat_keeps_first(self):
        concepts = self._concepts(["cup_raised"], ["cup_raised"])
        removed = _dedup_one_shot_milestones(
            concepts, {"one_shot_milestones": ["cup_raised"]},
        )
        self.assertEqual(
            [{"segment_id": "seg_2", "milestone": "cup_raised"}],
            removed,
        )
        self.assertEqual(["cup_raised"], concepts["seg_1"]["narrative"]["milestones"])
        self.assertEqual([], concepts["seg_2"]["narrative"]["milestones"])

    def test_dedup_keeps_reset_authorized_repeat(self):
        concepts = self._concepts(
            ["cup_raised"], ["cup_raised"], reset_events=["cup_raised"],
        )
        removed = _dedup_one_shot_milestones(
            concepts, {"one_shot_milestones": ["cup_raised"]},
        )
        self.assertEqual([], removed)
        self.assertEqual(["cup_raised"], concepts["seg_2"]["narrative"]["milestones"])

    def test_dedup_leaves_non_one_shot_milestones_untouched(self):
        concepts = self._concepts(["chorus_refrain"], ["chorus_refrain"])
        removed = _dedup_one_shot_milestones(
            concepts, {"one_shot_milestones": ["cup_raised"]},
        )
        self.assertEqual([], removed)
        self.assertEqual(["chorus_refrain"], concepts["seg_2"]["narrative"]["milestones"])

    def test_dedup_no_one_shot_contract_is_noop(self):
        concepts = self._concepts(["cup_raised"], ["cup_raised"])
        removed = _dedup_one_shot_milestones(concepts, {})
        self.assertEqual([], removed)
        self.assertEqual(["cup_raised"], concepts["seg_2"]["narrative"]["milestones"])

    def test_final_validation_dedups_repeated_one_shot_milestone(self):
        concepts = self._concepts(["cup_raised"], ["cup_raised"])
        annotated = validate_and_annotate_concept_chronology(
            concepts, {"one_shot_milestones": ["cup_raised"]},
        )
        self.assertEqual([], concepts["seg_2"]["narrative"]["milestones"])
        self.assertEqual("accepted", annotated["seg_2"]["semantic_validation"]["outcome"])


def _loc_concept(
    beat: str,
    *,
    location: str = "",
    cast_states: dict[str, str] | None = None,
) -> dict:
    return {
        "concept": beat,
        "narrative": {
            "story_beat": beat,
            "objective": "test objective",
            "action": "test action",
            "action_phase": "test phase",
            "milestones": [],
            "location": location,
            "cast_states": cast_states if cast_states is not None else {},
            "props": {},
        },
    }


class OneShotMilestoneDedupTests(unittest.TestCase):
    def test_dedup_removes_unauthorized_repeat(self):
        concepts = {
            "seg_1": _loc_concept("First.", cast_states={}),
            "seg_2": _loc_concept("Repeat.", cast_states={}),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["drink"]
        concepts["seg_2"]["narrative"]["milestones"] = ["drink"]
        contract = {"one_shot_milestones": ["drink"]}
        removed = _dedup_one_shot_milestones(concepts, contract)
        self.assertEqual(
            [{"segment_id": "seg_2", "milestone": "drink"}],
            removed,
        )
        self.assertEqual([], concepts["seg_2"]["narrative"]["milestones"])
        self.assertEqual(["drink"], concepts["seg_1"]["narrative"]["milestones"])

    def test_dedup_keeps_reset_authorized_repeat(self):
        concepts = {
            "seg_1": _loc_concept("First.", cast_states={}),
            "seg_2": _loc_concept("Reset.", cast_states={}),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["drink"]
        concepts["seg_2"]["narrative"]["milestones"] = ["drink"]
        concepts["seg_2"]["narrative"]["causal_events"] = ["drink_reset"]
        contract = {"one_shot_milestones": ["drink"], "reset_events": ["drink_reset"]}
        self.assertEqual([], _dedup_one_shot_milestones(concepts, contract))
        self.assertEqual(["drink"], concepts["seg_2"]["narrative"]["milestones"])

    def test_dedup_no_contract_is_noop(self):
        concepts = {
            "seg_1": _loc_concept("First.", cast_states={}),
            "seg_2": _loc_concept("Repeat.", cast_states={}),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["drink"]
        concepts["seg_2"]["narrative"]["milestones"] = ["drink"]
        self.assertEqual([], _dedup_one_shot_milestones(concepts, {}))
        self.assertEqual(["drink"], concepts["seg_2"]["narrative"]["milestones"])


class ActorLocationCoercionTests(unittest.TestCase):
    def test_coerce_forces_disallowed_actor_to_absent(self):
        concepts = {
            "seg_1": _loc_concept(
                "At the fog.",
                location="the_void_fog",
                cast_states={"stranger_reflection": "present"},
            ),
        }
        contract = {
            "actor_allowed_locations": {
                "stranger_reflection": ["mirror_threshold"],
            },
        }
        coerced = _coerce_disallowed_actor_locations(concepts, contract)
        self.assertEqual(
            [{"segment_id": "seg_1", "actor": "stranger_reflection"}],
            coerced,
        )
        self.assertEqual(
            "absent",
            concepts["seg_1"]["narrative"]["cast_states"]["stranger_reflection"],
        )

    def test_coerce_leaves_allowed_actor_untouched(self):
        concepts = {
            "seg_1": _loc_concept(
                "At the threshold.",
                location="mirror_threshold",
                cast_states={"stranger_reflection": "present"},
            ),
        }
        contract = {
            "actor_allowed_locations": {
                "stranger_reflection": ["mirror_threshold"],
            },
        }
        self.assertEqual([], _coerce_disallowed_actor_locations(concepts, contract))
        self.assertEqual(
            "present",
            concepts["seg_1"]["narrative"]["cast_states"]["stranger_reflection"],
        )

    def test_coerce_ignores_already_absent_actor(self):
        concepts = {
            "seg_1": _loc_concept(
                "At the fog.",
                location="the_void_fog",
                cast_states={"stranger_reflection": "absent"},
            ),
        }
        contract = {
            "actor_allowed_locations": {
                "stranger_reflection": ["mirror_threshold"],
            },
        }
        self.assertEqual([], _coerce_disallowed_actor_locations(concepts, contract))

    def test_coerce_no_contract_is_noop(self):
        concepts = {
            "seg_1": _loc_concept(
                "At the fog.",
                location="the_void_fog",
                cast_states={"stranger_reflection": "present"},
            ),
        }
        self.assertEqual([], _coerce_disallowed_actor_locations(concepts, {}))
        self.assertEqual(
            "present",
            concepts["seg_1"]["narrative"]["cast_states"]["stranger_reflection"],
        )

    def test_final_validation_coerces_disallowed_actor(self):
        concepts = {
            "seg_1": _loc_concept(
                "At the fog.",
                location="the_void_fog",
                cast_states={"stranger_reflection": "present"},
            ),
        }
        contract = {
            "actor_allowed_locations": {
                "stranger_reflection": ["mirror_threshold"],
            },
        }
        annotated = validate_and_annotate_concept_chronology(concepts, contract)
        self.assertEqual(
            "absent",
            concepts["seg_1"]["narrative"]["cast_states"]["stranger_reflection"],
        )
        self.assertEqual("accepted", annotated["seg_1"]["semantic_validation"]["outcome"])


class TerminalStateCoercionTests(unittest.TestCase):
    def test_coerce_terminal_state_sets_required_state(self):
        concepts = {
            "seg_1": _loc_concept(
                "The end.",
                location="the_void_fog",
                cast_states={"lead_subject": "dissolving_silhouette"},
            ),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["spiritual_erasure_drift"]
        contract = {
            "terminal_states": {
                "lead_subject": {
                    "milestone": "spiritual_erasure_drift",
                    "state": "drifting_endlessly",
                },
            },
        }
        coerced = _coerce_terminal_states(concepts, contract)
        self.assertEqual(
            [{"segment_id": "seg_1", "actor": "lead_subject"}],
            coerced,
        )
        self.assertEqual(
            "drifting_endlessly",
            concepts["seg_1"]["narrative"]["cast_states"]["lead_subject"],
        )

    def test_coerce_terminal_state_leaves_reset_authorized_untouched(self):
        concepts = {
            "seg_1": _loc_concept(
                "The return.",
                location="the_void_fog",
                cast_states={"lead_subject": "corporeal"},
            ),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["spiritual_erasure_drift"]
        concepts["seg_1"]["narrative"]["causal_events"] = ["lead_subject_returns"]
        contract = {
            "terminal_states": {
                "lead_subject": {
                    "milestone": "spiritual_erasure_drift",
                    "state": "drifting_endlessly",
                    "reset_event": "lead_subject_returns",
                },
            },
        }
        self.assertEqual([], _coerce_terminal_states(concepts, contract))
        self.assertEqual(
            "corporeal",
            concepts["seg_1"]["narrative"]["cast_states"]["lead_subject"],
        )

    def test_coerce_terminal_state_no_contract_is_noop(self):
        concepts = {
            "seg_1": _loc_concept(
                "The end.",
                location="the_void_fog",
                cast_states={"lead_subject": "dissolving_silhouette"},
            ),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["spiritual_erasure_drift"]
        self.assertEqual([], _coerce_terminal_states(concepts, {}))
        self.assertEqual(
            "dissolving_silhouette",
            concepts["seg_1"]["narrative"]["cast_states"]["lead_subject"],
        )


class MilestoneReorderTests(unittest.TestCase):
    def test_reorder_moves_out_of_order_milestone(self):
        concepts = {
            "seg_1": _loc_concept("Early.", location="dissolving_apartment", cast_states={}),
            "seg_2": _loc_concept("Jumping.", location="dissolving_apartment", cast_states={}),
            "seg_3": _loc_concept("The predecessor.", location="mirror_threshold", cast_states={}),
        }
        concepts["seg_2"]["narrative"]["milestones"] = ["internal_void_discovery"]
        concepts["seg_3"]["narrative"]["milestones"] = ["reflection_stranger_encounter"]
        contract = {
            "milestone_order": [
                "reflection_stranger_encounter",
                "internal_void_discovery",
            ],
        }
        moved = _reorder_out_of_order_milestones(concepts, contract)
        self.assertEqual(1, len(moved))
        self.assertNotIn(
            "internal_void_discovery",
            concepts["seg_2"]["narrative"]["milestones"],
        )
        self.assertIn(
            "internal_void_discovery",
            concepts["seg_3"]["narrative"]["milestones"],
        )

    def test_reorder_leaves_in_order_milestones_untouched(self):
        concepts = {
            "seg_1": _loc_concept("The predecessor.", location="mirror_threshold", cast_states={}),
            "seg_2": _loc_concept("The successor.", location="the_void_fog", cast_states={}),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["reflection_stranger_encounter"]
        concepts["seg_2"]["narrative"]["milestones"] = ["internal_void_discovery"]
        contract = {
            "milestone_order": [
                "reflection_stranger_encounter",
                "internal_void_discovery",
            ],
        }
        self.assertEqual([], _reorder_out_of_order_milestones(concepts, contract))

    def test_reorder_no_contract_is_noop(self):
        concepts = {
            "seg_1": _loc_concept("The successor.", location="the_void_fog", cast_states={}),
        }
        concepts["seg_1"]["narrative"]["milestones"] = ["internal_void_discovery"]
        self.assertEqual([], _reorder_out_of_order_milestones(concepts, {}))


if __name__ == "__main__":
    unittest.main()
