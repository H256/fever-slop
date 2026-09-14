import json
import unittest

from feverslop.prompting.concept_prompt_batcher import (
    ConceptPromptBatcher,
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
            "Concept batch: repairing 2 missing or invalid scene keys: seg_2, seg_3",
            progress,
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

if __name__ == "__main__":
    unittest.main()
