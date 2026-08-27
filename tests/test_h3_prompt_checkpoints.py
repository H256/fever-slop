from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.h3_prompt_checkpoints import H3PromptCheckpointStore
from feverslop.domain.canonical_render_plan import (
    PromptRole,
    build_canonical_scene,
    stable_scene_id,
)
from feverslop.domain.h3_prompt_checkpoint import (
    H3_CHECKPOINT_SCHEMA,
    H3PromptCheckpointInput,
)
from feverslop.errors import FeverSlopDataError


class H3PromptCheckpointStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project = Path(self.temp_dir.name)
        self.store = H3PromptCheckpointStore(self.project)

    def request(self, **changes) -> H3PromptCheckpointInput:
        values = {
            "scene_number": 1,
            "segment_id": "segment-a",
            "segment": {"scene": 1, "segment_id": "segment-a", "type": "vocals"},
            "concept": "A singer turns toward camera",
            "scene_details": {"camera_motion": "slow push"},
            "global_context": {"style": "cinematic"},
            "mode": "r2v",
            "video_type": "music_video",
            "audio_paths": {},
            "generator_revision": {"contract": 1, "model": "test-model", "guide": "guide-v1"},
        }
        values.update(changes)
        return H3PromptCheckpointInput(**values)

    def test_save_writes_complete_checkpoint_and_load_reuses_matching_input(self):
        request = self.request()
        result = {
            "prompt": "final judged prompt",
            "references": [],
            "prompt_judge": {"verdict": "good", "issues": []},
            "prompt_judge_attempts": [{"verdict": "bad"}, {"verdict": "good"}],
        }

        saved = self.store.save(request, result)
        payload = json.loads(saved.path.read_text(encoding="utf-8"))
        reused = self.store.load(request)

        self.assertEqual(H3_CHECKPOINT_SCHEMA, payload["schema"])
        self.assertEqual(1, payload["scene"])
        self.assertEqual(stable_scene_id("segment-a"), payload["scene_id"])
        self.assertEqual("segment-a", payload["segment_id"])
        self.assertEqual("good", payload["status"])
        self.assertTrue(payload["input_fingerprint"].startswith("sha256:"))
        self.assertEqual(result, payload["generated"])
        self.assertEqual(result["prompt_judge"], payload["judge"])
        self.assertEqual(result["prompt_judge_attempts"], payload["judge_attempts"])
        self.assertEqual("dspy-h3-prompt-builder", payload["provenance"]["source"])
        self.assertIsNotNone(reused)
        self.assertEqual(result, reused.generated)

    def test_save_persists_structured_payload_and_compiler_fingerprints(self):
        result = {
            "prompt": "compiled prompt",
            "prompt_provenance": {"compiler": "deterministic_h3_compiler", "compiler_version": 1},
            "creative_sections": {"shots": [{"shot_id": "shot-1", "visible_action": "turn"}]},
            "locked_facts": {"scene_id": "segment-a", "facts": [{"key": "wardrobe", "value": "cloak"}]},
        }
        saved = self.store.save(self.request(), result)
        provenance = json.loads(saved.path.read_text(encoding="utf-8"))["provenance"]
        self.assertEqual("deterministic_h3_compiler", provenance["compiler"])
        self.assertTrue(provenance["creative_sections_sha256"].startswith("sha256:"))
        self.assertTrue(provenance["locked_facts_sha256"].startswith("sha256:"))

    def test_save_reads_production_sections_shape_for_structured_fingerprints(self):
        result = {
            "prompt": "compiled prompt",
            "prompt_provenance": {"compiler": "deterministic_h3_compiler", "compiler_version": 2},
            "sections": {
                "h3_sections": {"subject": "hero"},
                "shots": [{"shot_id": "shot-1", "visible_action": "turn"}],
                "facts": {"scene_id": "segment-a", "facts": [{"key": "wardrobe", "value": "cloak"}]},
                "shot_windows": {"shot-1": [0, 12]},
            },
        }
        payload = json.loads(self.store.save(self.request(), result).path.read_text(encoding="utf-8"))

        self.assertEqual(2, payload["provenance"]["compiler_version"])
        self.assertTrue(payload["provenance"]["creative_sections_sha256"].startswith("sha256:"))
        self.assertTrue(payload["provenance"]["locked_facts_sha256"].startswith("sha256:"))

    def test_stage_fingerprints_identify_only_changed_checkpoint_inputs(self):
        request = self.request()
        saved = self.store.save(request, {"prompt": "cached"})

        self.assertEqual(frozenset(), self.store.invalidated_stages(request, saved))
        self.assertEqual(
            frozenset({"creative"}),
            self.store.invalidated_stages(
                self.request(concept="A different action"),
                saved,
            ),
        )
        self.assertEqual(
            frozenset({"locked_facts"}),
            self.store.invalidated_stages(
                self.request(segment={**request.segment, "locked_facts": {"wardrobe": "red"}}),
                saved,
            ),
        )
        self.assertEqual(
            frozenset({"compiler"}),
            self.store.invalidated_stages(
                self.request(generator_revision={"contract": 2}),
                saved,
            ),
        )

    def test_exhausted_bad_and_unjudged_results_have_distinct_statuses(self):
        bad = self.store.save(
            self.request(),
            {"prompt": "rejected prompt", "prompt_judge": {"verdict": "bad"}},
        )
        self.assertEqual("bad_exhausted", bad.status)

        unjudged = self.store.save(
            self.request(scene_number=2, segment_id="segment-b", segment={"scene": 2, "segment_id": "segment-b"}),
            {"prompt": "fallback prompt"},
        )
        self.assertEqual("unjudged", unjudged.status)

    def test_load_does_not_reuse_rejected_or_unjudged_checkpoints(self):
        request = self.request()
        self.store.save(request, {"prompt": "rejected", "prompt_judge": {"verdict": "bad"}})
        self.assertIsNone(self.store.load(request))

        request = self.request(scene_number=2, segment_id="segment-b", segment={"scene": 2, "segment_id": "segment-b"})
        self.store.save(request, {"prompt": "unjudged"})
        self.assertIsNone(self.store.load(request))

    def test_changed_input_or_generator_revision_is_not_reused(self):
        request = self.request()
        self.store.save(request, {"prompt": "cached", "prompt_judge": {"verdict": "good"}})

        changed_concept = self.request(concept="A different action")
        changed_revision = self.request(
            generator_revision={"contract": 1, "model": "test-model", "guide": "guide-v2"},
        )

        self.assertIsNone(self.store.load(changed_concept))
        self.assertIsNone(self.store.load(changed_revision))

    def test_model_or_transport_change_does_not_invalidate_checkpoint(self):
        request = self.request(
            generator_revision={
                "contract": 1,
                "model": "remote-model",
                "base_url": "https://remote.example/v1",
                "guide": "guide-v1",
            },
        )
        self.store.save(request, {"prompt": "cached", "prompt_judge": {"verdict": "good"}})

        switched = self.request(
            generator_revision={
                "contract": 1,
                "model": "local-model",
                "base_url": "http://localhost:1919/v1",
                "guide": "guide-v1",
            },
        )

        self.assertIsNotNone(self.store.load(switched))

    def test_scene_number_cannot_reuse_another_canonical_identity(self):
        self.store.save(
            self.request(),
            {"prompt": "cached", "prompt_judge": {"verdict": "good"}},
        )

        replacement = self.request(
            segment_id="segment-replacement",
            segment={"scene": 1, "segment_id": "segment-replacement"},
        )

        self.assertIsNone(self.store.load(replacement))

    def test_scene_local_prompt_inputs_invalidate_only_the_matching_checkpoint(self):
        base = self.request(segment={
            "scene": 1,
            "segment_id": "segment-a",
            "ltx": {"prompt_relay": [{"state": "singing", "prompt": "old relay"}]},
            "subject_directives": {"subjects": [{"id": "hero", "action": "stands"}]},
        })
        self.store.save(base, {"prompt": "cached"})

        variants = (
            self.request(segment={**base.segment, "ltx": {"prompt_relay": [{"state": "singing", "prompt": "new relay"}]}}),
            self.request(segment={**base.segment, "subject_directives": {"subjects": [{"id": "hero", "action": "runs"}]}}),
            self.request(scene_details={"camera_motion": "handheld"}),
            self.request(global_context={"style": "documentary"}),
        )

        for changed in variants:
            with self.subTest(changed=changed):
                self.assertIsNone(self.store.load(changed))

    def test_reference_file_content_change_invalidates_cached_fingerprint(self):
        reference = self.project / "reference.png"
        reference.write_bytes(b"version one")
        request = self.request(segment={
            "scene": 1,
            "segment_id": "segment-a",
            "references": {"reference_image_paths": [reference]},
        })
        self.store.save(request, {"prompt": "cached"})

        reference.write_bytes(b"version two is different")

        self.assertIsNone(self.store.load(request))

    def test_save_updates_canonical_generated_h3_without_touching_human_override(self):
        base = self.project / "output/render/plans/base.json"
        base.parent.mkdir(parents=True)
        canonical = build_canonical_scene(
            segment_id="segment-a",
            generated_roles={PromptRole.H3_VIDEO: "old generated"},
        )
        canonical["roles"][PromptRole.H3_VIDEO]["override"] = {
            "value": "human approved",
            "provenance": {"source": "human", "note": "keep exactly"},
        }
        base.write_text(json.dumps([{"scene": 1, "canonical": canonical}]), encoding="utf-8")
        expected_override = json.loads(json.dumps(canonical["roles"][PromptRole.H3_VIDEO]["override"]))

        saved = self.store.save(self.request(), {"prompt": "new judged prompt"})

        role = json.loads(base.read_text(encoding="utf-8"))[0]["canonical"]["roles"][PromptRole.H3_VIDEO]
        self.assertEqual("new judged prompt", role["generated"]["value"])
        self.assertEqual(expected_override, role["override"])
        self.assertEqual(saved.input_fingerprint, role["generated"]["provenance"]["input_fingerprint"])

    def test_reuse_syncs_checkpoint_created_before_canonical_base_existed(self):
        request = self.request()
        self.store.save(request, {"prompt": "checkpoint prompt", "prompt_judge": {"verdict": "good"}})
        base = self.project / "output/render/plans/base.json"
        base.parent.mkdir(parents=True)
        base.write_text(json.dumps([{
            "scene": 1,
            "canonical": build_canonical_scene(
                segment_id="segment-a",
                generated_roles={PromptRole.H3_VIDEO: "later base value"},
            ),
        }]), encoding="utf-8")

        self.store.load(request)

        role = json.loads(base.read_text(encoding="utf-8"))[0]["canonical"]["roles"][PromptRole.H3_VIDEO]
        self.assertEqual("checkpoint prompt", role["generated"]["value"])

    def test_reporting_names_scene_verdict_status_and_path_without_prompt_body(self):
        messages = []

        class Reporter:
            def message(self, message):
                messages.append(message)

        store = H3PromptCheckpointStore(self.project, reporter=Reporter())
        request = self.request()
        store.save(request, {
            "prompt": "SECRET PROMPT BODY",
            "prompt_judge": {"verdict": "good"},
        })
        store.load(request)

        combined = "\n".join(messages)
        self.assertIn("scene 1", combined)
        self.assertIn("judge GOOD", combined)
        self.assertIn("status good", combined)
        self.assertIn("h3_prompt.json", combined)
        self.assertIn("generated", combined)
        self.assertIn("reused", combined)
        self.assertNotIn("SECRET PROMPT BODY", combined)

    def test_canonical_sync_rejects_duplicate_scene_identity(self):
        base = self.project / "output/render/plans/base.json"
        base.parent.mkdir(parents=True)
        canonical = build_canonical_scene(
            segment_id="segment-a",
            generated_roles={PromptRole.H3_VIDEO: "old"},
        )
        base.write_text(json.dumps([
            {"scene": 1, "canonical": canonical},
            {"scene": 2, "canonical": json.loads(json.dumps(canonical))},
        ]), encoding="utf-8")

        with self.assertRaisesRegex(FeverSlopDataError, "duplicate canonical scene_id"):
            self.store.save(self.request(), {"prompt": "new"})


if __name__ == "__main__":
    unittest.main()
