from __future__ import annotations

import copy
import unittest

from feverslop.application.effective_render_plan import (
    canonical_scene_dependencies,
    canonical_plan_revision,
    project_effective_plan,
    project_effective_scene,
)
from feverslop.domain.canonical_render_plan import (
    PromptRole,
    build_canonical_scene,
)
from feverslop.errors import FeverSlopDataError


def _canonical_scene(*, segment_id: str = "segment-a") -> dict:
    canonical = build_canonical_scene(
        segment_id=segment_id,
        generated_roles={
            PromptRole.Z_IMAGE: "generated still",
            PromptRole.LTX_BASE: "generated base",
            PromptRole.LTX_I2V: "generated i2v",
            PromptRole.LTX_MSR_GLOBAL: "generated msr global",
            PromptRole.LTX_MSR_RELAY: [{"prompt": "generated msr relay", "frame_start": 0, "frame_end": 47}],
            PromptRole.INGREDIENTS_GLOBAL: "generated ingredients global",
            PromptRole.INGREDIENTS_RELAY: [{"prompt": "generated ingredients relay", "frame_start": 0, "frame_end": 47}],
            PromptRole.H3_VIDEO: "generated h3",
            PromptRole.PERFORMANCE_TIMING: {"intent": "singing"},
        },
    )
    canonical["roles"][PromptRole.Z_IMAGE]["override"] = {"value": "human still"}
    canonical["roles"][PromptRole.LTX_I2V]["override"] = {"value": "human i2v"}
    canonical["roles"][PromptRole.LTX_MSR_GLOBAL]["override"] = {"value": "human msr global"}
    canonical["roles"][PromptRole.LTX_MSR_RELAY]["override"] = {
        "value": [{"prompt": "human msr relay", "frame_start": 0, "frame_end": 47}],
    }
    canonical["roles"][PromptRole.INGREDIENTS_GLOBAL]["override"] = {
        "value": "human ingredients global",
    }
    canonical["roles"][PromptRole.INGREDIENTS_RELAY]["override"] = {
        "value": [{"prompt": "human ingredients relay", "frame_start": 0, "frame_end": 47}],
    }
    canonical["roles"][PromptRole.H3_VIDEO]["override"] = {"value": "human h3"}
    return {"scene": 1, "canonical": canonical}


def _derived_scene(canonical: dict) -> dict:
    return {
        "scene": 1,
        "metadata": {"segment_id": canonical["segment_id"]},
        "canonical": copy.deepcopy(canonical),
        "z_image": {"prompt": "stale still"},
        "ltx": {
            "base_prompt": "stale base",
            "i2v_prompt_from_t2i": "stale i2v",
            "original_style_i2v_prompt": "stale i2v",
            "msr_global_prompt": "stale msr global",
            "msr_prompt_relay": [{"prompt": "stale msr relay", "frame_start": 0, "frame_end": 47}],
        },
        "ingredients": {"global_prompt": "stale ingredients global"},
        "h3": {"prompt": "stale h3"},
        "performance_timing": {"intent": "instrumental"},
    }


class EffectiveRenderPlanProjectionTests(unittest.TestCase):
    def test_projects_current_canonical_overrides_into_backend_specific_fields(self):
        base = _canonical_scene()
        stale = _derived_scene(base["canonical"])
        original = copy.deepcopy(stale)

        projected = project_effective_scene(
            stale,
            canonical_scene=base,
            source_revision="base-sha",
        )

        self.assertEqual("human still", projected["z_image"]["prompt"])
        self.assertEqual("generated base", projected["ltx"]["base_prompt"])
        self.assertEqual("human i2v", projected["ltx"]["i2v_prompt_from_t2i"])
        self.assertEqual("human i2v", projected["ltx"]["original_style_i2v_prompt"])
        self.assertEqual("human msr global", projected["ltx"]["msr_global_prompt"])
        self.assertEqual("human msr relay", projected["ltx"]["msr_prompt_relay"][0]["prompt"])
        self.assertEqual("human ingredients global", projected["ingredients"]["global_prompt"])
        self.assertEqual("human ingredients relay", projected["ltx"]["prompt_relay"][0]["prompt"])
        self.assertEqual("human h3", projected["h3"]["prompt"])
        self.assertEqual({"intent": "singing"}, projected["performance_timing"])
        self.assertEqual(stale, original)

    def test_projection_records_canonical_source_identity_and_revision(self):
        base = _canonical_scene()

        projected = project_effective_plan([_derived_scene(base["canonical"])], [base])

        provenance = projected[0]["canonical_projection"]
        self.assertEqual("feverslop.canonical-projection/v1", provenance["schema"])
        self.assertEqual(base["canonical"]["scene_id"], provenance["scene_id"])
        self.assertEqual("output/render/plans/base.json", provenance["source"])
        self.assertEqual(canonical_plan_revision([base]), provenance["source_revision"])
        self.assertEqual(64, len(provenance["source_revision"]))

    def test_revision_is_deterministic_for_equivalent_mapping_order(self):
        first = _canonical_scene()
        second = {"canonical": first["canonical"], "scene": 1}

        self.assertEqual(canonical_plan_revision([first]), canonical_plan_revision([second]))

    def test_revision_covers_authoritative_operational_and_reference_fields(self):
        base = _canonical_scene()
        changed = copy.deepcopy(base)
        changed["width"] = 1920
        changed["references"] = {"actor_ids": ["dancer"]}

        self.assertNotEqual(
            canonical_plan_revision([base]),
            canonical_plan_revision([changed]),
        )

    def test_scene_dependencies_are_deterministic_and_scene_local(self):
        base = _canonical_scene()
        projected = project_effective_scene(_derived_scene(base["canonical"]), canonical_scene=base)
        reordered = {key: projected[key] for key in reversed(projected)}

        first = canonical_scene_dependencies(projected, canonical_scene=base)
        second = canonical_scene_dependencies(reordered, canonical_scene={
            "canonical": base["canonical"],
            "scene": 1,
        })

        self.assertEqual(first, second)
        self.assertEqual(64, len(first.workflow_fingerprint))
        self.assertEqual(64, len(first.reference_fingerprint))

    def test_workflow_fingerprint_covers_prompt_timing_resolution_seed_and_relay(self):
        base = _canonical_scene()
        base.update({"width": 1280, "height": 720, "seed": 7})
        projected = project_effective_scene(_derived_scene(base["canonical"]), canonical_scene=base)
        original = canonical_scene_dependencies(projected, canonical_scene=base)

        mutations = []
        prompt = copy.deepcopy(base)
        prompt["canonical"]["roles"][PromptRole.LTX_BASE]["generated"]["value"] = "new prompt"
        mutations.append(prompt)
        timing = copy.deepcopy(base)
        timing["canonical"]["roles"][PromptRole.PERFORMANCE_TIMING]["generated"]["value"] = {"intent": "dance"}
        mutations.append(timing)
        resolution = copy.deepcopy(base)
        resolution["width"] = 1920
        mutations.append(resolution)
        seed = copy.deepcopy(base)
        seed["seed"] = 8
        mutations.append(seed)
        relay = copy.deepcopy(base)
        relay["canonical"]["roles"][PromptRole.LTX_MSR_RELAY]["override"]["value"] = [
            {"prompt": "new relay", "frame_start": 0, "frame_end": 47},
        ]
        mutations.append(relay)

        for changed in mutations:
            with self.subTest(changed=changed):
                current = project_effective_scene(projected, canonical_scene=changed)
                dependencies = canonical_scene_dependencies(current, canonical_scene=changed)
                self.assertNotEqual(original.workflow_fingerprint, dependencies.workflow_fingerprint)
                self.assertEqual(original.reference_fingerprint, dependencies.reference_fingerprint)

    def test_reference_bindings_have_an_independent_fingerprint(self):
        base = _canonical_scene()
        base["references"] = {"actor_ids": ["singer"], "location_id": "stage"}
        projected = project_effective_scene(_derived_scene(base["canonical"]), canonical_scene=base)
        original = canonical_scene_dependencies(projected, canonical_scene=base)
        changed = copy.deepcopy(base)
        changed["references"] = {"actor_ids": ["dancer"], "location_id": "roof"}

        current = project_effective_scene(projected, canonical_scene=changed)
        dependencies = canonical_scene_dependencies(current, canonical_scene=changed)

        self.assertEqual(original.workflow_fingerprint, dependencies.workflow_fingerprint)
        self.assertNotEqual(original.reference_fingerprint, dependencies.reference_fingerprint)

    def test_projection_applies_authoritative_operational_fields_and_reference_bindings(self):
        base = _canonical_scene()
        base.update({
            "fps": 30,
            "frame_count": 61,
            "width": 1920,
            "height": 1080,
            "seed": 17,
            "references": {"actor_ids": ["dancer"], "location_id": "roof"},
        })
        derived = _derived_scene(base["canonical"])
        derived.update({
            "fps": 24,
            "frame_count": 49,
            "width": 1280,
            "height": 720,
            "seed": 7,
            "references": {
                "actor_ids": ["singer"],
                "location_id": "stage",
                "actor_msr_paths": ["output/references/actors/singer.png"],
            },
        })

        projected = project_effective_scene(derived, canonical_scene=base)

        self.assertEqual((30, 61, 1920, 1080, 17), tuple(
            projected[key] for key in ("fps", "frame_count", "width", "height", "seed")
        ))
        self.assertEqual(["dancer"], projected["references"]["actor_ids"])
        self.assertEqual("roof", projected["references"]["location_id"])
        self.assertEqual(
            ["output/references/actors/singer.png"],
            projected["references"]["actor_msr_paths"],
        )

    def test_projection_persists_dependency_fingerprints(self):
        base = _canonical_scene()

        projected = project_effective_plan([_derived_scene(base["canonical"])], [base])[0]

        dependencies = projected["canonical_projection"]["dependencies"]
        self.assertEqual("feverslop.canonical-dependencies/v1", dependencies["schema"])
        self.assertEqual(base["canonical"]["scene_id"], dependencies["scene_id"])
        self.assertEqual(64, len(dependencies["workflow_fingerprint"]))
        self.assertEqual(64, len(dependencies["reference_fingerprint"]))

    def test_legacy_scene_passes_through_without_projection_metadata(self):
        legacy = {"scene": 1, "z_image": {"prompt": "legacy"}, "ltx": {"base_prompt": "legacy video"}}

        self.assertEqual(legacy, project_effective_scene(legacy))
        self.assertEqual([legacy], project_effective_plan([legacy], []))

    def test_duplicate_canonical_identity_is_rejected(self):
        base = _canonical_scene()

        with self.assertRaisesRegex(FeverSlopDataError, "duplicate canonical scene_id"):
            project_effective_plan(
                [_derived_scene(base["canonical"])],
                [base, copy.deepcopy(base)],
            )


if __name__ == "__main__":
    unittest.main()
