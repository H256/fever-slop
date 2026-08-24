from __future__ import annotations

import copy
import unittest

from feverslop.application.effective_render_plan import (
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
